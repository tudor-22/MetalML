"""Synchronous compute sessions with bounded reuse of storage, never input values."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np


class BufferPool:
    def __init__(self, runtime, limit=64 * 1024 * 1024, *, private=False):
        self.runtime = runtime
        self.limit = limit
        self.cached_bytes = 0
        self.allocations = 0
        self.private = private
        self._free = OrderedDict()
        self._serial = 0

    def acquire(self, size):
        capacity = ((size + 4095) // 4096) * 4096
        for key, block in self._free.items():
            if block[2] == capacity:
                self._free.pop(key)
                self.cached_bytes -= capacity
                return block
        buf = self.runtime.device.newBufferWithLength_options_(
            capacity,
            self.runtime.metal.MTLResourceStorageModePrivate
            if self.private
            else self.runtime.metal.MTLResourceStorageModeShared,
        )
        if buf is None:
            raise MemoryError(f"Metal could not allocate {capacity} bytes")
        self.allocations += 1
        return buf, None if self.private else buf.contents().as_buffer(capacity), capacity

    def release(self, block):
        if block[2] > self.limit:
            return
        while self.cached_bytes + block[2] > self.limit:
            _, oldest = self._free.popitem(last=False)
            self.cached_bytes -= oldest[2]
        self._serial += 1
        self._free[self._serial] = block
        self.cached_bytes += block[2]


class Buffer:
    def __init__(self, block, shape, dtype, runtime):
        self.runtime = runtime
        self.shape = tuple(shape)
        self.metal = block[0]
        self.array = np.frombuffer(block[1], dtype=dtype, count=int(np.prod(shape))).reshape(shape)

    def write(self, value):
        np.copyto(self.array, value)
        self.runtime.uploaded_bytes += self.array.nbytes

    def read(self):
        # Never expose pooled storage to callers: the next operation may reuse it.
        return self.array.copy()


class Session:
    def __init__(self, runtime):
        self.runtime = runtime
        self.blocks = []
        self.pending = []
        self.private_buffers = []
        self.private_blocks = []

    def __enter__(self):
        self.runtime.lock.acquire()
        self.autorelease = self.runtime.objc.autorelease_pool()
        self.autorelease.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            # Even an interrupted wait must not recycle memory still in GPU use.
            for command in self.pending:
                command.waitUntilCompleted()
            for block in self.blocks:
                self.runtime.pool.release(block)
            for block in self.private_blocks:
                self.runtime.private_pool.release(block)
        finally:
            self.autorelease.__exit__(exc_type, exc, traceback)
            self.runtime.lock.release()

    def empty(self, shape, dtype=np.float32):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if size <= 0:
            raise ValueError("Metal buffer must be nonempty")
        block = self.runtime.pool.acquire(size)
        self.blocks.append(block)
        return Buffer(block, shape, dtype, self.runtime)

    def upload(self, value):
        a = np.asarray(value)
        buf = self.empty(a.shape, a.dtype)
        buf.write(a)
        return buf

    def private_copy(self, command, source, destination=None):
        """Stage random-access data into device-local memory on discrete GPUs."""
        size = source.array.nbytes
        if destination is None:
            destination = self.private_empty(size)
            destination.shape = source.shape
        encoder = command.blitCommandEncoder()
        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            source.metal, 0, destination.metal, 0, size
        )
        encoder.endEncoding()
        return destination

    def private_empty(self, size):
        block = self.runtime.private_pool.acquire(size)
        self.private_blocks.append(block)
        destination = SimpleNamespace(metal=block[0])
        self.private_buffers.append(destination)
        return destination

    def command(self):
        return self.runtime.queue.commandBuffer()

    def encode(self, command, name, buffers, params, grid):
        pipeline = self.runtime.pipeline(name)
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf.metal, 0, index)
        p = np.asarray(params, np.uint32).tobytes()
        encoder.setBytes_length_atIndex_(p, len(p), len(buffers))
        if name == "column_stats":
            group, counts = (256, 1, 1), (grid[0], 1, 1)
        elif name == "stats_partial":
            group, counts = (16, 16, 1), ((grid[0] + 15) // 16, grid[1], 1)
        elif name in {"gram_partial", "gram_centered"}:
            group, counts = (16, 16, 1), ((grid[0] + 15) // 16, (grid[1] + 15) // 16, grid[2])
        elif name in {"matmul", "distances_tiled", "projection_tiled"}:
            group, counts = (16, 16, 1), ((grid[0] + 15) // 16, (grid[1] + 15) // 16, 1)
        elif name in {
            "assign_coalesced",
            "distances_coalesced",
            "linear_small",
            "gaussian_scores",
            "svm_kernel",
            "knn_topk",
        }:
            width = pipeline.threadExecutionWidth()
            group, counts = (256, 1, 1), ((grid[0] * width + 255) // 256, 1, 1)
        else:
            width = min(256, pipeline.maxTotalThreadsPerThreadgroup())
            group, counts = (width, 1, 1), ((grid[0] + width - 1) // width, 1, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(counts, group)
        encoder.endEncoding()

    def submit(self, command, dispatches=1):
        self.pending.append(command)
        command.commit()
        command.waitUntilCompleted()
        self.pending.remove(command)
        if command.error() is not None:
            raise RuntimeError(f"Metal execution failed: {command.error()}")
        self.runtime.dispatches += dispatches
        self.runtime.submissions += 1

    def matrix(self, buf):
        rows, cols = buf.shape
        descriptor = self.runtime.mps.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            rows, cols, cols * 4, self.runtime.mps.MPSDataTypeFloat32
        )
        return self.runtime.mps.MPSMatrix.alloc().initWithBuffer_descriptor_(buf.metal, descriptor)

    def matmul(self, command, left, right, result, transpose_left=False):
        m, k = left.shape[::-1] if transpose_left else left.shape
        n = right.shape[1]
        matrices = [self.matrix(buf) for buf in (left, right, result)]
        kernel = self.runtime._matrix_kernel(m, n, k, transpose_left)
        kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(command, *matrices)
