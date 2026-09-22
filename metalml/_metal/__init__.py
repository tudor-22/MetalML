"""Native Metal runtime. No GPU handles are stored on fitted estimators."""

import platform
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._inference import InferenceRuntime
from ._session import BufferPool, Session


class MetalUnavailableError(RuntimeError):
    pass


class MetalRuntime(InferenceRuntime):
    def __init__(self):
        if platform.system() != "Darwin":
            raise MetalUnavailableError("Metal requires macOS")
        try:
            import Metal
            import objc
        except ImportError as exc:
            raise MetalUnavailableError("Install pyobjc-framework-Metal") from exc
        self.metal, self.objc = Metal, objc
        try:
            import MetalPerformanceShaders as mps
        except ImportError:
            mps = None
        self.mps = mps
        devices = list(Metal.MTLCopyAllDevices())
        if not devices:
            raise MetalUnavailableError("No Metal compute device found")
        # Prefer a high-performance discrete GPU on Intel Macs.
        self.device = min(devices, key=lambda d: bool(d.isLowPower()))
        self.name = str(self.device.name())
        self.queue = self.device.newCommandQueue()
        options = Metal.MTLCompileOptions.new()
        options.setFastMathEnabled_(False)
        source = Path(__file__).with_name("kernels.metal").read_text()
        self.library, error = self.device.newLibraryWithSource_options_error_(source, options, None)
        if self.library is None:
            raise RuntimeError(f"Metal kernel compilation failed: {error}")
        self.pipelines = {}
        self.lock = threading.RLock()
        self.dispatches = 0
        self.submissions = 0
        self.uploaded_bytes = 0
        self.pool = BufferPool(self)
        self.private_pool = BufferPool(self, private=True)

    def pipeline(self, name):
        if name not in self.pipelines:
            function = self.library.newFunctionWithName_(name)
            pipeline, error = self.device.newComputePipelineStateWithFunction_error_(function, None)
            if pipeline is None:
                raise RuntimeError(f"Metal pipeline {name}: {error}")
            self.pipelines[name] = pipeline
        return self.pipelines[name]

    def run(self, name, inputs, outputs, params, grid):
        with Session(self) as session:
            buffers = [session.upload(x) for x in inputs]
            results = [session.empty(shape, dtype) for shape, dtype in outputs]
            command = session.command()
            session.encode(command, name, buffers + results, params, grid)
            session.submit(command)
            return [buf.read() for buf in results]

    def stats(self, x):
        n, d = x.shape
        with Session(self) as session:
            xb, result = session.upload(x), session.empty((4, d))
            command = session.command()
            dispatches = self._encode_stats(session, command, xb, result, (n, d))
            session.submit(command, dispatches=dispatches)
            stats = result.read()
        return (
            stats[0].astype(float) + x[0].astype(float),
            stats[1].astype(float),
            stats[2],
            stats[3],
        )

    def _encode_stats(self, session, command, xb, result, shape):
        n, d = shape
        chunks = (n + 255) // 256
        if n >= 1024 and chunks * 4 * d * 4 <= 32 * 1024 * 1024:
            partial = session.empty((chunks, 4, d))
            session.encode(command, "stats_partial", [xb, partial], shape, (d, chunks))
            session.encode(command, "stats_finish", [partial, result], shape, (d,))
            return 2
        session.encode(command, "column_stats", [xb, result], shape, (d,))
        return 1

    def standardize(self, x, mean, scale):
        high = np.asarray(mean, np.float32)
        low = np.asarray(mean - high, np.float32)
        return self.run(
            "standardize",
            [x, high, low, np.asarray(scale, np.float32)],
            [(x.shape, np.float32)],
            x.shape,
            (x.size,),
        )[0]

    def affine(self, x, scale, bias):
        return self.run(
            "affine",
            [x, np.asarray(scale, np.float32), np.asarray(bias, np.float32)],
            [(x.shape, np.float32)],
            x.shape,
            (x.size,),
        )[0]

    def normalize(self, x, norm):
        return self.run(
            "normalize_rows",
            [x],
            [(x.shape, np.float32)],
            [*x.shape, {"max": 0, "l1": 1, "l2": 2}[norm]],
            (len(x),),
        )[0]

    def matmul(self, a, b):
        m, k = a.shape
        if b.shape[0] != k:
            raise ValueError("Incompatible matrix shapes")
        n = b.shape[1]
        if self.mps is not None:
            return self._mps_matmul(a, b)
        return self.run("matmul", [a, b], [((m, n), np.float32)], [m, n, k], (n, m))[0]

    @lru_cache(maxsize=32)  # noqa: B019 - the runtime is a process-lifetime singleton
    def _matrix_kernel(self, m, n, k, transpose_left=False):
        return self.mps.MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
            self.device, transpose_left, False, m, n, k, 1.0, 0.0
        )

    def _mps_matmul(self, a, b):
        with Session(self) as session:
            left, right = session.upload(a), session.upload(b)
            result = session.empty((a.shape[0], b.shape[1]))
            command = session.command()
            session.matmul(command, left, right, result)
            session.submit(command)
            return result.read()

    def _center(self, session, command, buffer, mean):
        if mean is None:
            return 0
        high = np.asarray(mean, np.float32).reshape(-1)
        low = np.asarray(np.asarray(mean).reshape(-1) - high, np.float32)
        constants = [session.upload(a) for a in (high, low, np.ones_like(high))]
        session.encode(
            command,
            "standardize",
            [buffer, *constants, buffer],
            buffer.shape,
            (int(np.prod(buffer.shape)),),
        )
        return 1

    def ridge_products(self, x, y, *, x_mean=None, y_mean=None):
        """Upload X once; transpose in MPS and batch both sufficient statistics."""
        if x.shape[1] <= 512 and x.shape[0] >= 1024:
            return self._split_ridge_products(x, y, x_mean=x_mean, y_mean=y_mean)
        if self.mps is None:
            if x_mean is not None:
                x = np.asarray(x - x_mean, np.float32)
            if y_mean is not None:
                y = np.asarray(y - y_mean, np.float32)
            return self.matmul(x.T, x), self.matmul(x.T, y)
        with Session(self) as session:
            xb, yb = session.upload(x), session.upload(y)
            gram = session.empty((x.shape[1], x.shape[1]))
            cross = session.empty((x.shape[1], y.shape[1]))
            command = session.command()
            if len(x) >= 1024:
                xb = session.private_copy(command, xb)
                yb = session.private_copy(command, yb)
            centered = self._center(session, command, xb, x_mean) + self._center(
                session, command, yb, y_mean
            )
            session.matmul(command, xb, xb, gram, transpose_left=True)
            session.matmul(command, xb, yb, cross, transpose_left=True)
            session.submit(command, dispatches=2 + centered)
            return gram.read(), cross.read()

    def covariance_products(self, x, mean):
        if (len(x) >= 1024 and x.shape[1] <= 512) or self.mps is None:
            return self._split_ridge_products(x, None, x_mean=mean)[0]
        with Session(self) as session:
            source = session.upload(x)
            result = session.empty((x.shape[1], x.shape[1]))
            command = session.command()
            xb = session.private_copy(command, source) if len(x) >= 1024 else source
            self._center(session, command, xb, mean)
            session.matmul(command, xb, xb, result, transpose_left=True)
            session.submit(command, dispatches=2)
            return result.read()

    def _split_ridge_products(self, x, y, *, x_mean=None, y_mean=None):
        n, d = x.shape
        targets = 0 if y is None else y.shape[1]
        max_chunks = max(1, (32 * 1024 * 1024) // (d * (d + targets) * 4))
        block = max(1024, ((n + max_chunks - 1) // max_chunks + 15) // 16 * 16)
        chunks = (n + block - 1) // block
        with Session(self) as session:
            xs = session.upload(x)
            ys = session.upload(y) if y is not None else None
            partial = session.private_empty(chunks * d * (d + targets) * 4)
            result = session.empty((d, d + targets))
            command = session.command()
            xb = session.private_copy(command, xs)
            yb = session.private_copy(command, ys) if ys is not None else xb
            means = xb
            if x_mean is not None or y_mean is not None:
                xm = np.zeros(d) if x_mean is None else np.asarray(x_mean).reshape(-1)
                ym = np.zeros(targets) if y_mean is None else np.asarray(y_mean).reshape(-1)
                xh, yh = xm.astype(np.float32), ym.astype(np.float32)
                means = session.upload(
                    np.concatenate([xh, xm - xh, yh, ym - yh]).astype(np.float32)
                )
            params = [n, d, targets, block, int(x_mean is not None), int(y_mean is not None)]
            session.encode(
                command, "gram_centered", [xb, yb, means, partial], params, (d + targets, d, chunks)
            )
            session.encode(
                command, "gram_symmetric_finish", [partial, result], params, (d * (d + targets),)
            )
            session.submit(command, dispatches=2)
            values = result.read()
            return values[:, :d], values[:, d:]

    def scale_fit_transform(self, x, with_mean, with_std):
        with Session(self) as session:
            xb = session.upload(x)
            stats, result = session.empty((4, x.shape[1])), session.empty(x.shape)
            command = session.command()
            dispatches = self._encode_stats(session, command, xb, stats, x.shape)
            session.encode(
                command,
                "standardize_fitted",
                [xb, stats, result],
                [*x.shape, int(with_mean), int(with_std)],
                (x.size,),
            )
            session.submit(command, dispatches=dispatches + 1)
            values = stats.read()
            return (
                values[0].astype(float) + x[0].astype(float),
                values[1].astype(float),
                result.read(),
            )

    def distances(self, x, centers):
        n, d = x.shape
        k = len(centers)
        return self.run(
            "distances_coalesced", [x, centers], [((n, k), np.float32)], [n, k, d], (n * k,)
        )[0]

    def assign(self, x, centers):
        n, d = x.shape
        with Session(self) as session:
            xb, cb = session.upload(x), session.upload(centers)
            labels, errors = session.empty((n,), np.int32), session.empty((n,))
            status = session.upload(np.zeros(1, np.uint32))
            command = session.command()
            session.encode(
                command,
                "assign_coalesced",
                [xb, cb, labels, errors, status],
                [n, len(centers), d],
                (n,),
            )
            session.submit(command)
            return labels.read(), errors.read()

    def kmeans_session(self, x, weights, n_clusters):
        from ._kmeans import KMeansSession

        return KMeansSession(self, x, weights, n_clusters)

    def logistic_irls(self, x, y, alpha, fit_intercept=True, max_iter=100, tol=1e-4):
        """Train binary L2 logistic regression with Newton/IRLS on the GPU."""
        from ._logistic import LogisticIRLS

        with LogisticIRLS(self, x, y, alpha, fit_intercept, max_iter, tol) as session:
            return session.run()

    def update(self, x, labels, weights, centers):
        n, d = x.shape
        return self.run(
            "update_centers",
            [x, labels, weights, centers],
            [(centers.shape, np.float32)],
            [n, len(centers), d],
            (centers.size,),
        )[0]


@lru_cache(maxsize=1)
def probe():
    try:
        return MetalRuntime(), None
    except (MetalUnavailableError, OSError) as exc:
        return None, str(exc)


def backend_info():
    runtime, reason = probe()
    return {
        "available": runtime is not None,
        "device": runtime.name if runtime else None,
        "reason": reason,
        "dispatches": runtime.dispatches if runtime else 0,
        "submissions": runtime.submissions if runtime else 0,
        "uploaded_bytes": runtime.uploaded_bytes if runtime else 0,
        "buffer_allocations": runtime.pool.allocations if runtime else 0,
        "pooled_bytes": runtime.pool.cached_bytes if runtime else 0,
        "private_buffer_allocations": runtime.private_pool.allocations if runtime else 0,
        "private_pooled_bytes": runtime.private_pool.cached_bytes if runtime else 0,
        "precision": "float32",
        "runtime": "native Metal via PyObjC",
        "matrix_backend": "Metal Performance Shaders"
        if runtime and runtime.mps
        else "custom Metal",
    }
