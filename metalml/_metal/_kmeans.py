"""Lloyd iterations reuse device data and return only compact state per iteration."""

import numpy as np

from ._session import Session


class KMeansFallback(Exception):
    """Input needs sklearn's exceptional-case handling."""


class KMeansSession(Session):
    def __init__(self, runtime, x, weights, n_clusters):
        super().__init__(runtime)
        self.x, self.weights, self.k = x, weights, n_clusters

    def __enter__(self):
        super().__enter__()
        try:
            n, d = self.x.shape
            self.xb, self.wb = self.upload(self.x), self.upload(self.weights)
            self.device_ready = False
            self.centers = self.empty((self.k, d))
            self.updated = self.empty((self.k, d))
            self.labels, self.errors = self.empty((n,), np.int32), self.empty((n,))
            self.status = self.empty((1,), np.uint32)
            # Bound the extra center-reduction workspace to ~32 MiB (one
            # cluster/feature slice is the minimum), independent of n_samples.
            max_chunks = max(1, (32 * 1024 * 1024) // (self.k * (d + 1) * 4))
            block = max(256, (n + max_chunks - 1) // max_chunks)
            self.partial = self.empty(((n + block - 1) // block, self.k, d + 1))
            self.params = (n, self.k, d, block)
            trials = 2 + int(np.log(self.k))
            self.candidates = self.empty((trials, d))
            self.distances = self.empty((n, trials))
            return self
        except BaseException:
            import sys

            super().__exit__(*sys.exc_info())
            raise

    def command(self):
        command = super().command()
        if not self.device_ready:
            # Stage once for every initialization and Lloyd iteration. Random
            # feature accesses on AMD should read its local VRAM, not PCIe.
            self.xb = self.private_copy(command, self.xb)
            self.wb = self.private_copy(command, self.wb)
            self.device_ready = True
        return command

    def initialize(self, random):
        """Greedy k-means++: GPU distances, CPU seeded sampling and potentials."""
        n, d = self.x.shape
        trials = 2 + int(np.log(self.k))
        centers = np.empty((self.k, d), np.float32)
        centers[0] = self.x[random.choice(n, p=self.weights / self.weights.sum())]
        # Allocate once, including candidate/output workspace shared across seeds.
        candidates, distances = self.candidates, self.distances

        def evaluate(values):
            count = len(values)
            candidates.array[:count] = values
            self.runtime.uploaded_bytes += values.nbytes
            command = self.command()
            self.encode(
                command,
                "distances_coalesced",
                [self.xb, candidates, distances],
                [n, count, d],
                (n * count,),
            )
            self.submit(command)
            return distances.array.ravel()[: n * count].reshape(n, count).copy()

        closest = evaluate(centers[:1])[:, 0]
        potential = closest @ self.weights
        for c in range(1, self.k):
            thresholds = random.uniform(size=trials) * potential
            ids = np.searchsorted(np.cumsum(self.weights * closest, dtype=np.float64), thresholds)
            np.clip(ids, 0, n - 1, out=ids)
            candidate_distances = evaluate(self.x[ids])
            np.minimum(candidate_distances, closest[:, None], out=candidate_distances)
            potentials = self.weights @ candidate_distances
            best = int(np.argmin(potentials))
            closest = candidate_distances[:, best].copy()
            potential = potentials[best]
            centers[c] = self.x[ids[best]]
        return centers

    def lloyd(self, centers, max_iter, tol):
        n, k, d, _ = self.params
        self.centers.write(centers)
        for iteration in range(1, max_iter + 1):  # noqa: B007 - returned after convergence
            self.status.array.fill(0)
            command = self.command()
            self.encode(
                command,
                "assign_coalesced",
                [self.xb, self.centers, self.labels, self.errors, self.status],
                self.params,
                (n,),
            )
            self.encode(
                command,
                "centers_partial",
                [self.xb, self.labels, self.wb, self.partial],
                self.params,
                (self.partial.array.size,),
            )
            self.encode(
                command,
                "centers_finish",
                [self.partial, self.centers, self.updated, self.status],
                self.params,
                (k * d,),
            )
            self.submit(command, dispatches=3)
            status = int(self.status.array[0])
            if status & 1:
                raise KMeansFallback("Distances exceed float32 range")
            if status & 2:
                raise KMeansFallback("Empty-cluster relocation uses sklearn")
            updated = self.updated.read()
            shift = np.sum((updated.astype(float) - centers) ** 2)
            centers = updated
            self.centers, self.updated = self.updated, self.centers
            if shift <= tol:
                break
        self.status.array.fill(0)
        command = self.command()
        self.encode(
            command,
            "assign_coalesced",
            [self.xb, self.centers, self.labels, self.errors, self.status],
            self.params,
            (n,),
        )
        self.submit(command)
        if self.status.array[0]:
            raise KMeansFallback("Distances exceed float32 range")
        inertia = float(np.dot(self.errors.array.astype(float), self.weights.astype(float)))
        return inertia, centers, self.labels.read(), iteration
