"""Native Metal Lloyd KMeans with sklearn fallback for unsupported settings."""

import warnings

import numpy as np
from sklearn import cluster as _sk
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._dispatch import BackendDiagnostics, as_float32, record, select
from ._metal._kmeans import KMeansFallback


def __getattr__(name):
    return getattr(_sk, name)


__all__ = _sk.__all__


class KMeans(BackendDiagnostics, _sk.KMeans):
    _metal_fitted_attribute = "cluster_centers_"

    def fit(self, X, y=None, sample_weight=None):
        self._validate_params()
        reason = None
        if self.algorithm != "lloyd" or callable(self.init):
            reason = "Elkan and callable initialization use sklearn"
        elif self.verbose:
            reason = "Verbose iteration reporting uses sklearn"
        runtime = select(self, "fit", X, reason=reason)
        if runtime is None:
            return super().fit(X, y, sample_weight=sample_weight)
        x = as_float32(validate_data(self, X, dtype=np.float32, order="C"))
        n, d = x.shape
        if n < self.n_clusters:
            raise ValueError(f"n_samples={n} should be >= n_clusters={self.n_clusters}")
        weights = (
            np.ones(n, np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )
        if weights.ndim == 0:
            weights = np.full(n, weights, np.float32)
        if weights.shape != (n,) or not np.isfinite(weights).all():
            raise ValueError("sample_weight must contain one finite weight per sample")
        if np.any(weights <= 0):
            select(self, "fit", X, reason="Nonpositive sample weights use sklearn")
            return super().fit(X, y, sample_weight=sample_weight)
        random = check_random_state(self.random_state)
        array_init = not isinstance(self.init, str)
        n_init = self.n_init
        if n_init == "auto":
            n_init = 10 if isinstance(self.init, str) and self.init == "random" else 1
        if array_init and n_init != 1:
            warnings.warn(
                "Explicit initial centers perform only one initialization",
                RuntimeWarning,
                stacklevel=2,
            )
            n_init = 1
        # Translation improves float32 distance accuracy without changing clustering.
        mean = x.mean(axis=0, dtype=np.float64)
        centered = np.empty_like(x)
        np.subtract(x, mean, out=centered)
        tol = float(np.var(centered, axis=0).mean() * self.tol)
        best = None
        try:
            with runtime.kmeans_session(centered, weights, self.n_clusters) as session:
                for _ in range(n_init):
                    if array_init:
                        centers = as_float32(np.asarray(self.init) - mean)
                        if centers.shape != (self.n_clusters, d) or not np.isfinite(centers).all():
                            raise ValueError(
                                "Initial centers must have shape (n_clusters, n_features) and be finite"
                            )
                    elif self.init == "k-means++":
                        centers = session.initialize(random)
                    else:
                        indices = random.choice(
                            n,
                            self.n_clusters,
                            replace=False,
                            p=weights.astype(float) / weights.sum(dtype=float),
                        )
                        centers = centered[indices].copy()
                    candidate = session.lloyd(centers, self.max_iter, tol)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
        except KMeansFallback as exc:
            select(self, "fit", X, reason=str(exc))
            return super().fit(X, y, sample_weight=sample_weight)
        self.inertia_, centers, self.labels_, self.n_iter_ = best
        self.cluster_centers_ = as_float32(centers + mean)
        self._n_threads = 1
        self._n_features_out = self.n_clusters
        self._n_init = n_init
        if len(np.unique(self.labels_)) < self.n_clusters:
            warnings.warn(
                "Fewer distinct clusters than requested", ConvergenceWarning, stacklevel=2
            )
        record(self, "fit", "metal+cpu", "Seed sampling and convergence checks use CPU")
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        runtime = select(self, "predict", X)
        if runtime is None:
            return super().predict(X)
        x = validate_data(self, X, reset=False, dtype=np.float32)
        return runtime.assign(as_float32(x), as_float32(self.cluster_centers_))[0]

    def transform(self, X):
        check_is_fitted(self, "cluster_centers_")
        runtime = select(self, "transform", X)
        if runtime is None:
            return super().transform(X)
        x = validate_data(self, X, reset=False, dtype=np.float32)
        return np.sqrt(runtime.distances(as_float32(x), as_float32(self.cluster_centers_)))

    def score(self, X, y=None, sample_weight=None):
        check_is_fitted(self, "cluster_centers_")
        runtime = select(self, "score", X)
        if runtime is None:
            return super().score(X, y, sample_weight=sample_weight)
        x = validate_data(self, X, reset=False, dtype=np.float32)
        errors = runtime.assign(as_float32(x), as_float32(self.cluster_centers_))[1]
        weights = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight)
        if weights.ndim == 0:
            weights = np.full(len(x), weights)
        if weights.shape != (len(x),) or not np.isfinite(weights).all():
            raise ValueError("sample_weight must contain one finite weight per sample")
        return -float(np.dot(errors.astype(float), weights))

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight=sample_weight).transform(X)
