"""Bounded-memory exact Euclidean neighbor search on Metal."""

from numbers import Integral

import numpy as np
from scipy import sparse
from sklearn import neighbors as _sk
from sklearn.neighbors._base import _get_weights
from sklearn.utils.validation import check_is_fitted, validate_data

from ._config import get_config
from ._dispatch import BackendDiagnostics, as_float32, prefer_cpu, record, select
from ._prediction import estimator

__all__ = _sk.__all__


def __getattr__(name):
    return getattr(_sk, name)


class _Neighbors(BackendDiagnostics):
    _metal_fitted_attribute = "_fit_X"

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self, "_fit_X")
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if not isinstance(k, Integral) or k <= 0 or k > self.n_samples_fit_:
            # Let sklearn issue its exact validation error.
            return super().kneighbors(X, n_neighbors=n_neighbors, return_distance=return_distance)
        reason = None
        if X is None:
            reason = "Self-excluding training queries use sklearn"
        elif self._fit_method != "brute" or self.effective_metric_ != "euclidean":
            reason = "Metal neighbors require brute-force Euclidean distance"
        elif k > 32:
            reason = "More than 32 neighbors use sklearn"
        elif sparse.issparse(self._fit_X):
            reason = "Sparse training data use sklearn"
        elif self._fit_X.dtype != np.float32 and get_config()["precision"] == "preserve":
            reason = "Training precision preserved; Metal neighbors require float32 training data"
        runtime = select(self, "kneighbors", X, reason=reason)
        if runtime is None:
            return super().kneighbors(X, n_neighbors=n_neighbors, return_distance=return_distance)
        if prefer_cpu(
            self,
            "kneighbors",
            runtime,
            self.n_samples_fit_ <= 1024
            and self.n_features_in_ <= 64
            and (len(X) < 8192 or self.n_samples_fit_ < 1024 or self.n_features_in_ < 64 or k > 8),
            "Small reference-set neighbor search favors CPU on the measured 5300M",
        ):
            return super().kneighbors(X, n_neighbors=n_neighbors, return_distance=return_distance)
        x = validate_data(self, X, reset=False, dtype=np.float32, ensure_all_finite=False)
        distances, indices = runtime.neighbors(as_float32(x), as_float32(self._fit_X), k)
        if not np.isfinite(distances).all() or np.any(indices >= self.n_samples_fit_):
            select(self, "kneighbors", X, reason="Distances exceed float32 range; using sklearn")
            return super().kneighbors(X, n_neighbors=n_neighbors, return_distance=return_distance)
        return (distances, indices) if return_distance else indices


class _Classifier(_Neighbors):
    def predict_proba(self, X):
        check_is_fitted(self, "_fit_X")
        if get_config()["backend"] == "cpu":
            record(self, "predict_proba", "cpu", "CPU explicitly requested")
            return super().predict_proba(X)
        # Always route through kneighbors, including sklearn's optimized
        # uniform-weight branch which normally bypasses that public method.
        distance, indices = self.kneighbors(X)
        if self.operation_backends_["kneighbors"] == "cpu":
            record(self, "predict_proba", "cpu", self.fallback_reasons_["kneighbors"])
            return super().predict_proba(X)
        weights = _get_weights(distance, self.weights)
        if weights is None:
            weights = np.ones_like(distance)
        if np.any(np.all(weights == 0, axis=1)):
            raise ValueError("All neighbors of some sample is getting zero weights.")
        targets = self._y if self.outputs_2d_ else self._y[:, None]
        classes = self.classes_ if self.outputs_2d_ else [self.classes_]
        result = []
        for output, labels in enumerate(classes):
            votes = np.zeros((len(indices), len(labels)), np.float64)
            for neighbor in range(indices.shape[1]):
                votes[np.arange(len(indices)), targets[indices[:, neighbor], output]] += weights[
                    :, neighbor
                ]
            votes /= votes.sum(axis=1, keepdims=True)
            result.append(votes)
        record(
            self,
            "predict_proba",
            "metal+cpu" if self.operation_backends_["kneighbors"] == "metal" else "cpu",
            self.fallback_reasons_["kneighbors"],
        )
        return result if self.outputs_2d_ else result[0]

    def predict(self, X):
        probabilities = self.predict_proba(X)
        if self.outputs_2d_:
            return np.column_stack(
                [
                    labels[p.argmax(axis=1)]
                    for labels, p in zip(self.classes_, probabilities, strict=True)
                ]
            )
        return self.classes_[probabilities.argmax(axis=1)]


NearestNeighbors = estimator("NearestNeighbors", _Neighbors, _sk.NearestNeighbors, __name__)
KNeighborsRegressor = estimator(
    "KNeighborsRegressor", _Neighbors, _sk.KNeighborsRegressor, __name__
)
KNeighborsClassifier = estimator(
    "KNeighborsClassifier", _Classifier, _sk.KNeighborsClassifier, __name__
)
