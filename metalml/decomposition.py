"""Metal covariance/projection with CPU eigensolvers and sklearn solver fallbacks."""

import numpy as np
from scipy.linalg import eigh
from sklearn import decomposition as _sk
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from ._dispatch import BackendDiagnostics, as_float32, prefer_cpu, record, select
from ._prediction import estimator


def __getattr__(name):
    return getattr(_sk, name)


__all__ = _sk.__all__


class PCA(BackendDiagnostics, _sk.PCA):
    _metal_fitted_attribute = "components_"

    def fit(self, X, y=None):
        self._validate_params()
        shape = getattr(X, "shape", ())
        # Match sklearn's tall-matrix auto policy. Other solvers/component
        # selection modes retain upstream behavior, including copy=False.
        eligible = (
            len(shape) == 2
            and shape[0] >= 10 * shape[1]
            and 0 < shape[1] <= 1000
            and self.svd_solver in ("auto", "covariance_eigh")
            and self.copy
            and (self.n_components is None or isinstance(self.n_components, (int, np.integer)))
        )
        if eligible:
            runtime = select(self, "fit", X)
            if runtime is not None and prefer_cpu(
                self,
                "fit",
                runtime,
                shape[0] <= 20000 and shape[1] <= 256,
                "Small PCA covariance fit favors CPU on the measured 5300M",
            ):
                return super().fit(X, y)
            if runtime is not None:
                x = validate_data(
                    self, X, dtype=np.float32, ensure_min_samples=2, ensure_all_finite=False
                )
                n, d = x.shape
                count = d if self.n_components is None else self.n_components
                if not 0 <= count <= min(n, d):
                    raise ValueError(
                        "n_components must be between 0 and min(n_samples, n_features)"
                    )
                mean = x.mean(axis=0, dtype=np.float64)
                gram = runtime.covariance_products(as_float32(x), mean)
                if np.isfinite(gram).all():
                    values, vectors = eigh(
                        gram.astype(np.float64) / (n - 1),
                        check_finite=False,
                        driver="evd" if d >= 128 else "evr",
                    )
                else:
                    values, vectors = np.array([0.0]), None
                # Covariance formation squares the condition number. Retain
                # sklearn for degenerate or poorly resolved spectra.
                if values[0] > 0 and values[-1] / values[0] <= 1e4:
                    values = values[::-1]
                    _, components = svd_flip(None, vectors[:, ::-1].T, u_based_decision=False)
                    self.mean_ = mean
                    self.n_samples_, self.n_components_ = n, count
                    self._fit_svd_solver = "covariance_eigh"
                    self.components_ = components[:count].astype(np.float32)
                    self.explained_variance_ = values[:count].astype(np.float32)
                    self.explained_variance_ratio_ = (values[:count] / values.sum()).astype(
                        np.float32
                    )
                    self.singular_values_ = np.sqrt(values[:count] * (n - 1)).astype(np.float32)
                    self.noise_variance_ = float(values[count:].mean()) if count < d else 0.0
                    record(self, "fit", "metal+cpu")
                    return self
                select(self, "fit", X, reason="Ill-conditioned float32 covariance; using sklearn")
        record(self, "fit", "cpu", "PCA decomposition uses sklearn's SVD/eigensolver")
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        shape = getattr(X, "shape", ())
        if (
            len(shape) == 2
            and shape[0] >= 10 * shape[1]
            and 0 < shape[1] <= 1000
            and self.svd_solver in ("auto", "covariance_eigh")
            and self.copy
            and (self.n_components is None or isinstance(self.n_components, (int, np.integer)))
            and self.n_components != 0
        ):
            self.fit(X, y)
            if self.operation_backends_["fit"] == "metal+cpu":
                result = self.transform(X)
                record(self, "fit_transform", "metal+cpu")
                return result
            record(self, "fit_transform", "cpu", self.fallback_reasons_["fit"])
            return _sk.PCA.transform(self, X)
        # sklearn's fit_transform handles copy=False without centering twice.
        record(self, "fit", "cpu", "PCA decomposition uses sklearn's SVD/eigensolver")
        record(self, "fit_transform", "cpu", "Uses sklearn's decomposition output directly")
        return super().fit_transform(X, y)

    def transform(self, X):
        check_is_fitted(self, "components_")
        runtime = select(
            self,
            "transform",
            X,
            reason="Zero-component projection uses sklearn" if self.n_components_ == 0 else None,
        )
        if runtime is None:
            return super().transform(X)
        if prefer_cpu(
            self,
            "transform",
            runtime,
            np.size(X) * self.n_components_ <= 81_920_000,
            "Small PCA projection favors CPU on the measured 5300M",
        ):
            return super().transform(X)
        x = validate_data(self, X, reset=False, dtype=np.float32, ensure_all_finite=False)
        components = self.components_.T
        if self.whiten:
            components = components / np.maximum(
                np.sqrt(self.explained_variance_), np.finfo(np.float32).eps
            )
        # Center on the device before multiplication to avoid cancellation on
        # large-offset data, with no full CPU-centered copy or extra GPU wait.
        return runtime.linear(as_float32(x), as_float32(components), 0, mean=self.mean_)

    def inverse_transform(self, X):
        check_is_fitted(self, "components_")
        runtime = select(self, "inverse_transform", X)
        if runtime is None:
            return super().inverse_transform(X)
        x = check_array(X, dtype=np.float32)
        if x.shape[1] != self.n_components_:
            raise ValueError(f"Expected {self.n_components_} components")
        components = self.components_
        if self.whiten:
            components = np.sqrt(self.explained_variance_)[:, None] * components
        return runtime.linear(as_float32(x), as_float32(components), self.mean_)


class _SVD(BackendDiagnostics):
    _metal_fitted_attribute = "components_"

    def fit_transform(self, X, y=None):
        record(self, "fit", "cpu", "Truncated SVD decomposition uses sklearn")
        record(self, "fit_transform", "cpu", "Uses sklearn's decomposition output directly")
        return super().fit_transform(X, y)

    def transform(self, X):
        check_is_fitted(self, "components_")
        runtime = select(self, "transform", X)
        if runtime is None:
            return super().transform(X)
        if prefer_cpu(
            self,
            "transform",
            runtime,
            np.size(X) * self.components_.shape[0] <= 20_480_000,
            "Small SVD projection favors CPU on the measured 5300M",
        ):
            return super().transform(X)
        x = validate_data(self, X, reset=False, dtype=np.float32)
        return runtime.linear(as_float32(x), as_float32(self.components_.T), 0)

    def inverse_transform(self, X):
        check_is_fitted(self, "components_")
        runtime = select(self, "inverse_transform", X)
        if runtime is None:
            return super().inverse_transform(X)
        x = check_array(X, dtype=np.float32)
        if x.shape[1] != self.components_.shape[0]:
            raise ValueError(f"Expected {self.components_.shape[0]} components")
        return runtime.linear(as_float32(x), as_float32(self.components_), 0)


TruncatedSVD = estimator("TruncatedSVD", _SVD, _sk.TruncatedSVD, __name__)
