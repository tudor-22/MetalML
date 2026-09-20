"""Accelerated dense float32 scalers. Other names delegate to sklearn."""

import numpy as np
from sklearn import preprocessing as _sk
from sklearn.utils.validation import check_is_fitted, validate_data

from ._dispatch import BackendDiagnostics, as_float32, copy_result, record, select


def __getattr__(name):
    return getattr(_sk, name)


__all__ = _sk.__all__


class StandardScaler(BackendDiagnostics, _sk.StandardScaler):
    _metal_fitted_attribute = "n_samples_seen_"

    def _set_statistics(self, mean, var, count):
        self.n_samples_seen_ = np.int64(count)
        self.mean_ = mean.astype(float) if self.with_mean or self.with_std else None
        self.var_ = var.astype(float) if self.with_std else None
        self.scale_ = None
        if self.with_std:
            self.scale_ = np.sqrt(self.var_)
            self.scale_[self.scale_ == 0] = 1

    def fit_transform(self, X, y=None, **fit_params):
        # Preserve sklearn's routing and errors for fit parameters not handled here.
        if fit_params:
            return super().fit_transform(X, y, **fit_params)
        self._validate_params()
        runtime = select(self, "fit", X)
        if runtime is None:
            return super().fit_transform(X, y)
        self._reset()
        x = as_float32(validate_data(self, X, dtype=np.float32))
        mean, var, result = runtime.scale_fit_transform(x, self.with_mean, self.with_std)
        if not np.isfinite(var).all():
            select(self, "fit", X, reason="Statistics exceed float32 range")
            # Re-enter the established CPU fit/transform dispatch on exceptional input.
            self.fit(X, y)
            return self.transform(X)
        self._set_statistics(mean, var, len(x))
        record(self, "transform", "metal")
        return copy_result(X, result, self.copy)

    def fit(self, X, y=None, sample_weight=None):
        self._validate_params()
        runtime = select(
            self,
            "fit",
            X,
            reason="Weighted statistics use sklearn" if sample_weight is not None else None,
        )
        if runtime is None:
            return super().fit(X, y, sample_weight=sample_weight)
        self._reset()
        x = validate_data(self, X, dtype=np.float32)
        mean, var, _, _ = runtime.stats(as_float32(x))
        if not np.isfinite(var).all():
            select(self, "fit", X, reason="Statistics exceed float32 range")
            return super().fit(X, y, sample_weight=sample_weight)
        self._set_statistics(mean, var, len(x))
        return self

    def partial_fit(self, X, y=None, sample_weight=None):
        record(self, "partial_fit", "cpu", "Incremental statistics use sklearn")
        return super().partial_fit(X, y, sample_weight=sample_weight)

    def transform(self, X, copy=None):
        check_is_fitted(self, "n_samples_seen_")
        runtime = select(self, "transform", X)
        if runtime is None:
            return super().transform(X, copy=copy)
        x = validate_data(self, X, reset=False, dtype=np.float32)
        scale = self.scale_ if self.with_std else np.ones(x.shape[1])
        mean = self.mean_ if self.with_mean else np.zeros(x.shape[1])
        return copy_result(
            X, runtime.standardize(as_float32(x), mean, scale), self.copy if copy is None else copy
        )

    def inverse_transform(self, X, copy=None):
        check_is_fitted(self, "n_samples_seen_")
        runtime = select(self, "inverse_transform", X)
        if runtime is None:
            return super().inverse_transform(X, copy=copy)
        x = validate_data(self, X, reset=False, dtype=np.float32)
        scale = self.scale_ if self.with_std else np.ones(x.shape[1])
        bias = self.mean_ if self.with_mean else np.zeros(x.shape[1])
        return copy_result(
            X, runtime.affine(as_float32(x), scale, bias), self.copy if copy is None else copy
        )


class MinMaxScaler(BackendDiagnostics, _sk.MinMaxScaler):
    _metal_fitted_attribute = "n_samples_seen_"

    def fit(self, X, y=None):
        self._validate_params()
        runtime = select(self, "fit", X)
        if runtime is None:
            return super().fit(X, y)
        if self.feature_range[0] >= self.feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller than maximum")
        self._reset()
        x = validate_data(self, X, dtype=np.float32)
        _, _, self.data_min_, self.data_max_ = runtime.stats(as_float32(x))
        self.data_range_ = self.data_max_ - self.data_min_
        denominator = self.data_range_.copy()
        denominator[denominator < 10 * np.finfo(np.float32).eps] = 1
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / denominator
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        self.n_samples_seen_ = len(x)
        return self

    def partial_fit(self, X, y=None):
        record(self, "partial_fit", "cpu", "Incremental statistics use sklearn")
        return super().partial_fit(X, y)

    def transform(self, X):
        check_is_fitted(self, "n_samples_seen_")
        runtime = select(self, "transform", X)
        if runtime is None:
            return super().transform(X)
        x = validate_data(self, X, reset=False, dtype=np.float32)
        result = runtime.affine(as_float32(x), self.scale_, self.min_)
        if self.clip:
            np.clip(result, *self.feature_range, out=result)
        return copy_result(X, result, self.copy)

    def inverse_transform(self, X):
        check_is_fitted(self, "n_samples_seen_")
        runtime = select(self, "inverse_transform", X)
        if runtime is None:
            return super().inverse_transform(X)
        x = validate_data(self, X, reset=False, dtype=np.float32)
        return copy_result(
            X, runtime.affine(as_float32(x), 1 / self.scale_, -self.min_ / self.scale_), self.copy
        )


class Normalizer(BackendDiagnostics, _sk.Normalizer):
    _metal_fitted_attribute = "n_features_in_"

    def fit(self, X, y=None):
        record(self, "fit", "cpu", "Normalizer fitting only validates input")
        return super().fit(X, y)

    def transform(self, X, copy=None):
        self._validate_params()
        runtime = select(self, "transform", X)
        if runtime is None:
            return super().transform(X, copy=copy)
        x = validate_data(self, X, reset=False, dtype=np.float32)
        return copy_result(
            X, runtime.normalize(as_float32(x), self.norm), self.copy if copy is None else copy
        )
