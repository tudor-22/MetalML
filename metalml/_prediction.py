"""Shared sklearn-compatible inference adapters; training stays upstream."""

from functools import wraps

import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_is_fitted, validate_data

from ._config import config_context
from ._dispatch import BackendDiagnostics, as_float32, prefer_cpu, record, select


def cpu_fit(source):
    @wraps(source.fit)
    def fit(self, *args, **kwargs):
        # Some optimizers call predict internally (for early stopping). Keep
        # their convergence and validation behavior wholly within sklearn.
        with config_context(backend="cpu"):
            result = source.fit(self, *args, **kwargs)
        record(self, "fit", "cpu", "Training uses sklearn; Metal accelerates inference")
        return result

    return fit


def estimator(name, mixin, source, module):
    return type(
        name,
        (mixin, source),
        {
            "__module__": module,
            "__doc__": f"{name}: sklearn training with Metal inference for dense float32 input.",
            "fit": cpu_fit(source),
        },
    )


def linear_scores(model, X, operation):
    check_is_fitted(model, "coef_")
    runtime = select(
        model,
        operation,
        X,
        reason="Sparse coefficients use sklearn" if sparse.issparse(model.coef_) else None,
    )
    if runtime is None:
        return None
    coef = np.asarray(model.coef_)
    outputs = 1 if coef.ndim == 1 else coef.shape[0]
    if prefer_cpu(
        model,
        operation,
        runtime,
        coef.dtype == np.float32 and outputs <= 16 and np.size(X) * outputs <= 20_480_000,
        "Small float32 linear prediction favors CPU on the measured 5300M",
    ):
        return None
    x = validate_data(model, X, reset=False, dtype=np.float32)
    weights = coef.reshape(1, -1).T if coef.ndim == 1 else coef.T
    result = runtime.linear(as_float32(x), as_float32(weights), model.intercept_)
    if not np.isfinite(result).all():
        select(model, operation, X, reason="Linear scores exceed float32 range; using sklearn")
        return None
    return result[:, 0] if coef.ndim == 1 else result


class LinearRegressor(BackendDiagnostics):
    _metal_fitted_attribute = "coef_"

    def _decision_function(self, X):
        scores = linear_scores(self, X, "predict")
        return super()._decision_function(X) if scores is None else scores


class LinearClassifier(BackendDiagnostics):
    _metal_fitted_attribute = "coef_"

    def decision_function(self, X):
        scores = linear_scores(self, X, "decision_function")
        if scores is None:
            return super().decision_function(X)
        return scores.ravel() if scores.ndim == 2 and scores.shape[1] == 1 else scores


class GeneralizedLinear(BackendDiagnostics):
    _metal_fitted_attribute = "coef_"

    def _linear_predictor(self, X):
        scores = linear_scores(self, X, "predict")
        return super()._linear_predictor(X) if scores is None else scores
