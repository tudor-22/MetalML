"""GPU Ridge sufficient statistics and dense linear-model prediction."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvalsh
from sklearn import linear_model as _sk
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._dispatch import BackendDiagnostics, as_float32, prefer_cpu, record, select
from ._metal._logistic import LogisticFallback
from ._prediction import (
    GeneralizedLinear,
    LinearClassifier,
    LinearRegressor,
    estimator,
    linear_scores,
)


def __getattr__(name):
    return getattr(_sk, name)


__all__ = _sk.__all__


class _MetalPredict(BackendDiagnostics):
    _metal_fitted_attribute = "coef_"

    def predict(self, X):
        result = linear_scores(self, X, "predict")
        return super().predict(X) if result is None else result


class LinearRegression(_MetalPredict, _sk.LinearRegression):
    """CPU least-squares fitting (preserves rank handling), Metal prediction."""

    def fit(self, X, y, sample_weight=None):
        record(self, "fit", "cpu", "Least-squares rank-revealing solver uses sklearn")
        return super().fit(X, y, sample_weight=sample_weight)


class LogisticRegression(BackendDiagnostics, _sk.LogisticRegression):
    """Metal IRLS training for binary L2 logistic regression; Metal scores.

    The Newton (IRLS) weighted Gram and score products run on the GPU. The
    small ``d x d`` solve, multiclass fits, non-L2 penalties, class weights,
    and sample weights use sklearn.
    """

    _metal_fitted_attribute = "coef_"

    def fit(self, X, y, sample_weight=None):
        self._validate_params()
        reason = None
        if sample_weight is not None:
            reason = "Weighted logistic regression uses sklearn"
        elif self.class_weight is not None:
            reason = "Class-weighted logistic regression uses sklearn"
        elif self.warm_start:
            reason = "Warm-started logistic regression uses sklearn"
        elif self.penalty not in (None, "l2"):
            reason = "Only L2 or unpenalized logistic regression trains on Metal"
        runtime = select(self, "fit", X, reason=reason)
        if runtime is None:
            return super().fit(X, y, sample_weight=sample_weight)

        y = np.asarray(y)
        if y.ndim != 1:
            select(self, "fit", X, reason="Multi-output logistic regression uses sklearn")
            return super().fit(X, y, sample_weight=sample_weight)
        check_classification_targets(y)
        classes = np.unique(y)
        if len(classes) != 2:
            select(self, "fit", X, reason="Multiclass logistic regression uses sklearn")
            return super().fit(X, y, sample_weight=sample_weight)
        if runtime.mps is None or np.shape(X)[1] > 1024:
            record(self, "fit", "cpu", "Logistic GPU training needs MPS and at most 1024 features")
            return super().fit(X, y, sample_weight=sample_weight)

        x = as_float32(validate_data(self, X, dtype=np.float32, ensure_all_finite=False))
        alpha = 0.0 if self.penalty is None else 1.0 / float(self.C)
        y01 = (y == classes[1]).astype(np.float32)
        try:
            coefficients, intercept, n_iter = runtime.logistic_irls(
                x, y01, alpha, self.fit_intercept, int(self.max_iter), float(self.tol)
            )
        except LogisticFallback as exc:
            select(self, "fit", X, reason=str(exc))
            return super().fit(X, y, sample_weight=sample_weight)

        self.classes_ = classes
        self.coef_ = coefficients.reshape(1, -1).astype(np.float32)
        self.intercept_ = np.asarray([intercept], dtype=np.float32)
        self.n_iter_ = np.asarray([n_iter], dtype=np.int32)
        record(self, "fit", "metal+cpu")
        return self

    def decision_function(self, X):
        check_is_fitted(self, "coef_")
        runtime = select(self, "decision_function", X)
        if runtime is None:
            return super().decision_function(X)
        x = validate_data(self, X, reset=False, dtype=np.float32)
        result = runtime.linear(as_float32(x), as_float32(self.coef_.T), self.intercept_)
        return result[:, 0] if result.shape[1] == 1 else result


class Ridge(_MetalPredict, _sk.Ridge):
    """Metal Gram/cross-products, CPU small-system solve, Metal prediction."""

    def fit(self, X, y, sample_weight=None):
        self._validate_params()
        reason = None
        if sample_weight is not None:
            reason = "Weighted regression uses sklearn"
        elif self.positive or self.solver not in {"auto", "cholesky"}:
            reason = "Requested Ridge solver uses sklearn"
        elif np.ndim(self.alpha) != 0 or self.alpha <= 0:
            reason = "Nonpositive or per-target regularization uses sklearn"
        elif len(getattr(X, "shape", ())) == 2 and (X.shape[1] > 2048 or X.shape[1] > X.shape[0]):
            reason = "Wide regression uses sklearn to avoid a large Gram matrix"
        runtime = select(self, "fit", X, reason=reason)
        if runtime is None:
            return super().fit(X, y, sample_weight=sample_weight)
        if prefer_cpu(
            self,
            "fit",
            runtime,
            len(getattr(X, "shape", ())) == 2 and X.shape[0] <= 20000 and 128 < X.shape[1] <= 256,
            "Moderate-width Ridge fitting favors CPU on the measured 5300M",
        ):
            return super().fit(X, y, sample_weight=sample_weight)
        # select() already checked finite X; y is checked again after casting.
        x, target = validate_data(
            self, X, y, dtype=np.float32, multi_output=True, y_numeric=True, ensure_all_finite=False
        )
        if x.shape[1] > 2048 or x.shape[1] > x.shape[0]:
            select(
                self, "fit", X, reason="Wide regression uses sklearn to avoid a large Gram matrix"
            )
            return super().fit(X, y, sample_weight=sample_weight)
        x = as_float32(x)
        target = as_float32(target)
        if not np.isfinite(target).all():
            raise ValueError("Targets exceed float32 range")
        x_mean = x.mean(axis=0, dtype=np.float64) if self.fit_intercept else np.zeros(x.shape[1])
        y_mean = (
            target.mean(axis=0, dtype=np.float64)
            if self.fit_intercept
            else np.zeros(target.shape[1:] or ())
        )
        gram, cross = runtime.ridge_products(
            x,
            target.reshape(len(x), -1),
            x_mean=x_mean if self.fit_intercept else None,
            y_mean=y_mean if self.fit_intercept else None,
        )
        gram = gram.astype(np.float64)
        gram.flat[:: gram.shape[0] + 1] += self.alpha
        # Normal equations amplify round-off. Preserve sklearn's robust path
        # when float32 sufficient statistics cannot safely resolve the system.
        safe = False
        if np.isfinite(gram).all():
            diagonal = np.diag(gram)
            radii = np.abs(gram).sum(axis=1) - np.abs(diagonal)
            lower, upper = np.min(diagonal - radii), np.max(diagonal + radii)
            margin = 4 * np.finfo(np.float64).eps * len(gram) * max(1.0, upper)
            lower, upper = lower - margin, upper + margin
            # Gershgorin bounds can certify the same condition threshold in
            # O(d^2). Difficult systems still get the full spectral check.
            safe = lower > 0 and upper / lower <= 1e4
            if not safe:
                eigenvalues = eigvalsh(gram, check_finite=False)
                safe = eigenvalues[0] > 0 and eigenvalues[-1] / eigenvalues[0] <= 1e4
        if not safe:
            select(self, "fit", X, reason="Ill-conditioned float32 Gram matrix; using sklearn")
            return super().fit(X, y, sample_weight=sample_weight)
        coef = cho_solve(
            cho_factor(gram, check_finite=False), cross.astype(np.float64), check_finite=False
        ).T
        self.coef_ = coef[0].astype(np.float32) if target.ndim == 1 else coef.astype(np.float32)
        self.intercept_ = np.asarray(y_mean - x_mean @ self.coef_.T, dtype=np.float32)
        self.n_iter_ = None
        self.solver_ = "cholesky"
        record(self, "fit", "metal+cpu")
        return self


# These estimators all expose the same linear score operation. Inherit their
# public predict methods so uncertainty, link functions, and class semantics
# remain sklearn's responsibility.
for _name in (
    "RidgeCV",
    "Lasso",
    "LassoCV",
    "ElasticNet",
    "ElasticNetCV",
    "Lars",
    "LarsCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "BayesianRidge",
    "ARDRegression",
    "HuberRegressor",
    "TheilSenRegressor",
    "SGDRegressor",
    "PassiveAggressiveRegressor",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
):
    globals()[_name] = estimator(_name, LinearRegressor, getattr(_sk, _name), __name__)

for _name in (
    "RidgeClassifier",
    "RidgeClassifierCV",
    "SGDClassifier",
    "Perceptron",
    "PassiveAggressiveClassifier",
    "LogisticRegressionCV",
):
    globals()[_name] = estimator(_name, LinearClassifier, getattr(_sk, _name), __name__)

for _name in ("PoissonRegressor", "GammaRegressor", "TweedieRegressor"):
    globals()[_name] = estimator(_name, GeneralizedLinear, getattr(_sk, _name), __name__)
