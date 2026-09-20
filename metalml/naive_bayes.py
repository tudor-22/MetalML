"""GPU likelihoods with sklearn training and probability normalization."""

import numpy as np
from sklearn import naive_bayes as _sk

from ._dispatch import BackendDiagnostics, as_float32, select
from ._prediction import estimator

__all__ = _sk.__all__


def __getattr__(name):
    return getattr(_sk, name)


class _Gaussian(BackendDiagnostics):
    _metal_fitted_attribute = "classes_"

    def _joint_log_likelihood(self, X):
        reason = (
            "Degenerate variances or priors use sklearn"
            if (np.any(self.var_ <= 0) or np.any(self.class_prior_ <= 0))
            else None
        )
        runtime = select(self, "predict", X, reason=reason)
        if runtime is None:
            return super()._joint_log_likelihood(X)
        result = runtime.gaussian_scores(as_float32(X), self.theta_, self.var_, self.class_prior_)
        if not np.isfinite(result).all():
            select(
                self,
                "predict",
                X,
                reason="Gaussian likelihoods exceed float32 range; using sklearn",
            )
            return super()._joint_log_likelihood(X)
        return result


class _Discrete(BackendDiagnostics):
    _metal_fitted_attribute = "classes_"

    def _joint_log_likelihood(self, X):
        weights, bias = self.feature_log_prob_, self.class_log_prior_
        if isinstance(self, _sk.BernoulliNB):
            negative = np.log1p(-np.exp(weights))
            weights, bias = weights - negative, bias + negative.sum(axis=1)
        elif isinstance(self, _sk.ComplementNB) and len(self.classes_) > 1:
            bias = np.zeros(len(self.classes_))
        reason = (
            "Nonfinite likelihood coefficients use sklearn"
            if not (np.isfinite(weights).all() and np.isfinite(bias).all())
            else None
        )
        runtime = select(self, "predict", X, reason=reason)
        if runtime is None:
            return super()._joint_log_likelihood(X)
        return runtime.linear(as_float32(X), as_float32(weights.T), bias)


GaussianNB = estimator("GaussianNB", _Gaussian, _sk.GaussianNB, __name__)
for _name in ("MultinomialNB", "BernoulliNB", "ComplementNB"):
    globals()[_name] = estimator(_name, _Discrete, getattr(_sk, _name), __name__)
