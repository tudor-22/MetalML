"""GPU-trained Naive Bayes classifiers with Metal likelihood inference."""

import numpy as np
from sklearn import naive_bayes as _sk
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_non_negative, column_or_1d, validate_data

from ._dispatch import BackendDiagnostics, as_float32, record, select

__all__ = _sk.__all__

# Dense feature counts beyond this width are cheaper to finish with sklearn.
_MAX_FEATURES = 8192


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


class GaussianNB(_Gaussian, _sk.GaussianNB):
    """GaussianNB with Metal class-conditional statistics and likelihoods."""

    def fit(self, X, y, sample_weight=None):
        self._validate_params()
        reason = None
        if sample_weight is not None:
            reason = "Weighted GaussianNB uses sklearn"
        elif self.priors is not None:
            reason = "Explicit priors use sklearn's GaussianNB validation"
        runtime = select(self, "fit", X, reason=reason)
        if runtime is None:
            return super().fit(X, y, sample_weight=sample_weight)

        y = np.asarray(y)
        if y.ndim != 1:
            select(self, "fit", X, reason="Non-1D targets use sklearn")
            return super().fit(X, y, sample_weight=sample_weight)
        check_classification_targets(y)
        classes = np.unique(y)
        if len(classes) < 2:
            select(self, "fit", X, reason="Single-class GaussianNB uses sklearn")
            return super().fit(X, y, sample_weight=sample_weight)

        x = as_float32(validate_data(self, X, dtype=np.float32, ensure_all_finite=False))
        labels = np.ascontiguousarray(np.searchsorted(classes, y).astype(np.int32))
        counts = np.bincount(labels, minlength=len(classes)).astype(np.float64)

        sums = runtime.class_sums(x, labels, len(classes))
        means = sums / counts[:, None]
        centered = runtime.class_centered_sums(x, labels, as_float32(means))

        # sklearn boosts variance by var_smoothing times the largest global
        # column variance, available from the existing GPU reduction.
        self.epsilon_ = self.var_smoothing * float(runtime.stats(x)[1].max())

        self.classes_ = classes
        self.class_count_ = counts
        self.theta_ = means.astype(np.float64)
        self.var_ = (centered / counts[:, None]).astype(np.float64) + self.epsilon_
        self.class_prior_ = self.class_count_ / self.class_count_.sum()
        record(self, "fit", "metal")
        return self


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


def _discrete_fit(self, X, y, sample_weight, source, binarize, check_nonneg):
    self._validate_params()
    reason = "Weighted discrete Naive Bayes uses sklearn" if sample_weight is not None else None
    runtime = select(self, "fit", X, reason=reason)
    if runtime is None:
        return source.fit(self, X, y, sample_weight=sample_weight)

    x, y = self._check_X_y(X, y)
    y = column_or_1d(np.asarray(y))
    check_classification_targets(y)
    classes = np.unique(y)
    if len(classes) < 2:
        select(self, "fit", X, reason="Single-class discrete Naive Bayes uses sklearn")
        return source.fit(self, X, y, sample_weight=sample_weight)
    if x.shape[1] > _MAX_FEATURES:
        select(self, "fit", X, reason="Very wide feature spaces use sklearn counts")
        return source.fit(self, X, y, sample_weight=sample_weight)
    if check_nonneg:
        check_non_negative(x, type(self).__name__)

    n_classes = len(classes)
    self.classes_ = classes
    self._init_counters(n_classes, x.shape[1])
    labels = np.ascontiguousarray(np.searchsorted(classes, y).astype(np.int32))

    # The GPU computes the event counts; sklearn's own smoothing and prior
    # updates then derive the log probabilities exactly as upstream.
    self.feature_count_ += runtime.class_sums(as_float32(x), labels, n_classes, binarize=binarize)
    self.class_count_ += np.bincount(labels, minlength=n_classes).astype(np.float64)
    alpha = self._check_alpha()
    if isinstance(self, _sk.ComplementNB):
        self.feature_all_ = self.feature_count_.sum(axis=0)
    self._update_feature_log_prob(alpha)
    self._update_class_log_prior(class_prior=self.class_prior)
    record(self, "fit", "metal")
    return self


class MultinomialNB(_Discrete, _sk.MultinomialNB):
    """MultinomialNB with Metal event counts and likelihoods."""

    def fit(self, X, y, sample_weight=None):
        return _discrete_fit(
            self, X, y, sample_weight, _sk.MultinomialNB, binarize=None, check_nonneg=True
        )


class BernoulliNB(_Discrete, _sk.BernoulliNB):
    """BernoulliNB with Metal event counts and likelihoods."""

    def fit(self, X, y, sample_weight=None):
        return _discrete_fit(
            self, X, y, sample_weight, _sk.BernoulliNB, binarize=self.binarize, check_nonneg=False
        )


class ComplementNB(_Discrete, _sk.ComplementNB):
    """ComplementNB with Metal event counts and likelihoods."""

    def fit(self, X, y, sample_weight=None):
        return _discrete_fit(
            self, X, y, sample_weight, _sk.ComplementNB, binarize=None, check_nonneg=True
        )
