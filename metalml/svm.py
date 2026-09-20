"""Metal kernel evaluation and scores; sklearn fits and calibrates SVMs."""

import numpy as np
from sklearn import svm as _sk
from sklearn.utils.multiclass import _ovr_decision_function
from sklearn.utils.validation import check_is_fitted, validate_data

from ._dispatch import BackendDiagnostics, as_float32, record, select
from ._prediction import LinearClassifier, LinearRegressor, estimator

__all__ = _sk.__all__


def __getattr__(name):
    return getattr(_sk, name)


class _Kernel(BackendDiagnostics):
    _metal_fitted_attribute = "support_"

    def _metal_scores(self, X, operation):
        check_is_fitted(self, "support_")
        reason = None
        if self._sparse or self.kernel not in ("linear", "rbf", "poly", "sigmoid"):
            reason = "Sparse, precomputed, or callable SVM kernels use sklearn"
        elif len(self.support_) == 0:
            reason = "Empty support-vector set uses sklearn"
        runtime = select(self, operation, X, reason=reason)
        if runtime is None:
            return None
        x = validate_data(self, X, reset=False, dtype=np.float32)
        if self._impl in ("c_svc", "nu_svc") and len(self.classes_) > 2:
            n = len(self.classes_)
            starts = np.r_[0, np.cumsum(self._n_support)]
            coef = np.zeros((len(self.support_), n * (n - 1) // 2), np.float32)
            pair = 0
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = slice(starts[i], starts[i + 1]), slice(starts[j], starts[j + 1])
                    coef[a, pair] = self.dual_coef_[j - 1, a]
                    coef[b, pair] = self.dual_coef_[i, b]
                    pair += 1
        else:
            coef = as_float32(self.dual_coef_.T)
        result = runtime.svm_scores(
            as_float32(x),
            self.support_vectors_,
            coef,
            self.intercept_,
            self.kernel,
            self._gamma,
            self.coef0,
            self.degree,
        )
        if not np.isfinite(result).all():
            select(self, operation, X, reason="SVM scores exceed float32 range; using sklearn")
            return None
        return result


class _Regressor(_Kernel):
    def predict(self, X):
        result = self._metal_scores(X, "predict")
        return super().predict(X) if result is None else result[:, 0]


class _Classifier(_Kernel):
    def decision_function(self, X):
        result = self._metal_scores(X, "decision_function")
        if result is None:
            return super().decision_function(X)
        if len(self.classes_) == 2:
            return result[:, 0]
        if self.decision_function_shape == "ovr":
            return _ovr_decision_function(result < 0, -result, len(self.classes_))
        return result

    def predict(self, X):
        check_is_fitted(self, "support_")
        if self.break_ties and self.decision_function_shape == "ovo":
            raise ValueError("break_ties must be False when decision_function_shape is 'ovo'")
        result = self._metal_scores(X, "predict")
        if result is None:
            return super().predict(X)
        if len(self.classes_) == 2:
            labels = (result[:, 0] > 0).astype(np.intp)
        elif self.break_ties:
            labels = _ovr_decision_function(result < 0, -result, len(self.classes_)).argmax(axis=1)
        else:
            votes = np.zeros((len(result), len(self.classes_)), np.int32)
            pair = 0
            for i in range(len(self.classes_)):
                for j in range(i + 1, len(self.classes_)):
                    votes[:, i] += result[:, pair] > 0
                    votes[:, j] += result[:, pair] <= 0
                    pair += 1
            labels = votes.argmax(axis=1)
        return self.classes_.take(labels)

    def _dense_predict_proba(self, X):
        record(self, "predict_proba", "cpu", "Calibrated SVM probabilities use sklearn")
        return super()._dense_predict_proba(X)


class _OneClass(_Kernel):
    def decision_function(self, X):
        result = self._metal_scores(X, "decision_function")
        return super().decision_function(X) if result is None else result[:, 0]

    def predict(self, X):
        result = self._metal_scores(X, "predict")
        if result is None:
            return super().predict(X)
        return np.where(result[:, 0] > 0, 1, -1)


for _name in ("SVC", "NuSVC"):
    globals()[_name] = estimator(_name, _Classifier, getattr(_sk, _name), __name__)
for _name in ("SVR", "NuSVR"):
    globals()[_name] = estimator(_name, _Regressor, getattr(_sk, _name), __name__)
OneClassSVM = estimator("OneClassSVM", _OneClass, _sk.OneClassSVM, __name__)
LinearSVC = estimator("LinearSVC", LinearClassifier, _sk.LinearSVC, __name__)


class _LinearSVR(LinearRegressor):
    def predict(self, X):
        return self._decision_function(X)


LinearSVR = estimator("LinearSVR", _LinearSVR, _sk.LinearSVR, __name__)
