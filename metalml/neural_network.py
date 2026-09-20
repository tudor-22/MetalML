"""Sklearn MLP training with all inference layers batched on Metal."""

import numpy as np
from sklearn import neural_network as _sk
from sklearn.utils.validation import check_is_fitted, validate_data

from ._dispatch import BackendDiagnostics, as_float32, select
from ._prediction import estimator

__all__ = _sk.__all__


def __getattr__(name):
    return getattr(_sk, name)


class _MLP(BackendDiagnostics):
    _metal_fitted_attribute = "coefs_"

    def _forward_pass_fast(self, X, check_input=True):
        check_is_fitted(self, "coefs_")
        runtime = select(self, "predict", X)
        if runtime is None:
            return super()._forward_pass_fast(X, check_input=check_input)
        x = validate_data(self, X, reset=False, dtype=np.float32) if check_input else X
        activations = [self.activation] * (len(self.coefs_) - 1) + [self.out_activation_]
        return runtime.dense_layers(as_float32(x), self.coefs_, self.intercepts_, activations)


MLPClassifier = estimator("MLPClassifier", _MLP, _sk.MLPClassifier, __name__)
MLPRegressor = estimator("MLPRegressor", _MLP, _sk.MLPRegressor, __name__)
