"""Tree traversal adapters shared by individual trees and forest ensembles."""

import numpy as np
from sklearn.utils.validation import check_is_fitted, validate_data

from ._dispatch import BackendDiagnostics, as_float32, select


def scores(model, X, classifier, forest, check_input=True):
    check_is_fitted(model, "estimators_" if forest else "tree_")
    reason = (
        "Multi-output classification uses sklearn" if classifier and model.n_outputs_ != 1 else None
    )
    runtime = select(model, "predict_proba" if classifier else "predict", X, reason=reason)
    if runtime is None:
        return None
    x = validate_data(model, X, reset=False, dtype=np.float32) if check_input else X
    trees = [t.tree_ for t in model.estimators_] if forest else [model.tree_]
    result = runtime.tree_scores(as_float32(x), trees, classifier)
    if not np.isfinite(result).all():
        select(
            model,
            "predict_proba" if classifier else "predict",
            X,
            reason="Tree values exceed float32 range; using sklearn",
        )
        return None
    return result


class TreeClassifier(BackendDiagnostics):
    _metal_fitted_attribute = "tree_"

    def predict_proba(self, X, check_input=True):
        result = scores(self, X, True, False, check_input)
        return super().predict_proba(X, check_input=check_input) if result is None else result

    def predict(self, X, check_input=True):
        result = scores(self, X, True, False, check_input)
        if result is None:
            return super().predict(X, check_input=check_input)
        return self.classes_.take(result.argmax(axis=1), axis=0)


class TreeRegressor(BackendDiagnostics):
    _metal_fitted_attribute = "tree_"

    def predict(self, X, check_input=True):
        result = scores(self, X, False, False, check_input)
        if result is None:
            return super().predict(X, check_input=check_input)
        return result[:, 0] if self.n_outputs_ == 1 else result


class ForestClassifier(BackendDiagnostics):
    _metal_fitted_attribute = "estimators_"

    def predict_proba(self, X):
        result = scores(self, X, True, True)
        return super().predict_proba(X) if result is None else result


class ForestRegressor(BackendDiagnostics):
    _metal_fitted_attribute = "estimators_"

    def predict(self, X):
        result = scores(self, X, False, True)
        if result is None:
            return super().predict(X)
        return result[:, 0] if self.n_outputs_ == 1 else result
