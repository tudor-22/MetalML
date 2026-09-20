"""CPU forest training, fused Metal traversal and averaging for prediction."""

from sklearn import ensemble as _sk

from ._prediction import estimator
from ._trees import ForestClassifier, ForestRegressor

__all__ = _sk.__all__


def __getattr__(name):
    return getattr(_sk, name)


for _name in ("RandomForestClassifier", "ExtraTreesClassifier"):
    globals()[_name] = estimator(_name, ForestClassifier, getattr(_sk, _name), __name__)
for _name in ("RandomForestRegressor", "ExtraTreesRegressor"):
    globals()[_name] = estimator(_name, ForestRegressor, getattr(_sk, _name), __name__)
