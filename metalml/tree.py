"""CPU tree construction, Metal dense prediction."""

from sklearn import tree as _sk

from ._prediction import estimator
from ._trees import TreeClassifier, TreeRegressor

__all__ = _sk.__all__


def __getattr__(name):
    return getattr(_sk, name)


for _name in ("DecisionTreeClassifier", "ExtraTreeClassifier"):
    globals()[_name] = estimator(_name, TreeClassifier, getattr(_sk, _name), __name__)
for _name in ("DecisionTreeRegressor", "ExtraTreeRegressor"):
    globals()[_name] = estimator(_name, TreeRegressor, getattr(_sk, _name), __name__)
