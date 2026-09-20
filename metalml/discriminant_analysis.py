"""Metal LDA decision scores; sklearn discriminant fitting."""

from sklearn import discriminant_analysis as _sk

from ._prediction import LinearClassifier, estimator

__all__ = _sk.__all__


def __getattr__(name):
    return getattr(_sk, name)


LinearDiscriminantAnalysis = estimator(
    "LinearDiscriminantAnalysis", LinearClassifier, _sk.LinearDiscriminantAnalysis, __name__
)
