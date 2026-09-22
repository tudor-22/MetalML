"""Tests for GPU (Metal) IRLS training of binary logistic regression.

These exercise the new training path added on top of MetalML's inference-only
runtime: the Newton weighted Gram and score products run on the GPU, and any
configuration the GPU path does not support must fall back cleanly to sklearn.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SklearnLogistic

import metalml
from metalml.linear_model import LogisticRegression

pytestmark = pytest.mark.skipif(
    not metalml.backend_info()["available"], reason="Metal GPU unavailable"
)


def binary_data(n=4000, d=24, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d).astype(np.float32)
    y = (x @ w + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    return x, y


def multiclass_data(n=3000, d=20, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d, classes)).astype(np.float32)
    y = np.argmax(x @ w + 0.3 * rng.normal(size=(n, classes)), axis=1)
    return x, y


def test_binary_training_matches_sklearn():
    x, y = binary_data()
    reference = SklearnLogistic(max_iter=1000, tol=1e-6).fit(x, y)
    model = LogisticRegression(max_iter=100, tol=1e-6).fit(x, y)

    np.testing.assert_allclose(model.coef_, reference.coef_, atol=2e-3, rtol=1e-3)
    np.testing.assert_allclose(model.intercept_, reference.intercept_, atol=2e-3)
    assert np.mean(model.predict(x) == reference.predict(x)) > 0.995
    np.testing.assert_allclose(model.predict_proba(x), reference.predict_proba(x), atol=1e-3)


def test_training_runs_on_metal():
    x, y = binary_data()
    before = metalml.backend_info()["dispatches"]
    model = LogisticRegression().fit(x, y)
    after = metalml.backend_info()["dispatches"]

    assert model.operation_backends_["fit"] == "metal+cpu"
    assert after > before  # the GPU actually executed work
    assert 0 < int(model.n_iter_[0]) <= model.max_iter


def test_fit_intercept_false():
    x, y = binary_data()
    reference = SklearnLogistic(fit_intercept=False, max_iter=1000, tol=1e-6).fit(x, y)
    model = LogisticRegression(fit_intercept=False, max_iter=100, tol=1e-6).fit(x, y)
    assert model.operation_backends_["fit"] == "metal+cpu"
    assert model.intercept_[0] == 0.0
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=2e-3, rtol=1e-3)


def test_unpenalized_training_matches_sklearn():
    x, y = binary_data(d=16)
    reference = SklearnLogistic(penalty=None, max_iter=1000, tol=1e-6).fit(x, y)
    model = LogisticRegression(penalty=None, max_iter=100, tol=1e-6).fit(x, y)
    assert model.operation_backends_["fit"] == "metal+cpu"
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=3e-3, rtol=1e-3)


def test_probabilities_are_valid():
    x, y = binary_data()
    proba = LogisticRegression().fit(x, y).predict_proba(x)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: LogisticRegression(penalty="l1", solver="liblinear"), id="l1"),
        pytest.param(lambda: LogisticRegression(class_weight="balanced"), id="class_weight"),
        pytest.param(
            lambda: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=50),
            id="elasticnet",
        ),
    ],
)
def test_unsupported_configs_fall_back(make):
    x, y = binary_data()
    model = make().fit(x, y)
    assert model.operation_backends_["fit"] == "cpu"
    assert model.fallback_reasons_["fit"]


def test_l2_objective_is_solver_independent():
    # Any L2 binary solver targets the same optimum, so the GPU IRLS path is
    # valid even when the estimator was configured with a different solver.
    x, y = binary_data(d=16)
    reference = SklearnLogistic(solver="lbfgs", max_iter=2000, tol=1e-8).fit(x, y)
    model = LogisticRegression(solver="saga", max_iter=100, tol=1e-8).fit(x, y)
    assert model.operation_backends_["fit"] == "metal+cpu"
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=3e-3, rtol=1e-3)


def test_multiclass_falls_back():
    x, y = multiclass_data(classes=3)
    model = LogisticRegression().fit(x, y)
    assert model.operation_backends_["fit"] == "cpu"
    assert "Multiclass" in model.fallback_reasons_["fit"]


def test_sample_weight_falls_back():
    x, y = binary_data()
    model = LogisticRegression().fit(x, y, sample_weight=np.ones(len(y)))
    assert model.operation_backends_["fit"] == "cpu"


def test_warm_start_falls_back():
    x, y = binary_data()
    model = LogisticRegression(warm_start=True).fit(x, y)
    assert model.operation_backends_["fit"] == "cpu"


def test_float64_preserves_precision_by_default():
    x, y = binary_data()
    model = LogisticRegression().fit(x.astype(np.float64), y)
    assert model.operation_backends_["fit"] == "cpu"


def test_precision_float32_allows_float64_input():
    x, y = binary_data()
    with metalml.config_context(precision="float32"):
        model = LogisticRegression().fit(x.astype(np.float64), y)
    assert model.operation_backends_["fit"] == "metal+cpu"
    assert model.coef_.dtype == np.float32


def test_strict_metal_raises_on_ineligible_input():
    x, y = binary_data()
    with metalml.config_context(backend="metal"), pytest.raises(metalml.MetalUnavailableError):
        LogisticRegression().fit(x.astype(np.float64), y)


def test_cpu_backend_uses_sklearn():
    x, y = binary_data()
    with metalml.config_context(backend="cpu"):
        model = LogisticRegression().fit(x, y)
    assert model.operation_backends_["fit"] == "cpu"


def test_sklearn_clone_and_get_params():
    from sklearn.base import clone

    model = LogisticRegression(C=0.5, max_iter=200, tol=1e-5)
    clone(model).get_params()  # round-trips without the Metal patch leaking
