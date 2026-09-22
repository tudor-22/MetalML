"""Tests for GPU-trained Naive Bayes and LinearRegression estimators."""

import numpy as np
import pytest
from sklearn import naive_bayes as sk_nb
from sklearn.linear_model import LinearRegression as SklearnLinear

import metalml
from metalml import naive_bayes as ml_nb
from metalml.linear_model import LinearRegression

pytestmark = pytest.mark.skipif(
    not metalml.backend_info()["available"], reason="Metal GPU unavailable"
)


def gaussian_data(n=6000, d=24, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    return x, y, rng


def regression_data(n=8000, d=24, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x @ rng.normal(size=d).astype(np.float32)).astype(np.float32)
    return x, y


# --------------------------------------------------------------------------
# Naive Bayes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["GaussianNB", "MultinomialNB", "BernoulliNB", "ComplementNB"],
)
def test_naive_bayes_matches_sklearn(name):
    x, y, rng = gaussian_data()
    data = np.abs(x) if name in ("MultinomialNB", "ComplementNB") else x
    reference = getattr(sk_nb, name)().fit(data, y)
    model = getattr(ml_nb, name)().fit(data, y)

    assert model.operation_backends_["fit"] == "metal"
    assert np.mean(model.predict(data) == reference.predict(data)) > 0.995
    np.testing.assert_allclose(model.predict_log_proba(data), reference.predict_log_proba(data), atol=1e-3)
    assert model.classes_.tolist() == reference.classes_.tolist()


def test_gaussian_nb_parameters_match():
    x, y, _ = gaussian_data()
    reference = sk_nb.GaussianNB().fit(x, y)
    model = ml_nb.GaussianNB().fit(x, y)
    np.testing.assert_allclose(model.theta_, reference.theta_, atol=1e-4)
    np.testing.assert_allclose(model.var_, reference.var_, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(model.class_prior_, reference.class_prior_)


def test_multinomial_counts_match():
    x, y, _ = gaussian_data()
    x = np.abs(x)
    reference = sk_nb.MultinomialNB().fit(x, y)
    model = ml_nb.MultinomialNB().fit(x, y)
    np.testing.assert_allclose(model.feature_count_, reference.feature_count_, atol=1e-2)
    np.testing.assert_allclose(model.class_count_, reference.class_count_)


def test_bernoulli_binarize_and_alpha():
    x, y, _ = gaussian_data()
    reference = sk_nb.BernoulliNB(binarize=0.2, alpha=0.5).fit(x, y)
    model = ml_nb.BernoulliNB(binarize=0.2, alpha=0.5).fit(x, y)
    assert model.operation_backends_["fit"] == "metal"
    np.testing.assert_allclose(model.feature_count_, reference.feature_count_, atol=1e-2)
    assert np.mean(model.predict(x) == reference.predict(x)) > 0.99


def test_naive_bayes_runs_on_metal():
    x, y, _ = gaussian_data()
    before = metalml.backend_info()["dispatches"]
    ml_nb.GaussianNB().fit(x, y)
    assert metalml.backend_info()["dispatches"] > before


@pytest.mark.parametrize("name", ["GaussianNB", "MultinomialNB", "BernoulliNB", "ComplementNB"])
def test_naive_bayes_sample_weight_falls_back(name):
    x, y, _ = gaussian_data()
    data = np.abs(x) if name in ("MultinomialNB", "ComplementNB") else x
    model = getattr(ml_nb, name)().fit(data, y, sample_weight=np.ones(len(y)))
    assert model.operation_backends_["fit"] == "cpu"


def test_naive_bayes_float64_falls_back():
    x, y, _ = gaussian_data()
    model = ml_nb.GaussianNB().fit(x.astype(np.float64), y)
    assert model.operation_backends_["fit"] == "cpu"


def test_gaussian_nb_explicit_priors_fall_back():
    x, y, _ = gaussian_data()
    model = ml_nb.GaussianNB(priors=[0.5, 0.5]).fit(x, y)
    assert model.operation_backends_["fit"] == "cpu"


# --------------------------------------------------------------------------
# LinearRegression
# --------------------------------------------------------------------------


def test_linear_regression_matches_sklearn():
    x, y = regression_data()
    reference = SklearnLinear().fit(x, y)
    model = LinearRegression().fit(x, y)

    assert model.operation_backends_["fit"] == "metal+cpu"
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(model.intercept_, reference.intercept_, atol=1e-3)
    assert model.rank_ == reference.rank_
    np.testing.assert_allclose(model.singular_, reference.singular_, rtol=1e-3)
    np.testing.assert_allclose(model.predict(x), reference.predict(x), atol=1e-2)


def test_linear_regression_no_intercept():
    x, y = regression_data()
    reference = SklearnLinear(fit_intercept=False).fit(x, y)
    model = LinearRegression(fit_intercept=False).fit(x, y)
    assert model.operation_backends_["fit"] == "metal+cpu"
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-4, rtol=1e-4)


def test_linear_regression_multioutput():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4000, 16)).astype(np.float32)
    y = np.c_[x @ rng.normal(size=16), x @ rng.normal(size=16)].astype(np.float32)
    reference = SklearnLinear().fit(x, y)
    model = LinearRegression().fit(x, y)
    assert model.coef_.shape == (2, 16)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "make,data",
    [
        pytest.param(lambda: LinearRegression(positive=True), "plain", id="positive"),
        pytest.param(lambda: LinearRegression(), "weighted", id="sample_weight"),
        pytest.param(lambda: LinearRegression(), "float64", id="float64"),
        pytest.param(lambda: LinearRegression(), "collinear", id="ill_conditioned"),
    ],
)
def test_linear_regression_fallbacks(make, data):
    x, y = regression_data()
    kwargs = {}
    if data == "weighted":
        kwargs["sample_weight"] = np.ones(len(y))
    elif data == "float64":
        x = x.astype(np.float64)
    elif data == "collinear":
        x = x.copy()
        x[:, 1] = x[:, 0]
    model = make().fit(x, y, **kwargs)
    assert model.operation_backends_["fit"] == "cpu"
    assert model.fallback_reasons_["fit"]
