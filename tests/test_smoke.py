"""End-to-end smoke tests: the Metal patch must not break the sklearn surface."""

import numpy as np
import pytest

import metalml as ml

pytestmark = pytest.mark.skipif(
    not ml.backend_info()["available"], reason="Metal GPU unavailable"
)


def regression_data(n=500, d=8):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x @ rng.normal(size=d).astype(np.float32)).astype(np.float32)
    return x, y


def test_version_and_backend_info():
    assert isinstance(ml.__version__, str)
    info = ml.backend_info()
    assert info["available"] is True
    assert info["device"]


def test_preprocessing_roundtrip():
    x, _ = regression_data()
    scaled = ml.preprocessing.StandardScaler().fit_transform(x)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-3)


def test_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge as SklearnRidge

    x, y = regression_data()
    model = ml.linear_model.Ridge(alpha=1.0).fit(x, y)
    reference = SklearnRidge(alpha=1.0).fit(x, y)
    np.testing.assert_allclose(model.predict(x), reference.predict(x), atol=1e-2)


def test_kmeans_labels_shape():
    x, _ = regression_data(n=400, d=8)
    labels = ml.cluster.KMeans(n_clusters=4, n_init=3, random_state=0).fit_predict(x)
    assert labels.shape == (len(x),)
    assert len(np.unique(labels)) <= 4


def test_pca_transform_shape():
    x, _ = regression_data(n=600, d=12)
    pca = ml.decomposition.PCA(n_components=4).fit(x)
    assert pca.transform(x).shape == (len(x), 4)


def test_pipeline_forwarding():
    x, y = regression_data()
    pipeline = ml.pipeline.make_pipeline(ml.preprocessing.StandardScaler(), ml.linear_model.Ridge())
    pipeline.fit(x, y)
    assert pipeline.predict(x).shape == (len(x),)


def test_unknown_sklearn_symbol_forwards():
    # A public sklearn name not overridden by MetalML still resolves.
    assert ml.metrics.accuracy_score([0, 1, 1], [0, 1, 0]) == 2 / 3
