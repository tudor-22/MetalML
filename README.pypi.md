# MetalML

**Scikit-learn workflows, powered by Apple's Metal GPU backend.**

MetalML brings native Metal compute kernels and Metal Performance Shaders to
selected scikit-learn operations on macOS. Designed for Apple Silicon, it keeps
the familiar estimator interface and delegates unsupported configurations to
scikit-learn on the CPU.

## Install

```sh
python -m pip install metalml
```

Python 3.10 or later is required; Python 3.12 is recommended. MetalML supports
scikit-learn 1.6.x and 1.7.x. On Apple Silicon, use native ARM64 Python.
Dependencies include NumPy, SciPy, scikit-learn and, on macOS, the PyObjC Metal
and Metal Performance Shaders bindings.

The Metal shaders are included in the package and compile on the selected GPU
during its first operation. No CUDA, PyTorch, MLX or separate Xcode project is
required. Other operating systems retain sklearn CPU fallback.

## Quick start

```python
import numpy as np
import metalml as ml

rng = np.random.default_rng(42)
X = rng.normal(size=(10000, 64)).astype(np.float32)
y = 2 * X[:, 0] - X[:, 1]

model = ml.linear_model.Ridge(alpha=1.0).fit(X, y)
predictions = model.predict(X[:100])
print(model.operation_backends_)
print(ml.backend_info())
```

For workflows using `import sklearn as ml`, replace that import with
`import metalml as ml`. Direct imports work too:

```python
from metalml.preprocessing import StandardScaler
from metalml.linear_model import Ridge
from metalml.pipeline import make_pipeline

pipeline = make_pipeline(StandardScaler(), Ridge())
pipeline.fit(X, y)
```

Existing sklearn imports and previously created estimators are not patched.
Public unimplemented modules and classes forward to the installed sklearn.
Private sklearn imports are outside the compatibility surface.

## Supported GPU operations

| Family | Metal operations | CPU work |
| --- | --- | --- |
| StandardScaler, MinMaxScaler, Normalizer | Dense statistics, scaling, normalization | Incremental fitting and unsupported settings |
| KMeans | Initialization distances, Lloyd iterations, prediction and distances | Seed sampling and iteration control |
| Ridge | Gram/cross-products and prediction | Condition checks and linear-system solve |
| LogisticRegression (binary, L2) | IRLS training and prediction | Small solve, multiclass, other penalties and weights |
| LinearRegression | Normal-equation training and prediction | Ill-conditioned, wide, weighted or constrained fits |
| Supported linear models, linear SVMs and LDA | Linear scores and prediction | Training and output conversion where needed |
| PCA | Eligible covariance computation, projection and inverse projection | Eigensolver and other fitting configurations |
| TruncatedSVD | Projection and inverse projection | Training |
| MLPClassifier, MLPRegressor | Dense layers and activations | Training and label decoding |
| GaussianNB, MultinomialNB, BernoulliNB, ComplementNB | Class-statistics training and likelihood scores | Probability normalization |
| DecisionTree, ExtraTree, RandomForest, ExtraTrees | Traversal and forest averaging | Training and label decoding |
| SVC, NuSVC, SVR, NuSVR, OneClassSVM | Supported kernel scores | Training, voting and calibrated SVC probabilities |
| NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor | Exact Euclidean distances and top-k | Index storage, votes and regression aggregation |

Support is operation-specific. Ridge GPU fitting requires scalar positive
regularization, an eligible solver, no sample weights or positivity constraint,
at most 2,048 features and at least as many samples as features. Binary L2
logistic regression trains on the GPU with dense float32 input and at most
1,024 features; multiclass, other penalties and weighted fits use sklearn.
LinearRegression trains dense float32 normal equations with at most 2,048
features, falling back for ill-conditioned, wide, weighted or constrained
fits. Naive Bayes trains class-conditional statistics for dense float32 input
up to 8,192 features.
PCA covariance fitting requires an eligible tall matrix, at most 1,000 features
and supported component selection. Ill-conditioned fits use sklearn. Neighbors
require dense brute-force Euclidean search with at most 32 neighbors. Other
settings retain CPU behavior; strict Metal mode raises when a GPU operation must
fall back.

## Precision and backend selection

Metal uses float32. By default, float64/integer input preserves its precision
through CPU fallback, and sparse input is never silently densified. Use float32
arrays or explicitly allow conversion:

```python
ml.set_config(precision="float32")

with ml.config_context(backend="cpu"):
    reference = ml.linear_model.Ridge().fit(X, y)

with ml.config_context(backend="metal"):
    accelerated = ml.linear_model.Ridge().fit(X, y)
```

Automatic dispatch is the default. `METALML_BACKEND=cpu` disables GPU dispatch.
Strict Metal mode applies to implemented GPU operations; documented CPU
training and forwarded estimators remain on CPU. Shader compilation and GPU
execution errors raise explicitly.

MetalML wrappers expose `operation_backends_`, `fallback_reasons_` and `backend_`.
Forwarded sklearn classes retain their original interface.

## Status and limitations

MetalML is experimental and is not a complete GPU port of scikit-learn.
Execution has been verified on an AMD Radeon Pro 5300M. Apple Silicon is the
target platform, but physical Apple Silicon validation and performance tuning
are still pending. Other operating systems have not been tested on physical
hosts. No general speedup is promised.

Small operations can favor CPU. Current performance routing rules are specific
to the measured 5300M. Float32 arithmetic, tied distances and small classification
margins can produce differences from CPU results. Measure complete operations
and numerical accuracy on your own data and hardware.

Estimator boundaries return NumPy arrays; whole pipelines are not kept on the
GPU. Shared and device-local caches each retain up to 64 MiB, with additional
memory needed for active operations. GPU calls within a process are serialized.
Use spawn/loky rather than forking after Metal initialization.

Pipelines, cloning, cross-validation, metadata routing and pandas output use
the sklearn-compatible interface. Saved pickle/joblib models contain NumPy
state rather than GPU handles; MetalML must be installed to load them. Only
load serialized models from trusted sources.

## License

Apache License 2.0. Independent project; unaffiliated with Apple, NVIDIA or
scikit-learn.
