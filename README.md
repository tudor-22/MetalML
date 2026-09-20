# MetalML

Scikit-learn-compatible imports with native Metal GPU operations on macOS.
MetalML delegates unsupported estimators and configurations to scikit-learn.
It is an experimental partial GPU implementation, not a complete GPU port of
scikit-learn or a performance-equivalent replacement for RAPIDS cuML.

## Installation

Python 3.10 or later is required; Python 3.12 is recommended. Supported
scikit-learn versions are 1.6.x and 1.7.x. From this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For editable development, use `python -m pip install -e '.[dev]'`.
On Apple Silicon, use a native ARM64 Python environment. Create a fresh
virtual environment on each machine instead of copying one between Macs.

On macOS, installation includes the PyObjC Metal and Metal Performance Shaders
bindings. The bundled Metal source compiles on the destination GPU during the
first GPU operation. No CUDA, PyTorch, MLX, global sklearn patch, or separate
Xcode project is required.

The runtime targets Metal-capable Apple Silicon and Intel/AMD Macs. Physical
execution has been verified on an AMD Radeon Pro 5300M. Apple Silicon execution
and performance still need hardware verification. Other operating systems use
sklearn CPU fallback.

## Usage

```python
import numpy as np
import metalml as ml  # replaces: import sklearn as ml

rng = np.random.default_rng(42)
X = rng.normal(size=(10000, 64)).astype(np.float32)
y = 2 * X[:, 0] - X[:, 1]

model = ml.linear_model.Ridge().fit(X, y)
predictions = model.predict(X[:100])
print(model.operation_backends_)
print(ml.backend_info())
```

Imports such as `from metalml.linear_model import Ridge` also work. Change each
`from sklearn...` prefix to `metalml` when using that style. Existing sklearn
imports, previously created estimators and sklearn's internal estimator
factories are not patched. Public unimplemented APIs forward to sklearn;
private sklearn module imports are outside this compatibility surface.

Estimators inherit sklearn constructor signatures and parameter handling.
Pipelines, cloning, cross-validation, pandas output, metadata routing, and
pickle/joblib serialization use the sklearn-compatible estimator interface.
Saved estimators contain NumPy state, not device handles; MetalML must be
installed when loading them.

## GPU coverage

GPU support is operation-specific and often covers inference only.

| Family | GPU operations | CPU operations / limits |
| --- | --- | --- |
| StandardScaler, MinMaxScaler, Normalizer | Dense scaling, statistics and normalization | Incremental fitting and unsupported configurations |
| KMeans | Initialization distances, Lloyd iterations, prediction and distances | Seed sampling/control, Elkan and unsupported initialization/settings |
| Ridge | Gram/cross-products and prediction | Condition checks, small-system solve and unsupported fits |
| Other supported linear models and linear SVMs | Linear scores and prediction | Training; uncertainty, labels and inverse links as applicable |
| PCA | Eligible tall-matrix covariance and projection | Eigensolver and other solvers/configurations |
| TruncatedSVD | Projection | Training |
| MLPClassifier, MLPRegressor | Dense layers and activations | Training and label decoding |
| GaussianNB, MultinomialNB, BernoulliNB, ComplementNB | Likelihood scores | Training and probability normalization |
| DecisionTree, ExtraTree, RandomForest, ExtraTrees | Traversal and forest averaging | Training, missing values and multi-output classification |
| SVC, NuSVC, SVR, NuSVR, OneClassSVM | Supported kernel scores | Training, voting and calibrated SVC probabilities |
| NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor | Exact brute-force Euclidean distance and top-k | Training/index storage, voting and regression aggregation |
| LinearDiscriminantAnalysis | Decision scores | Training and probability conversion |
| Other estimators | Original sklearn behavior | CPU |

Ridge GPU fitting requires scalar positive alpha, `auto` or `cholesky`, no
sample weights or positivity constraint, at most 2,048 features, and at least
as many samples as features. Ill-conditioned systems retain sklearn's robust
solver path.

PCA covariance fitting supports `auto`/`covariance_eigh`, `copy=True`, integer
or `None` component counts, at most 1,000 features, and at least ten times as
many samples as features. Poorly conditioned covariance uses sklearn.

Neighbors require dense brute-force Euclidean search with at most 32 neighbors.
Other metrics, tree indexes and self-excluding `X=None` queries use sklearn.
Equal-distance GPU ties use the lowest reference index; sklearn tie ordering
can differ. SVM kernels support `linear`, `rbf`, `poly` and `sigmoid`; callable,
precomputed and sparse kernels use sklearn.

## Precision and backend control

Metal computes in float32. By default, float64 and integer inputs retain their
precision through CPU fallback, and sparse inputs are never silently densified.
Explicitly opt into float32 conversion when suitable for the application:

```python
ml.set_config(precision="float32")

with ml.config_context(backend="cpu"):
    reference = ml.linear_model.Ridge().fit(X, y)

with ml.config_context(backend="metal"):
    accelerated = ml.linear_model.Ridge().fit(X, y)
```

`backend="auto"` is the default. It uses eligibility checks and fixed performance
rules for the measured Radeon Pro 5300M. Those timing rules are not calibrated
for Apple Silicon or other GPUs. `METALML_BACKEND=cpu` disables dispatch,
including in worker processes.

Strict Metal mode raises `MetalUnavailableError` when an implemented accelerated
operation must fall back. It does not turn documented CPU training or forwarded
sklearn estimators into GPU implementations. Unexpected shader compilation or
GPU execution errors raise rather than silently hiding failures.

Use `operation_backends_`, `fallback_reasons_`, and `backend_` on MetalML estimator
wrappers to inspect dispatch. Forwarded sklearn classes do not gain these
attributes. `backend_info()` reports the device and runtime counters.

## Performance and limits

GPU execution is not always faster than CPU. Dataset size, model structure,
transfers, synchronization and hardware determine the result. Measure complete
operations on the intended machine. The first GPU call includes compilation;
later calls reuse compiled pipelines and bounded storage pools.

MetalML returns NumPy arrays at estimator boundaries. Pipelines are not kept
entirely on the GPU. Device-local and shared buffer caches are each bounded at
64 MiB; active inputs, outputs and workspaces need additional memory. GPU work
within a process is serialized; multiple processes can contend for the device.
Use spawn/loky instead of forking after Metal initialization.

Float32 reductions and tied distances can change numerical results or labels.
Results are tolerance-compatible, not bit-for-bit copies. GPU training for
most model families, sparse GPU algorithms and broader device calibration
remain future work.

## Build

```sh
python -m pip install '.[dev]'
python -m build
```

This repository contains the Python package, required Metal shaders and
packaging files. Local environments, benchmark data and validation reports
are excluded.

## License

See [LICENSE](LICENSE). Independent project; not affiliated with scikit-learn,
Apple or NVIDIA.
