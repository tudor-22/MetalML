import numpy as np
from scipy import sparse

from ._config import get_config
from ._metal import MetalUnavailableError, probe


class BackendDiagnostics:
    @property
    def backend_(self):
        return self._metal_backend_state["backend"]

    def __sklearn_is_fitted__(self):
        # Diagnostics must never make an unsuccessful fit look successful.
        return hasattr(self, self._metal_fitted_attribute)


def record(estimator, operation, backend, reason=None):
    # These are diagnostics, not sklearn's fitted-state markers.
    if not hasattr(estimator, "operation_backends_"):
        estimator.operation_backends_ = {}
        estimator.fallback_reasons_ = {}
        estimator._metal_backend_state = {}
    estimator.operation_backends_[operation] = backend
    estimator.fallback_reasons_[operation] = reason
    estimator._metal_backend_state["backend"] = backend


def select(estimator, operation, x, *, reason=None):
    config = get_config()
    if config["backend"] == "cpu":
        record(estimator, operation, "cpu", "CPU explicitly requested")
        return None
    if reason is None:
        if sparse.issparse(x):
            reason = "Sparse inputs use sklearn"
        else:
            a = np.asarray(x)
            if a.ndim != 2 or a.size == 0 or a.dtype.kind not in "fiu":
                reason = "Metal requires a nonempty dense numeric matrix"
            elif a.dtype != np.float32 and config["precision"] == "preserve":
                reason = "Precision preserved; use float32 data or precision='float32'"
            elif not np.isfinite(a).all():
                reason = "Nonfinite inputs use sklearn validation and handling"
            elif a.dtype != np.float32 and np.max(np.abs(a)) > np.finfo(np.float32).max:
                reason = "Input exceeds float32 range"
    runtime = None
    if reason is None:
        runtime, reason = probe()
    if reason is not None:
        if config["backend"] == "metal":
            raise MetalUnavailableError(f"{type(estimator).__name__}.{operation}: {reason}")
        record(estimator, operation, "cpu", reason)
        return None
    record(estimator, operation, "metal")
    return runtime


def as_float32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def prefer_cpu(estimator, operation, runtime, condition, reason):
    """Conservative crossover rules measured on the local 5300M, never forced mode."""
    if get_config()["backend"] == "auto" and runtime.name == "AMD Radeon Pro 5300M" and condition:
        record(estimator, operation, "cpu", reason)
        return True
    return False


def copy_result(original, result, copy):
    if (
        not copy
        and isinstance(original, np.ndarray)
        and original.dtype.kind == "f"
        and original.flags.writeable
    ):
        original[...] = result
        return original
    return result
