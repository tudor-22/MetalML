"""Context-local dispatch settings, independent of sklearn's own settings."""

import os
from contextlib import contextmanager
from contextvars import ContextVar

import sklearn

_config = ContextVar("metalml_config", default=None)


def get_config():
    return {
        **sklearn.get_config(),
        **(
            _config.get()
            or {
                "backend": os.environ.get("METALML_BACKEND", "auto"),
                "precision": "preserve",
            }
        ),
    }


def set_config(*, backend=None, precision=None, **sklearn_options):
    """backend: auto/cpu/metal; precision: preserve or float32.

    Strict metal mode raises if an accelerated operation cannot use Metal.
    Unimplemented sklearn classes are always ordinary CPU classes.
    """
    current = get_config()
    config = {key: current[key] for key in ("backend", "precision")}
    if backend is not None:
        if backend not in {"auto", "cpu", "metal"}:
            raise ValueError("backend must be 'auto', 'cpu', or 'metal'")
        config["backend"] = backend
    if precision is not None:
        if precision not in {"preserve", "float32"}:
            raise ValueError("precision must be 'preserve' or 'float32'")
        config["precision"] = precision
    sklearn.set_config(**sklearn_options)
    _config.set(config)


@contextmanager
def config_context(*, backend=None, precision=None, **sklearn_options):
    token = _config.set(_config.get())
    try:
        with sklearn.config_context(**sklearn_options):
            set_config(backend=backend, precision=precision)
            yield
    finally:
        _config.reset(token)
