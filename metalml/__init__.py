"""Change `import sklearn as ml` to `import metalml as ml`.

Public sklearn modules are forwarded lazily; selected estimator classes have
native Metal implementations. Existing sklearn imports are never modified.
"""

import importlib

import sklearn as _sklearn

from ._compat import install as _install
from ._config import config_context, get_config, set_config
from ._metal import MetalUnavailableError, backend_info

__version__ = "0.2.1"
sklearn_version = _sklearn.__version__
_install()


def __getattr__(name):
    if name in _sklearn.__all__:
        try:
            value = importlib.import_module("metalml." + name)
        except ModuleNotFoundError as exc:
            if exc.name != "metalml." + name:
                raise
            value = getattr(_sklearn, name)
        globals()[name] = value
        return value
    return getattr(_sklearn, name)


def __dir__():
    return sorted(set(globals()) | set(_sklearn.__all__))


__all__ = list(_sklearn.__all__) + [
    "backend_info",
    "sklearn_version",
    "MetalUnavailableError",
    "config_context",
    "get_config",
    "set_config",
]
