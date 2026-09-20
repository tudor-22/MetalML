"""Lazy module proxies; never monkeypatch or reload sklearn implementation files."""

import importlib
import importlib.abc
import importlib.util
import sys
import types


class _Proxy(types.ModuleType):
    def __getattr__(self, name):
        source = self.__dict__.get("_source")
        if source is None:
            raise AttributeError(name)
        value = getattr(source, name)
        if isinstance(value, types.ModuleType) and value.__name__.startswith("sklearn."):
            tail = value.__name__[len("sklearn.") :]
            if not any(part.startswith("_") for part in tail.split(".")):
                return importlib.import_module("metalml." + tail)
        return value

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._source)))


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return _Proxy(spec.name)

    def exec_module(self, module):
        source = importlib.import_module("sklearn." + module.__name__[len("metalml.") :])
        module._source = source
        module.__doc__ = source.__doc__
        if hasattr(source, "__path__"):
            module.__path__ = []
        if hasattr(source, "__all__"):
            module.__all__ = source.__all__


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("metalml."):
            return None
        tail = fullname[len("metalml.") :]
        if any(part.startswith("_") for part in tail.split(".")):
            return None
        # Real MetalML files get first refusal; this finder is appended after PathFinder.
        try:
            spec = importlib.util.find_spec("sklearn." + tail)
        except (ModuleNotFoundError, AttributeError):
            return None
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _Loader(), is_package=spec.submodule_search_locations is not None
        )


def install():
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.append(_Finder())
