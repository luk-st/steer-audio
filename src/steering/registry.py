"""Method registry: ``name -> Controller subclass``.

Concrete steering methods register themselves so we can drive evaluation /
comparison loops generically::

    for name in list_methods():
        ctrl = get_method(name).from_pretrained(...)
        with model.steer(ctrl):
            ...
"""

from __future__ import annotations

from typing import Type

from .controller import Controller

_REGISTRY: dict[str, Type[Controller]] = {}


def register_method(name: str):
    """Class decorator that registers a Controller subclass under ``name``."""

    def _decorator(cls: Type[Controller]) -> Type[Controller]:
        if not issubclass(cls, Controller):
            raise TypeError(f"{cls.__name__} must subclass Controller to register.")
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"Method {name!r} is already registered to {_REGISTRY[name].__name__}."
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_method(name: str) -> Type[Controller]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown steering method {name!r}. Registered: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name]


def list_methods() -> list[str]:
    return sorted(_REGISTRY)
