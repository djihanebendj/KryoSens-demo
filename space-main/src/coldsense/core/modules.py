"""Domain-module contract and registry.

ColdSense is a platform; the cold-room capability is one domain module.
A new domain (e.g. reefer trucks, vaccine fridges) registers itself here
instead of being wired into the app by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ModuleError
from .governance import ModelCard


@dataclass(frozen=True)
class DomainModule:
    """Static description of a domain module."""

    key: str
    title: str
    version: str
    description: str
    model_cards: tuple[ModelCard, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, DomainModule] = {}


def register(module: DomainModule) -> DomainModule:
    if module.key in _REGISTRY and _REGISTRY[module.key] != module:
        raise ModuleError(f"domain module '{module.key}' is already registered with different metadata")
    _REGISTRY[module.key] = module
    return module


def get(key: str) -> DomainModule:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise ModuleError(f"unknown domain module '{key}'; registered: {sorted(_REGISTRY)}") from None


def all_modules() -> list[DomainModule]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]
