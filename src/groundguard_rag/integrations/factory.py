"""Load an explicitly named application factory for CLI integrations."""

from __future__ import annotations

import importlib
from typing import Any

from groundguard_rag.api import GroundGuard
from groundguard_rag.domain.exceptions import ConfigurationError


def load_guard(factory_spec: str) -> GroundGuard:
    """Load ``module:attribute`` and call it to obtain an assembled guard."""

    if not isinstance(factory_spec, str) or factory_spec.count(":") != 1:
        raise ConfigurationError("factory must use module:attribute syntax")
    module_name, attribute_name = factory_spec.split(":", 1)
    if not module_name.strip() or not attribute_name.strip():
        raise ConfigurationError("factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory: Any = getattr(module, attribute_name)
    guard = factory() if callable(factory) else factory
    if not isinstance(guard, GroundGuard):
        raise ConfigurationError("factory must return a GroundGuard instance")
    return guard
