"""Domain-level exceptions.

Kept intentionally small for stage 0: only what construction/validation of
the stage-0 contracts actually needs. Later stages (decomposition, retrieval,
repair policies, ...) will add their own exception types as that behavior is
implemented -- they must not be pre-declared here as placeholders.
"""

from __future__ import annotations


class GroundGuardError(Exception):
    """Base class for all GroundGuard-RAG domain errors."""


class DomainValidationError(GroundGuardError):
    """A domain model was constructed with invalid or inconsistent data.

    Raised from ``__post_init__`` validators on the frozen dataclasses in
    :mod:`groundguard_rag.domain.models`. Deliberately not caught and
    swallowed anywhere in this package -- invalid domain state must fail
    loudly at construction time rather than propagate silently.
    """


class ConfigurationError(GroundGuardError):
    """A configuration model was constructed with invalid or inconsistent data."""
