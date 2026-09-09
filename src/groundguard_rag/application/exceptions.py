"""Application-layer exceptions."""

from __future__ import annotations

from groundguard_rag.domain.exceptions import GroundGuardError


class ApplicationInvariantError(GroundGuardError):
    """The application service reached an internally inconsistent state.

    Unlike ``AdapterContractError``, this exception does not blame an injected
    adapter's returned value. It protects orchestration invariants explicitly
    so they remain enforced when Python runs with optimization enabled.
    """


class AdapterContractError(GroundGuardError):
    """An injected adapter's *return value* violated its port contract.

    Raised by ``VerifyService`` when a ``ClaimDecomposer``, ``EvidenceSelector``,
    ``Verifier``, or ``Calibrator`` returns something that does not satisfy
    the contract described in its port's docstring -- wrong type, an
    out-of-range span, a reference to evidence never supplied, a verdict
    for a different claim, an illegal calibrated confidence, and so on.

    This is distinct from an adapter *raising* its own exception: that is
    never caught or swallowed here, it propagates to the caller unchanged.
    ``AdapterContractError`` is only raised by ``VerifyService`` itself,
    after inspecting a value an adapter successfully returned.
    """
