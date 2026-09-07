from __future__ import annotations

from groundguard_rag import Chunk, VerificationRequest, VerificationState

from examples.demo_factory import build_groundguard


def test_demo_factory_runs_supported_and_not_checkable_paths():
    guard = build_groundguard()
    supported = guard.verify(
        VerificationRequest(
            "supported",
            "Paris is in France.",
            (Chunk("e1", "Paris is in France."),),
        )
    )
    greeting = guard.verify(
        VerificationRequest("greeting", "Hello!", (Chunk("e2", "Hello!"),))
    )
    assert supported.audit_report.verdicts[0].state is VerificationState.SUPPORTED
    assert (
        greeting.audit_report.verdicts[0].state
        is VerificationState.NOT_CHECKABLE
    )
