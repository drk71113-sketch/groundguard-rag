from __future__ import annotations

import pytest

from groundguard_rag import (
    Chunk,
    GroundGuard,
    HealConfig,
    HealingOutput,
    VerificationOutput,
    VerificationRequest,
    heal,
    verify,
)
from groundguard_rag.adapters.heal import (
    CallbackRewriter,
    ConservativeRepairPolicy,
    HardBoundsOnlyStopPolicy,
)
from groundguard_rag.application.heal_service import HealService
from groundguard_rag.domain.exceptions import ConfigurationError
from tests.stage8._helpers import StateProgress, make_verify_service


def _request(answer: str) -> VerificationRequest:
    return VerificationRequest(
        request_id="request-1",
        answer=answer,
        chunks=(Chunk("evidence-1", "Paris is in France."),),
        query="Where is Paris?",
    )


def test_public_verify_facade_and_one_shot_helper_return_dual_views():
    service = make_verify_service()
    result = GroundGuard(verify_service=service).verify(
        _request("Paris is in France.")
    )
    one_shot = verify(_request("Paris is in France."), service=service)

    assert isinstance(result, VerificationOutput)
    assert result.views.sidecar.answer == "Paris is in France."
    assert result.audit_report.verdicts[0].state.value == "SUPPORTED"
    assert one_shot.to_dict() == result.to_dict()


def test_heal_requires_explicit_service_instead_of_silent_verify_fallback():
    with pytest.raises(ConfigurationError):
        GroundGuard(verify_service=make_verify_service()).heal(
            _request("Paris is in Germany.")
        )


def test_end_to_end_public_heal_rewrites_reverifies_and_returns_candidate_only():
    verify_service = make_verify_service()
    heal_service = HealService(
        verify_service=verify_service,
        repair_policy=ConservativeRepairPolicy(),
        stop_policy=HardBoundsOnlyStopPolicy(),
        progress_evaluator=StateProgress(),
        config=HealConfig(
            max_rounds=2,
            max_attempts_per_claim=2,
            timeout_seconds=5,
            cost_budget=1,
            min_improvement=0.5,
        ),
        monotonic=lambda: 0.0,
        rewriter=CallbackRewriter(
            lambda claim, evidence: "Paris is in France."
        ),
    )
    request = _request("Paris is in Germany.")
    result = GroundGuard(
        verify_service=verify_service, heal_service=heal_service
    ).heal(request)
    one_shot = heal(
        request,
        verify_service=verify_service,
        heal_service=heal_service,
    )

    assert isinstance(result, HealingOutput)
    assert result.candidate_answer == "Paris is in France."
    assert result.audit_report.stop_reason == "all_grounded"
    assert result.audit_report.repair_actions[0].committed is True
    assert result.views.sidecar.answer == result.candidate_answer
    assert request.answer == "Paris is in Germany."
    assert one_shot.candidate_answer == result.candidate_answer


def test_guard_rejects_mismatched_verify_and_heal_service_graphs():
    first = make_verify_service()
    second = make_verify_service()
    healer = HealService(
        verify_service=second,
        repair_policy=ConservativeRepairPolicy(),
        stop_policy=HardBoundsOnlyStopPolicy(),
        progress_evaluator=StateProgress(),
        config=HealConfig(1, 1, 1, 1, 0),
        monotonic=lambda: 0.0,
    )
    with pytest.raises(ConfigurationError):
        GroundGuard(verify_service=first, heal_service=healer)
