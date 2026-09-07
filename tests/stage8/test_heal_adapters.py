from __future__ import annotations

import math

import pytest

from groundguard_rag.adapters.heal import (
    CalibratedProgressCallback,
    CallbackRetriever,
    CallbackRewriter,
    ConservativeRepairPolicy,
    ConservativeRepairPolicyConfig,
    HardBoundsOnlyStopPolicy,
)
from groundguard_rag.domain.enums import RepairAction, VerificationState
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import (
    AtomicClaim,
    Chunk,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceReference,
    HealPolicyState,
    LabelScores,
)


def _verdict(state: VerificationState) -> ClaimVerdict:
    claim = AtomicClaim("c1", "claim", 0, 5)
    if state in {VerificationState.SUPPORTED, VerificationState.CONTRADICTED}:
        values = (
            (0.9, 0.05, 0.05)
            if state is VerificationState.SUPPORTED
            else (0.05, 0.9, 0.05)
        )
        assessments = (
            EvidenceAssessment(
                reference=EvidenceReference("e1"),
                label_scores=LabelScores(*values, score_kind="probabilities"),
                state=state,
                rationale=None,
                verifier_id="test",
                verifier_revision="v1",
            ),
        )
    else:
        assessments = ()
    return ClaimVerdict(claim=claim, state=state, evidence_assessments=assessments)


def _state(
    *, attempts: int = 0, retriever: bool = True, rewriter: bool = True
) -> HealPolicyState:
    return HealPolicyState(
        round=1,
        completed_rounds=0,
        target_claim_id="c1",
        attempts_for_claim=attempts,
        attempts_by_claim={"c1": attempts},
        elapsed_seconds=0.0,
        estimated_cost_spent=0.0,
        estimated_cost_remaining=5.0,
        state_hash="sha256:test",
        request_has_query=True,
        retriever_available=retriever,
        rewriter_available=rewriter,
        verdict_states=(VerificationState.INSUFFICIENT_EVIDENCE,),
        abstained_claim_ids=(),
        accepted_claim_ids=(),
        max_rounds=3,
        max_attempts_per_claim=3,
        timeout_seconds=10.0,
        cost_budget=5.0,
        min_improvement=0.1,
    )


def test_conservative_policy_prefers_retrieval_for_missing_evidence():
    policy = ConservativeRepairPolicy(
        ConservativeRepairPolicyConfig(retrieval_estimated_cost=0.25)
    )
    decision = policy.decide(
        _verdict(VerificationState.INSUFFICIENT_EVIDENCE), _state()
    )
    assert decision.action is RepairAction.RETRIEVE
    assert decision.estimated_cost == 0.25


def test_conservative_policy_prefers_rewrite_for_contradiction():
    policy = ConservativeRepairPolicy(
        ConservativeRepairPolicyConfig(rewrite_estimated_cost=0.5)
    )
    decision = policy.decide(
        _verdict(VerificationState.CONTRADICTED), _state()
    )
    assert decision.action is RepairAction.REWRITE
    assert decision.estimated_cost == 0.5


def test_conservative_policy_abstains_without_safe_capability():
    decision = ConservativeRepairPolicy().decide(
        _verdict(VerificationState.INSUFFICIENT_EVIDENCE),
        _state(retriever=False, rewriter=False),
    )
    assert decision.action is RepairAction.ABSTAIN
    assert decision.estimated_cost == 0


@pytest.mark.parametrize("value", [-1, math.inf, math.nan, True])
def test_policy_config_rejects_invalid_estimated_cost(value):
    with pytest.raises(ConfigurationError):
        ConservativeRepairPolicyConfig(retrieval_estimated_cost=value)


def test_hard_bounds_only_policy_never_relaxes_or_adds_a_stop():
    assert HardBoundsOnlyStopPolicy().should_stop(_state()) is False
    with pytest.raises(DomainValidationError):
        HardBoundsOnlyStopPolicy().should_stop({})


def test_callback_adapters_are_explicit_and_copy_evidence_list():
    chunk = Chunk("new", "new evidence")
    retriever = CallbackRetriever(lambda query: [chunk])
    assert retriever.retrieve("query") == [chunk]

    observed = []

    def rewrite(claim, evidence):
        observed.append(evidence)
        return "replacement"

    rewriter = CallbackRewriter(rewrite)
    source = []
    assert rewriter.rewrite(AtomicClaim("c1", "claim", 0, 5), source) == "replacement"
    assert observed[0] is not source


def test_callback_adapters_validate_inputs_but_leave_output_to_heal_service():
    with pytest.raises(ConfigurationError):
        CallbackRetriever(None)
    with pytest.raises(DomainValidationError):
        CallbackRetriever(lambda query: []).retrieve(" ")
    with pytest.raises(DomainValidationError):
        CallbackRewriter(lambda claim, evidence: "x").rewrite("claim", [])


def test_calibrated_progress_callback_requires_traceable_metadata():
    evaluator = CalibratedProgressCallback(
        lambda verdict: 0.75,
        calibrator_id="groundedness-platt",
        calibrator_revision="sha256:fitted-artifact",
    )
    assert evaluator.evaluate(
        _verdict(VerificationState.INSUFFICIENT_EVIDENCE)
    ) == 0.75
    assert evaluator.calibrator_id == "groundedness-platt"
    assert evaluator.calibrator_revision == "sha256:fitted-artifact"

    with pytest.raises(ConfigurationError):
        CalibratedProgressCallback(
            lambda verdict: 0.5,
            calibrator_id="",
            calibrator_revision="revision",
        )
