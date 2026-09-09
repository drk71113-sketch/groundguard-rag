from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from groundguard_rag.adapters.decomposition.rule_based import RuleBasedClaimDecomposer
from groundguard_rag.application.exceptions import ApplicationInvariantError
from groundguard_rag.application.heal_service import HealResult, HealService
from groundguard_rag.application.input_hashing import compute_input_hash
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.config import HealConfig, VerifyConfig
from groundguard_rag.domain.enums import RepairAction, VerificationState
from groundguard_rag.domain.models import (
    AtomicClaim,
    Chunk,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceCandidate,
    EvidenceReference,
    LabelScores,
    RepairDecision,
    VerificationRequest,
)
from groundguard_rag.domain.ports import (
    EvidenceSelector,
    HealProgressEvaluator,
    RepairPolicy,
    Retriever,
    Rewriter,
    StopPolicy,
    Verifier,
)
from groundguard_rag.schema.semantic_validation import validate_audit_report_semantics


class _AllChunksSelector(EvidenceSelector):
    def select(
        self, claim: AtomicClaim, chunks: list[Chunk]
    ) -> list[EvidenceReference]:
        return [EvidenceReference(chunk_id=chunk.chunk_id) for chunk in chunks]


class _ContentVerifier(Verifier):
    """Deterministic test verifier; it is not a production detector."""

    def verify(
        self, claim: AtomicClaim, evidence: list[EvidenceCandidate]
    ) -> ClaimVerdict:
        claim_supported = claim.text.startswith("Grounded")
        claim_contradicted = claim.text.startswith("Contradicted")
        edge_states: list[VerificationState] = []
        for candidate in evidence:
            if claim_supported or "support evidence" in candidate.text:
                edge_states.append(VerificationState.SUPPORTED)
            elif claim_contradicted or "contradiction evidence" in candidate.text:
                edge_states.append(VerificationState.CONTRADICTED)
            else:
                edge_states.append(VerificationState.INSUFFICIENT_EVIDENCE)

        if VerificationState.SUPPORTED in edge_states and VerificationState.CONTRADICTED in edge_states:
            state = VerificationState.CONFLICTING_EVIDENCE
        elif VerificationState.CONTRADICTED in edge_states:
            state = VerificationState.CONTRADICTED
        elif VerificationState.SUPPORTED in edge_states:
            state = VerificationState.SUPPORTED
        else:
            state = VerificationState.INSUFFICIENT_EVIDENCE

        assessments = tuple(
            EvidenceAssessment(
                reference=candidate.reference,
                label_scores=_scores(edge_state),
                state=edge_state,
                rationale=None,
                verifier_id="test-verifier",
                verifier_revision="v1",
            )
            for candidate, edge_state in zip(evidence, edge_states)
        )
        return ClaimVerdict(
            claim=claim,
            state=state,
            evidence_assessments=assessments,
        )


def _scores(state: VerificationState) -> LabelScores:
    if state is VerificationState.SUPPORTED:
        values = (0.9, 0.05, 0.05)
    elif state is VerificationState.CONTRADICTED:
        values = (0.05, 0.9, 0.05)
    else:
        values = (0.05, 0.05, 0.9)
    return LabelScores(
        supported=values[0],
        contradicted=values[1],
        insufficient=values[2],
        score_kind="probabilities",
    )


class _QueuePolicy(RepairPolicy):
    def __init__(self, *decisions: RepairDecision):
        self.decisions = deque(decisions)
        self.states = []

    def decide(self, verdict, state):
        self.states.append(state)
        return self.decisions.popleft()


class _BarrierAcceptPolicy(RepairPolicy):
    """A reentrant test double that forces heal calls to overlap."""

    def __init__(self, parties):
        self._barrier = Barrier(parties)

    def decide(self, verdict, state):
        self._barrier.wait(timeout=10)
        return RepairDecision(action=RepairAction.ACCEPT, estimated_cost=0.0)


class _NeverStop(StopPolicy):
    def should_stop(self, state) -> bool:
        return False


class _FixedStop(StopPolicy):
    def __init__(self, value):
        self.value = value

    def should_stop(self, state):
        return self.value


class _Progress(HealProgressEvaluator):
    def __init__(self, scorer=None):
        self._scorer = scorer or self._default_score

    def evaluate(self, verdict):
        return self._scorer(verdict)

    @property
    def calibrator_id(self):
        return "test-groundedness-calibrator"

    @property
    def calibrator_revision(self):
        return "sha256:test-progress-v1"

    @staticmethod
    def _default_score(verdict):
        return {
            VerificationState.SUPPORTED: 0.9,
            VerificationState.CONTRADICTED: 0.05,
            VerificationState.INSUFFICIENT_EVIDENCE: 0.1,
            VerificationState.CONFLICTING_EVIDENCE: 0.02,
            VerificationState.NOT_CHECKABLE: 1.0,
        }[verdict.state]


class _StaticRewriter(Rewriter):
    def __init__(self, replacement: str):
        self.replacement = replacement
        self.calls = 0

    def rewrite(self, claim, evidence):
        self.calls += 1
        return self.replacement


class _QueueRewriter(Rewriter):
    def __init__(self, *replacements):
        self.replacements = deque(replacements)
        self.calls = 0

    def rewrite(self, claim, evidence):
        self.calls += 1
        return self.replacements.popleft()


class _RaisingRewriter(Rewriter):
    def rewrite(self, claim, evidence):
        raise RuntimeError("SECRET provider prompt must not enter audit")


class _StaticRetriever(Retriever):
    def __init__(self, chunks, on_call=None):
        self.chunks = chunks
        self.on_call = on_call
        self.calls = []

    def retrieve(self, query):
        self.calls.append(query)
        if self.on_call is not None:
            self.on_call()
        return list(self.chunks)


class _ManualClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class _SequenceClock:
    def __init__(self, *values):
        self.values = deque(values)
        self.last = values[-1]

    def __call__(self):
        if self.values:
            self.last = self.values.popleft()
        return self.last


def _verify_service():
    return VerifyService(
        decomposer=RuleBasedClaimDecomposer(),
        selector=_AllChunksSelector(),
        verifier=_ContentVerifier(),
        config=VerifyConfig(),
        model_revision="test-model-v1",
        threshold_version="test-threshold-v1",
        clock=lambda: "2026-09-02T00:00:00+00:00",
    )


def _config(**overrides):
    fields = dict(
        max_rounds=3,
        max_attempts_per_claim=2,
        timeout_seconds=30.0,
        cost_budget=1.0,
        min_improvement=0.5,
    )
    fields.update(overrides)
    return HealConfig(**fields)


def _request(answer="Unverified.", *, query="original question", chunks=None):
    return VerificationRequest(
        request_id="req-heal",
        answer=answer,
        chunks=tuple(chunks or [Chunk(chunk_id="seed", text="background")]),
        query=query,
    )


def _service(
    policy,
    *,
    rewriter=None,
    retriever=None,
    stop_policy=None,
    progress=None,
    config=None,
    clock=None,
):
    return HealService(
        verify_service=_verify_service(),
        repair_policy=policy,
        stop_policy=stop_policy or _NeverStop(),
        progress_evaluator=progress or _Progress(),
        config=config or _config(),
        monotonic=clock or _ManualClock(),
        retriever=retriever,
        rewriter=rewriter,
    )


def _decision(action, cost=0.0, **kwargs):
    return RepairDecision(action=action, estimated_cost=cost, **kwargs)


def test_already_grounded_stops_without_repair_rounds():
    result = _service(_QueuePolicy()).heal(_request(answer="Grounded."))
    assert result.audit_report.stop_reason == "all_grounded"
    assert result.audit_report.repair_rounds == 0
    assert result.candidate_answer == "Grounded."


def test_successful_rewrite_is_verified_then_committed():
    source = _request()
    rewriter = _StaticRewriter("Grounded.")
    result = _service(
        _QueuePolicy(_decision(RepairAction.REWRITE, cost=0.2)),
        rewriter=rewriter,
    ).heal(source)

    assert result.candidate_answer == "Grounded."
    assert source.answer == "Unverified."
    assert result.audit_report.stop_reason == "all_grounded"
    action = result.audit_report.repair_actions[0]
    assert action.committed is True
    assert action.state_before is VerificationState.INSUFFICIENT_EVIDENCE
    assert action.state_after is VerificationState.SUPPORTED
    assert action.calibrated_confidence_after - action.calibrated_confidence_before >= 0.5
    assert result.audit_report.metrics.estimated_cost == 0.2


def test_successful_retrieval_uses_targeted_query_and_adds_chunks():
    retriever = _StaticRetriever(
        [Chunk(chunk_id="new", text="support evidence for the claim")]
    )
    result = _service(
        _QueuePolicy(
            _decision(
                RepairAction.RETRIEVE,
                cost=0.1,
                retrieval_query="explicit targeted query",
            )
        ),
        retriever=retriever,
    ).heal(_request())

    assert retriever.calls == ["explicit targeted query"]
    assert [chunk.chunk_id for chunk in result.candidate_chunks] == ["seed", "new"]
    assert result.audit_report.repair_actions[0].evidence_ids_added == ("new",)
    assert result.audit_report.verdicts[0].state is VerificationState.SUPPORTED


def test_retrieval_without_original_query_is_never_called():
    retriever = _StaticRetriever([Chunk(chunk_id="new", text="support evidence")])
    result = _service(
        _QueuePolicy(_decision(RepairAction.RETRIEVE, cost=0.1)),
        retriever=retriever,
    ).heal(_request(query=None))
    assert retriever.calls == []
    assert result.audit_report.stop_reason == "action_unavailable"
    assert result.audit_report.repair_actions[0].committed is False


def test_missing_rewriter_is_audited_not_faked():
    result = _service(
        _QueuePolicy(_decision(RepairAction.REWRITE, cost=0.1))
    ).heal(_request())
    assert result.candidate_answer == "Unverified."
    assert result.audit_report.stop_reason == "action_unavailable"


def test_cost_quote_is_checked_before_external_call():
    rewriter = _StaticRewriter("Grounded.")
    result = _service(
        _QueuePolicy(_decision(RepairAction.REWRITE, cost=1.01)),
        rewriter=rewriter,
        config=_config(cost_budget=1.0),
    ).heal(_request())
    assert rewriter.calls == 0
    assert result.audit_report.stop_reason == "cost_budget"
    assert result.audit_report.repair_rounds == 0


def test_cumulative_cost_budget_stops_before_second_external_call():
    rewriter = _QueueRewriter("Still uncertain.", "Grounded.")
    progress = _Progress(
        lambda verdict: 0.7 if verdict.claim.text.startswith("Still") else _Progress._default_score(verdict)
    )
    result = _service(
        _QueuePolicy(
            _decision(RepairAction.REWRITE, cost=0.6),
            _decision(RepairAction.REWRITE, cost=0.5),
        ),
        rewriter=rewriter,
        progress=progress,
        config=_config(cost_budget=1.0, min_improvement=0.5),
    ).heal(_request())
    assert rewriter.calls == 1
    assert result.audit_report.stop_reason == "cost_budget"
    assert result.audit_report.metrics.estimated_cost == 0.6


def test_timeout_after_external_call_rolls_candidate_back():
    clock = _ManualClock()
    retriever = _StaticRetriever(
        [Chunk(chunk_id="new", text="support evidence")],
        on_call=lambda: setattr(clock, "value", 2.0),
    )
    result = _service(
        _QueuePolicy(_decision(RepairAction.RETRIEVE, cost=0.2)),
        retriever=retriever,
        clock=clock,
        config=_config(timeout_seconds=1.0),
    ).heal(_request())
    assert [chunk.chunk_id for chunk in result.candidate_chunks] == ["seed"]
    assert result.audit_report.stop_reason == "timeout"
    assert result.audit_report.repair_actions[0].committed is False
    assert result.audit_report.metrics.estimated_cost == 0.2


def test_timeout_before_policy_prevents_any_round():
    policy = _QueuePolicy(_decision(RepairAction.ACCEPT))
    result = _service(
        policy,
        clock=_SequenceClock(0.0, 2.0),
        config=_config(timeout_seconds=1.0),
    ).heal(_request())
    assert result.audit_report.stop_reason == "timeout"
    assert result.audit_report.repair_rounds == 0
    assert len(policy.decisions) == 1


def test_rewrite_to_same_state_hash_is_rejected_as_loop():
    result = _service(
        _QueuePolicy(_decision(RepairAction.REWRITE)),
        rewriter=_StaticRewriter("Unverified."),
    ).heal(_request())
    assert result.audit_report.stop_reason == "state_loop"
    assert result.audit_report.repair_actions[0].committed is False
    assert result.candidate_answer == "Unverified."


def test_minimum_improvement_failure_rolls_back_verified_candidate():
    result = _service(
        _QueuePolicy(_decision(RepairAction.REWRITE)),
        rewriter=_StaticRewriter("Still uncertain."),
    ).heal(_request())
    assert result.audit_report.stop_reason == "min_improvement"
    assert result.candidate_answer == "Unverified."
    action = result.audit_report.repair_actions[0]
    assert action.committed is False
    assert action.claim_text_after == "Still uncertain."


def test_rewrite_that_splits_one_claim_is_rejected_and_rolled_back():
    result = _service(
        _QueuePolicy(_decision(RepairAction.REWRITE)),
        rewriter=_StaticRewriter("Grounded. Extra."),
    ).heal(_request())
    assert result.audit_report.stop_reason == "adapter_contract_error"
    assert result.candidate_answer == "Unverified."
    assert result.audit_report.repair_actions[0].committed is False


def test_retriever_duplicate_existing_chunk_id_is_contract_failure():
    retriever = _StaticRetriever([Chunk(chunk_id="seed", text="replacement")])
    result = _service(
        _QueuePolicy(_decision(RepairAction.RETRIEVE, cost=0.1)),
        retriever=retriever,
    ).heal(_request())
    assert result.audit_report.stop_reason == "adapter_contract_error"
    assert len(result.candidate_chunks) == 1


def test_adapter_exception_is_sanitized_and_candidate_is_not_committed():
    result = _service(
        _QueuePolicy(_decision(RepairAction.REWRITE, cost=0.1)),
        rewriter=_RaisingRewriter(),
    ).heal(_request())
    assert result.audit_report.stop_reason == "adapter_error"
    serialized = str(result.audit_report.to_dict())
    assert "SECRET" not in serialized
    assert "RuntimeError" in serialized
    assert result.candidate_answer == "Unverified."


def test_delete_preserves_surviving_claim_id_and_audits_initial_snapshot():
    policy = _QueuePolicy(
        _decision(RepairAction.DELETE),
        _decision(RepairAction.ABSTAIN),
    )
    result = _service(policy).heal(_request(answer="Unverified. Another."))

    assert result.candidate_answer.strip() == "Another."
    assert len(result.audit_report.initial_verdicts) == 2
    assert len(result.audit_report.verdicts) == 1
    deleted = result.audit_report.repair_actions[0]
    assert deleted.action is RepairAction.DELETE
    assert deleted.state_after is None
    assert deleted.claim_id_after is None
    assert deleted.committed is True
    assert result.audit_report.verdicts[0].claim.claim_id == "claim-12-20"
    assert result.audit_report.stop_reason == "all_claims_handled"


def test_delete_of_only_claim_is_rejected_without_empty_answer():
    result = _service(_QueuePolicy(_decision(RepairAction.DELETE))).heal(_request())
    assert result.audit_report.stop_reason == "adapter_contract_error"
    assert result.candidate_answer == "Unverified."


def test_accept_and_abstain_are_terminal_audited_decisions_without_mutation():
    result = _service(
        _QueuePolicy(
            _decision(RepairAction.ACCEPT),
            _decision(RepairAction.ABSTAIN),
        )
    ).heal(_request(answer="Unverified. Another."))
    assert result.candidate_answer == "Unverified. Another."
    assert len(result.accepted_claim_ids) == 1
    assert len(result.abstained_claim_ids) == 1
    assert result.audit_report.stop_reason == "all_claims_handled"
    assert all(action.committed for action in result.audit_report.repair_actions)


def test_per_claim_attempt_bound_is_hard_even_when_policy_has_more_actions():
    progress = _Progress(
        lambda verdict: min(0.9, 0.1 + 0.2 * max(0, len(verdict.evidence_assessments) - 1))
    )
    retriever = _StaticRetriever([Chunk(chunk_id="weak", text="more background")])
    policy = _QueuePolicy(
        _decision(RepairAction.RETRIEVE),
        _decision(RepairAction.REWRITE),
    )
    result = _service(
        policy,
        retriever=retriever,
        rewriter=_StaticRewriter("Grounded."),
        progress=progress,
        config=_config(max_attempts_per_claim=1, min_improvement=0.1),
    ).heal(_request())
    assert result.audit_report.stop_reason == "max_attempts_per_claim"
    assert len(policy.decisions) == 1


def test_max_rounds_is_hard_after_a_committed_but_still_ungrounded_candidate():
    progress = _Progress(
        lambda verdict: 0.7 if verdict.claim.text.startswith("Still") else _Progress._default_score(verdict)
    )
    policy = _QueuePolicy(
        _decision(RepairAction.REWRITE),
        _decision(RepairAction.REWRITE),
    )
    result = _service(
        policy,
        rewriter=_QueueRewriter("Still uncertain.", "Grounded."),
        progress=progress,
        config=_config(max_rounds=1, min_improvement=0.5),
    ).heal(_request())
    assert result.candidate_answer == "Still uncertain."
    assert result.audit_report.stop_reason == "max_rounds"
    assert len(policy.decisions) == 1


def test_stop_policy_can_only_stop_earlier():
    policy = _QueuePolicy(_decision(RepairAction.ACCEPT))
    result = _service(policy, stop_policy=_FixedStop(True)).heal(_request())
    assert result.audit_report.stop_reason == "policy_stop"
    assert result.audit_report.repair_rounds == 0
    assert len(policy.decisions) == 1


def test_policy_cannot_mutate_read_only_state():
    class MutatingPolicy(RepairPolicy):
        def decide(self, verdict, state):
            state["round"] = 99
            return _decision(RepairAction.ACCEPT)

    result = _service(MutatingPolicy()).heal(_request())
    assert result.audit_report.stop_reason == "adapter_error"
    assert result.audit_report.repair_rounds == 0


def test_non_boolean_stop_policy_result_is_contract_error():
    result = _service(
        _QueuePolicy(_decision(RepairAction.ACCEPT)),
        stop_policy=_FixedStop("yes"),
    ).heal(_request())
    assert result.audit_report.stop_reason == "adapter_contract_error"


def test_bare_string_repair_policy_result_is_contract_error():
    class LegacyStringPolicy(RepairPolicy):
        def decide(self, verdict, state):
            return "rewrite"

    result = _service(LegacyStringPolicy()).heal(_request())
    assert result.audit_report.stop_reason == "adapter_contract_error"
    assert result.audit_report.repair_rounds == 0


def test_invalid_progress_output_is_rejected_before_action():
    result = _service(
        _QueuePolicy(_decision(RepairAction.REWRITE)),
        rewriter=_StaticRewriter("Grounded."),
        progress=_Progress(lambda verdict: float("nan")),
    ).heal(_request())
    assert result.audit_report.stop_reason == "adapter_contract_error"
    assert result.audit_report.repair_rounds == 0


def test_invalid_progress_output_after_verification_rolls_back():
    values = deque([0.1, float("nan")])
    result = _service(
        _QueuePolicy(_decision(RepairAction.REWRITE)),
        rewriter=_StaticRewriter("Grounded."),
        progress=_Progress(lambda verdict: values.popleft()),
    ).heal(_request())
    assert result.audit_report.stop_reason == "adapter_contract_error"
    assert result.candidate_answer == "Unverified."
    assert result.audit_report.repair_actions[0].state_after is None


def test_retrieval_default_query_is_derived_only_when_original_query_exists():
    retriever = _StaticRetriever(
        [Chunk(chunk_id="new", text="support evidence")]
    )
    _service(
        _QueuePolicy(_decision(RepairAction.RETRIEVE)),
        retriever=retriever,
    ).heal(_request(query="What happened?"))
    assert retriever.calls == ["What happened?\nUnverified."]


def test_result_hash_report_semantics_and_input_immutability_hold():
    request = _request()
    original_chunks = request.chunks
    result = _service(
        _QueuePolicy(_decision(RepairAction.REWRITE)),
        rewriter=_StaticRewriter("Grounded."),
    ).heal(request)

    assert isinstance(result, HealResult)
    assert result.output_hash == compute_input_hash(
        result.candidate_answer, result.candidate_chunks
    )
    assert result.audit_report.input_hash == compute_input_hash(
        request.answer, request.chunks
    )
    validate_audit_report_semantics(result.audit_report.to_dict())
    assert request.answer == "Unverified."
    assert request.chunks is original_chunks


def _initial_target(service, request):
    report = service.verify_service.verify(request)
    return report.verdicts[0], tuple(
        verdict.claim.claim_id for verdict in report.verdicts
    )


def test_retrieve_internal_guard_raises_explicitly_without_retriever():
    service = _service(_QueuePolicy())
    request = _request()
    target, stable_ids = _initial_target(service, request)

    with pytest.raises(ApplicationInvariantError, match="retriever"):
        service._build_candidate(
            decision=_decision(RepairAction.RETRIEVE),
            target=target,
            current_request=request,
            stable_ids=stable_ids,
        )


def test_retrieve_internal_guard_raises_explicitly_without_query():
    retriever = _StaticRetriever([])
    service = _service(_QueuePolicy(), retriever=retriever)
    request = _request(query=None)
    target, stable_ids = _initial_target(service, request)

    with pytest.raises(ApplicationInvariantError, match="query"):
        service._build_candidate(
            decision=_decision(RepairAction.RETRIEVE),
            target=target,
            current_request=request,
            stable_ids=stable_ids,
        )
    assert retriever.calls == []


def test_rewrite_internal_guard_raises_explicitly_without_rewriter():
    service = _service(_QueuePolicy())
    request = _request()
    target, stable_ids = _initial_target(service, request)

    with pytest.raises(ApplicationInvariantError, match="rewriter"):
        service._build_candidate(
            decision=_decision(RepairAction.REWRITE),
            target=target,
            current_request=request,
            stable_ids=stable_ids,
        )


def test_internal_invariant_is_not_misclassified_as_an_adapter_failure(monkeypatch):
    service = _service(_QueuePolicy(_decision(RepairAction.RETRIEVE)))
    monkeypatch.setattr(service, "_action_is_available", lambda action, request: True)

    with pytest.raises(ApplicationInvariantError, match="retriever"):
        service.heal(_request())


def test_same_heal_service_instance_is_reentrant_with_thread_safe_adapters():
    parties = 8
    service = _service(_BarrierAcceptPolicy(parties), clock=lambda: 0.0)
    requests = tuple(
        VerificationRequest(
            request_id=f"heal-concurrent-{index}",
            answer=f"Unverified claim {index}.",
            chunks=(Chunk(chunk_id=f"seed-{index}", text="background"),),
            query="original question",
        )
        for index in range(parties)
    )

    with ThreadPoolExecutor(max_workers=parties) as executor:
        results = tuple(executor.map(service.heal, requests))

    assert [result.audit_report.request_id for result in results] == [
        request.request_id for request in requests
    ]
    assert [result.candidate_answer for result in results] == [
        request.answer for request in requests
    ]
    assert all(result.audit_report.stop_reason == "all_claims_handled" for result in results)
    assert all(len(result.audit_report.repair_actions) == 1 for result in results)
    assert all(result.audit_report.repair_actions[0].committed for result in results)
