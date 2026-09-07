"""Stage 7: bounded, side-effect-contained self-heal orchestration.

The service owns no provider client and performs no hidden I/O. It can call
external retrieval or generation only through explicitly injected ports, one
claim/action per round, after all hard time/cost/attempt bounds pass. Every
candidate is verified before commit and the returned answer/chunks always
match the final audit verdicts.

``ClaimVerdict.calibrated_confidence`` is intentionally *not* used as the
repair objective: it estimates whether the current five-state verdict is
correct, not whether a claim is supported. The required
``HealProgressEvaluator`` supplies a separately calibrated
P(claim grounded/supported) for the ``min_improvement`` guard.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from groundguard_rag.application.exceptions import AdapterContractError
from groundguard_rag.application.input_hashing import (
    compute_answer_hash,
    compute_input_hash,
)
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.config import HealConfig
from groundguard_rag.domain.enums import (
    HealStopReason,
    RepairAction,
    RunMode,
    VerificationState,
)
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import (
    SCHEMA_VERSION,
    AuditReport,
    Chunk,
    ClaimVerdict,
    EvidenceCandidate,
    HealPolicyState,
    RepairActionRecord,
    RepairDecision,
    RunMetrics,
    VerificationRequest,
)
from groundguard_rag.domain.ports import (
    HealProgressEvaluator,
    RepairPolicy,
    Retriever,
    Rewriter,
    StopPolicy,
)
from groundguard_rag.schema.semantic_validation import validate_audit_report_semantics


_GROUNDED_STATES = frozenset(
    {VerificationState.SUPPORTED, VerificationState.NOT_CHECKABLE}
)
_ATTEMPTED_REPAIR_ACTIONS = frozenset(
    {RepairAction.RETRIEVE, RepairAction.REWRITE, RepairAction.DELETE}
)


@dataclass(frozen=True)
class HealResult:
    """A correction candidate plus the audit proof for its exact state.

    The result is deliberately passive: callers decide whether to display or
    write it back. ``output_hash`` covers candidate answer and ordered chunks,
    while the report's ``input_hash`` continues to identify the original input.
    """

    candidate_answer: str
    candidate_chunks: tuple[Chunk, ...]
    output_hash: str
    audit_report: AuditReport
    abstained_claim_ids: tuple[str, ...] = ()
    accepted_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_answer, str) or not self.candidate_answer.strip():
            raise DomainValidationError(
                "HealResult.candidate_answer must be a non-empty string"
            )
        chunks = tuple(self.candidate_chunks)
        if any(not isinstance(chunk, Chunk) for chunk in chunks):
            raise DomainValidationError(
                "HealResult.candidate_chunks must contain only Chunk instances"
            )
        object.__setattr__(self, "candidate_chunks", chunks)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise DomainValidationError(
                "HealResult.candidate_chunks must not contain duplicate chunk_id values"
            )
        expected_hash = compute_input_hash(self.candidate_answer, chunks)
        if self.output_hash != expected_hash:
            raise DomainValidationError(
                "HealResult.output_hash does not match candidate_answer/candidate_chunks"
            )
        if not isinstance(self.audit_report, AuditReport):
            raise DomainValidationError(
                "HealResult.audit_report must be an AuditReport"
            )
        if self.audit_report.run_mode is not RunMode.HEAL:
            raise DomainValidationError(
                "HealResult.audit_report must have run_mode == HEAL"
            )
        final_claim_ids: set[str] = set()
        available_chunk_ids = set(chunk_ids)
        for verdict in self.audit_report.verdicts:
            claim = verdict.claim
            if claim.end_char > len(self.candidate_answer) or self.candidate_answer[
                claim.start_char : claim.end_char
            ] != claim.text:
                raise DomainValidationError(
                    "HealResult final verdict claim spans must match candidate_answer"
                )
            final_claim_ids.add(claim.claim_id)
            if any(
                assessment.reference.chunk_id not in available_chunk_ids
                for assessment in verdict.evidence_assessments
            ):
                raise DomainValidationError(
                    "HealResult final verdict evidence must reference candidate_chunks"
                )
        for field_name in ("abstained_claim_ids", "accepted_claim_ids"):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise DomainValidationError(
                    f"HealResult.{field_name} must contain non-empty strings"
                )
            if len(values) != len(set(values)):
                raise DomainValidationError(
                    f"HealResult.{field_name} must not contain duplicates"
                )
            object.__setattr__(self, field_name, values)
            if not set(values) <= final_claim_ids:
                raise DomainValidationError(
                    f"HealResult.{field_name} must reference final verdict claim IDs"
                )
        if set(self.abstained_claim_ids) & set(self.accepted_claim_ids):
            raise DomainValidationError(
                "HealResult claim IDs cannot be both abstained and accepted"
            )


class HealService:
    """One-action-per-round bounded repair state machine.

    Hard limits are implemented here and cannot be relaxed by either policy.
    ``StopPolicy`` may only stop earlier. A monotonic clock is injected so
    timeout behavior is deterministic in tests; adapter calls are cooperative
    boundaries and must still configure their own provider-level timeouts.
    """

    def __init__(
        self,
        *,
        verify_service: VerifyService,
        repair_policy: RepairPolicy,
        stop_policy: StopPolicy,
        progress_evaluator: HealProgressEvaluator,
        config: HealConfig,
        monotonic: Callable[[], float],
        retriever: Retriever | None = None,
        rewriter: Rewriter | None = None,
    ) -> None:
        if not isinstance(verify_service, VerifyService):
            raise ConfigurationError(
                "HealService.verify_service must be a VerifyService instance"
            )
        if not isinstance(repair_policy, RepairPolicy):
            raise ConfigurationError(
                "HealService.repair_policy must implement RepairPolicy"
            )
        if not isinstance(stop_policy, StopPolicy):
            raise ConfigurationError(
                "HealService.stop_policy must implement StopPolicy"
            )
        if not isinstance(progress_evaluator, HealProgressEvaluator):
            raise ConfigurationError(
                "HealService.progress_evaluator must implement HealProgressEvaluator"
            )
        if not isinstance(config, HealConfig):
            raise ConfigurationError(
                "HealService.config must be a HealConfig instance"
            )
        if retriever is not None and not isinstance(retriever, Retriever):
            raise ConfigurationError(
                "HealService.retriever must implement Retriever or be None"
            )
        if rewriter is not None and not isinstance(rewriter, Rewriter):
            raise ConfigurationError(
                "HealService.rewriter must implement Rewriter or be None"
            )
        if not callable(monotonic):
            raise ConfigurationError("HealService.monotonic must be callable")

        try:
            progress_id = progress_evaluator.calibrator_id
            progress_revision = progress_evaluator.calibrator_revision
        except Exception as exc:
            raise ConfigurationError(
                "HealService could not read progress evaluator calibration metadata"
            ) from exc
        for name, value in (
            ("progress_evaluator.calibrator_id", progress_id),
            ("progress_evaluator.calibrator_revision", progress_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"HealService.{name} must be non-empty")

        self._verify_service = verify_service
        self._repair_policy = repair_policy
        self._stop_policy = stop_policy
        self._progress_evaluator = progress_evaluator
        self._config = config
        self._monotonic = monotonic
        self._retriever = retriever
        self._rewriter = rewriter
        self._progress_id = progress_id
        self._progress_revision = progress_revision

    @property
    def verify_service(self) -> VerifyService:
        """The exact verify service used for initial and final snapshots."""

        return self._verify_service

    def heal(self, request: VerificationRequest) -> HealResult:
        """Return a bounded correction candidate without mutating the input.

        Initial verification errors propagate because no heal run exists yet.
        Once the initial snapshot is available, plugin/action failures become
        deterministic, auditable stop results and no partial candidate is
        committed.
        """
        if not isinstance(request, VerificationRequest):
            raise DomainValidationError(
                "HealService.heal request must be a VerificationRequest"
            )

        started_at = self._now()
        initial_report = self._verify_service.verify(request)
        current_request = request
        current_report = initial_report
        stable_ids = tuple(
            verdict.claim.claim_id for verdict in initial_report.verdicts
        )
        seen_state_hashes = {
            compute_input_hash(current_request.answer, current_request.chunks)
        }
        attempts: dict[str, int] = {claim_id: 0 for claim_id in stable_ids}
        abstained: set[str] = set()
        accepted: set[str] = set()
        actions: list[RepairActionRecord] = []
        repair_rounds = 0
        spent_estimate = 0.0
        stop_reason: HealStopReason | None = None

        while stop_reason is None:
            bad_verdicts = [
                verdict
                for verdict in current_report.verdicts
                if verdict.state not in _GROUNDED_STATES
            ]
            if not bad_verdicts:
                stop_reason = HealStopReason.ALL_GROUNDED
                break

            if self._elapsed(started_at) >= self._config.timeout_seconds:
                stop_reason = HealStopReason.TIMEOUT
                break
            if repair_rounds >= self._config.max_rounds:
                stop_reason = HealStopReason.MAX_ROUNDS
                break

            unresolved = [
                verdict
                for verdict in bad_verdicts
                if verdict.claim.claim_id not in abstained | accepted
            ]
            if not unresolved:
                stop_reason = HealStopReason.ALL_CLAIMS_HANDLED
                break

            eligible = [
                verdict
                for verdict in unresolved
                if attempts.get(verdict.claim.claim_id, 0)
                < self._config.max_attempts_per_claim
            ]
            if not eligible:
                stop_reason = HealStopReason.MAX_ATTEMPTS_PER_CLAIM
                break
            target = eligible[0]

            state = self._policy_state(
                target=target,
                current_report=current_report,
                current_request=current_request,
                attempts=attempts,
                repair_rounds=repair_rounds,
                spent_estimate=spent_estimate,
                started_at=started_at,
                abstained=abstained,
                accepted=accepted,
            )
            try:
                policy_stop = self._stop_policy.should_stop(state)
                if not isinstance(policy_stop, bool):
                    stop_reason = HealStopReason.ADAPTER_CONTRACT_ERROR
                    break
                if policy_stop:
                    stop_reason = HealStopReason.POLICY_STOP
                    break
            except Exception:
                stop_reason = HealStopReason.ADAPTER_ERROR
                break

            if self._elapsed(started_at) >= self._config.timeout_seconds:
                stop_reason = HealStopReason.TIMEOUT
                break

            try:
                decision = self._repair_policy.decide(target, state)
            except Exception:
                stop_reason = HealStopReason.ADAPTER_ERROR
                break
            if not isinstance(decision, RepairDecision):
                stop_reason = HealStopReason.ADAPTER_CONTRACT_ERROR
                break

            if self._elapsed(started_at) >= self._config.timeout_seconds:
                stop_reason = HealStopReason.TIMEOUT
                break
            if spent_estimate + decision.estimated_cost > self._config.cost_budget:
                stop_reason = HealStopReason.COST_BUDGET
                break

            try:
                before_progress = self._evaluate_progress(target)
            except AdapterContractError:
                stop_reason = HealStopReason.ADAPTER_CONTRACT_ERROR
                break
            except Exception:
                stop_reason = HealStopReason.ADAPTER_ERROR
                break

            repair_rounds += 1
            round_number = repair_rounds
            if decision.action in _ATTEMPTED_REPAIR_ACTIONS:
                attempts[target.claim.claim_id] = (
                    attempts.get(target.claim.claim_id, 0) + 1
                )

            if decision.action in {RepairAction.ABSTAIN, RepairAction.ACCEPT}:
                terminal = abstained if decision.action is RepairAction.ABSTAIN else accepted
                terminal.add(target.claim.claim_id)
                actions.append(
                    self._record(
                        round_number=round_number,
                        target=target,
                        decision=decision,
                        state_after=target.state,
                        claim_id_after=target.claim.claim_id,
                        claim_text_after=target.claim.text,
                        committed=True,
                        answer_before=current_request.answer,
                        answer_after=current_request.answer,
                        confidence_before=before_progress,
                        confidence_after=before_progress,
                        elapsed_ms=0.0,
                    )
                )
                continue

            if not self._action_is_available(decision.action, request):
                actions.append(
                    self._record(
                        round_number=round_number,
                        target=target,
                        decision=decision,
                        state_after=target.state,
                        claim_id_after=target.claim.claim_id,
                        claim_text_after=target.claim.text,
                        committed=False,
                        answer_before=current_request.answer,
                        answer_after=current_request.answer,
                        confidence_before=before_progress,
                        confidence_after=before_progress,
                        elapsed_ms=0.0,
                        failure=HealStopReason.ACTION_UNAVAILABLE.value,
                    )
                )
                stop_reason = HealStopReason.ACTION_UNAVAILABLE
                break

            action_started = self._now()
            external_action = decision.action in {
                RepairAction.RETRIEVE,
                RepairAction.REWRITE,
            }
            if external_action:
                # Charge the declared quote immediately before the external
                # call. A provider failure may still incur work/cost, so a
                # failed call does not refund the audit estimate.
                spent_estimate += decision.estimated_cost

            try:
                candidate_request, candidate_ids, added_ids, removed_ids = (
                    self._build_candidate(
                        decision=decision,
                        target=target,
                        current_request=current_request,
                        stable_ids=stable_ids,
                    )
                )
            except AdapterContractError as exc:
                actions.append(
                    self._failed_record(
                        round_number,
                        target,
                        decision,
                        current_request.answer,
                        before_progress,
                        action_started,
                        HealStopReason.ADAPTER_CONTRACT_ERROR,
                        exc,
                    )
                )
                stop_reason = HealStopReason.ADAPTER_CONTRACT_ERROR
                break
            except Exception as exc:
                actions.append(
                    self._failed_record(
                        round_number,
                        target,
                        decision,
                        current_request.answer,
                        before_progress,
                        action_started,
                        HealStopReason.ADAPTER_ERROR,
                        exc,
                    )
                )
                stop_reason = HealStopReason.ADAPTER_ERROR
                break

            candidate_state_hash = compute_input_hash(
                candidate_request.answer, candidate_request.chunks
            )
            if self._elapsed(started_at) >= self._config.timeout_seconds:
                actions.append(
                    self._record(
                        round_number=round_number,
                        target=target,
                        decision=decision,
                        state_after=None,
                        claim_id_after=target.claim.claim_id,
                        claim_text_after=target.claim.text,
                        committed=False,
                        answer_before=current_request.answer,
                        answer_after=candidate_request.answer,
                        confidence_before=before_progress,
                        confidence_after=None,
                        elapsed_ms=self._elapsed_ms(action_started),
                        evidence_ids_added=added_ids,
                        evidence_ids_removed=removed_ids,
                        failure=HealStopReason.TIMEOUT.value,
                    )
                )
                stop_reason = HealStopReason.TIMEOUT
                break
            if candidate_state_hash in seen_state_hashes:
                actions.append(
                    self._record(
                        round_number=round_number,
                        target=target,
                        decision=decision,
                        state_after=None,
                        claim_id_after=target.claim.claim_id,
                        claim_text_after=target.claim.text,
                        committed=False,
                        answer_before=current_request.answer,
                        answer_after=candidate_request.answer,
                        confidence_before=before_progress,
                        confidence_after=None,
                        elapsed_ms=self._elapsed_ms(action_started),
                        evidence_ids_added=added_ids,
                        evidence_ids_removed=removed_ids,
                        failure=HealStopReason.STATE_LOOP.value,
                    )
                )
                stop_reason = HealStopReason.STATE_LOOP
                break

            try:
                verified_candidate = self._verify_service.verify(candidate_request)
                verified_candidate = self._stabilize_claim_ids(
                    verified_candidate, candidate_ids
                )
                self._validate_candidate_shape(
                    decision.action,
                    target,
                    current_report,
                    verified_candidate,
                    candidate_request,
                )
            except AdapterContractError as exc:
                actions.append(
                    self._failed_record(
                        round_number,
                        target,
                        decision,
                        current_request.answer,
                        before_progress,
                        action_started,
                        HealStopReason.ADAPTER_CONTRACT_ERROR,
                        exc,
                        answer_after=candidate_request.answer,
                    )
                )
                stop_reason = HealStopReason.ADAPTER_CONTRACT_ERROR
                break
            except Exception as exc:
                actions.append(
                    self._failed_record(
                        round_number,
                        target,
                        decision,
                        current_request.answer,
                        before_progress,
                        action_started,
                        HealStopReason.ADAPTER_ERROR,
                        exc,
                        answer_after=candidate_request.answer,
                    )
                )
                stop_reason = HealStopReason.ADAPTER_ERROR
                break

            if decision.action is RepairAction.DELETE:
                after_verdict = None
                after_progress = None
                claim_id_after = None
                claim_text_after = None
            else:
                after_verdict = self._verdict_by_id(
                    verified_candidate, target.claim.claim_id
                )
                claim_id_after = after_verdict.claim.claim_id
                claim_text_after = after_verdict.claim.text
                try:
                    after_progress = self._evaluate_progress(after_verdict)
                except AdapterContractError as exc:
                    actions.append(
                        self._failed_record(
                            round_number,
                            target,
                            decision,
                            current_request.answer,
                            before_progress,
                            action_started,
                            HealStopReason.ADAPTER_CONTRACT_ERROR,
                            exc,
                            answer_after=candidate_request.answer,
                        )
                    )
                    stop_reason = HealStopReason.ADAPTER_CONTRACT_ERROR
                    break
                except Exception as exc:
                    actions.append(
                        self._failed_record(
                            round_number,
                            target,
                            decision,
                            current_request.answer,
                            before_progress,
                            action_started,
                            HealStopReason.ADAPTER_ERROR,
                            exc,
                            answer_after=candidate_request.answer,
                        )
                    )
                    stop_reason = HealStopReason.ADAPTER_ERROR
                    break

            if self._elapsed(started_at) >= self._config.timeout_seconds:
                actions.append(
                    self._record(
                        round_number=round_number,
                        target=target,
                        decision=decision,
                        state_after=(after_verdict.state if after_verdict else None),
                        claim_id_after=(claim_id_after or target.claim.claim_id),
                        claim_text_after=(claim_text_after or target.claim.text),
                        committed=False,
                        answer_before=current_request.answer,
                        answer_after=candidate_request.answer,
                        confidence_before=before_progress,
                        confidence_after=after_progress,
                        elapsed_ms=self._elapsed_ms(action_started),
                        evidence_ids_added=added_ids,
                        evidence_ids_removed=removed_ids,
                        failure=HealStopReason.TIMEOUT.value,
                    )
                )
                stop_reason = HealStopReason.TIMEOUT
                break

            if decision.action in {RepairAction.RETRIEVE, RepairAction.REWRITE}:
                assert after_progress is not None  # validated float from evaluator
                if after_progress - before_progress < self._config.min_improvement:
                    actions.append(
                        self._record(
                            round_number=round_number,
                            target=target,
                            decision=decision,
                            state_after=after_verdict.state,
                            claim_id_after=claim_id_after,
                            claim_text_after=claim_text_after,
                            committed=False,
                            answer_before=current_request.answer,
                            answer_after=candidate_request.answer,
                            confidence_before=before_progress,
                            confidence_after=after_progress,
                            elapsed_ms=self._elapsed_ms(action_started),
                            evidence_ids_added=added_ids,
                            evidence_ids_removed=removed_ids,
                            failure=HealStopReason.MIN_IMPROVEMENT.value,
                        )
                    )
                    stop_reason = HealStopReason.MIN_IMPROVEMENT
                    break

            # Candidate passed every guard: atomically replace the in-memory
            # candidate only now. The caller's request and host remain untouched.
            previous_request = current_request
            current_request = candidate_request
            current_report = verified_candidate
            stable_ids = candidate_ids
            seen_state_hashes.add(candidate_state_hash)
            actions.append(
                self._record(
                    round_number=round_number,
                    target=target,
                    decision=decision,
                    state_after=(after_verdict.state if after_verdict else None),
                    claim_id_after=claim_id_after,
                    claim_text_after=claim_text_after,
                    committed=True,
                    answer_before=previous_request.answer,
                    answer_after=current_request.answer,
                    confidence_before=before_progress,
                    confidence_after=after_progress,
                    elapsed_ms=self._elapsed_ms(action_started),
                    evidence_ids_added=added_ids,
                    evidence_ids_removed=removed_ids,
                )
            )

        assert stop_reason is not None
        return self._finish(
            original_request=request,
            current_request=current_request,
            current_report=current_report,
            initial_report=initial_report,
            actions=tuple(actions),
            repair_rounds=repair_rounds,
            stop_reason=stop_reason,
            spent_estimate=spent_estimate,
            started_at=started_at,
            abstained=abstained,
            accepted=accepted,
        )

    def _build_candidate(
        self,
        *,
        decision: RepairDecision,
        target: ClaimVerdict,
        current_request: VerificationRequest,
        stable_ids: tuple[str, ...],
    ) -> tuple[VerificationRequest, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        target_index = stable_ids.index(target.claim.claim_id)
        if decision.action is RepairAction.RETRIEVE:
            assert self._retriever is not None
            base_query = current_request.query
            assert base_query is not None and base_query.strip()
            query = decision.retrieval_query or f"{base_query.strip()}\n{target.claim.text}"
            chunks = self._retriever.retrieve(query)
            if not isinstance(chunks, list):
                raise AdapterContractError("Retriever.retrieve must return a list")
            if any(not isinstance(chunk, Chunk) for chunk in chunks):
                raise AdapterContractError(
                    "Retriever.retrieve must return only Chunk instances"
                )
            new_ids = [chunk.chunk_id for chunk in chunks]
            if len(new_ids) != len(set(new_ids)):
                raise AdapterContractError(
                    "Retriever.retrieve returned duplicate chunk_id values"
                )
            existing = {chunk.chunk_id for chunk in current_request.chunks}
            collision = next((chunk_id for chunk_id in new_ids if chunk_id in existing), None)
            if collision is not None:
                raise AdapterContractError(
                    f"Retriever.retrieve returned existing chunk_id {collision!r}"
                )
            return (
                VerificationRequest(
                    request_id=current_request.request_id,
                    answer=current_request.answer,
                    chunks=current_request.chunks + tuple(chunks),
                    query=current_request.query,
                ),
                stable_ids,
                tuple(new_ids),
                (),
            )

        if decision.action is RepairAction.REWRITE:
            assert self._rewriter is not None
            chunks_by_id = {chunk.chunk_id: chunk for chunk in current_request.chunks}
            candidates = [
                EvidenceCandidate.from_chunk(
                    assessment.reference, chunks_by_id[assessment.reference.chunk_id]
                )
                for assessment in target.evidence_assessments
            ]
            replacement = self._rewriter.rewrite(target.claim, candidates)
            if not isinstance(replacement, str):
                raise AdapterContractError("Rewriter.rewrite must return a string")
            if not replacement.strip() or replacement != replacement.strip():
                raise AdapterContractError(
                    "Rewriter.rewrite must return non-empty text without leading/trailing whitespace"
                )
            answer = (
                current_request.answer[: target.claim.start_char]
                + replacement
                + current_request.answer[target.claim.end_char :]
            )
            return (
                VerificationRequest(
                    request_id=current_request.request_id,
                    answer=answer,
                    chunks=current_request.chunks,
                    query=current_request.query,
                ),
                stable_ids,
                (),
                (),
            )

        if decision.action is RepairAction.DELETE:
            answer = (
                current_request.answer[: target.claim.start_char]
                + current_request.answer[target.claim.end_char :]
            )
            if not answer.strip():
                raise AdapterContractError(
                    "DELETE would produce an empty/whitespace-only answer"
                )
            candidate_ids = stable_ids[:target_index] + stable_ids[target_index + 1 :]
            return (
                VerificationRequest(
                    request_id=current_request.request_id,
                    answer=answer,
                    chunks=current_request.chunks,
                    query=current_request.query,
                ),
                candidate_ids,
                (),
                (),
            )

        raise AdapterContractError(
            f"unsupported mutating RepairAction {decision.action.value!r}"
        )

    def _validate_candidate_shape(
        self,
        action: RepairAction,
        target: ClaimVerdict,
        before: AuditReport,
        after: AuditReport,
        candidate_request: VerificationRequest,
    ) -> None:
        expected_count = len(before.verdicts) - (1 if action is RepairAction.DELETE else 0)
        if len(after.verdicts) != expected_count:
            raise AdapterContractError(
                f"{action.value} changed claim count from {len(before.verdicts)} "
                f"to {len(after.verdicts)}; constrained repair expected {expected_count}"
            )
        if action is RepairAction.REWRITE:
            after_target = self._verdict_by_id(after, target.claim.claim_id)
            replacement = candidate_request.answer[
                after_target.claim.start_char : after_target.claim.end_char
            ]
            if replacement != after_target.claim.text:
                raise AdapterContractError(
                    "rewritten claim span does not match candidate answer"
                )

    def _stabilize_claim_ids(
        self, report: AuditReport, stable_ids: tuple[str, ...]
    ) -> AuditReport:
        if len(report.verdicts) != len(stable_ids):
            raise AdapterContractError(
                "candidate decomposition does not match expected claim lineage count"
            )
        verdicts: list[ClaimVerdict] = []
        for stable_id, verdict in zip(stable_ids, report.verdicts):
            claim = dataclasses.replace(verdict.claim, claim_id=stable_id)
            verdicts.append(dataclasses.replace(verdict, claim=claim))
        stabilized = dataclasses.replace(report, verdicts=tuple(verdicts))
        validate_audit_report_semantics(stabilized.to_dict())
        return stabilized

    def _action_is_available(
        self, action: RepairAction, original_request: VerificationRequest
    ) -> bool:
        if action is RepairAction.RETRIEVE:
            return (
                self._retriever is not None
                and original_request.query is not None
                and bool(original_request.query.strip())
            )
        if action is RepairAction.REWRITE:
            return self._rewriter is not None
        return True

    def _evaluate_progress(self, verdict: ClaimVerdict) -> float:
        value = self._progress_evaluator.evaluate(verdict)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise AdapterContractError(
                "HealProgressEvaluator.evaluate must return a finite number in [0, 1]"
            )
        return float(value)

    def _policy_state(
        self,
        *,
        target: ClaimVerdict,
        current_report: AuditReport,
        current_request: VerificationRequest,
        attempts: Mapping[str, int],
        repair_rounds: int,
        spent_estimate: float,
        started_at: float,
        abstained: set[str],
        accepted: set[str],
    ) -> HealPolicyState:
        """Return the stable, immutable stage-7 policy protocol model."""
        return HealPolicyState(
            round=repair_rounds + 1,
            completed_rounds=repair_rounds,
            target_claim_id=target.claim.claim_id,
            attempts_for_claim=attempts.get(target.claim.claim_id, 0),
            attempts_by_claim=attempts,
            elapsed_seconds=self._elapsed(started_at),
            estimated_cost_spent=spent_estimate,
            estimated_cost_remaining=max(
                0.0, self._config.cost_budget - spent_estimate
            ),
            state_hash=compute_input_hash(
                current_request.answer, current_request.chunks
            ),
            request_has_query=bool(
                current_request.query and current_request.query.strip()
            ),
            retriever_available=self._retriever is not None,
            rewriter_available=self._rewriter is not None,
            verdict_states=tuple(
                verdict.state for verdict in current_report.verdicts
            ),
            abstained_claim_ids=tuple(sorted(abstained)),
            accepted_claim_ids=tuple(sorted(accepted)),
            max_rounds=self._config.max_rounds,
            max_attempts_per_claim=self._config.max_attempts_per_claim,
            timeout_seconds=self._config.timeout_seconds,
            cost_budget=self._config.cost_budget,
            min_improvement=self._config.min_improvement,
        )

    def _record(
        self,
        *,
        round_number: int,
        target: ClaimVerdict,
        decision: RepairDecision,
        state_after: VerificationState | None,
        claim_id_after: str | None,
        claim_text_after: str | None,
        committed: bool,
        answer_before: str,
        answer_after: str,
        confidence_before: float | None,
        confidence_after: float | None,
        elapsed_ms: float,
        evidence_ids_added: tuple[str, ...] = (),
        evidence_ids_removed: tuple[str, ...] = (),
        failure: str | None = None,
    ) -> RepairActionRecord:
        return RepairActionRecord(
            round=round_number,
            claim_id=target.claim.claim_id,
            action=decision.action,
            state_before=target.state,
            state_after=state_after,
            claim_id_after=claim_id_after,
            claim_text_before=target.claim.text,
            claim_text_after=claim_text_after,
            committed=committed,
            note=decision.note,
            evidence_ids_added=evidence_ids_added,
            evidence_ids_removed=evidence_ids_removed,
            answer_hash_before=compute_answer_hash(answer_before),
            answer_hash_after=compute_answer_hash(answer_after),
            calibrated_confidence_before=confidence_before,
            calibrated_confidence_after=confidence_after,
            elapsed_ms=elapsed_ms,
            estimated_cost=decision.estimated_cost,
            failure_or_stop_reason=failure,
        )

    def _failed_record(
        self,
        round_number: int,
        target: ClaimVerdict,
        decision: RepairDecision,
        answer_before: str,
        before_progress: float,
        action_started: float,
        reason: HealStopReason,
        exc: Exception,
        *,
        answer_after: str | None = None,
    ) -> RepairActionRecord:
        # Exception messages may contain provider prompts, URLs, or secrets;
        # only the exception type is safe and still useful for classification.
        failure = f"{reason.value}:{type(exc).__name__}"
        return self._record(
            round_number=round_number,
            target=target,
            decision=decision,
            state_after=None,
            claim_id_after=target.claim.claim_id,
            claim_text_after=target.claim.text,
            committed=False,
            answer_before=answer_before,
            answer_after=answer_after if answer_after is not None else answer_before,
            confidence_before=before_progress,
            confidence_after=None,
            elapsed_ms=self._elapsed_ms(action_started),
            failure=failure,
        )

    def _finish(
        self,
        *,
        original_request: VerificationRequest,
        current_request: VerificationRequest,
        current_report: AuditReport,
        initial_report: AuditReport,
        actions: tuple[RepairActionRecord, ...],
        repair_rounds: int,
        stop_reason: HealStopReason,
        spent_estimate: float,
        started_at: float,
        abstained: set[str],
        accepted: set[str],
    ) -> HealResult:
        elapsed_ms = self._elapsed(started_at) * 1000.0
        report = AuditReport(
            schema_version=SCHEMA_VERSION,
            request_id=original_request.request_id,
            input_hash=compute_input_hash(
                original_request.answer, original_request.chunks
            ),
            model_revision=current_report.model_revision,
            threshold_version=current_report.threshold_version,
            created_at=current_report.created_at,
            run_mode=RunMode.HEAL,
            verdicts=current_report.verdicts,
            initial_verdicts=initial_report.verdicts,
            calibrator_id=current_report.calibrator_id,
            calibrator_revision=current_report.calibrator_revision,
            heal_progress_calibrator_id=self._progress_id,
            heal_progress_calibrator_revision=self._progress_revision,
            repair_rounds=repair_rounds,
            repair_actions=actions,
            stop_reason=stop_reason.value,
            metrics=RunMetrics(
                total_latency_ms=elapsed_ms,
                stage_latencies_ms=MappingProxyType(
                    {"heal_total": elapsed_ms}
                ),
                estimated_cost=spent_estimate,
            ),
        )
        validate_audit_report_semantics(report.to_dict())
        return HealResult(
            candidate_answer=current_request.answer,
            candidate_chunks=current_request.chunks,
            output_hash=compute_input_hash(
                current_request.answer, current_request.chunks
            ),
            audit_report=report,
            abstained_claim_ids=tuple(sorted(abstained)),
            accepted_claim_ids=tuple(sorted(accepted)),
        )

    @staticmethod
    def _verdict_by_id(report: AuditReport, claim_id: str) -> ClaimVerdict:
        for verdict in report.verdicts:
            if verdict.claim.claim_id == claim_id:
                return verdict
        raise AdapterContractError(
            f"verified candidate lost expected claim_id {claim_id!r}"
        )

    def _now(self) -> float:
        value = self._monotonic()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ConfigurationError(
                "HealService.monotonic must return a finite real number"
            )
        return float(value)

    def _elapsed(self, started_at: float) -> float:
        value = self._now() - started_at
        if value < 0:
            raise ConfigurationError(
                "HealService.monotonic moved backwards during a heal run"
            )
        return value

    def _elapsed_ms(self, started_at: float) -> float:
        return self._elapsed(started_at) * 1000.0
