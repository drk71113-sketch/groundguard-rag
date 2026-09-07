"""Conservative deterministic policies for the bounded heal state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass

from groundguard_rag.domain.enums import RepairAction, VerificationState
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import ClaimVerdict, HealPolicyState, RepairDecision
from groundguard_rag.domain.ports import RepairPolicy, StopPolicy


def _cost(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a non-bool real number")
    if not math.isfinite(value) or value < 0:
        raise ConfigurationError(f"{name} must be finite and >= 0")


@dataclass(frozen=True)
class ConservativeRepairPolicyConfig:
    """Caller-declared cost estimates used before external adapter calls."""

    retrieval_estimated_cost: float = 0.0
    rewrite_estimated_cost: float = 0.0

    def __post_init__(self) -> None:
        _cost(self.retrieval_estimated_cost, "retrieval_estimated_cost")
        _cost(self.rewrite_estimated_cost, "rewrite_estimated_cost")


class ConservativeRepairPolicy(RepairPolicy):
    """Prefer evidence acquisition, then constrained rewrite, else abstain.

    The policy never deletes or accepts an unresolved claim automatically.
    Hard limits remain the responsibility of ``HealService`` and cannot be
    weakened here.
    """

    def __init__(self, config: ConservativeRepairPolicyConfig | None = None) -> None:
        if config is None:
            config = ConservativeRepairPolicyConfig()
        if not isinstance(config, ConservativeRepairPolicyConfig):
            raise ConfigurationError(
                "ConservativeRepairPolicy.config must be a "
                "ConservativeRepairPolicyConfig"
            )
        self._config = config

    @property
    def config(self) -> ConservativeRepairPolicyConfig:
        return self._config

    def decide(
        self, verdict: ClaimVerdict, state: HealPolicyState
    ) -> RepairDecision:
        if not isinstance(verdict, ClaimVerdict):
            raise DomainValidationError(
                "ConservativeRepairPolicy.verdict must be a ClaimVerdict"
            )
        if not isinstance(state, HealPolicyState):
            raise DomainValidationError(
                "ConservativeRepairPolicy.state must be a HealPolicyState"
            )
        if verdict.claim.claim_id != state.target_claim_id:
            raise DomainValidationError(
                "ConservativeRepairPolicy target claim does not match policy state"
            )

        if verdict.state in {
            VerificationState.SUPPORTED,
            VerificationState.NOT_CHECKABLE,
        }:
            return RepairDecision(
                action=RepairAction.ACCEPT,
                estimated_cost=0.0,
                note="claim is already terminal for groundedness handling",
            )

        # Contradicted/conflicting claims already have evidence to constrain a
        # rewrite, so prefer correction over collecting more context.
        if verdict.state in {
            VerificationState.CONTRADICTED,
            VerificationState.CONFLICTING_EVIDENCE,
        } and state.rewriter_available:
            return RepairDecision(
                action=RepairAction.REWRITE,
                estimated_cost=self._config.rewrite_estimated_cost,
                note="rewrite unresolved claim against its supplied evidence",
            )

        # For missing evidence, try retrieval once before considering rewrite.
        if (
            state.retriever_available
            and state.request_has_query
            and state.attempts_for_claim == 0
        ):
            return RepairDecision(
                action=RepairAction.RETRIEVE,
                estimated_cost=self._config.retrieval_estimated_cost,
                note="retrieve additional evidence for unresolved claim",
            )

        if state.rewriter_available:
            return RepairDecision(
                action=RepairAction.REWRITE,
                estimated_cost=self._config.rewrite_estimated_cost,
                note="rewrite unresolved claim without an additional retrieval",
            )

        return RepairDecision(
            action=RepairAction.ABSTAIN,
            estimated_cost=0.0,
            note="no safe repair capability is available",
        )


class HardBoundsOnlyStopPolicy(StopPolicy):
    """Request no extra early stop and rely on ``HealService`` hard bounds."""

    def should_stop(self, state: HealPolicyState) -> bool:
        if not isinstance(state, HealPolicyState):
            raise DomainValidationError(
                "HardBoundsOnlyStopPolicy.state must be a HealPolicyState"
            )
        return False
