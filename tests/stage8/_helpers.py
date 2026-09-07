from __future__ import annotations

from groundguard_rag.adapters.decomposition import RuleBasedClaimDecomposer
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.config import VerifyConfig
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.models import (
    AtomicClaim,
    Chunk,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceCandidate,
    EvidenceReference,
    LabelScores,
)
from groundguard_rag.domain.ports import (
    EvidenceSelector,
    HealProgressEvaluator,
    Verifier,
)


class AllChunksSelector(EvidenceSelector):
    def select(
        self, claim: AtomicClaim, chunks: list[Chunk]
    ) -> list[EvidenceReference]:
        return [EvidenceReference(chunk_id=chunk.chunk_id) for chunk in chunks]


class GermanyFranceVerifier(Verifier):
    """Deterministic test double, not a production detector."""

    def verify(
        self, claim: AtomicClaim, evidence: list[EvidenceCandidate]
    ) -> ClaimVerdict:
        if "Germany" in claim.text:
            state = VerificationState.CONTRADICTED
            values = (0.05, 0.9, 0.05)
        elif evidence:
            state = VerificationState.SUPPORTED
            values = (0.9, 0.05, 0.05)
        else:
            return ClaimVerdict(
                claim=claim,
                state=VerificationState.INSUFFICIENT_EVIDENCE,
                evidence_assessments=(),
            )
        assessments = tuple(
            EvidenceAssessment(
                reference=item.reference,
                label_scores=LabelScores(
                    supported=values[0],
                    contradicted=values[1],
                    insufficient=values[2],
                    score_kind="probabilities",
                ),
                state=state,
                rationale=None,
                verifier_id="stage8-test-verifier",
                verifier_revision="v1",
            )
            for item in evidence
        )
        return ClaimVerdict(
            claim=claim,
            state=state,
            evidence_assessments=assessments,
        )


class StateProgress(HealProgressEvaluator):
    def evaluate(self, verdict: ClaimVerdict) -> float:
        return {
            VerificationState.SUPPORTED: 0.95,
            VerificationState.CONTRADICTED: 0.05,
            VerificationState.INSUFFICIENT_EVIDENCE: 0.1,
            VerificationState.CONFLICTING_EVIDENCE: 0.02,
            VerificationState.NOT_CHECKABLE: 1.0,
        }[verdict.state]

    @property
    def calibrator_id(self) -> str:
        return "stage8-test-groundedness"

    @property
    def calibrator_revision(self) -> str:
        return "sha256:stage8-test-only"


def make_verify_service() -> VerifyService:
    return VerifyService(
        decomposer=RuleBasedClaimDecomposer(),
        selector=AllChunksSelector(),
        verifier=GermanyFranceVerifier(),
        config=VerifyConfig(),
        model_revision="stage8-test-v1",
        threshold_version="stage8-threshold-v1",
        clock=lambda: "2026-09-03T12:00:00+00:00",
    )
