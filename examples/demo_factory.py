"""Dependency-free pipeline demo and MCP factory.

``DemoExactEvidenceVerifier`` is intentionally a test double for exercising
the integration pipeline.  It cannot detect paraphrases or contradictions and
must not be used as a production grounding detector.
"""

from __future__ import annotations

from datetime import datetime, timezone

from groundguard_rag import GroundGuard, VerifyConfig
from groundguard_rag.adapters.decomposition import RuleBasedClaimDecomposer
from groundguard_rag.adapters.evidence_selection import LexicalEvidenceSelector
from groundguard_rag.adapters.verification import RuleBasedCheckabilityVerifier
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.models import (
    AtomicClaim,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceCandidate,
    LabelScores,
)
from groundguard_rag.domain.ports import Verifier


def _normalized(text: str) -> str:
    return text.casefold().strip().strip(".?!。！？")


class DemoExactEvidenceVerifier(Verifier):
    """Exact-substring test double used only by the runnable examples."""

    def verify(
        self, claim: AtomicClaim, evidence: list[EvidenceCandidate]
    ) -> ClaimVerdict:
        claim_text = _normalized(claim.text)
        assessments: list[EvidenceAssessment] = []
        for candidate in evidence:
            supported = claim_text in _normalized(candidate.text)
            state = (
                VerificationState.SUPPORTED
                if supported
                else VerificationState.INSUFFICIENT_EVIDENCE
            )
            assessments.append(
                EvidenceAssessment(
                    reference=candidate.reference,
                    label_scores=LabelScores(
                        supported=1.0 if supported else 0.0,
                        contradicted=0.0,
                        insufficient=0.0 if supported else 1.0,
                        score_kind="demo-exact-match",
                    ),
                    state=state,
                    rationale="demo exact substring match" if supported else None,
                    verifier_id="demo-exact-match",
                    verifier_revision="not-for-production",
                )
            )
        overall = (
            VerificationState.SUPPORTED
            if any(
                item.state is VerificationState.SUPPORTED for item in assessments
            )
            else VerificationState.INSUFFICIENT_EVIDENCE
        )
        return ClaimVerdict(
            claim=claim,
            state=overall,
            evidence_assessments=tuple(assessments),
            rationale="pipeline demonstration only",
        )


def build_groundguard() -> GroundGuard:
    """Return the explicit local demo assembly used by CLI examples."""

    service = VerifyService(
        decomposer=RuleBasedClaimDecomposer(),
        selector=LexicalEvidenceSelector(),
        verifier=RuleBasedCheckabilityVerifier(DemoExactEvidenceVerifier()),
        config=VerifyConfig(),
        model_revision="demo-exact-match/not-for-production",
        threshold_version="demo-v1",
        clock=lambda: datetime.now(timezone.utc).isoformat(),
    )
    return GroundGuard(verify_service=service)
