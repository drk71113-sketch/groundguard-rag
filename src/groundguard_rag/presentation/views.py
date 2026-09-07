"""Build sidecar and inline answer views without changing audit state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import AuditReport

GROUNDING_SCOPE_NOTE = (
    "Verification states describe grounding against the supplied chunks only; "
    "INSUFFICIENT_EVIDENCE is not a real-world falsity judgement."
)


@dataclass(frozen=True)
class EvidenceAnnotation:
    """Presentation-safe summary of one claim-evidence audit edge."""

    chunk_id: str
    start_char: int | None
    end_char: int | None
    relevance_score: float | None
    state: VerificationState

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "relevance_score": self.relevance_score,
            "state": self.state.value,
        }


@dataclass(frozen=True)
class ClaimAnnotation:
    """Claim-level sidecar entry; raw model scores are intentionally absent."""

    claim_id: str
    text: str
    start_char: int
    end_char: int
    state: VerificationState
    calibrated_confidence: float | None
    evidence: tuple[EvidenceAnnotation, ...]
    rationale: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "state": self.state.value,
            "calibrated_confidence": self.calibrated_confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SidecarView:
    """Original answer plus separate claim annotations."""

    answer: str
    claims: tuple[ClaimAnnotation, ...]
    scope_note: str = GROUNDING_SCOPE_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "claims": [claim.to_dict() for claim in self.claims],
            "scope_note": self.scope_note,
        }


@dataclass(frozen=True)
class AnswerViews:
    """The two required presentation forms for one verified answer."""

    sidecar: SidecarView
    inline: str

    def to_dict(self) -> dict[str, Any]:
        return {"sidecar": self.sidecar.to_dict(), "inline": self.inline}


class AnswerViewRenderer:
    """Render immutable views from the report's final claim graph."""

    def render(self, answer: str, report: AuditReport) -> AnswerViews:
        if not isinstance(answer, str) or not answer.strip():
            raise DomainValidationError(
                "AnswerViewRenderer.answer must be a non-empty string"
            )
        if not isinstance(report, AuditReport):
            raise DomainValidationError(
                "AnswerViewRenderer.report must be an AuditReport"
            )

        annotations: list[ClaimAnnotation] = []
        previous_end = 0
        for verdict in report.verdicts:
            claim = verdict.claim
            if claim.start_char < previous_end:
                raise DomainValidationError(
                    "AnswerViewRenderer verdict claim spans must be ordered and non-overlapping"
                )
            if claim.end_char > len(answer) or answer[
                claim.start_char : claim.end_char
            ] != claim.text:
                raise DomainValidationError(
                    "AnswerViewRenderer verdict claim spans must match answer"
                )
            previous_end = claim.end_char
            evidence = tuple(
                EvidenceAnnotation(
                    chunk_id=edge.reference.chunk_id,
                    start_char=edge.reference.start_char,
                    end_char=edge.reference.end_char,
                    relevance_score=edge.reference.relevance_score,
                    state=edge.state,
                )
                for edge in verdict.evidence_assessments
            )
            annotations.append(
                ClaimAnnotation(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    start_char=claim.start_char,
                    end_char=claim.end_char,
                    state=verdict.state,
                    calibrated_confidence=verdict.calibrated_confidence,
                    evidence=evidence,
                    rationale=verdict.rationale,
                )
            )

        inline = answer
        for annotation in reversed(annotations):
            marker = self._marker(annotation)
            inline = inline[: annotation.end_char] + marker + inline[annotation.end_char :]

        return AnswerViews(
            sidecar=SidecarView(answer=answer, claims=tuple(annotations)),
            inline=inline,
        )

    @staticmethod
    def _marker(annotation: ClaimAnnotation) -> str:
        confidence = (
            "unavailable"
            if annotation.calibrated_confidence is None
            else f"{annotation.calibrated_confidence:.3f}"
        )
        evidence_ids = [item.chunk_id for item in annotation.evidence]
        encoded_evidence = json.dumps(
            evidence_ids, ensure_ascii=False, separators=(",", ":")
        )
        return (
            f" [GG state={annotation.state.value} "
            f"confidence={confidence} evidence={encoded_evidence}]"
        )
