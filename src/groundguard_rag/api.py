"""Stable public facade over the explicit verify/heal application services."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from groundguard_rag.application.heal_service import HealResult, HealService
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import AuditReport, Chunk, VerificationRequest
from groundguard_rag.presentation.views import AnswerViewRenderer, AnswerViews


def _thaw_json(value: Any) -> Any:
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "source": chunk.source,
        "metadata": _thaw_json(chunk.metadata),
    }


@dataclass(frozen=True)
class VerificationOutput:
    """Public verify result containing the answer, both views, and audit JSON."""

    answer: str
    views: AnswerViews
    audit_report: AuditReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "views": self.views.to_dict(),
            "audit_report": self.audit_report.to_dict(),
        }


@dataclass(frozen=True)
class HealingOutput:
    """Public bounded-heal result; it remains a candidate until the host accepts it."""

    candidate_answer: str
    candidate_chunks: tuple[Chunk, ...]
    output_hash: str
    views: AnswerViews
    audit_report: AuditReport
    abstained_claim_ids: tuple[str, ...]
    accepted_claim_ids: tuple[str, ...]

    @classmethod
    def from_result(
        cls, result: HealResult, renderer: AnswerViewRenderer
    ) -> HealingOutput:
        return cls(
            candidate_answer=result.candidate_answer,
            candidate_chunks=result.candidate_chunks,
            output_hash=result.output_hash,
            views=renderer.render(result.candidate_answer, result.audit_report),
            audit_report=result.audit_report,
            abstained_claim_ids=result.abstained_claim_ids,
            accepted_claim_ids=result.accepted_claim_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_answer": self.candidate_answer,
            "candidate_chunks": [
                _chunk_to_dict(chunk) for chunk in self.candidate_chunks
            ],
            "output_hash": self.output_hash,
            "views": self.views.to_dict(),
            "audit_report": self.audit_report.to_dict(),
            "abstained_claim_ids": list(self.abstained_claim_ids),
            "accepted_claim_ids": list(self.accepted_claim_ids),
        }


class GroundGuard:
    """Explicitly assembled GroundGuard-RAG integration facade.

    Construction never creates provider clients.  Verify and heal preserve the
    application services' no-writeback behavior and only add presentation
    views to their immutable results.
    """

    def __init__(
        self,
        *,
        verify_service: VerifyService,
        heal_service: HealService | None = None,
        renderer: AnswerViewRenderer | None = None,
    ) -> None:
        if not isinstance(verify_service, VerifyService):
            raise ConfigurationError(
                "GroundGuard.verify_service must be a VerifyService"
            )
        if heal_service is not None and not isinstance(heal_service, HealService):
            raise ConfigurationError(
                "GroundGuard.heal_service must be a HealService or None"
            )
        if heal_service is not None and heal_service.verify_service is not verify_service:
            raise ConfigurationError(
                "GroundGuard verify_service must be the same instance used by HealService"
            )
        if renderer is None:
            renderer = AnswerViewRenderer()
        if not isinstance(renderer, AnswerViewRenderer):
            raise ConfigurationError(
                "GroundGuard.renderer must be an AnswerViewRenderer"
            )
        self._verify_service = verify_service
        self._heal_service = heal_service
        self._renderer = renderer

    @property
    def heal_enabled(self) -> bool:
        return self._heal_service is not None

    def verify(self, request: VerificationRequest) -> VerificationOutput:
        if not isinstance(request, VerificationRequest):
            raise DomainValidationError(
                "GroundGuard.verify request must be a VerificationRequest"
            )
        report = self._verify_service.verify(request)
        return VerificationOutput(
            answer=request.answer,
            views=self._renderer.render(request.answer, report),
            audit_report=report,
        )

    def heal(self, request: VerificationRequest) -> HealingOutput:
        if not isinstance(request, VerificationRequest):
            raise DomainValidationError(
                "GroundGuard.heal request must be a VerificationRequest"
            )
        if self._heal_service is None:
            raise ConfigurationError(
                "GroundGuard.heal requires an explicitly configured HealService"
            )
        result = self._heal_service.heal(request)
        return HealingOutput.from_result(result, self._renderer)


def verify(
    request: VerificationRequest,
    *,
    service: VerifyService,
    renderer: AnswerViewRenderer | None = None,
) -> VerificationOutput:
    """One-shot public verify helper with no implicit dependencies."""

    return GroundGuard(verify_service=service, renderer=renderer).verify(request)


def heal(
    request: VerificationRequest,
    *,
    verify_service: VerifyService,
    heal_service: HealService,
    renderer: AnswerViewRenderer | None = None,
) -> HealingOutput:
    """One-shot public heal helper with explicitly supplied services."""

    return GroundGuard(
        verify_service=verify_service,
        heal_service=heal_service,
        renderer=renderer,
    ).heal(request)
