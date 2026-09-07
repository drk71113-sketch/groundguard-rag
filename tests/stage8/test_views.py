from __future__ import annotations

import dataclasses

import pytest

from groundguard_rag.domain.enums import RunMode, VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import (
    SCHEMA_VERSION,
    AtomicClaim,
    AuditReport,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceReference,
    LabelScores,
)
from groundguard_rag.presentation import AnswerViewRenderer


def _report() -> AuditReport:
    claim = AtomicClaim("c1", "Paris is in France.", 0, 19)
    edge = EvidenceAssessment(
        reference=EvidenceReference("chunk-1", relevance_score=0.8),
        label_scores=LabelScores(0.9, 0.05, 0.05, "probabilities"),
        state=VerificationState.SUPPORTED,
        rationale="entailed",
        verifier_id="test",
        verifier_revision="v1",
    )
    verdict = ClaimVerdict(
        claim=claim,
        state=VerificationState.SUPPORTED,
        evidence_assessments=(edge,),
        calibrated_confidence=0.87,
    )
    return AuditReport(
        schema_version=SCHEMA_VERSION,
        request_id="r1",
        input_hash="sha256:test",
        model_revision="m1",
        threshold_version="t1",
        created_at="2026-09-03T12:00:00+00:00",
        run_mode=RunMode.VERIFY,
        verdicts=(verdict,),
        calibrator_id="calibrator",
        calibrator_revision="cal-v1",
    )


def test_renderer_preserves_sidecar_answer_and_adds_inline_annotation():
    answer = "Paris is in France."
    views = AnswerViewRenderer().render(answer, _report())

    assert views.sidecar.answer == answer
    assert views.sidecar.claims[0].calibrated_confidence == 0.87
    assert views.sidecar.claims[0].evidence[0].chunk_id == "chunk-1"
    assert "state=SUPPORTED" in views.inline
    assert "confidence=0.870" in views.inline
    assert 'evidence=["chunk-1"]' in views.inline
    assert "raw_score" not in str(views.to_dict())


def test_renderer_never_substitutes_raw_score_for_missing_confidence():
    report = _report()
    verdict = dataclasses.replace(
        report.verdicts[0], raw_score=0.999, calibrated_confidence=None
    )
    report = dataclasses.replace(
        report,
        verdicts=(verdict,),
        calibrator_id=None,
        calibrator_revision=None,
    )
    views = AnswerViewRenderer().render("Paris is in France.", report)
    assert "confidence=unavailable" in views.inline
    assert "0.999" not in views.inline


def test_renderer_rejects_report_whose_spans_do_not_match_answer():
    with pytest.raises(DomainValidationError):
        AnswerViewRenderer().render("Different answer.", _report())
