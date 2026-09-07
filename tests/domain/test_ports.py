"""Port contract tests: abstract enforcement, signatures, and minimal fakes.

Concrete adapter algorithms are tested in their own adapter/application files.
"""

import inspect
import typing

import pytest

from groundguard_rag.domain.enums import RepairAction, RunMode, VerificationState
from groundguard_rag.domain.models import (
    AtomicClaim,
    AuditReport,
    Chunk,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceCandidate,
    EvidenceReference,
    LabelScores,
    RepairDecision,
)
from groundguard_rag.domain.ports import (
    AuditStore,
    Calibrator,
    ClaimDecomposer,
    EvidenceSelector,
    HealProgressEvaluator,
    RepairPolicy,
    Retriever,
    Rewriter,
    StopPolicy,
    Verifier,
)

PORT_CLASSES = [
    ClaimDecomposer,
    EvidenceSelector,
    Verifier,
    Calibrator,
    Retriever,
    Rewriter,
    HealProgressEvaluator,
    RepairPolicy,
    StopPolicy,
    AuditStore,
]


@pytest.mark.parametrize("port_class", PORT_CLASSES)
def test_port_cannot_be_instantiated_directly(port_class):
    with pytest.raises(TypeError):
        port_class()


def test_claim_decomposer_requires_decompose_implementation():
    class Incomplete(ClaimDecomposer):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_minimal_claim_decomposer_subclass_is_instantiable_and_callable():
    class StubDecomposer(ClaimDecomposer):
        def decompose(self, answer: str) -> list[AtomicClaim]:
            return [AtomicClaim(claim_id="c1", text=answer, start_char=0, end_char=len(answer))]

    decomposer = StubDecomposer()
    claims = decomposer.decompose("Paris is in France.")
    assert len(claims) == 1
    assert isinstance(claims[0], AtomicClaim)


def test_minimal_evidence_selector_subclass_is_instantiable():
    class StubSelector(EvidenceSelector):
        def select(self, claim, chunks):
            return [EvidenceReference(chunk_id=chunks[0].chunk_id)] if chunks else []

    selector = StubSelector()
    chunk = Chunk(chunk_id="c1", text="hi")
    refs = selector.select(claim=None, chunks=[chunk])
    assert refs[0].chunk_id == "c1"


def test_minimal_verifier_subclass_is_instantiable():
    class StubVerifier(Verifier):
        def verify(self, claim, evidence):
            return ClaimVerdict(
                claim=claim,
                state=VerificationState.INSUFFICIENT_EVIDENCE,
                evidence_assessments=(),
            )

    claim = AtomicClaim(claim_id="c1", text="x", start_char=0, end_char=1)
    candidate = EvidenceCandidate(reference=EvidenceReference(chunk_id="c1"), text="hi")
    verdict = StubVerifier().verify(claim, [candidate])
    assert verdict.state is VerificationState.INSUFFICIENT_EVIDENCE


def _evidence_param_annotation_text(method) -> str:
    hints = typing.get_type_hints(method)
    annotation = hints.get("evidence", inspect.signature(method).parameters["evidence"].annotation)
    return str(annotation)


def test_verifier_verify_takes_evidence_candidates_not_bare_references():
    # Stage 0.1 regression guard: Verifier.verify must accept
    # EvidenceCandidate (reference + actual text), not a bare
    # EvidenceReference -- otherwise an adapter has no explicit way to get
    # evidence content and would be pushed toward a hidden global lookup.
    annotation_text = _evidence_param_annotation_text(Verifier.verify)
    assert "EvidenceCandidate" in annotation_text
    assert "EvidenceReference" not in annotation_text


def test_rewriter_rewrite_takes_evidence_candidates_not_bare_references():
    annotation_text = _evidence_param_annotation_text(Rewriter.rewrite)
    assert "EvidenceCandidate" in annotation_text
    assert "EvidenceReference" not in annotation_text


def test_minimal_calibrator_subclass_is_instantiable():
    class StubCalibrator(Calibrator):
        def calibrate(self, verdict: ClaimVerdict) -> float:
            return 0.5

    assessment = EvidenceAssessment(
        reference=EvidenceReference(chunk_id="c1"),
        label_scores=LabelScores(supported=2.0, contradicted=-1.0, insufficient=-0.5, score_kind="logits"),
        state=VerificationState.SUPPORTED,
        rationale=None,
        verifier_id="verifier-a",
        verifier_revision="v1",
    )
    verdict = ClaimVerdict(
        claim=AtomicClaim(claim_id="c1", text="x", start_char=0, end_char=1),
        state=VerificationState.SUPPORTED,
        evidence_assessments=[assessment],
    )
    assert StubCalibrator().calibrate(verdict) == 0.5


def test_calibrator_calibrate_takes_a_claim_verdict_not_a_bare_assessment_or_float():
    # Stage 0.3 regression guard: Calibrator.calibrate must receive the full
    # ClaimVerdict (all evidence_assessments + claim-level state), not a
    # single EvidenceAssessment or a bare float -- a claim-level calibrated
    # confidence needs every edge, not just one.
    hints = typing.get_type_hints(Calibrator.calibrate)
    annotation = hints.get(
        "verdict", inspect.signature(Calibrator.calibrate).parameters["verdict"].annotation
    )
    assert "ClaimVerdict" in str(annotation)
    assert "EvidenceAssessment" not in str(annotation)


def test_minimal_retriever_subclass_is_instantiable():
    class StubRetriever(Retriever):
        def retrieve(self, query: str) -> list[Chunk]:
            return [Chunk(chunk_id="c1", text=query)]

    result = StubRetriever().retrieve("query text")
    assert result[0].text == "query text"


def test_minimal_rewriter_subclass_is_instantiable():
    class StubRewriter(Rewriter):
        def rewrite(self, claim, evidence) -> str:
            return "rewritten"

    candidate = EvidenceCandidate(reference=EvidenceReference(chunk_id="c1"), text="hi")
    assert StubRewriter().rewrite(None, [candidate]) == "rewritten"


def test_minimal_repair_policy_subclass_is_instantiable():
    class StubRepairPolicy(RepairPolicy):
        def decide(self, verdict, state):
            return RepairDecision(action=RepairAction.ABSTAIN, estimated_cost=0.0)

    assert StubRepairPolicy().decide(None, {}).action is RepairAction.ABSTAIN


def test_policy_ports_use_typed_heal_state_and_structured_decision():
    repair_hints = typing.get_type_hints(RepairPolicy.decide)
    stop_hints = typing.get_type_hints(StopPolicy.should_stop)
    assert "HealPolicyState" in str(repair_hints["state"])
    assert "RepairDecision" in str(repair_hints["return"])
    assert "HealPolicyState" in str(stop_hints["state"])


def test_minimal_heal_progress_evaluator_subclass_is_instantiable():
    class StubProgress(HealProgressEvaluator):
        def evaluate(self, verdict):
            return 0.5

        @property
        def calibrator_id(self):
            return "progress"

        @property
        def calibrator_revision(self):
            return "v1"

    assert StubProgress().evaluate(None) == 0.5


def test_minimal_stop_policy_subclass_is_instantiable():
    class StubStopPolicy(StopPolicy):
        def should_stop(self, state) -> bool:
            return True

    assert StubStopPolicy().should_stop({}) is True


def test_minimal_audit_store_subclass_is_instantiable():
    class StubAuditStore(AuditStore):
        def __init__(self):
            self._saved = {}

        def save(self, report: AuditReport) -> None:
            self._saved[report.request_id] = report

        def load(self, request_id: str) -> AuditReport:
            return self._saved[request_id]

    store = StubAuditStore()
    report = AuditReport(
        schema_version="1.1.0",
        request_id="req-1",
        input_hash="hash",
        model_revision="rev",
        threshold_version="thr",
        created_at="2026-09-01T00:00:00+00:00",
        run_mode=RunMode.VERIFY,
        verdicts=(),
    )
    store.save(report)
    assert store.load("req-1") is report
