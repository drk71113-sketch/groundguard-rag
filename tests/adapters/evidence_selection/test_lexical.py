import math

import pytest

from groundguard_rag.adapters.decomposition import RuleBasedClaimDecomposer
from groundguard_rag.adapters.evidence_selection import (
    LexicalEvidenceSelector,
    LexicalSelectorConfig,
)
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.config import VerifyConfig
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import (
    AtomicClaim,
    Chunk,
    ClaimVerdict,
    EvidenceAssessment,
    LabelScores,
    VerificationRequest,
)
from groundguard_rag.domain.ports import Verifier


def _claim(text="Paris is the capital of France."):
    return AtomicClaim(
        claim_id=f"claim-0-{len(text)}",
        text=text,
        start_char=0,
        end_char=len(text),
    )


def _chunk(chunk_id, text, metadata=None):
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=f"{chunk_id}.md",
        metadata={} if metadata is None else metadata,
    )


def test_ranks_full_overlap_above_partial_and_excludes_unrelated_chunks():
    chunks = [
        _chunk("partial", "Paris is a city."),
        _chunk("best", "France capital Paris reference material."),
        _chunk("unrelated", "Tokyo is in Japan."),
    ]

    references = LexicalEvidenceSelector().select(_claim(), chunks)

    assert [reference.chunk_id for reference in references] == ["best", "partial"]
    assert references[0].relevance_score > references[1].relevance_score > 0


def test_top_k_limits_results_after_ranking():
    selector = LexicalEvidenceSelector(LexicalSelectorConfig(top_k=1, min_score=0))
    chunks = [
        _chunk("partial", "Paris city"),
        _chunk("best", "Paris capital France"),
    ]

    references = selector.select(_claim(), chunks)

    assert [reference.chunk_id for reference in references] == ["best"]


def test_min_score_filters_weak_overlap():
    selector = LexicalEvidenceSelector(LexicalSelectorConfig(top_k=5, min_score=0.5))
    chunks = [
        _chunk("weak", "Paris city"),
        _chunk("strong", "Paris capital France"),
    ]

    references = selector.select(_claim(), chunks)

    assert [reference.chunk_id for reference in references] == ["strong"]


def test_zero_overlap_is_never_returned_even_when_min_score_is_zero():
    selector = LexicalEvidenceSelector(LexicalSelectorConfig(min_score=0))
    assert selector.select(_claim(), [_chunk("other", "Tokyo Japan")]) == []


def test_equal_scores_preserve_original_chunk_order():
    claim = _claim("alpha")
    chunks = [_chunk("first", "alpha one"), _chunk("second", "alpha two")]

    references = LexicalEvidenceSelector().select(claim, chunks)

    assert [reference.chunk_id for reference in references] == ["first", "second"]


def test_casefolding_and_stopword_filtering_keep_content_terms():
    claim = _claim("THE Cat is IN the House")
    chunks = [
        _chunk("match", "A HOUSE contains one cat."),
        _chunk("function-words", "the is in a an and"),
    ]

    references = LexicalEvidenceSelector().select(claim, chunks)

    assert [reference.chunk_id for reference in references] == ["match"]


def test_cjk_bigrams_select_matching_entities_and_phrases():
    claim = _claim("巴黎是法国首都")
    chunks = [
        _chunk("match", "法国首都是巴黎"),
        _chunk("other", "东京是日本城市"),
    ]

    references = LexicalEvidenceSelector().select(claim, chunks)

    assert [reference.chunk_id for reference in references] == ["match"]
    assert references[0].relevance_score > 0


@pytest.mark.parametrize("text", ["!!!", "the is of and", "的了是在"])
def test_claims_without_effective_tokens_return_no_evidence(text):
    assert LexicalEvidenceSelector().select(_claim(text), [_chunk("c1", text)]) == []


def test_empty_chunks_return_empty_list():
    assert LexicalEvidenceSelector().select(_claim(), []) == []


def test_references_are_whole_chunk_and_scores_are_bounded_relevance_only():
    reference = LexicalEvidenceSelector().select(
        _claim(), [_chunk("c1", "Paris capital France")]
    )[0]

    assert reference.start_char is None
    assert reference.end_char is None
    assert reference.relevance_score is not None
    assert 0.0 < reference.relevance_score <= 1.0


def test_selection_is_deterministic():
    claim = _claim()
    chunks = [_chunk("a", "Paris France"), _chunk("b", "capital France")]
    selector = LexicalEvidenceSelector()

    assert selector.select(claim, chunks) == selector.select(claim, list(chunks))


def test_selector_does_not_mutate_chunk_list_or_nested_metadata():
    claim = _claim()
    chunks = [_chunk("c1", "Paris capital France", {"nested": {"page": 1}})]
    original_chunk = chunks[0]
    original_metadata = dict(original_chunk.metadata)

    LexicalEvidenceSelector().select(claim, chunks)

    assert chunks == [original_chunk]
    assert dict(chunks[0].metadata) == original_metadata
    assert chunks[0].metadata["nested"]["page"] == 1


def test_metadata_and_source_do_not_affect_score():
    claim = _claim("alpha")
    chunks = [
        Chunk("first", "alpha", source="a", metadata={"priority": 1}),
        Chunk("second", "alpha", source="b", metadata={"priority": 999}),
    ]

    references = LexicalEvidenceSelector().select(claim, chunks)

    assert references[0].relevance_score == references[1].relevance_score
    assert [reference.chunk_id for reference in references] == ["first", "second"]


@pytest.mark.parametrize("top_k", [True, False, 0, -1, 1.5, "2"])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(ConfigurationError):
        LexicalSelectorConfig(top_k=top_k)


@pytest.mark.parametrize(
    "min_score",
    [True, False, -0.01, 1.01, float("nan"), float("inf"), "0.2"],
)
def test_invalid_min_score_is_rejected(min_score):
    with pytest.raises(ConfigurationError):
        LexicalSelectorConfig(min_score=min_score)


def test_min_score_boundaries_are_allowed():
    assert LexicalSelectorConfig(min_score=0).min_score == 0
    assert LexicalSelectorConfig(min_score=1).min_score == 1


def test_selector_rejects_wrong_config_type():
    with pytest.raises(ConfigurationError):
        LexicalEvidenceSelector(config={"top_k": 2})


def test_selector_rejects_wrong_claim_type():
    with pytest.raises(DomainValidationError):
        LexicalEvidenceSelector().select("not-a-claim", [])


@pytest.mark.parametrize("chunks", [(), "not-a-list", ["not-a-chunk"]])
def test_selector_rejects_wrong_chunks_container_or_items(chunks):
    with pytest.raises(DomainValidationError):
        LexicalEvidenceSelector().select(_claim(), chunks)


def test_selector_rejects_duplicate_chunk_ids_when_called_directly():
    chunks = [_chunk("same", "Paris"), _chunk("same", "France")]
    with pytest.raises(DomainValidationError, match="duplicate chunk_id"):
        LexicalEvidenceSelector().select(_claim(), chunks)


class InsufficientVerifier(Verifier):
    def __init__(self):
        self.received = []

    def verify(self, claim, evidence):
        self.received.append(list(evidence))
        assessments = tuple(
            EvidenceAssessment(
                reference=candidate.reference,
                label_scores=LabelScores(0.1, 0.1, 0.8, "probabilities"),
                state=VerificationState.INSUFFICIENT_EVIDENCE,
                rationale="lexical relevance is not entailment",
                verifier_id="test-insufficient",
                verifier_revision="v1",
            )
            for candidate in evidence
        )
        return ClaimVerdict(
            claim=claim,
            state=VerificationState.INSUFFICIENT_EVIDENCE,
            evidence_assessments=assessments,
            calibrated_confidence=None,
        )


def test_integrates_via_evidence_selector_port_without_becoming_a_verifier():
    verifier = InsufficientVerifier()
    service = VerifyService(
        decomposer=RuleBasedClaimDecomposer(),
        selector=LexicalEvidenceSelector(LexicalSelectorConfig(top_k=1)),
        verifier=verifier,
        calibrator=None,
        config=VerifyConfig(),
        model_revision="no-real-verifier-v1",
        threshold_version="lexical-selector-v1",
        clock=lambda: "2026-09-02T00:00:00+00:00",
    )
    request = VerificationRequest(
        request_id="request-1",
        answer="Paris is the capital of France.",
        chunks=(
            _chunk("supporting-candidate", "Paris capital France"),
            _chunk("unrelated", "Tokyo Japan"),
        ),
    )

    report = service.verify(request)

    assert len(verifier.received) == 1
    assert [candidate.reference.chunk_id for candidate in verifier.received[0]] == [
        "supporting-candidate"
    ]
    verdict = report.verdicts[0]
    assert verdict.state is VerificationState.INSUFFICIENT_EVIDENCE
    assert verdict.calibrated_confidence is None
    assert verdict.evidence_assessments[0].reference.relevance_score > 0
    assert math.isfinite(verdict.evidence_assessments[0].reference.relevance_score)
