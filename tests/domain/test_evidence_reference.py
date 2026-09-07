import pytest

from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import EvidenceReference


def test_whole_chunk_reference_constructs_with_no_span():
    ref = EvidenceReference(chunk_id="c1")
    assert ref.start_char is None
    assert ref.end_char is None
    assert ref.relevance_score is None


def test_span_reference_constructs():
    ref = EvidenceReference(chunk_id="c1", start_char=0, end_char=5, relevance_score=0.8)
    assert (ref.start_char, ref.end_char) == (0, 5)
    assert ref.relevance_score == 0.8


def test_to_dict_round_trips_fields():
    ref = EvidenceReference(chunk_id="c1", start_char=0, end_char=5, relevance_score=0.5)
    assert ref.to_dict() == {
        "chunk_id": "c1",
        "start_char": 0,
        "end_char": 5,
        "relevance_score": 0.5,
    }


def test_empty_chunk_id_rejected():
    with pytest.raises(DomainValidationError):
        EvidenceReference(chunk_id="")


@pytest.mark.parametrize(
    "start_char,end_char",
    [(0, None), (None, 5)],
)
def test_partial_span_rejected(start_char, end_char):
    # Either both offsets are given (a span) or neither (whole chunk) --
    # a half-specified span is ambiguous.
    with pytest.raises(DomainValidationError):
        EvidenceReference(chunk_id="c1", start_char=start_char, end_char=end_char)


def test_end_char_not_after_start_char_rejected():
    with pytest.raises(DomainValidationError):
        EvidenceReference(chunk_id="c1", start_char=5, end_char=5)


def test_negative_start_char_rejected():
    with pytest.raises(DomainValidationError):
        EvidenceReference(chunk_id="c1", start_char=-1, end_char=2)


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_relevance_score_out_of_range_rejected(score):
    with pytest.raises(DomainValidationError):
        EvidenceReference(chunk_id="c1", relevance_score=score)


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_relevance_score_boundary_values_accepted(score):
    ref = EvidenceReference(chunk_id="c1", relevance_score=score)
    assert ref.relevance_score == score


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_relevance_score_non_finite_rejected(score):
    with pytest.raises(DomainValidationError):
        EvidenceReference(chunk_id="c1", relevance_score=score)


@pytest.mark.parametrize("field_name", ["start_char", "end_char"])
@pytest.mark.parametrize("bad_value", [True, False, 1.5, "1"])
def test_offset_fields_reject_bool_and_non_int(field_name, bad_value):
    kwargs = dict(chunk_id="c1", start_char=0, end_char=5)
    kwargs[field_name] = bad_value
    with pytest.raises(DomainValidationError):
        EvidenceReference(**kwargs)
