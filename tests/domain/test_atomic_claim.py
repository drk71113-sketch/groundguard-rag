import pytest

from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import AtomicClaim


def test_valid_claim_constructs():
    claim = AtomicClaim(claim_id="claim-1", text="Paris is the capital of France.", start_char=0, end_char=32)
    assert claim.claim_id == "claim-1"
    assert claim.start_char == 0
    assert claim.end_char == 32


def test_to_dict_round_trips_fields():
    claim = AtomicClaim(claim_id="claim-1", text="x", start_char=1, end_char=2)
    assert claim.to_dict() == {
        "claim_id": "claim-1",
        "text": "x",
        "start_char": 1,
        "end_char": 2,
    }


@pytest.mark.parametrize("claim_id", ["", "  "])
def test_empty_claim_id_rejected(claim_id):
    with pytest.raises(DomainValidationError):
        AtomicClaim(claim_id=claim_id, text="x", start_char=0, end_char=1)


def test_blank_text_rejected():
    with pytest.raises(DomainValidationError):
        AtomicClaim(claim_id="c1", text="   ", start_char=0, end_char=1)


def test_negative_start_char_rejected():
    with pytest.raises(DomainValidationError):
        AtomicClaim(claim_id="c1", text="x", start_char=-1, end_char=1)


def test_end_char_equal_to_start_char_rejected():
    # A zero-length span is not a claim.
    with pytest.raises(DomainValidationError):
        AtomicClaim(claim_id="c1", text="x", start_char=5, end_char=5)


def test_end_char_before_start_char_rejected():
    with pytest.raises(DomainValidationError):
        AtomicClaim(claim_id="c1", text="x", start_char=5, end_char=3)


@pytest.mark.parametrize("field_name", ["start_char", "end_char"])
@pytest.mark.parametrize("bad_value", [True, False, 1.5, "1"])
def test_offset_fields_reject_bool_and_non_int(field_name, bad_value):
    # bool is a subclass of int in Python -- True/False must not silently
    # pass as 1/0 for a character offset.
    kwargs = dict(claim_id="c1", text="x", start_char=0, end_char=5)
    kwargs[field_name] = bad_value
    with pytest.raises(DomainValidationError):
        AtomicClaim(**kwargs)


def test_non_factual_text_is_still_a_valid_atomic_claim():
    # Resolves the stage 0.1 review's "AtomicClaim vs NOT_CHECKABLE"
    # semantic tension: AtomicClaim is a decomposed *unit of text*, not an
    # assertion that the unit is a checkable factual proposition. A
    # ClaimDecomposer must not pre-filter greetings/opinions/instructions
    # out before they even become an AtomicClaim -- that filtering
    # judgement is exactly what a Verifier's NOT_CHECKABLE verdict
    # represents later (see test_claim_verdict.py).
    greeting = AtomicClaim(claim_id="c1", text="Hi there, how are you?", start_char=0, end_char=22)
    assert greeting.text == "Hi there, how are you?"
