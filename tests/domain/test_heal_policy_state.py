from dataclasses import FrozenInstanceError

import pytest

from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import HealPolicyState


def _state(**overrides):
    fields = dict(
        round=1,
        completed_rounds=0,
        target_claim_id="c1",
        attempts_for_claim=0,
        attempts_by_claim={"c1": 0},
        elapsed_seconds=0.1,
        estimated_cost_spent=0.2,
        estimated_cost_remaining=0.8,
        state_hash="sha256:state",
        request_has_query=True,
        retriever_available=True,
        rewriter_available=False,
        verdict_states=(VerificationState.INSUFFICIENT_EVIDENCE,),
        abstained_claim_ids=(),
        accepted_claim_ids=(),
        max_rounds=3,
        max_attempts_per_claim=2,
        timeout_seconds=30.0,
        cost_budget=1.0,
        min_improvement=0.1,
    )
    fields.update(overrides)
    return HealPolicyState(**fields)


def test_valid_state_is_deeply_read_only_for_policy_bookkeeping():
    state = _state()
    with pytest.raises(FrozenInstanceError):
        state.round = 2
    with pytest.raises(TypeError):
        state.attempts_by_claim["c1"] = 1


def test_round_must_be_next_completed_round():
    with pytest.raises(DomainValidationError):
        _state(round=3, completed_rounds=1)


def test_target_attempt_count_must_match_mapping():
    with pytest.raises(DomainValidationError):
        _state(attempts_for_claim=1)


@pytest.mark.parametrize(
    "field_name", ["request_has_query", "retriever_available", "rewriter_available"]
)
def test_capability_flags_require_literal_bool(field_name):
    with pytest.raises(DomainValidationError):
        _state(**{field_name: 1})


def test_verdict_states_require_enum_members():
    with pytest.raises(DomainValidationError):
        _state(verdict_states=("INSUFFICIENT_EVIDENCE",))


def test_terminal_claim_sets_must_be_disjoint():
    with pytest.raises(DomainValidationError):
        _state(abstained_claim_ids=("c1",), accepted_claim_ids=("c1",))


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_min_improvement_stays_on_unit_interval(value):
    with pytest.raises(DomainValidationError):
        _state(min_improvement=value)
