import pytest

from groundguard_rag.domain.enums import RepairAction
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import RepairDecision


def test_external_action_requires_explicit_finite_nonnegative_quote():
    decision = RepairDecision(action=RepairAction.REWRITE, estimated_cost=0.25)
    assert decision.estimated_cost == 0.25


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, None])
def test_invalid_estimated_cost_rejected(value):
    with pytest.raises(DomainValidationError):
        RepairDecision(action=RepairAction.REWRITE, estimated_cost=value)


def test_retrieval_query_only_allowed_for_retrieve():
    decision = RepairDecision(
        action=RepairAction.RETRIEVE,
        estimated_cost=0.0,
        retrieval_query="targeted query",
    )
    assert decision.retrieval_query == "targeted query"
    with pytest.raises(DomainValidationError):
        RepairDecision(
            action=RepairAction.REWRITE,
            estimated_cost=0.0,
            retrieval_query="not legal here",
        )


@pytest.mark.parametrize(
    "action", [RepairAction.DELETE, RepairAction.ABSTAIN, RepairAction.ACCEPT]
)
def test_local_terminal_actions_must_quote_zero_cost(action):
    with pytest.raises(DomainValidationError):
        RepairDecision(action=action, estimated_cost=0.01)


def test_action_requires_enum_not_a_bare_string():
    with pytest.raises(DomainValidationError):
        RepairDecision(action="rewrite", estimated_cost=0.0)
