import pytest

from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import LabelScores


def test_valid_logit_scores_construct():
    scores = LabelScores(supported=2.1, contradicted=-1.5, insufficient=-0.3, score_kind="logits")
    assert scores.supported == 2.1
    assert scores.contradicted == -1.5
    assert scores.insufficient == -0.3
    assert scores.score_kind == "logits"


def test_valid_probability_scores_construct():
    scores = LabelScores(supported=0.7, contradicted=0.1, insufficient=0.2, score_kind="probabilities")
    assert scores.score_kind == "probabilities"


def test_round_trip_via_to_dict_preserves_all_three_raw_values():
    # This is the core regression guard for stage 0.2's item 1: the three
    # NLI-style values (entailment/contradiction/neutral) must all survive
    # a round trip, not collapse to a single winning-label scalar.
    scores = LabelScores(supported=2.1, contradicted=-1.5, insufficient=-0.3, score_kind="logits")
    payload = scores.to_dict()
    assert payload == {
        "supported": 2.1,
        "contradicted": -1.5,
        "insufficient": -0.3,
        "score_kind": "logits",
    }
    reconstructed = LabelScores(**payload)
    assert reconstructed == scores


@pytest.mark.parametrize("field_name", ["supported", "contradicted", "insufficient"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_score_rejected(field_name, bad_value):
    kwargs = dict(supported=1.0, contradicted=0.5, insufficient=0.1, score_kind="logits")
    kwargs[field_name] = bad_value
    with pytest.raises(DomainValidationError):
        LabelScores(**kwargs)


def test_empty_score_kind_rejected():
    with pytest.raises(DomainValidationError):
        LabelScores(supported=1.0, contradicted=0.0, insufficient=0.0, score_kind="")


@pytest.mark.parametrize("field_name", ["supported", "contradicted", "insufficient"])
@pytest.mark.parametrize("bad_value", [-0.01, 1.01])
def test_probabilities_out_of_unit_range_rejected(field_name, bad_value):
    kwargs = dict(supported=0.3, contradicted=0.3, insufficient=0.4, score_kind="probabilities")
    kwargs[field_name] = bad_value
    with pytest.raises(DomainValidationError):
        LabelScores(**kwargs)


def test_logits_are_not_bounded_to_unit_range():
    # score_kind == "logits" carries no [0, 1] constraint -- only
    # "probabilities" does.
    scores = LabelScores(supported=15.0, contradicted=-20.0, insufficient=3.0, score_kind="logits")
    assert scores.supported == 15.0


def test_probabilities_summing_to_exactly_one_accepted():
    scores = LabelScores(supported=0.5, contradicted=0.3, insufficient=0.2, score_kind="probabilities")
    assert scores.supported == 0.5


def test_probabilities_summing_to_one_within_float_tolerance_accepted():
    # 0.1 + 0.2 + 0.7 != 1.0 exactly in binary floating point -- the check
    # must use a tolerance, not exact equality.
    scores = LabelScores(supported=0.1, contradicted=0.2, insufficient=0.7, score_kind="probabilities")
    assert abs(scores.supported + scores.contradicted + scores.insufficient - 1.0) < 1e-9


@pytest.mark.parametrize(
    "supported,contradicted,insufficient",
    [
        (0.5, 0.5, 0.5),  # sums to 1.5
        (0.1, 0.1, 0.1),  # sums to 0.3
        (0.9, 0.09, 0.0),  # sums to 0.99, outside tolerance
    ],
)
def test_probabilities_not_summing_to_one_rejected(supported, contradicted, insufficient):
    with pytest.raises(DomainValidationError):
        LabelScores(
            supported=supported,
            contradicted=contradicted,
            insufficient=insufficient,
            score_kind="probabilities",
        )


def test_logits_are_not_required_to_sum_to_one():
    # The sum-to-1 constraint only applies when score_kind == "probabilities".
    scores = LabelScores(supported=15.0, contradicted=-20.0, insufficient=3.0, score_kind="logits")
    assert scores.supported + scores.contradicted + scores.insufficient == -2.0
