import pytest

from groundguard_rag.domain.config import HealConfig, VerifyConfig
from groundguard_rag.domain.exceptions import ConfigurationError


def test_verify_config_defaults_to_no_network():
    config = VerifyConfig()
    assert config.allow_network is False


def test_verify_config_can_opt_into_network_explicitly():
    config = VerifyConfig(allow_network=True)
    assert config.allow_network is True


def _minimal_heal_config(**overrides):
    fields = dict(
        max_rounds=3,
        max_attempts_per_claim=2,
        timeout_seconds=30.0,
        cost_budget=1.0,
        min_improvement=0.05,
    )
    fields.update(overrides)
    return HealConfig(**fields)


def test_valid_heal_config_constructs():
    config = _minimal_heal_config()
    assert config.max_rounds == 3


@pytest.mark.parametrize("max_rounds", [0, -1])
def test_max_rounds_must_be_at_least_one(max_rounds):
    with pytest.raises(ConfigurationError):
        _minimal_heal_config(max_rounds=max_rounds)


@pytest.mark.parametrize("max_attempts_per_claim", [0, -1])
def test_max_attempts_per_claim_must_be_at_least_one(max_attempts_per_claim):
    with pytest.raises(ConfigurationError):
        _minimal_heal_config(max_attempts_per_claim=max_attempts_per_claim)


@pytest.mark.parametrize("timeout_seconds", [0, -1.0])
def test_timeout_seconds_must_be_positive(timeout_seconds):
    with pytest.raises(ConfigurationError):
        _minimal_heal_config(timeout_seconds=timeout_seconds)


@pytest.mark.parametrize("cost_budget", [0, -1.0])
def test_cost_budget_must_be_positive(cost_budget):
    with pytest.raises(ConfigurationError):
        _minimal_heal_config(cost_budget=cost_budget)


def test_min_improvement_cannot_be_negative():
    with pytest.raises(ConfigurationError):
        _minimal_heal_config(min_improvement=-0.01)


def test_min_improvement_zero_is_allowed():
    config = _minimal_heal_config(min_improvement=0.0)
    assert config.min_improvement == 0.0


def test_min_improvement_one_is_allowed():
    # Stage 0.3: min_improvement is compared against calibrated-confidence
    # deltas (RepairActionRecord.calibrated_confidence_before/_after),
    # which are bounded to [0, 1] -- 1.0 is the boundary, not out of range.
    config = _minimal_heal_config(min_improvement=1.0)
    assert config.min_improvement == 1.0


@pytest.mark.parametrize("min_improvement", [1.01, 2.0, 100.0])
def test_min_improvement_above_one_rejected(min_improvement):
    # Stage 0.3 tightening: min_improvement used to be an unbounded
    # positive float; it is now bounded to [0, 1] since raw
    # verifier-specific scores must never be used for cross-round
    # improvement comparisons (core_requirements #8.3).
    with pytest.raises(ConfigurationError):
        _minimal_heal_config(min_improvement=min_improvement)


@pytest.mark.parametrize("field_name", ["max_rounds", "max_attempts_per_claim"])
def test_bool_rejected_for_int_fields(field_name):
    # bool is a subclass of int in Python -- True/False must not silently
    # pass as 1/0 for a round or attempt count.
    with pytest.raises(ConfigurationError):
        _minimal_heal_config(**{field_name: True})


@pytest.mark.parametrize(
    "field_name", ["timeout_seconds", "cost_budget", "min_improvement"]
)
def test_bool_rejected_for_float_fields(field_name):
    with pytest.raises(ConfigurationError):
        _minimal_heal_config(**{field_name: True})


@pytest.mark.parametrize(
    "field_name", ["timeout_seconds", "cost_budget", "min_improvement"]
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_rejected_for_float_fields(field_name, bad_value):
    with pytest.raises(ConfigurationError):
        _minimal_heal_config(**{field_name: bad_value})
