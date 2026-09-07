import pytest

from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import RunMetrics


def test_all_none_by_default():
    metrics = RunMetrics()
    assert metrics.total_latency_ms is None
    assert metrics.stage_latencies_ms is None
    assert metrics.model_call_count is None
    assert metrics.input_tokens is None
    assert metrics.output_tokens is None
    assert metrics.estimated_cost is None


def test_fully_populated_metrics_construct():
    metrics = RunMetrics(
        total_latency_ms=120.5,
        stage_latencies_ms={"decompose": 10.0, "verify": 100.5},
        model_call_count=3,
        input_tokens=512,
        output_tokens=64,
        estimated_cost=0.002,
    )
    assert metrics.total_latency_ms == 120.5
    assert dict(metrics.stage_latencies_ms) == {"decompose": 10.0, "verify": 100.5}
    assert metrics.model_call_count == 3


def test_stage_latencies_ms_is_frozen():
    metrics = RunMetrics(stage_latencies_ms={"decompose": 10.0})
    with pytest.raises(TypeError):
        metrics.stage_latencies_ms["decompose"] = 999.0


def test_stage_latencies_ms_is_isolated_from_caller_dict():
    source = {"decompose": 10.0}
    metrics = RunMetrics(stage_latencies_ms=source)
    source["decompose"] = 999.0
    assert metrics.stage_latencies_ms["decompose"] == 10.0


@pytest.mark.parametrize(
    "field_name", ["total_latency_ms", "estimated_cost"]
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_fields_rejected(field_name, bad_value):
    with pytest.raises(DomainValidationError):
        RunMetrics(**{field_name: bad_value})


@pytest.mark.parametrize("field_name", ["total_latency_ms", "estimated_cost"])
def test_negative_float_fields_rejected(field_name):
    with pytest.raises(DomainValidationError):
        RunMetrics(**{field_name: -0.01})


@pytest.mark.parametrize(
    "field_name", ["model_call_count", "input_tokens", "output_tokens"]
)
def test_negative_int_fields_rejected(field_name):
    with pytest.raises(DomainValidationError):
        RunMetrics(**{field_name: -1})


@pytest.mark.parametrize(
    "field_name", ["model_call_count", "input_tokens", "output_tokens"]
)
def test_bool_rejected_for_int_fields(field_name):
    with pytest.raises(DomainValidationError):
        RunMetrics(**{field_name: True})


@pytest.mark.parametrize("field_name", ["total_latency_ms", "estimated_cost"])
def test_bool_rejected_for_float_fields(field_name):
    with pytest.raises(DomainValidationError):
        RunMetrics(**{field_name: True})


def test_negative_stage_latency_rejected():
    with pytest.raises(DomainValidationError):
        RunMetrics(stage_latencies_ms={"decompose": -1.0})


def test_non_finite_stage_latency_rejected():
    with pytest.raises(DomainValidationError):
        RunMetrics(stage_latencies_ms={"decompose": float("nan")})


def test_non_string_stage_key_rejected():
    with pytest.raises(DomainValidationError):
        RunMetrics(stage_latencies_ms={123: 10.0})


def test_stage_latencies_ms_must_be_a_mapping():
    with pytest.raises(DomainValidationError):
        RunMetrics(stage_latencies_ms=[("decompose", 10.0)])


def test_to_dict_shape():
    metrics = RunMetrics(
        total_latency_ms=120.5,
        stage_latencies_ms={"decompose": 10.0},
        model_call_count=3,
        input_tokens=512,
        output_tokens=64,
        estimated_cost=0.002,
    )
    assert metrics.to_dict() == {
        "total_latency_ms": 120.5,
        "stage_latencies_ms": {"decompose": 10.0},
        "model_call_count": 3,
        "input_tokens": 512,
        "output_tokens": 64,
        "estimated_cost": 0.002,
    }


def test_to_dict_shape_when_empty():
    assert RunMetrics().to_dict() == {
        "total_latency_ms": None,
        "stage_latencies_ms": None,
        "model_call_count": None,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost": None,
    }
