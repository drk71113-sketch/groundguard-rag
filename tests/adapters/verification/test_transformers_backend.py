from __future__ import annotations

import math
from contextlib import AbstractContextManager
from types import SimpleNamespace

import pytest

import groundguard_rag.adapters.verification.transformers_backend as backend_module
from groundguard_rag.adapters.verification import (
    NliBackendContractError,
    NliLabelMapping,
    NliVerifier,
    NliVerifierConfig,
    TransformersNliBackend,
)
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import (
    AtomicClaim,
    EvidenceCandidate,
    EvidenceReference,
)


class FakeDeviceValue:
    def __init__(self) -> None:
        self.devices: list[str] = []

    def to(self, device: str) -> "FakeDeviceValue":
        self.devices.append(device)
        return self


class FakeTokenizer:
    def __init__(self, output=None) -> None:
        self.calls: list[dict[str, object]] = []
        self.output = output

    def __call__(
        self,
        premises,
        *,
        text_pair,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        self.calls.append(
            {
                "premises": premises,
                "hypotheses": text_pair,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        if self.output is not None:
            return self.output
        return {
            "input_ids": FakeDeviceValue(),
            "attention_mask": FakeDeviceValue(),
        }


class FakeTensor:
    def __init__(self, rows) -> None:
        self.rows = rows

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self):
        return self.rows


class FakeModel:
    def __init__(self, batches, *, num_labels: int = 3, output_factory=None) -> None:
        self.config = SimpleNamespace(num_labels=num_labels)
        self.batches = list(batches)
        self.output_factory = output_factory
        self.to_calls: list[str] = []
        self.eval_calls = 0
        self.calls: list[dict[str, object]] = []

    def to(self, device: str) -> "FakeModel":
        self.to_calls.append(device)
        return self

    def eval(self) -> None:
        self.eval_calls += 1

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.output_factory is not None:
            return self.output_factory()
        return SimpleNamespace(logits=FakeTensor(self.batches.pop(0)))


class FakeInferenceContext(AbstractContextManager):
    def __init__(self, runtime: "FakeTorch") -> None:
        self.runtime = runtime

    def __enter__(self):
        self.runtime.entries += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.runtime.exits += 1
        return False


class FakeTorch:
    def __init__(self) -> None:
        self.entries = 0
        self.exits = 0

    def inference_mode(self) -> FakeInferenceContext:
        return FakeInferenceContext(self)


MAPPING = NliLabelMapping(
    entailment_index=2,
    contradiction_index=0,
    neutral_index=1,
)


def make_backend(
    batches,
    *,
    tokenizer=None,
    model=None,
    torch_module=None,
    batch_size=16,
    max_length=512,
    device="cpu",
    mapping=MAPPING,
):
    tokenizer = tokenizer or FakeTokenizer()
    model = model or FakeModel(batches)
    torch_module = torch_module or FakeTorch()
    backend = TransformersNliBackend(
        tokenizer=tokenizer,
        model=model,
        label_mapping=mapping,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        torch_module=torch_module,
    )
    return backend, tokenizer, model, torch_module


def test_explicit_label_mapping_reorders_logits_into_domain_semantics():
    backend, _, _, _ = make_backend([[[7.0, 3.0, 11.0]]])

    result = backend.predict_batch([("evidence", "claim")])

    assert len(result) == 1
    assert result[0].supported == 11.0
    assert result[0].contradicted == 7.0
    assert result[0].insufficient == 3.0
    assert result[0].score_kind == "logits"


def test_predict_batch_chunks_work_and_preserves_order():
    batches = [
        [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]],
        [[3.0, 30.0, 300.0], [4.0, 40.0, 400.0]],
        [[5.0, 50.0, 500.0]],
    ]
    backend, tokenizer, model, torch_module = make_backend(batches, batch_size=2)
    pairs = [(f"premise-{i}", f"claim-{i}") for i in range(5)]

    result = backend.predict_batch(pairs)

    assert [scores.supported for scores in result] == [100, 200, 300, 400, 500]
    assert len(tokenizer.calls) == 3
    assert len(model.calls) == 3
    assert torch_module.entries == 3
    assert torch_module.exits == 3


def test_tokenizer_receives_premise_and_hypothesis_as_paired_batches():
    backend, tokenizer, _, _ = make_backend(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]],
        max_length=123,
    )

    backend.predict_batch([("evidence-a", "claim-a"), ("evidence-b", "claim-b")])

    assert tokenizer.calls == [
        {
            "premises": ["evidence-a", "evidence-b"],
            "hypotheses": ["claim-a", "claim-b"],
            "padding": True,
            "truncation": True,
            "max_length": 123,
            "return_tensors": "pt",
        }
    ]


def test_constructor_places_model_and_prediction_tensors_on_explicit_device():
    tokenizer = FakeTokenizer()
    backend, _, model, _ = make_backend(
        [[[0.0, 0.0, 1.0]]], tokenizer=tokenizer, device="cuda:7"
    )

    backend.predict_batch([("evidence", "claim")])

    assert model.to_calls == ["cuda:7"]
    assert model.eval_calls == 1
    assert all(
        tensor.devices == ["cuda:7"] for tensor in model.calls[0].values()
    )


def test_empty_batch_never_loads_runtime_or_calls_tokenizer_or_model(monkeypatch):
    tokenizer = FakeTokenizer()
    model = FakeModel([])
    backend = TransformersNliBackend(
        tokenizer=tokenizer,
        model=model,
        label_mapping=MAPPING,
        torch_module=None,
    )
    monkeypatch.setattr(
        backend_module,
        "_load_torch",
        lambda: pytest.fail("optional runtime must not load for an empty batch"),
    )

    assert backend.predict_batch([]) == []
    assert tokenizer.calls == []
    assert model.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entailment_index": True, "contradiction_index": 0, "neutral_index": 1},
        {"entailment_index": -1, "contradiction_index": 0, "neutral_index": 1},
        {"entailment_index": 0, "contradiction_index": 0, "neutral_index": 1},
    ],
)
def test_label_mapping_rejects_invalid_or_duplicate_indexes(kwargs):
    with pytest.raises(ConfigurationError):
        NliLabelMapping(**kwargs)


def test_constructor_rejects_mapping_that_does_not_cover_three_label_model():
    mapping = NliLabelMapping(0, 1, 3)
    with pytest.raises(ConfigurationError, match="0, 1, and 2"):
        make_backend([], mapping=mapping)


@pytest.mark.parametrize("num_labels", [2, 4, True, None])
def test_constructor_requires_exactly_three_model_labels(num_labels):
    model = FakeModel([], num_labels=num_labels)
    with pytest.raises(ConfigurationError):
        make_backend([], model=model)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("batch_size", 1.5),
        ("max_length", 0),
        ("max_length", False),
        ("max_length", 1.5),
        ("device", ""),
        ("device", "   "),
        ("device", None),
    ],
)
def test_constructor_rejects_invalid_runtime_configuration(field, value):
    kwargs = {field: value}
    with pytest.raises(ConfigurationError):
        make_backend([], **kwargs)


def test_constructor_rejects_non_callable_provider_objects():
    with pytest.raises(ConfigurationError, match="tokenizer"):
        TransformersNliBackend(
            tokenizer=object(),
            model=FakeModel([]),
            label_mapping=MAPPING,
            torch_module=FakeTorch(),
        )
    with pytest.raises(ConfigurationError, match="model"):
        TransformersNliBackend(
            tokenizer=FakeTokenizer(),
            model=object(),
            label_mapping=MAPPING,
            torch_module=FakeTorch(),
        )


@pytest.mark.parametrize(
    "pairs",
    [
        (("evidence", "claim"),),
        [["evidence", "claim"]],
        [("evidence",)],
        [("", "claim")],
        [("evidence", "")],
        [(1, "claim")],
    ],
)
def test_predict_batch_validates_direct_inputs(pairs):
    backend, _, _, _ = make_backend([])
    with pytest.raises(DomainValidationError):
        backend.predict_batch(pairs)


@pytest.mark.parametrize(
    "tokenizer_output",
    [
        [],
        {},
        {1: FakeDeviceValue()},
        {"input_ids": object()},
    ],
)
def test_tokenizer_output_contract_is_checked(tokenizer_output):
    tokenizer = FakeTokenizer(output=tokenizer_output)
    backend, _, _, _ = make_backend([], tokenizer=tokenizer)

    with pytest.raises(NliBackendContractError):
        backend.predict_batch([("evidence", "claim")])


def test_runtime_must_provide_inference_mode():
    backend, _, _, _ = make_backend([], torch_module=object())
    with pytest.raises(NliBackendContractError, match="inference_mode"):
        backend.predict_batch([("evidence", "claim")])


@pytest.mark.parametrize(
    "output_factory",
    [
        lambda: object(),
        lambda: SimpleNamespace(logits=object()),
        lambda: SimpleNamespace(logits=FakeTensor("not-a-list")),
        lambda: SimpleNamespace(logits=FakeTensor([])),
        lambda: SimpleNamespace(logits=FakeTensor([[1.0, 2.0]])),
        lambda: SimpleNamespace(logits=FakeTensor([[1.0, 2.0, 3.0, 4.0]])),
    ],
)
def test_model_output_shape_contract_is_checked(output_factory):
    model = FakeModel([], output_factory=output_factory)
    backend, _, _, _ = make_backend([], model=model)

    with pytest.raises(NliBackendContractError):
        backend.predict_batch([("evidence", "claim")])


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, True, "1.0"])
def test_non_numeric_or_non_finite_model_logits_are_backend_contract_errors(bad_value):
    model = FakeModel([[[bad_value, 0.0, 1.0]]])
    backend, _, _, _ = make_backend([], model=model)

    with pytest.raises(NliBackendContractError, match="finite"):
        backend.predict_batch([("evidence", "claim")])


class FakeAutoLoader:
    def __init__(self, value) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, model_name_or_path, **kwargs):
        self.calls.append((model_name_or_path, kwargs))
        return self.value


def test_from_pretrained_is_offline_and_remote_code_disabled_by_default(monkeypatch):
    torch_module = FakeTorch()
    tokenizer = FakeTokenizer()
    model = FakeModel([])
    tokenizer_loader = FakeAutoLoader(tokenizer)
    model_loader = FakeAutoLoader(model)
    monkeypatch.setattr(
        backend_module,
        "_load_optional_dependencies",
        lambda: (torch_module, tokenizer_loader, model_loader),
    )

    result = TransformersNliBackend.from_pretrained(
        "org/model",
        label_mapping=MAPPING,
        revision="immutable-commit",
    )

    assert isinstance(result, TransformersNliBackend)
    expected = (
        "org/model",
        {
            "local_files_only": True,
            "trust_remote_code": False,
            "revision": "immutable-commit",
        },
    )
    assert tokenizer_loader.calls == [expected]
    assert model_loader.calls == [expected]


def test_from_pretrained_network_and_remote_code_require_explicit_opt_in(monkeypatch):
    tokenizer_loader = FakeAutoLoader(FakeTokenizer())
    model_loader = FakeAutoLoader(FakeModel([]))
    monkeypatch.setattr(
        backend_module,
        "_load_optional_dependencies",
        lambda: (FakeTorch(), tokenizer_loader, model_loader),
    )

    TransformersNliBackend.from_pretrained(
        "org/model",
        label_mapping=MAPPING,
        local_files_only=False,
        trust_remote_code=True,
    )

    assert tokenizer_loader.calls[0][1] == {
        "local_files_only": False,
        "trust_remote_code": True,
    }
    assert model_loader.calls[0][1] == tokenizer_loader.calls[0][1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name_or_path": ""},
        {"model_name_or_path": "model", "revision": ""},
        {"model_name_or_path": "model", "local_files_only": 1},
        {"model_name_or_path": "model", "trust_remote_code": 0},
    ],
)
def test_from_pretrained_validates_loading_policy_before_runtime_import(
    monkeypatch, kwargs
):
    monkeypatch.setattr(
        backend_module,
        "_load_optional_dependencies",
        lambda: pytest.fail("runtime must not load before policy validation"),
    )
    with pytest.raises(ConfigurationError):
        TransformersNliBackend.from_pretrained(label_mapping=MAPPING, **kwargs)


@pytest.mark.parametrize(
    "invalid_settings",
    [
        {"label_mapping": NliLabelMapping(0, 1, 3)},
        {"batch_size": 0},
        {"max_length": True},
        {"device": ""},
    ],
)
def test_from_pretrained_validates_runtime_settings_before_loading(
    monkeypatch, invalid_settings
):
    monkeypatch.setattr(
        backend_module,
        "_load_optional_dependencies",
        lambda: pytest.fail("checkpoint must not load before settings validation"),
    )
    kwargs = {"label_mapping": MAPPING, **invalid_settings}
    with pytest.raises(ConfigurationError):
        TransformersNliBackend.from_pretrained("model", **kwargs)


def test_injected_provider_only_lazy_loads_torch_not_transformers_loader(monkeypatch):
    runtime = FakeTorch()
    monkeypatch.setattr(backend_module, "_load_torch", lambda: runtime)
    monkeypatch.setattr(
        backend_module,
        "_load_optional_dependencies",
        lambda: pytest.fail("injected model must not import the Transformers loader"),
    )
    backend = TransformersNliBackend(
        tokenizer=FakeTokenizer(),
        model=FakeModel([[[0.0, 0.0, 1.0]]]),
        label_mapping=MAPPING,
        torch_module=None,
    )

    result = backend.predict_batch([("evidence", "claim")])

    assert result[0].supported == 1.0
    assert runtime.entries == 1


def test_optional_dependency_error_has_actionable_install_hint(monkeypatch):
    error = ConfigurationError("install groundguard-rag[nli-transformers]")
    monkeypatch.setattr(
        backend_module,
        "_load_optional_dependencies",
        lambda: (_ for _ in ()).throw(error),
    )

    with pytest.raises(ConfigurationError, match="nli-transformers"):
        TransformersNliBackend.from_pretrained("model", label_mapping=MAPPING)


def test_backend_integrates_with_nli_verifier_without_creating_confidence():
    # Model column order is contradiction, neutral, entailment.  The explicit
    # mapping makes the edge supported; the raw model softmax remains an edge
    # probability and never becomes a calibrated claim confidence.
    backend, _, _, _ = make_backend([[[0.0, -2.0, 8.0]]])
    verifier = NliVerifier(
        backend,
        NliVerifierConfig(
            verifier_id="transformers-nli",
            verifier_revision="fixture-revision",
            decision_threshold=0.8,
        ),
    )
    claim = AtomicClaim("claim-0-5", "claim", 0, 5)
    evidence = [
        EvidenceCandidate(
            reference=EvidenceReference("chunk-1"),
            text="supporting evidence",
        )
    ]

    verdict = verifier.verify(claim, evidence)

    assert verdict.state is VerificationState.SUPPORTED
    assert verdict.raw_score is None
    assert verdict.calibrated_confidence is None
    assert verdict.evidence_assessments[0].label_scores.score_kind == "probabilities"
