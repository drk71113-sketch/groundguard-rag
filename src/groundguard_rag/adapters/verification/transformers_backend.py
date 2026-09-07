"""Optional Transformers sequence-classification backend for three-way NLI.

This adapter deliberately keeps provider loading outside the domain and
application layers.  Importing GroundGuard-RAG never imports torch or
transformers; those optional packages are loaded only by ``from_pretrained``
or immediately before inference for an explicitly injected model.

The backend emits raw logits in domain-semantic order.  It does not apply
softmax, thresholds, calibration, or claim aggregation: those responsibilities
remain in ``NliVerifier`` and future calibrator stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from groundguard_rag.adapters.verification.nli import (
    NliBackend,
    NliBackendContractError,
)
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import LabelScores


_OPTIONAL_EXTRA = "groundguard-rag[nli-transformers]"


@dataclass(frozen=True)
class NliLabelMapping:
    """Explicitly map a model's three output columns to NLI semantics.

    Hugging Face checkpoints do not share one universal numeric label order,
    and generic labels such as ``LABEL_0`` carry no safe semantic meaning.
    Requiring this mapping prevents a silently inverted entailment and
    contradiction decision.
    """

    entailment_index: int
    contradiction_index: int
    neutral_index: int

    def __post_init__(self) -> None:
        named_indexes = (
            ("entailment_index", self.entailment_index),
            ("contradiction_index", self.contradiction_index),
            ("neutral_index", self.neutral_index),
        )
        for name, index in named_indexes:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ConfigurationError(f"NliLabelMapping.{name} must be an integer")
            if index < 0:
                raise ConfigurationError(f"NliLabelMapping.{name} must be >= 0")
        if len({index for _, index in named_indexes}) != 3:
            raise ConfigurationError("NliLabelMapping indexes must be unique")


def _load_torch() -> Any:
    """Load only torch for inference with already-injected provider objects."""
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as exc:
        raise ConfigurationError(
            "TransformersNliBackend inference requires an optional dependency; install "
            f"{_OPTIONAL_EXTRA!r}"
        ) from exc
    return torch


def _load_optional_dependencies() -> tuple[Any, Any, Any]:
    """Load torch plus Transformers only for the checkpoint-loading path."""

    torch = _load_torch()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except (ImportError, ModuleNotFoundError) as exc:
        raise ConfigurationError(
            "TransformersNliBackend loading requires optional dependencies; install "
            f"{_OPTIONAL_EXTRA!r}"
        ) from exc
    return torch, AutoTokenizer, AutoModelForSequenceClassification


class TransformersNliBackend(NliBackend):
    """Run an injected three-label Transformers classifier in bounded batches.

    ``model`` and ``tokenizer`` are explicit dependencies.  ``from_pretrained``
    is the convenience loader; it is offline-only and remote-code-disabled by
    default.  The default device is CPU so hardware placement never changes
    implicitly.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        label_mapping: NliLabelMapping,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "cpu",
        torch_module: Any | None = None,
    ) -> None:
        self._validate_config(
            tokenizer=tokenizer,
            model=model,
            label_mapping=label_mapping,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        self._tokenizer = tokenizer
        self._label_mapping = label_mapping
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device
        self._torch_module = torch_module

        # The injected model is considered owned by this adapter.  Preparing it
        # for deterministic inference is explicit and never mutates request data.
        moved_model = model.to(device)
        self._model = model if moved_model is None else moved_model
        self._model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        label_mapping: NliLabelMapping,
        revision: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "cpu",
    ) -> "TransformersNliBackend":
        """Load an NLI checkpoint with explicit network and code-trust policy.

        Set ``local_files_only=False`` explicitly to permit Hugging Face Hub
        resolution.  ``trust_remote_code`` remains independently opt-in.
        Production deployments should also provide an immutable ``revision``
        when loading a remote checkpoint.
        """

        if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
            raise ConfigurationError("model_name_or_path must be a non-empty string")
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            raise ConfigurationError("revision must be a non-empty string or None")
        if not isinstance(local_files_only, bool):
            raise ConfigurationError("local_files_only must be a bool")
        if not isinstance(trust_remote_code, bool):
            raise ConfigurationError("trust_remote_code must be a bool")

        # Reject bad settings before importing a heavy runtime or resolving a
        # checkpoint.  This is especially important when network access was
        # explicitly enabled: invalid local configuration must never cause an
        # avoidable external request.
        cls._validate_runtime_settings(
            label_mapping=label_mapping,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )

        torch_module, auto_tokenizer, auto_model = _load_optional_dependencies()
        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if revision is not None:
            load_kwargs["revision"] = revision

        tokenizer = auto_tokenizer.from_pretrained(model_name_or_path, **load_kwargs)
        model = auto_model.from_pretrained(model_name_or_path, **load_kwargs)
        return cls(
            tokenizer=tokenizer,
            model=model,
            label_mapping=label_mapping,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            torch_module=torch_module,
        )

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[LabelScores]:
        self._validate_pairs(pairs)
        if not pairs:
            return []

        torch_module = self._torch_module
        if torch_module is None:
            torch_module = _load_torch()

        inference_mode = getattr(torch_module, "inference_mode", None)
        if not callable(inference_mode):
            raise NliBackendContractError(
                "torch runtime must provide callable inference_mode"
            )

        results: list[LabelScores] = []
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start : start + self._batch_size]
            premises = [premise for premise, _ in batch]
            hypotheses = [hypothesis for _, hypothesis in batch]
            encoded = self._tokenizer(
                premises,
                text_pair=hypotheses,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            model_inputs = self._move_inputs_to_device(encoded)
            with inference_mode():
                output = self._model(**model_inputs)
            rows = self._extract_logit_rows(output, expected_rows=len(batch))
            results.extend(self._map_rows(rows))
        return results

    @staticmethod
    def _validate_config(
        *,
        tokenizer: Any,
        model: Any,
        label_mapping: NliLabelMapping,
        batch_size: int,
        max_length: int,
        device: str,
    ) -> None:
        if not callable(tokenizer):
            raise ConfigurationError("tokenizer must be callable")
        if not callable(model):
            raise ConfigurationError("model must be callable")
        if not callable(getattr(model, "to", None)):
            raise ConfigurationError("model must provide callable to(device)")
        if not callable(getattr(model, "eval", None)):
            raise ConfigurationError("model must provide callable eval()")
        TransformersNliBackend._validate_runtime_settings(
            label_mapping=label_mapping,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        config = getattr(model, "config", None)
        num_labels = getattr(config, "num_labels", None)
        if isinstance(num_labels, bool) or not isinstance(num_labels, int):
            raise ConfigurationError("model.config.num_labels must be an integer")
        if num_labels != 3:
            raise ConfigurationError(
                "TransformersNliBackend requires a three-label NLI model"
            )

    @staticmethod
    def _validate_runtime_settings(
        *,
        label_mapping: NliLabelMapping,
        batch_size: int,
        max_length: int,
        device: str,
    ) -> None:
        if not isinstance(label_mapping, NliLabelMapping):
            raise ConfigurationError("label_mapping must be an NliLabelMapping")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ConfigurationError("batch_size must be a positive integer")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ConfigurationError("max_length must be a positive integer")
        if not isinstance(device, str) or not device.strip():
            raise ConfigurationError("device must be a non-empty string")

        indexes = {
            label_mapping.entailment_index,
            label_mapping.contradiction_index,
            label_mapping.neutral_index,
        }
        if indexes != {0, 1, 2}:
            raise ConfigurationError(
                "NliLabelMapping must map each of the model's indexes 0, 1, and 2 exactly once"
            )

    @staticmethod
    def _validate_pairs(pairs: list[tuple[str, str]]) -> None:
        if not isinstance(pairs, list):
            raise DomainValidationError(
                "TransformersNliBackend.pairs must be a list"
            )
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise DomainValidationError(
                    "each NLI pair must be a (premise, hypothesis) tuple"
                )
            premise, hypothesis = pair
            if not isinstance(premise, str) or not premise:
                raise DomainValidationError("NLI premise must be a non-empty string")
            if not isinstance(hypothesis, str) or not hypothesis:
                raise DomainValidationError("NLI hypothesis must be a non-empty string")

    def _move_inputs_to_device(self, encoded: Any) -> dict[str, Any]:
        if not isinstance(encoded, Mapping):
            raise NliBackendContractError("tokenizer output must be a mapping")
        moved: dict[str, Any] = {}
        for name, value in encoded.items():
            if not isinstance(name, str):
                raise NliBackendContractError("tokenizer output keys must be strings")
            move = getattr(value, "to", None)
            if not callable(move):
                raise NliBackendContractError(
                    f"tokenizer output {name!r} must provide callable to(device)"
                )
            moved[name] = move(self._device)
        if not moved:
            raise NliBackendContractError("tokenizer output must not be empty")
        return moved

    @staticmethod
    def _extract_logit_rows(output: Any, *, expected_rows: int) -> list[list[float]]:
        logits = getattr(output, "logits", None)
        try:
            rows = logits.detach().cpu().tolist()
        except (AttributeError, TypeError) as exc:
            raise NliBackendContractError(
                "model output.logits must support detach().cpu().tolist()"
            ) from exc
        if not isinstance(rows, list) or len(rows) != expected_rows:
            actual = len(rows) if isinstance(rows, list) else type(rows).__name__
            raise NliBackendContractError(
                "model logits must contain exactly one row per input pair "
                f"(expected {expected_rows}, got {actual})"
            )
        for row in rows:
            if not isinstance(row, list) or len(row) != 3:
                raise NliBackendContractError(
                    "each model logits row must contain exactly three values"
                )
        return rows

    def _map_rows(self, rows: list[list[float]]) -> list[LabelScores]:
        mapping = self._label_mapping
        mapped: list[LabelScores] = []
        for row in rows:
            try:
                mapped.append(
                    LabelScores(
                        supported=row[mapping.entailment_index],
                        contradicted=row[mapping.contradiction_index],
                        insufficient=row[mapping.neutral_index],
                        score_kind="logits",
                    )
                )
            except DomainValidationError as exc:
                raise NliBackendContractError(
                    "model logits must be finite, non-bool real numbers"
                ) from exc
        return mapped
