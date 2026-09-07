import dataclasses

import pytest

from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import Chunk


def test_valid_chunk_constructs():
    chunk = Chunk(chunk_id="c1", text="hello world", source="doc.md")
    assert chunk.chunk_id == "c1"
    assert chunk.text == "hello world"
    assert chunk.source == "doc.md"
    assert dict(chunk.metadata) == {}


def test_chunk_metadata_defaults_to_empty_and_is_read_only():
    chunk = Chunk(chunk_id="c1", text="hi")
    with pytest.raises(TypeError):
        chunk.metadata["x"] = 1  # MappingProxyType must reject writes


def test_chunk_metadata_is_isolated_from_caller_dict():
    source_metadata = {"page": 1}
    chunk = Chunk(chunk_id="c1", text="hi", metadata=source_metadata)
    source_metadata["page"] = 2
    assert chunk.metadata["page"] == 1


def test_chunk_metadata_nested_dict_is_also_read_only():
    # A frozen dataclass only stops *reassigning* chunk.metadata; a plain
    # dict stored inside it would still be mutable in place unless nested
    # containers are frozen too.
    chunk = Chunk(chunk_id="c1", text="hi", metadata={"nested": {"page": 1}})
    assert isinstance(chunk.metadata["nested"], type(chunk.metadata))
    with pytest.raises(TypeError):
        chunk.metadata["nested"]["page"] = 2


def test_chunk_metadata_nested_list_becomes_an_immutable_tuple():
    chunk = Chunk(chunk_id="c1", text="hi", metadata={"tags": ["a", "b"]})
    assert chunk.metadata["tags"] == ("a", "b")
    assert isinstance(chunk.metadata["tags"], tuple)


def test_chunk_metadata_nested_dict_inside_list_is_frozen():
    chunk = Chunk(chunk_id="c1", text="hi", metadata={"items": [{"x": 1}]})
    inner = chunk.metadata["items"][0]
    assert isinstance(inner, type(chunk.metadata))
    with pytest.raises(TypeError):
        inner["x"] = 2


def test_chunk_metadata_nested_container_is_isolated_from_caller():
    nested = {"page": 1}
    source_metadata = {"nested": nested}
    chunk = Chunk(chunk_id="c1", text="hi", metadata=source_metadata)
    nested["page"] = 999
    assert chunk.metadata["nested"]["page"] == 1


def test_chunk_is_immutable():
    chunk = Chunk(chunk_id="c1", text="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "changed"


@pytest.mark.parametrize("chunk_id", ["", "   "])
def test_empty_chunk_id_rejected(chunk_id):
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id=chunk_id, text="hi")


def test_empty_text_rejected():
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="")


def test_source_none_is_allowed():
    chunk = Chunk(chunk_id="c1", text="hi", source=None)
    assert chunk.source is None


def test_source_empty_string_rejected():
    # source is "non-empty string or None" -- unlike rationale/note/
    # stop_reason (which allow empty string), an empty source is
    # meaningless as a provenance label.
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="hi", source="")


def test_source_non_string_rejected():
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="hi", source=123)


def test_metadata_must_be_a_mapping():
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="hi", metadata=["not", "a", "mapping"])


def test_metadata_rejects_set_values():
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="hi", metadata={"tags": {"a", "b"}})


def test_metadata_rejects_custom_mutable_object_values():
    class Thing:
        pass

    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="hi", metadata={"obj": Thing()})


def test_metadata_rejects_non_string_keys():
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="hi", metadata={123: "value"})


def test_metadata_rejects_non_string_keys_in_nested_dict():
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="hi", metadata={"nested": {123: "value"}})


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_metadata_rejects_non_finite_float_values(bad_value):
    # Stage 0.3: metadata must reject NaN/Infinity/-Infinity, not just
    # non-JSON-compatible container types.
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="hi", metadata={"score": bad_value})


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_metadata_rejects_non_finite_float_values_nested_in_list(bad_value):
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="hi", metadata={"scores": [1.0, bad_value]})


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_metadata_rejects_non_finite_float_values_nested_in_dict(bad_value):
    with pytest.raises(DomainValidationError):
        Chunk(chunk_id="c1", text="hi", metadata={"nested": {"score": bad_value}})


def test_metadata_allows_json_compatible_scalars():
    chunk = Chunk(
        chunk_id="c1",
        text="hi",
        metadata={"str": "x", "int": 1, "float": 1.5, "bool": True, "none": None},
    )
    assert chunk.metadata["str"] == "x"
    assert chunk.metadata["bool"] is True
    assert chunk.metadata["none"] is None
