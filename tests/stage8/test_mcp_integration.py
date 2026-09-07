from __future__ import annotations

import asyncio

import pytest

from groundguard_rag import GroundGuard
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.integrations.mcp_server import create_mcp_server
from groundguard_rag.integrations.serialization import request_from_json
from tests.stage8._helpers import make_verify_service


class FakeMcpServer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tools = {}

    def tool(self):
        def decorator(function):
            self.tools[function.__name__] = function
            return function

        return decorator


def test_mcp_server_registers_and_executes_verify_without_sdk_import():
    server = create_mcp_server(
        GroundGuard(verify_service=make_verify_service()),
        server_factory=lambda **kwargs: FakeMcpServer(**kwargs),
    )

    assert set(server.tools) == {"groundguard_verify", "groundguard_heal"}
    result = server.tools["groundguard_verify"](
        request_id="mcp-1",
        answer="Paris is in France.",
        chunks=[{"chunk_id": "c1", "text": "Paris is in France."}],
    )
    assert result["audit_report"]["run_mode"] == "VERIFY"
    assert result["views"]["sidecar"]["answer"] == "Paris is in France."


def test_mcp_heal_fails_explicitly_when_not_configured():
    server = create_mcp_server(
        GroundGuard(verify_service=make_verify_service()),
        server_factory=lambda **kwargs: FakeMcpServer(**kwargs),
    )
    with pytest.raises(ConfigurationError):
        server.tools["groundguard_heal"]("r", "answer", [])


def test_json_boundary_rejects_unknown_fields_and_duplicate_chunk_ids():
    with pytest.raises(DomainValidationError):
        request_from_json(
            request_id="r",
            answer="answer",
            chunks=[{"chunk_id": "c", "text": "text", "secret": "x"}],
        )
    with pytest.raises(DomainValidationError):
        request_from_json(
            request_id="r",
            answer="answer",
            chunks=[
                {"chunk_id": "c", "text": "one"},
                {"chunk_id": "c", "text": "two"},
            ],
        )


def test_mcp_factory_contract_is_validated():
    with pytest.raises(ConfigurationError):
        create_mcp_server(object(), server_factory=FakeMcpServer)
    with pytest.raises(ConfigurationError):
        create_mcp_server(
            GroundGuard(verify_service=make_verify_service()),
            server_factory=lambda **kwargs: object(),
        )


def test_installed_official_mcp_sdk_accepts_the_tool_schemas():
    pytest.importorskip("mcp")
    server = create_mcp_server(GroundGuard(verify_service=make_verify_service()))
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "groundguard_verify",
        "groundguard_heal",
    }
    result = asyncio.run(
        server.call_tool(
            "groundguard_verify",
            {
                "request_id": "sdk-1",
                "answer": "Paris is in France.",
                "chunks": [
                    {"chunk_id": "sdk-chunk", "text": "Paris is in France."}
                ],
            },
        )
    )
    assert result.is_error is False
    assert result.structured_content["audit_report"]["run_mode"] == "VERIFY"
