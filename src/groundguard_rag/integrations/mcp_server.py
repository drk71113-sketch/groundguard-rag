"""Optional MCP server exposing an explicitly assembled ``GroundGuard``.

The MCP SDK is imported only inside ``create_mcp_server``.  Importing the core
package therefore remains dependency-free and cannot start a server or load a
provider as a side effect.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from groundguard_rag.api import GroundGuard
from groundguard_rag.domain.exceptions import ConfigurationError
from groundguard_rag.integrations.serialization import request_from_json


def _default_server_factory(*, name: str, instructions: str) -> Any:
    try:
        # MCP Python SDK v2 name.
        from mcp.server import MCPServer

        server_class = MCPServer
    except ImportError:
        try:
            # MCP Python SDK v1 compatibility path.
            from mcp.server.fastmcp import FastMCP

            server_class = FastMCP
        except ImportError as exc:
            raise ConfigurationError(
                "MCP support is optional; install groundguard-rag[mcp]"
            ) from exc
    return server_class(name=name, instructions=instructions)


def create_mcp_server(
    guard: GroundGuard,
    *,
    server_factory: Callable[..., Any] | None = None,
) -> Any:
    """Create, but do not start, an MCP server with verify/heal tools."""

    if not isinstance(guard, GroundGuard):
        raise ConfigurationError("create_mcp_server.guard must be a GroundGuard")
    factory = server_factory or _default_server_factory
    if not callable(factory):
        raise ConfigurationError("server_factory must be callable")
    server = factory(
        name="GroundGuard-RAG",
        instructions=(
            "Verify RAG answers only against caller-supplied chunks. "
            "Healing is bounded and never writes back to the host."
        ),
    )
    tool = getattr(server, "tool", None)
    if not callable(tool):
        raise ConfigurationError("MCP server factory result must expose tool()")

    @server.tool()
    def groundguard_verify(
        request_id: str,
        answer: str,
        chunks: list[dict[str, Any]],
        query: str | None = None,
    ) -> dict[str, Any]:
        """Verify an answer against only the supplied chunks."""

        request = request_from_json(
            request_id=request_id, answer=answer, chunks=chunks, query=query
        )
        return guard.verify(request).to_dict()

    @server.tool()
    def groundguard_heal(
        request_id: str,
        answer: str,
        chunks: list[dict[str, Any]],
        query: str | None = None,
    ) -> dict[str, Any]:
        """Run explicitly configured bounded healing and return a candidate."""

        request = request_from_json(
            request_id=request_id, answer=answer, chunks=chunks, query=query
        )
        return guard.heal(request).to_dict()

    return server
