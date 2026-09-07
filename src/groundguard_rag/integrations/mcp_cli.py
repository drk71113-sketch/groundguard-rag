"""Console launcher for an application-supplied GroundGuard factory."""

from __future__ import annotations

import argparse
from groundguard_rag.integrations.factory import load_guard
from groundguard_rag.integrations.mcp_server import create_mcp_server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a GroundGuard-RAG MCP server")
    parser.add_argument(
        "--factory",
        required=True,
        help="explicit application factory in module:attribute form",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
    )
    args = parser.parse_args(argv)
    server = create_mcp_server(load_guard(args.factory))
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
