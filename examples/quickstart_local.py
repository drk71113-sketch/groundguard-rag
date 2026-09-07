"""Run the dependency-free integration demo from the repository root."""

from __future__ import annotations

import json

from groundguard_rag import Chunk, VerificationRequest

# Direct-script execution puts ``examples/`` (not the repository root) first
# on sys.path, so import the sibling module rather than the package path.
from demo_factory import build_groundguard


def main() -> None:
    result = build_groundguard().verify(
        VerificationRequest(
            request_id="demo-request",
            answer="Paris is in France.",
            chunks=(Chunk("demo-evidence", "Paris is in France."),),
        )
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
