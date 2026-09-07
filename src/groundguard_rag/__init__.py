"""Public API for pluggable RAG verification and bounded self-healing.

Importing this module is side-effect free: it does not import optional model,
framework, or MCP runtimes and never creates a provider client.
"""

from groundguard_rag.api import (
    GroundGuard,
    HealingOutput,
    VerificationOutput,
    heal,
    verify,
)
from groundguard_rag.domain.config import HealConfig, VerifyConfig
from groundguard_rag.domain.enums import (
    HealStopReason,
    RepairAction,
    RunMode,
    VerificationState,
)
from groundguard_rag.domain.models import Chunk, VerificationRequest
from groundguard_rag.presentation.views import AnswerViewRenderer, AnswerViews

__version__ = "0.1.0"

__all__ = [
    "AnswerViewRenderer",
    "AnswerViews",
    "Chunk",
    "GroundGuard",
    "HealConfig",
    "HealStopReason",
    "HealingOutput",
    "RepairAction",
    "RunMode",
    "VerificationOutput",
    "VerificationRequest",
    "VerificationState",
    "VerifyConfig",
    "heal",
    "verify",
]
