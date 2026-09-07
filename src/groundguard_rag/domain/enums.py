"""The five-way verification state that every claim verdict is expressed in.

This enum is the project's central contract: it must stay five-way and must
never be silently collapsed to a three-way (supported/contradicted/baseless)
classification inside the domain layer. A presentation layer is free to map
these onto a simpler display vocabulary (e.g. showing INSUFFICIENT_EVIDENCE
and CONFLICTING_EVIDENCE both as "unverified"), but that mapping is a view
concern, not a domain concern, and must not happen here.
"""

from __future__ import annotations

import enum


class VerificationState(enum.Enum):
    """Verdict of a single atomic claim relative to the *given* evidence chunks.

    All five states are judgements about groundedness in the input chunks,
    not about real-world truth. In particular INSUFFICIENT_EVIDENCE must
    never be read as "the claim is false" -- it only means the supplied
    chunks did not contain enough signal either way.
    """

    #: The supplied evidence supports the claim.
    SUPPORTED = "SUPPORTED"

    #: The supplied evidence explicitly conflicts with the claim.
    CONTRADICTED = "CONTRADICTED"

    #: The supplied evidence is insufficient to support or refute the claim.
    #: This is NOT a real-world falsity judgement -- see class docstring.
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

    #: Different pieces of supplied evidence disagree with each other about
    #: the claim (distinct from INSUFFICIENT_EVIDENCE, where evidence is
    #: merely absent or too weak rather than mutually contradictory).
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"

    #: Opinion, greeting, instruction, or other content not suited to
    #: evidence-based fact checking in the first place.
    NOT_CHECKABLE = "NOT_CHECKABLE"


class RunMode(enum.Enum):
    """Which mode produced an ``AuditReport``.

    Recorded on every report per core_requirements #4/#5 so an audit
    consumer can tell a verify-only run (no external calls, no repair
    activity) apart from a heal run (adapters injected, bounded repair
    loop) without having to infer it from whether repair_rounds is zero.
    """

    #: verify only: no Retriever/Rewriter adapters invoked, no repair activity.
    VERIFY = "VERIFY"

    #: heal: caller-injected adapters were available and bounded repair may
    #: have run (see AuditReport.repair_rounds/repair_actions/stop_reason).
    HEAL = "HEAL"


class RepairAction(enum.Enum):
    """The only actions a stage-7 repair policy may authorize.

    Values are lower-case because they are serialized into the audit JSON
    and are intended to remain stable plugin-protocol identifiers.
    """

    RETRIEVE = "retrieve"
    REWRITE = "rewrite"
    DELETE = "delete"
    ABSTAIN = "abstain"
    ACCEPT = "accept"


class HealStopReason(enum.Enum):
    """Deterministic terminal reasons emitted by ``HealService``."""

    ALL_GROUNDED = "all_grounded"
    ALL_CLAIMS_HANDLED = "all_claims_handled"
    MAX_ROUNDS = "max_rounds"
    MAX_ATTEMPTS_PER_CLAIM = "max_attempts_per_claim"
    TIMEOUT = "timeout"
    COST_BUDGET = "cost_budget"
    MIN_IMPROVEMENT = "min_improvement"
    STATE_LOOP = "state_loop"
    POLICY_STOP = "policy_stop"
    ACTION_UNAVAILABLE = "action_unavailable"
    INVALID_REPAIR_OUTPUT = "invalid_repair_output"
    ADAPTER_ERROR = "adapter_error"
    ADAPTER_CONTRACT_ERROR = "adapter_contract_error"
