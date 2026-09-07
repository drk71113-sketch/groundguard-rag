"""Ports (plugin interfaces) that adapters implement.

Every class here is an ``abc.ABC`` contract: instantiating one directly raises
``TypeError`` and a concrete adapter must implement every abstract member.
Stage 7 replaces the originally deferred free-form repair-policy mapping with
the stable ``HealPolicyState`` model and requires ``RepairDecision`` rather
than a bare action string.

Stage 0.1: ``Verifier.verify`` and ``Rewriter.rewrite`` take
``list[EvidenceCandidate]`` rather than ``list[EvidenceReference]``. A bare
``EvidenceReference`` is only a pointer (chunk id + optional span); an
adapter that needs to actually read the evidence would otherwise have no
way to do so except by resolving ``chunk_id`` against some out-of-band
store it was not explicitly given, which core_requirements #4/#5 forbid
("不隐藏外部...不通过全局客户端绕过显式 adapter"). ``EvidenceCandidate`` carries
the resolved text explicitly instead.

Stage 0.3 (core_requirements #8.2): ``Calibrator.calibrate`` takes the full
``ClaimVerdict`` (not a single ``EvidenceAssessment`` or a bare float) and
returns the claim-level calibrated confidence. ``Verifier.verify`` remains
responsible for producing a ``ClaimVerdict`` whose ``evidence_assessments``
already satisfy ``ClaimVerdict``'s own construction-time consistency rules
(see ``groundguard_rag.domain.models._check_claim_verdict_edge_consistency``)
-- there is no separate "populate evidence_assessments however you like"
allowance, since an inconsistent verdict cannot be constructed at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from groundguard_rag.domain.models import (
    AtomicClaim,
    AuditReport,
    Chunk,
    ClaimVerdict,
    EvidenceCandidate,
    EvidenceReference,
    HealPolicyState,
    RepairDecision,
)


class ClaimDecomposer(ABC):
    """Splits an answer string into atomic units for downstream checking.

    Despite the name, this must decompose the *whole* answer -- including
    opinions, greetings, and instructions -- into ``AtomicClaim`` units,
    rather than pre-filtering for only "checkable" content itself. Deciding
    whether a given unit is actually checkable is a ``Verifier``'s job
    (expressed as ``VerificationState.NOT_CHECKABLE`` on the resulting
    ``ClaimVerdict``), not a ``ClaimDecomposer``'s -- doing that filtering
    here would be the same classification judgement made silently and
    without an auditable verdict.
    """

    @abstractmethod
    def decompose(self, answer: str) -> list[AtomicClaim]:
        """Return the atomic units found in ``answer``.

        Implementations must return units whose ``start_char``/``end_char``
        are valid offsets into ``answer``, and must not omit non-checkable
        content -- see the class docstring.
        """
        raise NotImplementedError


class EvidenceSelector(ABC):
    """Chooses which chunks (and optionally spans) are evidence for a claim."""

    @abstractmethod
    def select(
        self, claim: AtomicClaim, chunks: list[Chunk]
    ) -> list[EvidenceReference]:
        """Return the evidence references relevant to ``claim`` among ``chunks``.

        Implementations must not mutate ``chunks`` and must only reference
        ``chunk_id`` values that are present in ``chunks``.
        """
        raise NotImplementedError


class Verifier(ABC):
    """Judges whether evidence supports, contradicts, or fails to settle a claim.

    Takes ``EvidenceCandidate`` (reference + actual text), not a bare
    ``EvidenceReference`` -- see the module docstring. Implementations must
    resolve evidence content solely from the ``evidence`` argument; they
    must not look up chunk content from a hidden global registry.
    """

    @abstractmethod
    def verify(
        self, claim: AtomicClaim, evidence: list[EvidenceCandidate]
    ) -> ClaimVerdict:
        """Return the verdict for ``claim`` given ``evidence``.

        The returned ``ClaimVerdict.calibrated_confidence`` may be ``None``
        -- calibration is a separate concern, see ``Calibrator``. The
        returned ``ClaimVerdict``'s ``evidence_assessments`` must satisfy
        ``ClaimVerdict``'s own construction-time consistency rules (its
        claim-level ``state`` must actually match the edges' states, edges
        must be unique, etc.) -- ``ClaimVerdict``'s constructor enforces
        this, so an implementation cannot return an inconsistent verdict.
        """
        raise NotImplementedError


class Calibrator(ABC):
    """Maps a claim verdict's raw label scores onto a claim-level calibrated
    confidence.

    Stage 0.3 (core_requirements #8.2): takes the full ``ClaimVerdict``
    rather than a single ``EvidenceAssessment`` or a bare ``float``. A real
    claim-level calibration needs to see every edge's ``label_scores``
    (not just one edge, and not just the winning label's scalar) plus each
    edge's ``verifier_id``/``verifier_revision`` to select the right
    calibration parameters -- ``ClaimVerdict.evidence_assessments`` is
    exactly that full context. Only the interface is fixed here; no
    calibration algorithm is implemented, and no ``VerdictAggregator`` port
    is introduced this stage -- whether aggregation should be a separate
    port is deferred until a concrete need for it is identified.
    """

    @abstractmethod
    def calibrate(self, verdict: ClaimVerdict) -> float:
        """Return the claim-level calibrated confidence for ``verdict``, in [0, 1].

        Implementations must derive the result from
        ``verdict.evidence_assessments`` (each entry's ``label_scores``,
        ``verifier_id``, ``verifier_revision``) -- not just
        ``verdict.state`` or ``verdict.raw_score`` -- and must not simply
        return a raw label score unchanged unless that identity mapping
        has itself been validated as calibrated. See core_requirements #6
        on not repackaging raw softmax output as confidence.
        """
        raise NotImplementedError

    # Stage 6: optional self-description lets the orchestrator detect wiring
    # mistakes without breaking generic/legacy Calibrator implementations.
    # Implementations backed by a fitted artifact should override all three.
    @property
    def calibrator_id(self) -> str | None:
        return None

    @property
    def calibrator_revision(self) -> str | None:
        return None

    @property
    def threshold_version(self) -> str | None:
        return None


class Retriever(ABC):
    """External evidence retrieval, used only by heal-mode components.

    ``verify`` must never hold a reference to a ``Retriever`` implicitly;
    callers inject one explicitly into heal mode only when they want it
    used, per core_requirements #4.
    """

    @abstractmethod
    def retrieve(self, query: str) -> list[Chunk]:
        """Return newly retrieved chunks for ``query``."""
        raise NotImplementedError


class Rewriter(ABC):
    """Produces a constrained rewrite of a claim given (possibly new) evidence.

    Takes ``EvidenceCandidate`` for the same reason ``Verifier.verify``
    does: a rewrite grounded in evidence needs the evidence's actual text,
    not just a chunk id pointer, and must not fetch it from a hidden
    global store.
    """

    @abstractmethod
    def rewrite(self, claim: AtomicClaim, evidence: list[EvidenceCandidate]) -> str:
        """Return replacement text for ``claim`` grounded in ``evidence``."""
        raise NotImplementedError


class HealProgressEvaluator(ABC):
    """Produces a calibrated probability that a claim is grounded/supported.

    This is deliberately separate from ``Calibrator``. The latter estimates
    whether the current five-state verdict is correct; a highly confident
    CONTRADICTED verdict is not progress toward a supported answer. A heal
    progress implementation therefore needs its own labeled validation data
    and traceable identity.
    """

    @abstractmethod
    def evaluate(self, verdict: ClaimVerdict) -> float:
        """Return calibrated ``P(claim grounded/supported)`` in ``[0, 1]``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def calibrator_id(self) -> str:
        """Stable identifier of the progress calibration method/artifact."""
        raise NotImplementedError

    @property
    @abstractmethod
    def calibrator_revision(self) -> str:
        """Immutable revision of the fitted progress calibration artifact."""
        raise NotImplementedError


class RepairPolicy(ABC):
    """Decides what repair action, if any, to take for a claim verdict.

    ``state`` is the immutable ``HealPolicyState`` protocol: round count,
    prior attempts, budget spent, adapter availability, verdict states and
    every hard limit have stable names and types.
    """

    @abstractmethod
    def decide(
        self, verdict: ClaimVerdict, state: HealPolicyState
    ) -> RepairDecision:
        """Return a structured action and pre-call estimated cost.

        A bare string is not accepted: without an explicit quote the heal
        orchestrator could not enforce its cost budget before an external
        Retriever/Rewriter call.
        """
        raise NotImplementedError


class StopPolicy(ABC):
    """Decides whether a self-heal run must stop.

    ``HealService`` enforces all hard round/time/cost/improvement/loop bounds
    itself. This plugin may only request an earlier stop; it cannot override
    or relax those built-in guards.
    """

    @abstractmethod
    def should_stop(self, state: HealPolicyState) -> bool:
        """Return True if the heal loop must stop given ``state``."""
        raise NotImplementedError


class AuditStore(ABC):
    """Persists and retrieves ``AuditReport`` instances.

    Neither ``verify`` nor ``heal`` writes to storage on its own (see
    core_requirements #4/#5) -- an ``AuditStore`` is only ever invoked by a
    caller-provided adapter, never reached for implicitly.
    """

    @abstractmethod
    def save(self, report: AuditReport) -> None:
        """Persist ``report``."""
        raise NotImplementedError

    @abstractmethod
    def load(self, request_id: str) -> AuditReport:
        """Return the previously saved report for ``request_id``."""
        raise NotImplementedError
