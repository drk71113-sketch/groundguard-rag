"""Stage-0 domain data models: the contracts, not the logic.

Every model here is a frozen (immutable) dataclass that validates itself in
``__post_init__``. Immutability is a deliberate enforcement of the
``verify`` side-effect boundary in core_requirements.md #4: a verifier
cannot accidentally mutate a ``Chunk`` or ``VerificationRequest`` it was
handed, because the attribute assignment itself is disallowed by Python at
runtime (``dataclasses.FrozenInstanceError``).

None of these classes contain claim-decomposition, retrieval, scoring,
calibration, or repair-loop logic -- they only describe the shape of the
data those later stages will produce and consume, plus the structural
invariants (non-empty ids, valid spans, score ranges) that make the shape
trustworthy regardless of who constructs it.

Stage 0.1 additions: ``EvidenceCandidate`` (reference + actual text for
``Verifier``/``Rewriter``), ``EvidenceAssessment`` (one claim-evidence audit
edge), ``RepairActionRecord`` (structured repair-history entry).

Stage 0.2 additions/tightening (see project_meta/shared_config for the
acceptance review that requested them):

* ``LabelScores`` -- the full three-way raw NLI-style score
  (entailment/contradiction/neutral), replacing ``EvidenceAssessment``'s
  old single-scalar ``raw_label_score``.
* ``ClaimVerdict`` now enforces that its claim-level ``state`` is actually
  consistent with the states of its ``evidence_assessments`` edges (see
  ``_check_claim_verdict_edge_consistency``), and rejects duplicate edges.
* ``RunMetrics`` -- minimal structured performance/cost record.
* ``AuditReport`` gained ``run_mode`` and ``metrics``; ``RepairActionRecord``
  gained structured (not free-text) fields for evidence/answer/score deltas.
* Numerous type/immutability tightenings: bool rejected where an int is
  expected, ``AuditReport.schema_version`` must equal ``SCHEMA_VERSION``
  exactly, container membership type checks, ``Chunk.metadata`` restricted
  to JSON-compatible types and recursively frozen with rejected non-string
  keys.

Stage 0.3 additions/tightening (see core_requirements.md #8):

* ``AuditReport`` enforces a ``run_mode`` <-> repair-field invariant
  (VERIFY implies no repair activity at all; HEAL always carries a
  non-empty ``stop_reason``), and directly rejects a duplicate
  ``claim_id`` across ``verdicts`` or a ``repair_actions`` entry
  referencing a ``claim_id`` absent from ``verdicts``.
* ``RepairActionRecord.score_before``/``score_after`` were renamed to
  ``calibrated_confidence_before``/``calibrated_confidence_after`` and
  bounded to [0, 1], preventing verifier-specific raw scores from being
  used as a cross-round scale.
* ``Chunk.metadata`` rejects non-finite (NaN/Infinity) float values.
* ``LabelScores`` with ``score_kind == "probabilities"`` additionally
  requires the three values to sum to 1 within floating-point tolerance.

Stage 7 additions/tightening (see core_requirements.md #15):

* ``RepairDecision`` provides a typed action and pre-call estimated cost.
* ``RepairActionRecord`` records commit/rollback and claim lineage; its HEAL
  confidence fields carry separately calibrated groundedness progress, not
  stage-6 verdict-correctness confidence.
* ``AuditReport`` preserves initial and final verdict graphs plus the heal
  progress calibrator identity. The audit JSON version is therefore 1.1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping

from groundguard_rag.domain.enums import RepairAction, RunMode, VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError

#: Version of the audit report JSON shape produced by AuditReport.to_dict().
#: Bump this whenever the shape changes in a way that isn't purely
#: additive-and-optional, and keep
#: src/groundguard_rag/schema/audit_report.v1.schema.json's declared
#: version ("const") and AuditReport.schema_version (checked for *exact*
#: equality, not just "looks like a version") in sync.
SCHEMA_VERSION = "1.1.0"

#: Verdict states a single claim-evidence edge can hold. NOT_CHECKABLE is
#: excluded because it is a whole-claim classification made *before*
#: evidence is even sought (there is no edge to assess). CONFLICTING_EVIDENCE
#: is excluded because it only emerges from *comparing* multiple edges --
#: a single edge, in isolation, cannot be internally conflicting.
_EDGE_LEVEL_STATES = frozenset(
    {
        VerificationState.SUPPORTED,
        VerificationState.CONTRADICTED,
        VerificationState.INSUFFICIENT_EVIDENCE,
    }
)

#: JSON-compatible scalar types allowed inside Chunk.metadata (see
#: _freeze_json_compatible). Deliberately excludes set/frozenset, bytes,
#: and arbitrary objects -- none of those survive a JSON round-trip, and
#: allowing them here would let non-reproducible or silently-lossy data
#: into an otherwise "frozen" structure.
_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))

#: Float-equality tolerance for LabelScores' "probabilities must sum to 1"
#: check. Floating point addition of three independently-computed values
#: is not exact, so this must be a tolerance, not an exact ==.
_PROBABILITY_SUM_TOLERANCE = 1e-6


# --------------------------------------------------------------------------
# Shared validation helpers
# --------------------------------------------------------------------------


def _require_nonempty_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError(f"{field_name} must be a non-empty string")


def _require_str_or_none(value: Any, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise DomainValidationError(f"{field_name} must be a string or None")


def _require_int(value: Any, field_name: str) -> None:
    # bool is a subclass of int in Python, so isinstance(True, int) is True
    # and True/False would otherwise silently pass as 1/0 for an offset or
    # round number -- reject that explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise DomainValidationError(
            f"{field_name} must be an int, not {type(value).__name__}"
        )


def _require_int_or_none(value: Any, field_name: str) -> None:
    if value is not None:
        _require_int(value, field_name)


def _require_nonneg_int_or_none(value: Any, field_name: str) -> None:
    _require_int_or_none(value, field_name)
    if value is not None and value < 0:
        raise DomainValidationError(f"{field_name} must be >= 0")


def _require_finite(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainValidationError(
            f"{field_name} must be a real number, not {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise DomainValidationError(f"{field_name} must be finite (no NaN/Infinity)")


def _require_finite_or_none(value: float | None, field_name: str) -> None:
    # Every float that can end up in the audit JSON must be finite. Plain
    # JSON has no NaN/Infinity literals, but Python's json module parses the
    # non-standard "NaN"/"Infinity"/"-Infinity" tokens by default, so a
    # value can arrive as a real (non-finite) float without ever failing a
    # JSON Schema "type: number" check downstream. Reject it at the domain
    # boundary instead of relying on schema validation to catch it.
    if value is not None:
        _require_finite(value, field_name)


def _require_nonneg_finite_or_none(value: float | None, field_name: str) -> None:
    _require_finite_or_none(value, field_name)
    if value is not None and value < 0:
        raise DomainValidationError(f"{field_name} must be >= 0")


def _require_unit_interval_or_none(value: float | None, field_name: str) -> None:
    # For fields that are specifically a *calibrated confidence* or a
    # confidence-scale improvement threshold -- must be finite and, unlike
    # ClaimVerdict.raw_score/LabelScores' raw values, bounded to [0, 1].
    _require_finite_or_none(value, field_name)
    if value is not None and not (0.0 <= value <= 1.0):
        raise DomainValidationError(f"{field_name} must be within [0, 1]")


def _require_tz_aware_iso8601(value: str, field_name: str) -> None:
    # datetime.fromisoformat only grew "Z" support in 3.11; normalize it
    # ourselves so stage 0 doesn't silently depend on interpreter version.
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise DomainValidationError(
            f"{field_name} must be an ISO-8601 timestamp, got {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise DomainValidationError(
            f"{field_name} must include a UTC offset/timezone, got {value!r}"
        )


def _freeze_json_compatible(value: Any, field_name: str) -> Any:
    """Recursively validate ``value`` is JSON-compatible and freeze it.

    dict -> MappingProxyType (keys must be str); list/tuple -> tuple;
    str/int/float/bool/None pass through unchanged; anything else (set,
    bytes, a custom mutable object, ...) is rejected. A frozen dataclass
    only stops *reassigning* a field -- a mutable value stored inside it
    (a dict, a set, a dict nested inside a list) is still mutable in place
    unless every container is frozen, and unless disallowed container
    types are rejected outright rather than silently let through.
    """
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DomainValidationError(
                    f"{field_name} keys must be strings "
                    f"(got {key!r} of type {type(key).__name__})"
                )
            frozen[key] = _freeze_json_compatible(item, f"{field_name}[{key!r}]")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_compatible(item, f"{field_name}[...]") for item in value
        )
    if isinstance(value, _JSON_SCALAR_TYPES):
        # bool is not a float subclass in Python, so this only catches a
        # genuinely non-finite float value (NaN/Infinity/-Infinity), not
        # bool/int/str/None.
        if isinstance(value, float) and not math.isfinite(value):
            raise DomainValidationError(
                f"{field_name} must be finite (no NaN/Infinity) -- "
                f"got {value!r}"
            )
        return value
    raise DomainValidationError(
        f"{field_name} contains a non-JSON-compatible value of type "
        f"{type(value).__name__} (only str/int/float/bool/None/dict/list "
        "are allowed)"
    )


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One retrieved evidence chunk, exactly as the host RAG produced it.

    ``verify`` must treat these as read-only inputs (see core_requirements
    #4), so ``metadata`` is deep-frozen and type-checked (see
    ``_freeze_json_compatible``) rather than left as a plain ``dict`` with
    only its top level protected.
    """

    chunk_id: str
    text: str
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_str(self.chunk_id, "Chunk.chunk_id")
        if not self.text:
            raise DomainValidationError("Chunk.text must be non-empty")
        if self.source is not None:
            _require_nonempty_str(self.source, "Chunk.source")
        if not isinstance(self.metadata, Mapping):
            raise DomainValidationError(
                f"Chunk.metadata must be a mapping, not {type(self.metadata).__name__}"
            )
        object.__setattr__(
            self, "metadata", _freeze_json_compatible(self.metadata, "Chunk.metadata")
        )


@dataclass(frozen=True)
class AtomicClaim:
    """One atomic unit of an answer that a ``ClaimDecomposer`` extracted for
    checking.

    Despite the name, ``AtomicClaim`` does not assert that its text is a
    checkable factual proposition -- whether it turns out to be one is
    exactly what verification decides (see ``VerificationState.NOT_CHECKABLE``
    on the eventual ``ClaimVerdict``). A ``ClaimDecomposer`` must therefore
    decompose the *whole* answer into these units, including opinions,
    greetings, and instructions, rather than pre-filtering for
    checkability itself -- pre-filtering would require the same
    classification judgement a ``Verifier`` is responsible for, just done
    silently and without an auditable verdict. ``start_char``/``end_char``
    locate the unit within the *answer* string it came from (character
    offsets, half-open interval), which is what makes claim-evidence
    traceability possible later.
    """

    claim_id: str
    text: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        _require_nonempty_str(self.claim_id, "AtomicClaim.claim_id")
        if not self.text.strip():
            raise DomainValidationError("AtomicClaim.text must be non-empty")
        _require_int(self.start_char, "AtomicClaim.start_char")
        _require_int(self.end_char, "AtomicClaim.end_char")
        if self.start_char < 0:
            raise DomainValidationError("AtomicClaim.start_char must be >= 0")
        if self.end_char <= self.start_char:
            raise DomainValidationError(
                "AtomicClaim.end_char must be greater than start_char"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass(frozen=True)
class EvidenceReference:
    """A pointer from a claim to the evidence chunk (and optional span) that
    an ``EvidenceSelector`` chose for it.

    ``relevance_score`` is the *selector's* retrieval/relevance score. It is
    intentionally a separate concept from a ``ClaimVerdict``'s aggregate
    scores and an ``EvidenceAssessment``'s ``label_scores`` -- conflating
    "how relevant is this evidence" with "how much does this evidence
    support the claim" is an easy mistake and the two must not be merged
    into one field.

    This is a *pointer only* (chunk id + optional span + score) -- it does
    not carry evidence text. See ``EvidenceCandidate`` for the runtime type
    that pairs a reference with actual text for ``Verifier``/``Rewriter``.
    """

    chunk_id: str
    start_char: int | None = None
    end_char: int | None = None
    relevance_score: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.chunk_id, "EvidenceReference.chunk_id")
        span_present = (self.start_char is not None, self.end_char is not None)
        if span_present[0] != span_present[1]:
            raise DomainValidationError(
                "EvidenceReference.start_char and end_char must both be set "
                "(a chunk-level span) or both be None (whole-chunk reference)"
            )
        if self.start_char is not None and self.end_char is not None:
            _require_int(self.start_char, "EvidenceReference.start_char")
            _require_int(self.end_char, "EvidenceReference.end_char")
            if self.start_char < 0:
                raise DomainValidationError(
                    "EvidenceReference.start_char must be >= 0"
                )
            if self.end_char <= self.start_char:
                raise DomainValidationError(
                    "EvidenceReference.end_char must be greater than start_char"
                )
        _require_finite_or_none(
            self.relevance_score, "EvidenceReference.relevance_score"
        )
        if self.relevance_score is not None and not (0.0 <= self.relevance_score <= 1.0):
            raise DomainValidationError(
                "EvidenceReference.relevance_score must be within [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "relevance_score": self.relevance_score,
        }


@dataclass(frozen=True)
class EvidenceCandidate:
    """Runtime input to ``Verifier``/``Rewriter``: a reference plus its actual text.

    Stage 0's original ``Verifier.verify(claim, evidence: list[EvidenceReference])``
    signature gave an adapter only chunk ids and offsets -- to actually judge
    support it would have had to resolve those ids against *something*, and
    the only way to do that without this type is an implicit global chunk
    store, which core_requirements #4/#5 explicitly forbid ("不隐藏外部...
    不通过全局客户端绕过显式 adapter"). ``EvidenceCandidate`` closes that gap:
    callers that already hold the matching ``Chunk`` (from the same
    ``VerificationRequest.chunks`` they were given) can build one with
    ``EvidenceCandidate.from_chunk`` and pass real text explicitly.
    """

    reference: EvidenceReference
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference, EvidenceReference):
            raise DomainValidationError(
                "EvidenceCandidate.reference must be an EvidenceReference"
            )
        if not self.text:
            raise DomainValidationError("EvidenceCandidate.text must be non-empty")
        start, end = self.reference.start_char, self.reference.end_char
        if start is not None and end is not None and len(self.text) != end - start:
            raise DomainValidationError(
                "EvidenceCandidate.text length must match the "
                "reference's span length (end_char - start_char)"
            )

    @classmethod
    def from_chunk(cls, reference: EvidenceReference, chunk: Chunk) -> "EvidenceCandidate":
        """Build a candidate by slicing ``chunk.text`` according to ``reference``.

        Requires the caller to already hold the matching ``Chunk`` in hand
        (e.g. from the same request's ``chunks``) -- there is deliberately
        no id-based lookup performed here.
        """
        if reference.chunk_id != chunk.chunk_id:
            raise DomainValidationError(
                "EvidenceCandidate.from_chunk: reference.chunk_id "
                f"({reference.chunk_id!r}) does not match chunk.chunk_id "
                f"({chunk.chunk_id!r})"
            )
        if reference.start_char is not None and reference.end_char is not None:
            text = chunk.text[reference.start_char : reference.end_char]
        else:
            text = chunk.text
        return cls(reference=reference, text=text)


@dataclass(frozen=True)
class LabelScores:
    """The full three-way raw NLI-style score for one claim-evidence edge.

    A DeBERTa-MNLI-style verifier emits entailment/contradiction/neutral
    scores together; collapsing that to a single "winning label" scalar (as
    stage 0.1's ``EvidenceAssessment.raw_label_score`` did) throws away
    exactly the information a real ``Calibrator`` needs (e.g. temperature
    scaling over the full logit vector, not just the argmax value). This
    class fixes the shape of that full score; nothing here interprets or
    aggregates it.

    ``score_kind`` names the scale the three values are on (e.g. "logits",
    "probabilities") -- deliberately a free-form non-empty string rather
    than a closed enum, since stage 0 should not pre-commit to the finite
    set of scales every future ``Verifier`` might emit. When it is exactly
    ``"probabilities"``, each value is additionally checked to be within
    [0, 1] (a probability outside that range is a contradiction in terms,
    independent of any calibration judgement), and the three values must
    sum to 1 within floating-point tolerance -- three independent
    "probabilities" that don't sum to 1 are not actually a probability
    distribution over the three edge outcomes, regardless of what a
    Verifier intended.
    """

    supported: float
    contradicted: float
    insufficient: float
    score_kind: str

    def __post_init__(self) -> None:
        _require_finite(self.supported, "LabelScores.supported")
        _require_finite(self.contradicted, "LabelScores.contradicted")
        _require_finite(self.insufficient, "LabelScores.insufficient")
        _require_nonempty_str(self.score_kind, "LabelScores.score_kind")
        if self.score_kind == "probabilities":
            for name, value in (
                ("supported", self.supported),
                ("contradicted", self.contradicted),
                ("insufficient", self.insufficient),
            ):
                if not (0.0 <= value <= 1.0):
                    raise DomainValidationError(
                        f"LabelScores.{name} must be within [0, 1] when "
                        "score_kind == 'probabilities'"
                    )
            total = self.supported + self.contradicted + self.insufficient
            if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
                raise DomainValidationError(
                    "LabelScores.supported + contradicted + insufficient must "
                    f"sum to 1 (within {_PROBABILITY_SUM_TOLERANCE}) when "
                    f"score_kind == 'probabilities' (got sum={total!r})"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "contradicted": self.contradicted,
            "insufficient": self.insufficient,
            "score_kind": self.score_kind,
        }


@dataclass(frozen=True)
class EvidenceAssessment:
    """One claim-evidence audit edge: what a specific ``Verifier`` found when
    checking one claim against one piece of evidence.

    Stage 0's original ``ClaimVerdict`` stored only a flat list of
    ``EvidenceReference`` pointers alongside one claim-level aggregate
    state, which discarded *why* each individual piece of evidence was or
    was not supportive -- there was no way to reconstruct or audit the
    claim's evidence graph, only its final verdict. This class is that
    missing edge, and ``label_scores`` (rather than a single scalar) is the
    full raw score behind its ``state``. Only the structure is defined
    here; no aggregation algorithm (turning a set of these into a
    claim-level state) is implemented -- that is later-stage scope.
    """

    reference: EvidenceReference
    label_scores: LabelScores
    state: VerificationState
    rationale: str | None
    verifier_id: str
    verifier_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference, EvidenceReference):
            raise DomainValidationError(
                "EvidenceAssessment.reference must be an EvidenceReference"
            )
        if not isinstance(self.label_scores, LabelScores):
            raise DomainValidationError(
                "EvidenceAssessment.label_scores must be a LabelScores instance"
            )
        if not isinstance(self.state, VerificationState):
            raise DomainValidationError(
                "EvidenceAssessment.state must be a VerificationState member"
            )
        if self.state not in _EDGE_LEVEL_STATES:
            raise DomainValidationError(
                "EvidenceAssessment.state must be one of "
                "SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE -- "
                "NOT_CHECKABLE is a whole-claim classification made before "
                "evidence is sought, and CONFLICTING_EVIDENCE only emerges "
                "from comparing multiple edges; neither applies to a "
                "single edge in isolation"
            )
        _require_str_or_none(self.rationale, "EvidenceAssessment.rationale")
        _require_nonempty_str(self.verifier_id, "EvidenceAssessment.verifier_id")
        _require_nonempty_str(
            self.verifier_revision, "EvidenceAssessment.verifier_revision"
        )

    def edge_key(self) -> tuple[str, int | None, int | None, str, str]:
        """The suggested uniqueness key for this edge within one ClaimVerdict:
        chunk_id + start_char + end_char + verifier_id + verifier_revision.
        """
        return (
            self.reference.chunk_id,
            self.reference.start_char,
            self.reference.end_char,
            self.verifier_id,
            self.verifier_revision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.to_dict(),
            "label_scores": self.label_scores.to_dict(),
            "state": self.state.value,
            "rationale": self.rationale,
            "verifier_id": self.verifier_id,
            "verifier_revision": self.verifier_revision,
        }


def _check_claim_verdict_edge_consistency(
    state: VerificationState, edge_states: frozenset[VerificationState]
) -> None:
    """Structural consistency rules between a ClaimVerdict's aggregate
    ``state`` and the states of its ``evidence_assessments`` edges.

    These are audit-integrity rules ("does the declared summary match the
    detail it claims to summarize"), not an aggregation *algorithm* -- they
    never compute ``state`` from ``edge_states``, only reject a summary
    that is inconsistent with the detail it is paired with.
    """
    if state is VerificationState.NOT_CHECKABLE:
        if edge_states:
            raise DomainValidationError(
                "NOT_CHECKABLE requires evidence_assessments to be empty"
            )
    elif state is VerificationState.INSUFFICIENT_EVIDENCE:
        if edge_states - {VerificationState.INSUFFICIENT_EVIDENCE}:
            raise DomainValidationError(
                "INSUFFICIENT_EVIDENCE requires no evidence_assessments, or "
                "evidence_assessments that are all themselves "
                "INSUFFICIENT_EVIDENCE"
            )
    elif state is VerificationState.SUPPORTED:
        if VerificationState.SUPPORTED not in edge_states:
            raise DomainValidationError(
                "SUPPORTED requires at least one SUPPORTED evidence_assessments entry"
            )
        if VerificationState.CONTRADICTED in edge_states:
            raise DomainValidationError(
                "SUPPORTED must not coexist with a CONTRADICTED "
                "evidence_assessments entry (that combination is CONFLICTING_EVIDENCE)"
            )
    elif state is VerificationState.CONTRADICTED:
        if VerificationState.CONTRADICTED not in edge_states:
            raise DomainValidationError(
                "CONTRADICTED requires at least one CONTRADICTED evidence_assessments entry"
            )
        if VerificationState.SUPPORTED in edge_states:
            raise DomainValidationError(
                "CONTRADICTED must not coexist with a SUPPORTED "
                "evidence_assessments entry (that combination is CONFLICTING_EVIDENCE)"
            )
    elif state is VerificationState.CONFLICTING_EVIDENCE:
        if (
            VerificationState.SUPPORTED not in edge_states
            or VerificationState.CONTRADICTED not in edge_states
        ):
            raise DomainValidationError(
                "CONFLICTING_EVIDENCE requires at least one SUPPORTED and at "
                "least one CONTRADICTED evidence_assessments entry"
            )


@dataclass(frozen=True)
class ClaimVerdict:
    """The verification outcome for a single ``AtomicClaim``.

    ``evidence_assessments`` holds the per-edge results (see
    ``EvidenceAssessment``); ``state`` remains the claim-level *aggregated*
    outcome, and is checked for consistency against the edges' own states
    (see ``_check_claim_verdict_edge_consistency``) -- this rejects, for
    example, a claim declared SUPPORTED whose only edge is CONTRADICTED, or
    a CONFLICTING_EVIDENCE claim with no actual disagreement between edges.
    Duplicate edges (same chunk_id/span/verifier_id/verifier_revision) are
    rejected outright.

    ``raw_score``/``calibrated_confidence`` here are claim-level aggregate
    scores (distinct from each edge's own ``label_scores``). ``raw_score``
    is deliberately left unbounded: it is whatever scale the plugged-in
    ``Verifier``/aggregation happens to emit. Presenting it to a user as if
    it were a calibrated confidence is exactly the mistake core_requirements
    #6 prohibits, so the two stay distinct, separately optional fields.
    """

    claim: AtomicClaim
    state: VerificationState
    evidence_assessments: tuple[EvidenceAssessment, ...]
    raw_score: float | None = None
    calibrated_confidence: float | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.claim, AtomicClaim):
            raise DomainValidationError("ClaimVerdict.claim must be an AtomicClaim")
        if not isinstance(self.state, VerificationState):
            raise DomainValidationError(
                "ClaimVerdict.state must be a VerificationState member"
            )
        assessments = tuple(self.evidence_assessments)
        for assessment in assessments:
            if not isinstance(assessment, EvidenceAssessment):
                raise DomainValidationError(
                    "ClaimVerdict.evidence_assessments must contain only "
                    "EvidenceAssessment instances"
                )
        object.__setattr__(self, "evidence_assessments", assessments)

        seen_keys: set[tuple[str, int | None, int | None, str, str]] = set()
        for assessment in assessments:
            key = assessment.edge_key()
            if key in seen_keys:
                raise DomainValidationError(
                    "ClaimVerdict.evidence_assessments contains a duplicate "
                    f"edge (chunk_id/span/verifier_id/verifier_revision) {key!r}"
                )
            seen_keys.add(key)

        _require_finite_or_none(self.raw_score, "ClaimVerdict.raw_score")
        _require_finite_or_none(
            self.calibrated_confidence, "ClaimVerdict.calibrated_confidence"
        )
        if self.calibrated_confidence is not None and not (
            0.0 <= self.calibrated_confidence <= 1.0
        ):
            raise DomainValidationError(
                "ClaimVerdict.calibrated_confidence must be within [0, 1]"
            )
        _require_str_or_none(self.rationale, "ClaimVerdict.rationale")

        edge_states = frozenset(assessment.state for assessment in assessments)
        _check_claim_verdict_edge_consistency(self.state, edge_states)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "state": self.state.value,
            "evidence_assessments": [
                assessment.to_dict() for assessment in self.evidence_assessments
            ],
            "raw_score": self.raw_score,
            "calibrated_confidence": self.calibrated_confidence,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class VerificationRequest:
    """The input to ``verify``: an answer plus the chunks it should be
    checked against.

    ``query`` is optional and, per core_requirements #4, must be ignored by
    ``verify`` itself -- it exists only so the *same* request object can
    later be handed to ``heal`` mode components (a ``Retriever``) that do
    need it. ``chunks`` may legitimately be empty: that is a valid input
    that later stages should resolve to INSUFFICIENT_EVIDENCE for every
    claim, not a construction error.
    """

    request_id: str
    answer: str
    chunks: tuple[Chunk, ...]
    query: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.request_id, "VerificationRequest.request_id")
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise DomainValidationError(
                "VerificationRequest.answer must be a non-empty, non-whitespace string"
            )
        chunks = tuple(self.chunks)
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise DomainValidationError(
                    "VerificationRequest.chunks must contain only Chunk instances"
                )
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise DomainValidationError(
                "VerificationRequest.chunks must not contain duplicate chunk_id values"
            )
        object.__setattr__(self, "chunks", chunks)
        if self.query is not None and not isinstance(self.query, str):
            raise DomainValidationError(
                "VerificationRequest.query must be a string or None"
            )


@dataclass(frozen=True)
class HealPolicyState:
    """Stable, read-only snapshot passed to repair and stop policies.

    Stage 0 deliberately used an untyped ``Mapping[str, Any]`` while the heal
    loop did not exist. Stage 7 freezes the actual plugin protocol as a domain
    model so a misspelled key or changed value type fails visibly instead of
    silently altering policy behavior.
    """

    round: int
    completed_rounds: int
    target_claim_id: str
    attempts_for_claim: int
    attempts_by_claim: Mapping[str, int]
    elapsed_seconds: float
    estimated_cost_spent: float
    estimated_cost_remaining: float
    state_hash: str
    request_has_query: bool
    retriever_available: bool
    rewriter_available: bool
    verdict_states: tuple[VerificationState, ...]
    abstained_claim_ids: tuple[str, ...]
    accepted_claim_ids: tuple[str, ...]
    max_rounds: int
    max_attempts_per_claim: int
    timeout_seconds: float
    cost_budget: float
    min_improvement: float

    def __post_init__(self) -> None:
        for field_name in (
            "round",
            "completed_rounds",
            "attempts_for_claim",
            "max_rounds",
            "max_attempts_per_claim",
        ):
            value = getattr(self, field_name)
            _require_int(value, f"HealPolicyState.{field_name}")
            minimum = 1 if field_name in {"round", "max_rounds", "max_attempts_per_claim"} else 0
            if value < minimum:
                raise DomainValidationError(
                    f"HealPolicyState.{field_name} must be >= {minimum}"
                )
        if self.round != self.completed_rounds + 1:
            raise DomainValidationError(
                "HealPolicyState.round must equal completed_rounds + 1"
            )
        _require_nonempty_str(
            self.target_claim_id, "HealPolicyState.target_claim_id"
        )
        if not isinstance(self.attempts_by_claim, Mapping):
            raise DomainValidationError(
                "HealPolicyState.attempts_by_claim must be a mapping"
            )
        attempts: dict[str, int] = {}
        for claim_id, count in self.attempts_by_claim.items():
            _require_nonempty_str(
                claim_id, "HealPolicyState.attempts_by_claim key"
            )
            _require_int(count, f"HealPolicyState.attempts_by_claim[{claim_id!r}]")
            if count < 0:
                raise DomainValidationError(
                    "HealPolicyState attempt counts must be >= 0"
                )
            attempts[claim_id] = count
        if attempts.get(self.target_claim_id) != self.attempts_for_claim:
            raise DomainValidationError(
                "HealPolicyState.attempts_for_claim must match attempts_by_claim"
            )
        if self.attempts_for_claim >= self.max_attempts_per_claim:
            raise DomainValidationError(
                "HealPolicyState target must still be below max_attempts_per_claim"
            )
        object.__setattr__(self, "attempts_by_claim", MappingProxyType(attempts))

        for field_name in (
            "elapsed_seconds",
            "estimated_cost_spent",
            "estimated_cost_remaining",
            "timeout_seconds",
            "cost_budget",
            "min_improvement",
        ):
            value = getattr(self, field_name)
            _require_nonneg_finite_or_none(value, f"HealPolicyState.{field_name}")
            if value is None:
                raise DomainValidationError(
                    f"HealPolicyState.{field_name} must be a number"
                )
        if self.timeout_seconds <= 0 or self.cost_budget <= 0:
            raise DomainValidationError(
                "HealPolicyState timeout_seconds and cost_budget must be > 0"
            )
        if self.completed_rounds >= self.max_rounds:
            raise DomainValidationError(
                "HealPolicyState completed_rounds must be below max_rounds"
            )
        if self.estimated_cost_spent > self.cost_budget:
            raise DomainValidationError(
                "HealPolicyState estimated_cost_spent exceeds cost_budget"
            )
        if not math.isclose(
            self.estimated_cost_spent + self.estimated_cost_remaining,
            self.cost_budget,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise DomainValidationError(
                "HealPolicyState cost spent + remaining must equal cost_budget"
            )
        if not 0.0 <= self.min_improvement <= 1.0:
            raise DomainValidationError(
                "HealPolicyState.min_improvement must be within [0, 1]"
            )
        _require_nonempty_str(self.state_hash, "HealPolicyState.state_hash")
        for field_name in (
            "request_has_query",
            "retriever_available",
            "rewriter_available",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise DomainValidationError(
                    f"HealPolicyState.{field_name} must be a bool"
                )

        states = tuple(self.verdict_states)
        if any(not isinstance(state, VerificationState) for state in states):
            raise DomainValidationError(
                "HealPolicyState.verdict_states must contain VerificationState members"
            )
        if not states:
            raise DomainValidationError(
                "HealPolicyState.verdict_states must not be empty"
            )
        object.__setattr__(self, "verdict_states", states)
        terminal_sets: list[set[str]] = []
        for field_name in ("abstained_claim_ids", "accepted_claim_ids"):
            values = tuple(getattr(self, field_name))
            for value in values:
                _require_nonempty_str(value, f"HealPolicyState.{field_name} item")
            if len(values) != len(set(values)):
                raise DomainValidationError(
                    f"HealPolicyState.{field_name} must not contain duplicates"
                )
            object.__setattr__(self, field_name, values)
            terminal_sets.append(set(values))
        if terminal_sets[0] & terminal_sets[1]:
            raise DomainValidationError(
                "HealPolicyState claim IDs cannot be both abstained and accepted"
            )


@dataclass(frozen=True)
class RepairDecision:
    """A policy's explicit authorization for one bounded repair round.

    ``estimated_cost`` is a caller/plugin supplied quote for the complete
    round (including any external action it authorizes). ``HealService``
    checks it before invoking a Retriever/Rewriter and records it as an
    estimate, never as an actual provider bill. ``retrieval_query`` is only
    legal for RETRIEVE and never bypasses the separate requirement that the
    original ``VerificationRequest`` contain a non-empty query.
    """

    action: RepairAction
    estimated_cost: float
    retrieval_query: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, RepairAction):
            raise DomainValidationError(
                "RepairDecision.action must be a RepairAction member"
            )
        _require_nonneg_finite_or_none(
            self.estimated_cost, "RepairDecision.estimated_cost"
        )
        # The field is not optional even though the shared helper accepts
        # None; keeping this explicit produces a clearer contract error.
        if self.estimated_cost is None:
            raise DomainValidationError(
                "RepairDecision.estimated_cost must be a non-negative finite number"
            )
        _require_str_or_none(
            self.retrieval_query, "RepairDecision.retrieval_query"
        )
        if self.retrieval_query is not None and not self.retrieval_query.strip():
            raise DomainValidationError(
                "RepairDecision.retrieval_query must be non-empty when provided"
            )
        if (
            self.action is not RepairAction.RETRIEVE
            and self.retrieval_query is not None
        ):
            raise DomainValidationError(
                "RepairDecision.retrieval_query is only valid for RETRIEVE"
            )
        if self.action in {
            RepairAction.DELETE,
            RepairAction.ABSTAIN,
            RepairAction.ACCEPT,
        } and self.estimated_cost != 0:
            raise DomainValidationError(
                f"RepairDecision.{self.action.value} must have estimated_cost == 0"
            )
        _require_str_or_none(self.note, "RepairDecision.note")


@dataclass(frozen=True)
class RepairActionRecord:
    """One structured entry in an AuditReport's repair history.

    Stage 0.1 fixed the minimal round/claim/action/state-transition shape;
    stage 0.2 adds structured (not free-text) fields for the evidence,
    answer, score, timing, cost, and stop-reason deltas core_requirements
    #4's "自愈动作" record needs, so the stage-7 loop writes queryable fields
    instead of encoding essential detail into unvalidated ``note`` prose.

    ``round`` is 1-indexed. Whether a given record's ``round`` is
    consistent with its owning ``AuditReport.repair_rounds`` is checked by
    ``AuditReport`` itself (this class does not know the total round
    count of a report it is not yet attached to). Likewise, whether
    ``claim_id`` refers to a claim that actually appears in the owning
    report is a cross-object check performed by
    ``groundguard_rag.schema.semantic_validation``, not here.

    In a HEAL report, ``calibrated_confidence_before``/``_after`` carry the
    separately calibrated P(claim grounded/supported) supplied by
    ``HealProgressEvaluator``. They are not raw NLI logits/softmax and are
    not ``ClaimVerdict.calibrated_confidence`` (the probability that the
    current five-state verdict is correct). The report records the progress
    calibrator ID/revision so consumers can tell these meanings apart.
    """

    round: int
    claim_id: str
    action: RepairAction
    state_before: VerificationState
    state_after: VerificationState | None
    claim_id_after: str | None = None
    claim_text_before: str | None = None
    claim_text_after: str | None = None
    committed: bool = True
    note: str | None = None
    evidence_ids_added: tuple[str, ...] = ()
    evidence_ids_removed: tuple[str, ...] = ()
    answer_hash_before: str | None = None
    answer_hash_after: str | None = None
    calibrated_confidence_before: float | None = None
    calibrated_confidence_after: float | None = None
    elapsed_ms: float | None = None
    estimated_cost: float | None = None
    failure_or_stop_reason: str | None = None

    def __post_init__(self) -> None:
        _require_int(self.round, "RepairActionRecord.round")
        if self.round < 1:
            raise DomainValidationError("RepairActionRecord.round must be >= 1")
        _require_nonempty_str(self.claim_id, "RepairActionRecord.claim_id")
        if not isinstance(self.action, RepairAction):
            raise DomainValidationError(
                "RepairActionRecord.action must be a RepairAction member"
            )
        if not isinstance(self.state_before, VerificationState):
            raise DomainValidationError(
                "RepairActionRecord.state_before must be a VerificationState member"
            )
        if self.state_after is not None and not isinstance(
            self.state_after, VerificationState
        ):
            raise DomainValidationError(
                "RepairActionRecord.state_after must be a VerificationState member or None"
            )
        if (
            self.state_after is None
            and self.committed
            and self.action is not RepairAction.DELETE
        ):
            raise DomainValidationError(
                "a committed non-DELETE requires a verified state_after"
            )
        _require_str_or_none(
            self.claim_id_after, "RepairActionRecord.claim_id_after"
        )
        if self.claim_id_after is not None and not self.claim_id_after.strip():
            raise DomainValidationError(
                "RepairActionRecord.claim_id_after must be non-empty when provided"
            )
        if self.action is RepairAction.DELETE and self.committed:
            if self.claim_id_after is not None:
                raise DomainValidationError(
                    "a committed DELETE requires claim_id_after to be None"
                )
        elif self.claim_id_after is None:
            raise DomainValidationError(
                "a non-deleting or uncommitted action requires claim_id_after"
            )
        for field_name, value in (
            ("claim_text_before", self.claim_text_before),
            ("claim_text_after", self.claim_text_after),
        ):
            _require_str_or_none(value, f"RepairActionRecord.{field_name}")
            if value is not None and not value.strip():
                raise DomainValidationError(
                    f"RepairActionRecord.{field_name} must be non-empty when provided"
                )
        if self.action is RepairAction.DELETE and self.committed:
            if self.claim_text_after is not None:
                raise DomainValidationError(
                    "a committed DELETE requires claim_text_after to be None"
                )
        elif self.claim_text_after is None:
            raise DomainValidationError(
                "a non-deleting or uncommitted action requires claim_text_after"
            )
        if not isinstance(self.committed, bool):
            raise DomainValidationError(
                "RepairActionRecord.committed must be a bool"
            )
        _require_str_or_none(self.note, "RepairActionRecord.note")

        added = tuple(self.evidence_ids_added)
        for item in added:
            _require_nonempty_str(item, "RepairActionRecord.evidence_ids_added item")
        object.__setattr__(self, "evidence_ids_added", added)

        removed = tuple(self.evidence_ids_removed)
        for item in removed:
            _require_nonempty_str(item, "RepairActionRecord.evidence_ids_removed item")
        object.__setattr__(self, "evidence_ids_removed", removed)

        _require_str_or_none(
            self.answer_hash_before, "RepairActionRecord.answer_hash_before"
        )
        _require_str_or_none(
            self.answer_hash_after, "RepairActionRecord.answer_hash_after"
        )
        _require_unit_interval_or_none(
            self.calibrated_confidence_before,
            "RepairActionRecord.calibrated_confidence_before",
        )
        _require_unit_interval_or_none(
            self.calibrated_confidence_after,
            "RepairActionRecord.calibrated_confidence_after",
        )
        _require_nonneg_finite_or_none(
            self.elapsed_ms, "RepairActionRecord.elapsed_ms"
        )
        _require_nonneg_finite_or_none(
            self.estimated_cost, "RepairActionRecord.estimated_cost"
        )
        _require_str_or_none(
            self.failure_or_stop_reason, "RepairActionRecord.failure_or_stop_reason"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "claim_id": self.claim_id,
            "action": self.action.value,
            "state_before": self.state_before.value,
            "state_after": self.state_after.value if self.state_after is not None else None,
            "claim_id_after": self.claim_id_after,
            "claim_text_before": self.claim_text_before,
            "claim_text_after": self.claim_text_after,
            "committed": self.committed,
            "note": self.note,
            "evidence_ids_added": list(self.evidence_ids_added),
            "evidence_ids_removed": list(self.evidence_ids_removed),
            "answer_hash_before": self.answer_hash_before,
            "answer_hash_after": self.answer_hash_after,
            "calibrated_confidence_before": self.calibrated_confidence_before,
            "calibrated_confidence_after": self.calibrated_confidence_after,
            "elapsed_ms": self.elapsed_ms,
            "estimated_cost": self.estimated_cost,
            "failure_or_stop_reason": self.failure_or_stop_reason,
        }


@dataclass(frozen=True)
class RunMetrics:
    """Minimal structured performance/cost record for an ``AuditReport``.

    core_requirements #4 requires the audit JSON to carry "性能信息"
    (performance information); stage 0 through 0.1 left that unrepresented.
    This only fixes the shape -- nothing in stage 0.2 collects these
    values. Every field may be ``None`` (not yet measured), but a value
    that *is* provided must be non-negative and finite.
    """

    total_latency_ms: float | None = None
    stage_latencies_ms: Mapping[str, float] | None = None
    model_call_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost: float | None = None

    def __post_init__(self) -> None:
        _require_nonneg_finite_or_none(
            self.total_latency_ms, "RunMetrics.total_latency_ms"
        )
        if self.stage_latencies_ms is not None:
            if not isinstance(self.stage_latencies_ms, Mapping):
                raise DomainValidationError(
                    "RunMetrics.stage_latencies_ms must be a mapping or None"
                )
            frozen: dict[str, float] = {}
            for stage_name, latency in self.stage_latencies_ms.items():
                _require_nonempty_str(
                    stage_name, "RunMetrics.stage_latencies_ms key"
                )
                _require_nonneg_finite_or_none(
                    latency, f"RunMetrics.stage_latencies_ms[{stage_name!r}]"
                )
                frozen[stage_name] = latency
            object.__setattr__(self, "stage_latencies_ms", MappingProxyType(frozen))
        _require_nonneg_int_or_none(
            self.model_call_count, "RunMetrics.model_call_count"
        )
        _require_nonneg_int_or_none(self.input_tokens, "RunMetrics.input_tokens")
        _require_nonneg_int_or_none(self.output_tokens, "RunMetrics.output_tokens")
        _require_nonneg_finite_or_none(
            self.estimated_cost, "RunMetrics.estimated_cost"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_latency_ms": self.total_latency_ms,
            "stage_latencies_ms": (
                dict(self.stage_latencies_ms)
                if self.stage_latencies_ms is not None
                else None
            ),
            "model_call_count": self.model_call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": self.estimated_cost,
        }


@dataclass(frozen=True)
class AuditReport:
    """Versioned, machine-readable record of a verify or bounded-heal run.

    ``run_mode`` records whether this was a verify-only or heal run (see
    ``RunMode``). ``repair_rounds``/``repair_actions``/``stop_reason``
    describe self-heal activity per core_requirements #4; a verify-only
    report leaves them at their defaults (0 rounds, no actions, no stop
    reason). HEAL additionally stores ``initial_verdicts`` beside final
    ``verdicts`` and identifies the separate groundedness-progress
    calibrator used for the improvement guard. ``metrics`` is a
    ``RunMetrics`` (defaulting to all-``None`` when not collected).

    Stage 0.3 locks a cross-field invariant between ``run_mode`` and the
    repair-related fields (core_requirements #8.1): ``VERIFY`` means no
    repair activity happened at all (``repair_rounds == 0``,
    ``repair_actions == ()``, ``stop_reason is None``) -- a verify-only run
    never touches a repair loop, so it cannot have a stop reason or repair
    history. ``HEAL`` always has a ``stop_reason`` (a non-empty string):
    every heal run, even one that repaired nothing because every claim
    already passed, terminates for some determinate reason
    (core_requirements #2's "确定的停止原因"), and ``repair_rounds == 0`` is
    explicitly still allowed under ``HEAL`` for exactly that "nothing to
    repair" case.

    Construction rejects duplicate claim IDs, incomplete calibration
    metadata, out-of-range rounds, and action lineage that cannot resolve to
    the initial/final verdict graphs. ``semantic_validation`` mirrors these
    checks for foreign JSON documents that bypass this constructor.
    """

    schema_version: str
    request_id: str
    input_hash: str
    model_revision: str
    threshold_version: str
    created_at: str
    run_mode: RunMode
    verdicts: tuple[ClaimVerdict, ...]
    initial_verdicts: tuple[ClaimVerdict, ...] = ()
    calibrator_id: str | None = None
    calibrator_revision: str | None = None
    heal_progress_calibrator_id: str | None = None
    heal_progress_calibrator_revision: str | None = None
    repair_rounds: int = 0
    repair_actions: tuple[RepairActionRecord, ...] = ()
    stop_reason: str | None = None
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise DomainValidationError(
                f"AuditReport.schema_version must equal {SCHEMA_VERSION!r} "
                f"(got {self.schema_version!r})"
            )
        _require_nonempty_str(self.request_id, "AuditReport.request_id")
        _require_nonempty_str(self.input_hash, "AuditReport.input_hash")
        _require_nonempty_str(self.model_revision, "AuditReport.model_revision")
        _require_nonempty_str(self.threshold_version, "AuditReport.threshold_version")
        _require_tz_aware_iso8601(self.created_at, "AuditReport.created_at")
        if not isinstance(self.run_mode, RunMode):
            raise DomainValidationError("AuditReport.run_mode must be a RunMode member")

        verdicts = tuple(self.verdicts)
        claim_ids: set[str] = set()
        for verdict in verdicts:
            if not isinstance(verdict, ClaimVerdict):
                raise DomainValidationError(
                    "AuditReport.verdicts must contain only ClaimVerdict instances"
                )
            claim_id = verdict.claim.claim_id
            if claim_id in claim_ids:
                raise DomainValidationError(
                    f"AuditReport.verdicts contains a duplicate claim_id {claim_id!r}"
                )
            claim_ids.add(claim_id)
        object.__setattr__(self, "verdicts", verdicts)

        initial_verdicts = tuple(self.initial_verdicts)
        initial_claim_ids: set[str] = set()
        for verdict in initial_verdicts:
            if not isinstance(verdict, ClaimVerdict):
                raise DomainValidationError(
                    "AuditReport.initial_verdicts must contain only ClaimVerdict instances"
                )
            claim_id = verdict.claim.claim_id
            if claim_id in initial_claim_ids:
                raise DomainValidationError(
                    "AuditReport.initial_verdicts contains a duplicate "
                    f"claim_id {claim_id!r}"
                )
            initial_claim_ids.add(claim_id)
        object.__setattr__(self, "initial_verdicts", initial_verdicts)

        for field_name, value in (
            ("calibrator_id", self.calibrator_id),
            ("calibrator_revision", self.calibrator_revision),
            ("heal_progress_calibrator_id", self.heal_progress_calibrator_id),
            (
                "heal_progress_calibrator_revision",
                self.heal_progress_calibrator_revision,
            ),
        ):
            if value is not None:
                _require_nonempty_str(value, f"AuditReport.{field_name}")
        if (self.calibrator_id is None) != (self.calibrator_revision is None):
            raise DomainValidationError(
                "AuditReport.calibrator_id and calibrator_revision must both be set or both be None"
            )
        if (self.heal_progress_calibrator_id is None) != (
            self.heal_progress_calibrator_revision is None
        ):
            raise DomainValidationError(
                "AuditReport.heal_progress_calibrator_id and "
                "heal_progress_calibrator_revision must both be set or both be None"
            )
        if any(
            verdict.calibrated_confidence is not None
            for verdict in verdicts + initial_verdicts
        ) and self.calibrator_id is None:
            raise DomainValidationError(
                "AuditReport with calibrated confidence requires calibrator metadata"
            )

        _require_int(self.repair_rounds, "AuditReport.repair_rounds")
        if self.repair_rounds < 0:
            raise DomainValidationError("AuditReport.repair_rounds must be >= 0")

        actions = tuple(self.repair_actions)
        for action in actions:
            if not isinstance(action, RepairActionRecord):
                raise DomainValidationError(
                    "AuditReport.repair_actions must contain only RepairActionRecord instances"
                )
            if action.round > self.repair_rounds:
                raise DomainValidationError(
                    "AuditReport.repair_actions entries must have "
                    "round <= repair_rounds "
                    f"(got round={action.round}, repair_rounds={self.repair_rounds})"
                )
            if action.claim_id not in claim_ids | initial_claim_ids:
                raise DomainValidationError(
                    f"AuditReport.repair_actions entry references claim_id "
                    f"{action.claim_id!r}, which is not present in initial or final verdicts"
                )
            if (
                action.claim_id_after is not None
                and action.claim_id_after not in claim_ids | initial_claim_ids
            ):
                raise DomainValidationError(
                    "AuditReport.repair_actions entry claim_id_after must "
                    "reference an initial or final verdict"
                )
        object.__setattr__(self, "repair_actions", actions)

        if any(
            action.calibrated_confidence_before is not None
            or action.calibrated_confidence_after is not None
            for action in actions
        ) and self.heal_progress_calibrator_id is None:
            raise DomainValidationError(
                "AuditReport repair progress confidences require heal progress calibrator metadata"
            )

        _require_str_or_none(self.stop_reason, "AuditReport.stop_reason")

        if not isinstance(self.metrics, RunMetrics):
            raise DomainValidationError(
                "AuditReport.metrics must be a RunMetrics instance"
            )

        # RunMode <-> repair-field cross-field invariant (core_requirements
        # #8.1): checked last so the more basic per-field validation above
        # has already run and error messages stay specific.
        if self.run_mode is RunMode.VERIFY:
            if initial_verdicts:
                raise DomainValidationError(
                    "AuditReport.run_mode == VERIFY requires initial_verdicts to be empty"
                )
            if self.repair_rounds != 0:
                raise DomainValidationError(
                    "AuditReport.run_mode == VERIFY requires repair_rounds == 0"
                )
            if actions:
                raise DomainValidationError(
                    "AuditReport.run_mode == VERIFY requires repair_actions to be empty"
                )
            if self.stop_reason is not None:
                raise DomainValidationError(
                    "AuditReport.run_mode == VERIFY requires stop_reason to be None"
                )
            if self.heal_progress_calibrator_id is not None:
                raise DomainValidationError(
                    "AuditReport.run_mode == VERIFY requires heal progress calibrator metadata to be None"
                )
        elif self.run_mode is RunMode.HEAL:
            if not initial_verdicts:
                raise DomainValidationError(
                    "AuditReport.run_mode == HEAL requires non-empty initial_verdicts"
                )
            if self.stop_reason is None or not self.stop_reason.strip():
                raise DomainValidationError(
                    "AuditReport.run_mode == HEAL requires a non-empty stop_reason"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-compatible shape described by
        ``schema/audit_report.v1.schema.json``."""
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "input_hash": self.input_hash,
            "model_revision": self.model_revision,
            "threshold_version": self.threshold_version,
            "created_at": self.created_at,
            "run_mode": self.run_mode.value,
            "verdicts": [verdict.to_dict() for verdict in self.verdicts],
            "initial_verdicts": [
                verdict.to_dict() for verdict in self.initial_verdicts
            ],
            "calibrator_id": self.calibrator_id,
            "calibrator_revision": self.calibrator_revision,
            "heal_progress_calibrator_id": self.heal_progress_calibrator_id,
            "heal_progress_calibrator_revision": self.heal_progress_calibrator_revision,
            "repair_rounds": self.repair_rounds,
            "repair_actions": [action.to_dict() for action in self.repair_actions],
            "stop_reason": self.stop_reason,
            "metrics": self.metrics.to_dict(),
        }
