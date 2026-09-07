"""Cross-field / cross-object / numeric-finiteness checks JSON Schema cannot express.

JSON Schema (draft 2020-12) has no standard, portable way to compare two
sibling or cross-object property values against each other (e.g. "end_char
must be greater than start_char", "a nested repair_actions[].round must be
<= the top-level repair_rounds", "repair_actions[].claim_id must reference
a claim_id that actually appears in verdicts", "no two evidence_assessments
entries within one verdict share the same edge key", or "three sibling
LabelScores values must sum to 1"): those require either a custom keyword
or the non-standard ``$data`` reference extension, which the ``jsonschema``
package used in this project does not implement. Plain "type": "number"
also does not exclude NaN/Infinity, because Python's ``json`` module parses
the non-standard "NaN"/"Infinity"/"-Infinity" tokens into real (non-finite)
``float`` instances by default, which then pass a "number" type check.

The single-object ``run_mode`` <-> repair-field invariant (stage 0.3,
core_requirements #8.1) *is* expressible via plain if/then in the JSON
Schema (and is expressed there too, for defense in depth), but this module
re-checks it as well since it is also enforced directly by
``AuditReport``'s constructor and a foreign document should get the same
guarantee independent of whether it happens to also be schema-validated.

This module exists to catch these gaps for a raw audit-report *document*
(a plain dict) that did not necessarily go through this package's Python
dataclass constructors -- e.g. one produced by another language's client,
hand-crafted, or loaded from disk. It intentionally re-checks invariants
the dataclasses in ``groundguard_rag.domain.models`` already enforce at
construction time, because those constructors are not in the trust path
for a foreign document; ``AuditReport.to_dict()`` output is expected to
always pass these checks (see the round-trip tests).

Only structural shape checks belong here -- no aggregation algorithm, no
claim decomposition, no calibration. A document that fails
``jsonschema.validate`` against ``audit_report.v1.schema.json`` should be
rejected before it ever reaches this module; this module assumes the
document already has the right shape and only checks the relationships
plain JSON Schema cannot (or, for the run_mode invariant, checks again).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from groundguard_rag.domain.exceptions import DomainValidationError

#: Edge-level states (mirrors groundguard_rag.domain.models._EDGE_LEVEL_STATES,
#: duplicated here as plain strings since this module deliberately works on
#: raw dicts, not domain enum instances).
_SUPPORTED = "SUPPORTED"
_CONTRADICTED = "CONTRADICTED"
_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
_CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
_NOT_CHECKABLE = "NOT_CHECKABLE"

#: Float-equality tolerance for LabelScores' "probabilities must sum to 1"
#: check -- mirrors groundguard_rag.domain.models._PROBABILITY_SUM_TOLERANCE.
_PROBABILITY_SUM_TOLERANCE = 1e-6


def validate_audit_report_semantics(document: Mapping[str, Any]) -> None:
    """Raise ``DomainValidationError`` on the first schema-inexpressible (or,
    for the run_mode invariant, defense-in-depth) violation found in an
    audit-report ``document`` (as produced by ``AuditReport.to_dict()`` or
    an equivalent foreign JSON document).
    """
    _check_run_mode_consistency(document)
    _check_calibration_metadata(document)

    claim_ids_seen = _check_verdict_collection(
        document.get("verdicts", []), "verdicts"
    )
    initial_claim_ids_seen = _check_verdict_collection(
        document.get("initial_verdicts", []), "initial_verdicts"
    )

    repair_rounds = document.get("repair_rounds", 0)
    for action in document.get("repair_actions", []):
        round_number = action.get("round")
        if round_number is None or round_number > repair_rounds:
            raise DomainValidationError(
                "repair_actions entries must have round <= repair_rounds "
                f"(got round={round_number!r}, repair_rounds={repair_rounds!r})"
            )
        action_claim_id = action.get("claim_id")
        if action_claim_id not in claim_ids_seen | initial_claim_ids_seen:
            raise DomainValidationError(
                f"repair_action.claim_id {action_claim_id!r} does not reference "
                "any claim_id present in the initial or final verdicts"
            )
        claim_id_after = action.get("claim_id_after")
        if claim_id_after is not None and claim_id_after not in (
            claim_ids_seen | initial_claim_ids_seen
        ):
            raise DomainValidationError(
                f"repair_action.claim_id_after {claim_id_after!r} does not reference "
                "a claim_id present in initial or final verdicts"
            )
        _check_repair_action_shape(action)
        _check_finite(
            action.get("calibrated_confidence_before"),
            "repair_action.calibrated_confidence_before",
        )
        _check_finite(
            action.get("calibrated_confidence_after"),
            "repair_action.calibrated_confidence_after",
        )
        _check_finite(action.get("elapsed_ms"), "repair_action.elapsed_ms")
        _check_finite(action.get("estimated_cost"), "repair_action.estimated_cost")

    metrics = document.get("metrics") or {}
    _check_finite(metrics.get("total_latency_ms"), "metrics.total_latency_ms")
    _check_finite(metrics.get("estimated_cost"), "metrics.estimated_cost")
    for stage_name, latency in (metrics.get("stage_latencies_ms") or {}).items():
        _check_finite(latency, f"metrics.stage_latencies_ms[{stage_name!r}]")


def _check_verdict_collection(
    verdicts: Any, field_name: str
) -> set[str]:
    """Validate one initial/final verdict collection and return its IDs."""
    claim_ids_seen: set[str] = set()
    for verdict in verdicts:
        claim = verdict["claim"]
        claim_id = claim["claim_id"]
        if claim_id in claim_ids_seen:
            raise DomainValidationError(
                f"duplicate claim_id {claim_id!r} across {field_name} in one AuditReport"
            )
        claim_ids_seen.add(claim_id)

        _check_span(claim["start_char"], claim["end_char"], f"claim {claim_id!r}")
        _check_finite(verdict.get("raw_score"), "verdict.raw_score")
        _check_finite(
            verdict.get("calibrated_confidence"), "verdict.calibrated_confidence"
        )

        assessments = verdict.get("evidence_assessments", [])
        edge_keys_seen: set[tuple[Any, ...]] = set()
        edge_states: set[str] = set()
        for assessment in assessments:
            _check_evidence_assessment(assessment)
            key = _edge_key(assessment)
            if key in edge_keys_seen:
                raise DomainValidationError(
                    "duplicate evidence_assessments edge "
                    f"(chunk_id/span/verifier_id/verifier_revision) {key!r} "
                    f"within claim {claim_id!r}"
                )
            edge_keys_seen.add(key)
            edge_states.add(assessment["state"])

        _check_claim_verdict_edge_consistency(claim_id, verdict["state"], edge_states)
    return claim_ids_seen


def _check_repair_action_shape(action: Mapping[str, Any]) -> None:
    """Mirror RepairActionRecord's delete/commit nullability invariants."""
    is_committed_delete = (
        action.get("action") == "delete" and action.get("committed") is True
    )
    if is_committed_delete:
        if action.get("state_after") is not None:
            raise DomainValidationError(
                "committed delete repair_action requires state_after == null"
            )
        if action.get("claim_id_after") is not None:
            raise DomainValidationError(
                "committed delete repair_action requires claim_id_after == null"
            )
        if action.get("claim_text_after") is not None:
            raise DomainValidationError(
                "committed delete repair_action requires claim_text_after == null"
            )
    else:
        required_after_fields = ["claim_id_after", "claim_text_after"]
        if action.get("committed") is True:
            required_after_fields.append("state_after")
        for field_name in required_after_fields:
            if action.get(field_name) is None:
                raise DomainValidationError(
                    f"non-deleting or uncommitted repair_action requires {field_name}"
                )


def _check_run_mode_consistency(document: Mapping[str, Any]) -> None:
    """Mirrors AuditReport's own run_mode <-> repair-field invariant check
    (core_requirements #8.1) for a foreign document.
    """
    run_mode = document.get("run_mode")
    if run_mode == "VERIFY":
        if document.get("initial_verdicts"):
            raise DomainValidationError(
                "run_mode == VERIFY requires initial_verdicts to be empty"
            )
        if document.get("repair_rounds", 0) != 0:
            raise DomainValidationError(
                "run_mode == VERIFY requires repair_rounds == 0"
            )
        if document.get("repair_actions"):
            raise DomainValidationError(
                "run_mode == VERIFY requires repair_actions to be empty"
            )
        if document.get("stop_reason") is not None:
            raise DomainValidationError(
                "run_mode == VERIFY requires stop_reason to be None"
            )
        if document.get("heal_progress_calibrator_id") is not None or document.get(
            "heal_progress_calibrator_revision"
        ) is not None:
            raise DomainValidationError(
                "run_mode == VERIFY requires heal progress calibrator metadata to be null"
            )
    elif run_mode == "HEAL":
        if not document.get("initial_verdicts"):
            raise DomainValidationError(
                "run_mode == HEAL requires non-empty initial_verdicts"
            )
        stop_reason = document.get("stop_reason")
        if not isinstance(stop_reason, str) or not stop_reason.strip():
            raise DomainValidationError(
                "run_mode == HEAL requires a non-empty stop_reason"
            )


def _check_calibration_metadata(document: Mapping[str, Any]) -> None:
    """Require traceable calibration metadata for every calibrated result."""

    calibrator_id = document.get("calibrator_id")
    calibrator_revision = document.get("calibrator_revision")
    if (calibrator_id is None) != (calibrator_revision is None):
        raise DomainValidationError(
            "calibrator_id and calibrator_revision must both be set or both be null"
        )
    for name, value in (
        ("calibrator_id", calibrator_id),
        ("calibrator_revision", calibrator_revision),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise DomainValidationError(f"{name} must be a non-empty string or null")
    has_calibrated_verdict = any(
        verdict.get("calibrated_confidence") is not None
        for verdict in (
            list(document.get("verdicts", []))
            + list(document.get("initial_verdicts", []))
        )
    )
    if has_calibrated_verdict and calibrator_id is None:
        raise DomainValidationError(
            "calibrated verdicts require calibrator_id and calibrator_revision"
        )

    progress_id = document.get("heal_progress_calibrator_id")
    progress_revision = document.get("heal_progress_calibrator_revision")
    if (progress_id is None) != (progress_revision is None):
        raise DomainValidationError(
            "heal_progress_calibrator_id and heal_progress_calibrator_revision "
            "must both be set or both be null"
        )
    for name, value in (
        ("heal_progress_calibrator_id", progress_id),
        ("heal_progress_calibrator_revision", progress_revision),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise DomainValidationError(f"{name} must be a non-empty string or null")
    has_repair_progress = any(
        action.get("calibrated_confidence_before") is not None
        or action.get("calibrated_confidence_after") is not None
        for action in document.get("repair_actions", [])
    )
    if has_repair_progress and progress_id is None:
        raise DomainValidationError(
            "repair progress confidences require heal progress calibrator metadata"
        )


def _check_span(start_char: Any, end_char: Any, context: str) -> None:
    if end_char <= start_char:
        raise DomainValidationError(
            f"{context}: end_char must be greater than start_char "
            f"(got start_char={start_char!r}, end_char={end_char!r})"
        )


def _check_evidence_assessment(assessment: Mapping[str, Any]) -> None:
    reference = assessment["reference"]
    start, end = reference.get("start_char"), reference.get("end_char")
    if start is not None and end is not None:
        _check_span(start, end, f"evidence reference to chunk {reference.get('chunk_id')!r}")
    _check_finite(reference.get("relevance_score"), "evidence reference.relevance_score")
    label_scores = assessment.get("label_scores") or {}
    _check_finite(label_scores.get("supported"), "label_scores.supported")
    _check_finite(label_scores.get("contradicted"), "label_scores.contradicted")
    _check_finite(label_scores.get("insufficient"), "label_scores.insufficient")
    if label_scores.get("score_kind") == "probabilities":
        values = (
            label_scores.get("supported"),
            label_scores.get("contradicted"),
            label_scores.get("insufficient"),
        )
        if all(isinstance(value, (int, float)) for value in values):
            total = sum(values)
            if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
                raise DomainValidationError(
                    "label_scores.supported + contradicted + insufficient must "
                    f"sum to 1 (within {_PROBABILITY_SUM_TOLERANCE}) when "
                    f"score_kind == 'probabilities' (got sum={total!r})"
                )


def _edge_key(assessment: Mapping[str, Any]) -> tuple[Any, ...]:
    reference = assessment["reference"]
    return (
        reference.get("chunk_id"),
        reference.get("start_char"),
        reference.get("end_char"),
        assessment.get("verifier_id"),
        assessment.get("verifier_revision"),
    )


def _check_claim_verdict_edge_consistency(
    claim_id: str, state: str, edge_states: set[str]
) -> None:
    """Mirrors groundguard_rag.domain.models._check_claim_verdict_edge_consistency
    for a foreign document that did not go through ClaimVerdict's constructor.
    """
    context = f"claim {claim_id!r}"
    if state == _NOT_CHECKABLE:
        if edge_states:
            raise DomainValidationError(
                f"{context}: NOT_CHECKABLE requires evidence_assessments to be empty"
            )
    elif state == _INSUFFICIENT_EVIDENCE:
        if edge_states - {_INSUFFICIENT_EVIDENCE}:
            raise DomainValidationError(
                f"{context}: INSUFFICIENT_EVIDENCE requires no evidence_assessments, "
                "or evidence_assessments that are all themselves INSUFFICIENT_EVIDENCE"
            )
    elif state == _SUPPORTED:
        if _SUPPORTED not in edge_states:
            raise DomainValidationError(
                f"{context}: SUPPORTED requires at least one SUPPORTED evidence_assessments entry"
            )
        if _CONTRADICTED in edge_states:
            raise DomainValidationError(
                f"{context}: SUPPORTED must not coexist with a CONTRADICTED evidence_assessments entry"
            )
    elif state == _CONTRADICTED:
        if _CONTRADICTED not in edge_states:
            raise DomainValidationError(
                f"{context}: CONTRADICTED requires at least one CONTRADICTED evidence_assessments entry"
            )
        if _SUPPORTED in edge_states:
            raise DomainValidationError(
                f"{context}: CONTRADICTED must not coexist with a SUPPORTED evidence_assessments entry"
            )
    elif state == _CONFLICTING_EVIDENCE:
        if _SUPPORTED not in edge_states or _CONTRADICTED not in edge_states:
            raise DomainValidationError(
                f"{context}: CONFLICTING_EVIDENCE requires at least one SUPPORTED "
                "and at least one CONTRADICTED evidence_assessments entry"
            )


def _check_finite(value: Any, field_name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DomainValidationError(f"{field_name} must be finite (no NaN/Infinity)")
