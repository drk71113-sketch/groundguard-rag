"""Held-out claim-level sigmoid/Platt confidence calibration.

The raw claim-state strength extracted here is *not* confidence.  Only a
successfully fitted ``PlattClaimCalibrator`` may map it to an empirical
probability that the complete claim verdict is correct.  Fitting and
inference are local, deterministic, dependency-free, and never mutate a
``ClaimVerdict``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import (
    ConfigurationError,
    DomainValidationError,
    GroundGuardError,
)
from groundguard_rag.domain.models import ClaimVerdict
from groundguard_rag.domain.ports import Calibrator


CALIBRATION_ARTIFACT_SCHEMA_VERSION = "1.0.0"
CLAIM_STATE_STRENGTH_VERSION = "claim-state-strength.v1"
_HASH_PREFIX = "sha256:"


class CalibrationFitError(GroundGuardError):
    """The labeled calibration data could not produce a trustworthy fit."""


class CalibrationArtifactError(GroundGuardError):
    """A serialized or in-memory calibration artifact is invalid or tampered."""


def _require_nonempty_string(value: Any, name: str, error_type=ConfigurationError) -> None:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{name} must be a non-empty string")


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")


def _require_finite(value: Any, name: str, error_type=ConfigurationError) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{name} must be a non-bool real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise error_type(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class CalibrationSample:
    """One independently labeled claim verdict used for fit or evaluation."""

    sample_id: str
    verdict: ClaimVerdict
    expected_state: VerificationState

    def __post_init__(self) -> None:
        _require_nonempty_string(
            self.sample_id, "CalibrationSample.sample_id", DomainValidationError
        )
        if not isinstance(self.verdict, ClaimVerdict):
            raise DomainValidationError(
                "CalibrationSample.verdict must be a ClaimVerdict"
            )
        if self.verdict.calibrated_confidence is not None:
            raise DomainValidationError(
                "CalibrationSample.verdict must be uncalibrated to prevent leakage"
            )
        if not isinstance(self.expected_state, VerificationState):
            raise DomainValidationError(
                "CalibrationSample.expected_state must be a VerificationState"
            )

    @property
    def is_correct(self) -> bool:
        return self.verdict.state is self.expected_state


@dataclass(frozen=True)
class PlattFitConfig:
    """Numerical and data sufficiency limits for deterministic Platt fitting."""

    min_samples: int = 30
    max_iterations: int = 200
    tolerance: float = 1e-10
    l2: float = 1e-4
    probability_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        _require_positive_int(self.min_samples, "PlattFitConfig.min_samples")
        if self.min_samples < 2:
            raise ConfigurationError("PlattFitConfig.min_samples must be >= 2")
        _require_positive_int(
            self.max_iterations, "PlattFitConfig.max_iterations"
        )
        tolerance = _require_finite(
            self.tolerance, "PlattFitConfig.tolerance"
        )
        if tolerance <= 0.0:
            raise ConfigurationError("PlattFitConfig.tolerance must be > 0")
        l2 = _require_finite(self.l2, "PlattFitConfig.l2")
        if l2 <= 0.0:
            raise ConfigurationError(
                "PlattFitConfig.l2 must be > 0 to keep separated fits finite"
            )
        epsilon = _require_finite(
            self.probability_epsilon,
            "PlattFitConfig.probability_epsilon",
        )
        if not (0.0 < epsilon < 0.5):
            raise ConfigurationError(
                "PlattFitConfig.probability_epsilon must be within (0, 0.5)"
            )


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _HASH_PREFIX + hashlib.sha256(encoded).hexdigest()


def _artifact_revision_payload(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: fields[key]
        for key in (
            "schema_version",
            "feature_version",
            "calibrator_id",
            "verifier_id",
            "verifier_revision",
            "threshold_version",
            "slope",
            "intercept",
            "sample_count",
            "correct_count",
            "incorrect_count",
            "probability_epsilon",
            "fit_data_hash",
        )
    }


@dataclass(frozen=True)
class PlattCalibrationArtifact:
    """Portable, content-addressed parameters for claim-level calibration."""

    schema_version: str
    feature_version: str
    calibrator_id: str
    verifier_id: str
    verifier_revision: str
    threshold_version: str
    slope: float
    intercept: float
    sample_count: int
    correct_count: int
    incorrect_count: int
    probability_epsilon: float
    fit_data_hash: str
    calibrator_revision: str

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_ARTIFACT_SCHEMA_VERSION:
            raise CalibrationArtifactError(
                "PlattCalibrationArtifact.schema_version must equal "
                f"{CALIBRATION_ARTIFACT_SCHEMA_VERSION!r}"
            )
        if self.feature_version != CLAIM_STATE_STRENGTH_VERSION:
            raise CalibrationArtifactError(
                "PlattCalibrationArtifact.feature_version must equal "
                f"{CLAIM_STATE_STRENGTH_VERSION!r}"
            )
        for name in (
            "calibrator_id",
            "verifier_id",
            "verifier_revision",
            "threshold_version",
        ):
            _require_nonempty_string(
                getattr(self, name),
                f"PlattCalibrationArtifact.{name}",
                CalibrationArtifactError,
            )
        slope = _require_finite(
            self.slope, "PlattCalibrationArtifact.slope", CalibrationArtifactError
        )
        _require_finite(
            self.intercept,
            "PlattCalibrationArtifact.intercept",
            CalibrationArtifactError,
        )
        if slope < 0.0:
            raise CalibrationArtifactError(
                "PlattCalibrationArtifact.slope must be >= 0"
            )
        counts = (self.sample_count, self.correct_count, self.incorrect_count)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise CalibrationArtifactError("artifact counts must be integers")
        if self.sample_count < 2 or self.correct_count <= 0 or self.incorrect_count <= 0:
            raise CalibrationArtifactError(
                "artifact must contain at least two samples and both outcomes"
            )
        if self.correct_count + self.incorrect_count != self.sample_count:
            raise CalibrationArtifactError(
                "correct_count + incorrect_count must equal sample_count"
            )
        epsilon = _require_finite(
            self.probability_epsilon,
            "PlattCalibrationArtifact.probability_epsilon",
            CalibrationArtifactError,
        )
        if not (0.0 < epsilon < 0.5):
            raise CalibrationArtifactError(
                "artifact probability_epsilon must be within (0, 0.5)"
            )
        self._validate_hash(self.fit_data_hash, "fit_data_hash")
        self._validate_hash(self.calibrator_revision, "calibrator_revision")
        expected_revision = _canonical_hash(_artifact_revision_payload(self.to_dict()))
        if self.calibrator_revision != expected_revision:
            raise CalibrationArtifactError(
                "calibrator_revision does not match artifact contents"
            )

    @staticmethod
    def _validate_hash(value: Any, name: str) -> None:
        if (
            not isinstance(value, str)
            or not value.startswith(_HASH_PREFIX)
            or len(value) != len(_HASH_PREFIX) + 64
        ):
            raise CalibrationArtifactError(f"{name} must be a sha256:<64 hex> string")
        try:
            int(value[len(_HASH_PREFIX) :], 16)
        except ValueError as exc:
            raise CalibrationArtifactError(
                f"{name} must contain lowercase or uppercase hexadecimal digits"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_version": self.feature_version,
            "calibrator_id": self.calibrator_id,
            "verifier_id": self.verifier_id,
            "verifier_revision": self.verifier_revision,
            "threshold_version": self.threshold_version,
            "slope": self.slope,
            "intercept": self.intercept,
            "sample_count": self.sample_count,
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "probability_epsilon": self.probability_epsilon,
            "fit_data_hash": self.fit_data_hash,
            "calibrator_revision": self.calibrator_revision,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "PlattCalibrationArtifact":
        if not isinstance(document, Mapping):
            raise CalibrationArtifactError("artifact document must be a mapping")
        expected_keys = set(cls.__dataclass_fields__)
        actual_keys = set(document)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unknown = sorted(actual_keys - expected_keys)
            raise CalibrationArtifactError(
                f"artifact fields mismatch (missing={missing}, unknown={unknown})"
            )
        try:
            return cls(**dict(document))
        except TypeError as exc:
            raise CalibrationArtifactError("artifact document has invalid values") from exc


def _claim_state_strength_and_identity(
    verdict: ClaimVerdict,
) -> tuple[float, str, str]:
    if not isinstance(verdict, ClaimVerdict):
        raise DomainValidationError("calibration verdict must be a ClaimVerdict")
    assessments = verdict.evidence_assessments
    if not assessments:
        raise DomainValidationError(
            "claim-state calibration requires at least one evidence assessment"
        )

    identities = {
        (assessment.verifier_id, assessment.verifier_revision)
        for assessment in assessments
    }
    if len(identities) != 1:
        raise DomainValidationError(
            "all evidence assessments must share one verifier ID and revision"
        )
    for assessment in assessments:
        if assessment.label_scores.score_kind != "probabilities":
            raise DomainValidationError(
                "claim-state calibration requires probability label scores, not logits"
            )

    state = verdict.state
    if state is VerificationState.SUPPORTED:
        score = max(
            assessment.label_scores.supported
            for assessment in assessments
            if assessment.state is VerificationState.SUPPORTED
        )
    elif state is VerificationState.CONTRADICTED:
        score = max(
            assessment.label_scores.contradicted
            for assessment in assessments
            if assessment.state is VerificationState.CONTRADICTED
        )
    elif state is VerificationState.INSUFFICIENT_EVIDENCE:
        score = 1.0 - max(
            max(
                assessment.label_scores.supported,
                assessment.label_scores.contradicted,
            )
            for assessment in assessments
        )
    elif state is VerificationState.CONFLICTING_EVIDENCE:
        support_strength = max(
            assessment.label_scores.supported
            for assessment in assessments
            if assessment.state is VerificationState.SUPPORTED
        )
        contradiction_strength = max(
            assessment.label_scores.contradicted
            for assessment in assessments
            if assessment.state is VerificationState.CONTRADICTED
        )
        score = min(support_strength, contradiction_strength)
    else:
        raise DomainValidationError(
            "NOT_CHECKABLE verdicts have no evidence-based calibration feature"
        )

    # LabelScores already guarantees probability bounds.  Clamp the tiny
    # subtraction noise possible in the insufficient feature only.
    bounded_score = min(1.0, max(0.0, float(score)))
    verifier_id, verifier_revision = next(iter(identities))
    return bounded_score, verifier_id, verifier_revision


def claim_state_strength(verdict: ClaimVerdict) -> float:
    """Return the uncalibrated ``claim-state-strength.v1`` scalar feature."""

    score, _, _ = _claim_state_strength_and_identity(verdict)
    return score


def _clipped_logit(probability: float, epsilon: float) -> float:
    clipped = min(1.0 - epsilon, max(epsilon, probability))
    return math.log(clipped) - math.log1p(-clipped)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _objective(
    slope: float,
    intercept: float,
    features: list[float],
    targets: list[int],
    l2: float,
) -> float:
    loss = 0.5 * l2 * slope * slope
    for feature, target in zip(features, targets):
        linear = slope * feature + intercept
        loss += _softplus(linear) - target * linear
    return loss


def _fit_monotonic_logistic(
    features: list[float],
    targets: list[int],
    config: PlattFitConfig,
) -> tuple[float, float]:
    positive_rate = sum(targets) / len(targets)
    slope = 1.0
    intercept = _clipped_logit(positive_rate, config.probability_epsilon) - slope * (
        sum(features) / len(features)
    )
    current_loss = _objective(slope, intercept, features, targets, config.l2)

    for _ in range(config.max_iterations):
        gradient_slope = config.l2 * slope
        gradient_intercept = 0.0
        hessian_slope = config.l2
        hessian_cross = 0.0
        hessian_intercept = 0.0
        for feature, target in zip(features, targets):
            probability = _sigmoid(slope * feature + intercept)
            residual = probability - target
            curvature = probability * (1.0 - probability)
            gradient_slope += residual * feature
            gradient_intercept += residual
            hessian_slope += curvature * feature * feature
            hessian_cross += curvature * feature
            hessian_intercept += curvature

        determinant = (
            hessian_slope * hessian_intercept
            - hessian_cross * hessian_cross
        )
        if not math.isfinite(determinant) or determinant <= 1e-18:
            raise CalibrationFitError(
                "Platt fit Hessian is singular; calibration data are numerically degenerate"
            )
        delta_slope = (
            gradient_slope * hessian_intercept
            - gradient_intercept * hessian_cross
        ) / determinant
        delta_intercept = (
            hessian_slope * gradient_intercept
            - hessian_cross * gradient_slope
        ) / determinant

        target_slope = max(0.0, slope - delta_slope)
        target_intercept = intercept - delta_intercept
        direction_slope = target_slope - slope
        direction_intercept = target_intercept - intercept
        if max(abs(direction_slope), abs(direction_intercept)) <= config.tolerance:
            return slope, intercept

        step = 1.0
        accepted = False
        while step >= 2.0**-30:
            candidate_slope = max(0.0, slope + step * direction_slope)
            candidate_intercept = intercept + step * direction_intercept
            candidate_loss = _objective(
                candidate_slope,
                candidate_intercept,
                features,
                targets,
                config.l2,
            )
            if math.isfinite(candidate_loss) and candidate_loss <= current_loss:
                slope = candidate_slope
                intercept = candidate_intercept
                improvement = current_loss - candidate_loss
                current_loss = candidate_loss
                accepted = True
                if improvement <= config.tolerance:
                    return slope, intercept
                break
            step *= 0.5
        if not accepted:
            raise CalibrationFitError(
                "Platt fit line search failed to find a finite improving step"
            )

    raise CalibrationFitError(
        f"Platt fit did not converge within {config.max_iterations} iterations"
    )


class PlattClaimCalibrator(Calibrator):
    """Apply one fitted, verifier-specific claim correctness calibration."""

    def __init__(self, artifact: PlattCalibrationArtifact) -> None:
        if not isinstance(artifact, PlattCalibrationArtifact):
            raise ConfigurationError(
                "PlattClaimCalibrator.artifact must be a PlattCalibrationArtifact"
            )
        self._artifact = artifact

    @property
    def artifact(self) -> PlattCalibrationArtifact:
        return self._artifact

    @property
    def calibrator_id(self) -> str:
        return self._artifact.calibrator_id

    @property
    def calibrator_revision(self) -> str:
        return self._artifact.calibrator_revision

    @property
    def threshold_version(self) -> str:
        return self._artifact.threshold_version

    @classmethod
    def fit(
        cls,
        samples: list[CalibrationSample],
        *,
        calibrator_id: str,
        threshold_version: str,
        config: PlattFitConfig | None = None,
    ) -> "PlattClaimCalibrator":
        _require_nonempty_string(calibrator_id, "calibrator_id")
        _require_nonempty_string(threshold_version, "threshold_version")
        if config is None:
            config = PlattFitConfig()
        if not isinstance(config, PlattFitConfig):
            raise ConfigurationError("config must be a PlattFitConfig")
        if not isinstance(samples, list):
            raise DomainValidationError("calibration samples must be a list")
        if len(samples) < config.min_samples:
            raise CalibrationFitError(
                f"calibration requires at least {config.min_samples} samples"
            )

        records: list[tuple[str, float, int, str, str]] = []
        seen_ids: set[str] = set()
        for sample in samples:
            if not isinstance(sample, CalibrationSample):
                raise DomainValidationError(
                    "calibration samples must contain only CalibrationSample instances"
                )
            if sample.sample_id in seen_ids:
                raise CalibrationFitError(
                    f"duplicate calibration sample_id {sample.sample_id!r}"
                )
            seen_ids.add(sample.sample_id)
            strength, verifier_id, verifier_revision = (
                _claim_state_strength_and_identity(sample.verdict)
            )
            records.append(
                (
                    sample.sample_id,
                    strength,
                    int(sample.is_correct),
                    verifier_id,
                    verifier_revision,
                )
            )

        identities = {(record[3], record[4]) for record in records}
        if len(identities) != 1:
            raise CalibrationFitError(
                "all calibration samples must use one verifier ID and revision"
            )
        outcomes = {record[2] for record in records}
        if outcomes != {0, 1}:
            raise CalibrationFitError(
                "calibration samples must contain both correct and incorrect verdicts"
            )
        if len({record[1] for record in records}) < 2:
            raise CalibrationFitError(
                "calibration samples must contain at least two distinct state strengths"
            )

        records.sort(key=lambda item: item[0])
        feature_logits = [
            _clipped_logit(record[1], config.probability_epsilon)
            for record in records
        ]
        targets = [record[2] for record in records]
        slope, intercept = _fit_monotonic_logistic(
            feature_logits, targets, config
        )
        fit_payload = [
            {
                "sample_id": sample_id,
                "state_strength": strength,
                "is_correct": bool(target),
                "verifier_id": verifier_id,
                "verifier_revision": verifier_revision,
            }
            for sample_id, strength, target, verifier_id, verifier_revision in records
        ]
        fit_data_hash = _canonical_hash(fit_payload)
        verifier_id, verifier_revision = next(iter(identities))
        artifact_fields: dict[str, Any] = {
            "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
            "feature_version": CLAIM_STATE_STRENGTH_VERSION,
            "calibrator_id": calibrator_id,
            "verifier_id": verifier_id,
            "verifier_revision": verifier_revision,
            "threshold_version": threshold_version,
            "slope": slope,
            "intercept": intercept,
            "sample_count": len(records),
            "correct_count": sum(targets),
            "incorrect_count": len(targets) - sum(targets),
            "probability_epsilon": config.probability_epsilon,
            "fit_data_hash": fit_data_hash,
        }
        artifact_fields["calibrator_revision"] = _canonical_hash(
            _artifact_revision_payload(artifact_fields)
        )
        return cls(PlattCalibrationArtifact(**artifact_fields))

    def calibrate(self, verdict: ClaimVerdict) -> float:
        strength, verifier_id, verifier_revision = (
            _claim_state_strength_and_identity(verdict)
        )
        artifact = self._artifact
        if (
            verifier_id != artifact.verifier_id
            or verifier_revision != artifact.verifier_revision
        ):
            raise DomainValidationError(
                "verdict verifier identity does not match calibration artifact"
            )
        feature = _clipped_logit(strength, artifact.probability_epsilon)
        confidence = _sigmoid(artifact.slope * feature + artifact.intercept)
        if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
            raise CalibrationArtifactError(
                "calibration artifact produced an invalid confidence"
            )
        return confidence


@dataclass(frozen=True)
class ProbabilityMetrics:
    count: int
    accuracy: float
    mean_confidence: float
    log_loss: float
    brier_score: float
    expected_calibration_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "accuracy": self.accuracy,
            "mean_confidence": self.mean_confidence,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
        }


@dataclass(frozen=True)
class CalibrationEvaluation:
    raw: ProbabilityMetrics
    calibrated: ProbabilityMetrics
    n_bins: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw.to_dict(),
            "calibrated": self.calibrated.to_dict(),
            "n_bins": self.n_bins,
        }


def _probability_metrics(
    probabilities: list[float], targets: list[int], n_bins: int
) -> ProbabilityMetrics:
    count = len(probabilities)
    epsilon = 1e-15
    log_loss = 0.0
    brier = 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for probability, target in zip(probabilities, targets):
        clipped = min(1.0 - epsilon, max(epsilon, probability))
        log_loss -= target * math.log(clipped) + (1 - target) * math.log1p(-clipped)
        brier += (probability - target) ** 2
        index = min(int(probability * n_bins), n_bins - 1)
        bins[index].append((probability, target))

    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        bucket_confidence = sum(item[0] for item in bucket) / len(bucket)
        bucket_accuracy = sum(item[1] for item in bucket) / len(bucket)
        ece += (len(bucket) / count) * abs(bucket_confidence - bucket_accuracy)
    return ProbabilityMetrics(
        count=count,
        accuracy=sum(targets) / count,
        mean_confidence=sum(probabilities) / count,
        log_loss=log_loss / count,
        brier_score=brier / count,
        expected_calibration_error=ece,
    )


def evaluate_calibrator(
    calibrator: Calibrator,
    samples: list[CalibrationSample],
    *,
    n_bins: int = 10,
) -> CalibrationEvaluation:
    """Evaluate on caller-supplied labeled data, ideally disjoint from fit data."""

    if not isinstance(calibrator, Calibrator):
        raise ConfigurationError("calibrator must implement Calibrator")
    if not isinstance(samples, list) or not samples:
        raise DomainValidationError("evaluation samples must be a non-empty list")
    _require_positive_int(n_bins, "n_bins")

    raw_probabilities: list[float] = []
    calibrated_probabilities: list[float] = []
    targets: list[int] = []
    seen_ids: set[str] = set()
    for sample in samples:
        if not isinstance(sample, CalibrationSample):
            raise DomainValidationError(
                "evaluation samples must contain only CalibrationSample instances"
            )
        if sample.sample_id in seen_ids:
            raise DomainValidationError(
                f"duplicate evaluation sample_id {sample.sample_id!r}"
            )
        seen_ids.add(sample.sample_id)
        raw_probability = claim_state_strength(sample.verdict)
        calibrated_probability = calibrator.calibrate(sample.verdict)
        if (
            isinstance(calibrated_probability, bool)
            or not isinstance(calibrated_probability, (int, float))
            or not math.isfinite(calibrated_probability)
            or not (0.0 <= calibrated_probability <= 1.0)
        ):
            raise DomainValidationError(
                "calibrator returned an invalid probability during evaluation"
            )
        raw_probabilities.append(raw_probability)
        calibrated_probabilities.append(float(calibrated_probability))
        targets.append(int(sample.is_correct))

    return CalibrationEvaluation(
        raw=_probability_metrics(raw_probabilities, targets, n_bins),
        calibrated=_probability_metrics(calibrated_probabilities, targets, n_bins),
        n_bins=n_bins,
    )
