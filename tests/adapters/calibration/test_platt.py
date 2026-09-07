from __future__ import annotations

import math

import pytest

from groundguard_rag.adapters.calibration import (
    CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    CLAIM_STATE_STRENGTH_VERSION,
    CalibrationArtifactError,
    CalibrationFitError,
    CalibrationSample,
    PlattCalibrationArtifact,
    PlattClaimCalibrator,
    PlattFitConfig,
    claim_state_strength,
    evaluate_calibrator,
)
from groundguard_rag.adapters.decomposition import RuleBasedClaimDecomposer
from groundguard_rag.adapters.evidence_selection import (
    LexicalEvidenceSelector,
    LexicalSelectorConfig,
)
from groundguard_rag.adapters.verification import (
    NliBackend,
    NliVerifier,
    NliVerifierConfig,
)
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.config import VerifyConfig
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import (
    AtomicClaim,
    Chunk,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceReference,
    LabelScores,
    VerificationRequest,
)
from groundguard_rag.domain.ports import Calibrator


def _assessment(
    chunk_id: str,
    state: VerificationState,
    *,
    supported: float,
    contradicted: float,
    insufficient: float,
    verifier_id: str = "nli",
    verifier_revision: str = "rev-1",
    score_kind: str = "probabilities",
) -> EvidenceAssessment:
    return EvidenceAssessment(
        reference=EvidenceReference(chunk_id),
        label_scores=LabelScores(
            supported=supported,
            contradicted=contradicted,
            insufficient=insufficient,
            score_kind=score_kind,
        ),
        state=state,
        rationale=None,
        verifier_id=verifier_id,
        verifier_revision=verifier_revision,
    )


def _supported_verdict(
    strength: float,
    *,
    claim_id: str = "claim",
    verifier_id: str = "nli",
    verifier_revision: str = "rev-1",
) -> ClaimVerdict:
    remainder = 1.0 - strength
    return ClaimVerdict(
        claim=AtomicClaim(claim_id, "supported claim", 0, 15),
        state=VerificationState.SUPPORTED,
        evidence_assessments=(
            _assessment(
                f"chunk-{claim_id}",
                VerificationState.SUPPORTED,
                supported=strength,
                contradicted=remainder * 0.4,
                insufficient=remainder * 0.6,
                verifier_id=verifier_id,
                verifier_revision=verifier_revision,
            ),
        ),
    )


def _sample(sample_id: str, strength: float, correct: bool) -> CalibrationSample:
    verdict = _supported_verdict(strength, claim_id=sample_id)
    expected = (
        VerificationState.SUPPORTED
        if correct
        else VerificationState.CONTRADICTED
    )
    return CalibrationSample(sample_id, verdict, expected)


def _fit_samples() -> list[CalibrationSample]:
    # Increasing empirical correctness with overlapping outcomes avoids a
    # perfectly separated synthetic fit while remaining clearly monotonic.
    samples: list[CalibrationSample] = []
    group_specs = (
        (0.55, 3),
        (0.65, 5),
        (0.75, 7),
        (0.85, 8),
        (0.95, 9),
    )
    for group, (strength, correct_count) in enumerate(group_specs):
        for index in range(10):
            samples.append(
                _sample(
                    f"g{group}-s{index}",
                    strength,
                    correct=index < correct_count,
                )
            )
    return samples


def _fit_calibrator(samples=None) -> PlattClaimCalibrator:
    return PlattClaimCalibrator.fit(
        _fit_samples() if samples is None else samples,
        calibrator_id="claim-platt",
        threshold_version="threshold-v1",
        config=PlattFitConfig(min_samples=10),
    )


def test_supported_strength_uses_strongest_supported_edge():
    verdict = ClaimVerdict(
        claim=AtomicClaim("c", "claim", 0, 5),
        state=VerificationState.SUPPORTED,
        evidence_assessments=(
            _assessment(
                "a", VerificationState.SUPPORTED, supported=0.7, contradicted=0.1, insufficient=0.2
            ),
            _assessment(
                "b", VerificationState.SUPPORTED, supported=0.9, contradicted=0.05, insufficient=0.05
            ),
            _assessment(
                "c", VerificationState.INSUFFICIENT_EVIDENCE, supported=0.4, contradicted=0.2, insufficient=0.4
            ),
        ),
    )
    assert claim_state_strength(verdict) == pytest.approx(0.9)


def test_contradicted_strength_uses_strongest_contradicted_edge():
    verdict = ClaimVerdict(
        claim=AtomicClaim("c", "claim", 0, 5),
        state=VerificationState.CONTRADICTED,
        evidence_assessments=(
            _assessment(
                "a", VerificationState.CONTRADICTED, supported=0.1, contradicted=0.75, insufficient=0.15
            ),
            _assessment(
                "b", VerificationState.CONTRADICTED, supported=0.02, contradicted=0.95, insufficient=0.03
            ),
        ),
    )
    assert claim_state_strength(verdict) == pytest.approx(0.95)


def test_insufficient_strength_uses_weakest_all_edges_boundary():
    verdict = ClaimVerdict(
        claim=AtomicClaim("c", "claim", 0, 5),
        state=VerificationState.INSUFFICIENT_EVIDENCE,
        evidence_assessments=(
            _assessment(
                "a", VerificationState.INSUFFICIENT_EVIDENCE, supported=0.45, contradicted=0.1, insufficient=0.45
            ),
            _assessment(
                "b", VerificationState.INSUFFICIENT_EVIDENCE, supported=0.2, contradicted=0.6, insufficient=0.2
            ),
        ),
    )
    assert claim_state_strength(verdict) == pytest.approx(0.4)


def test_conflicting_strength_uses_weaker_side_of_conflict():
    verdict = ClaimVerdict(
        claim=AtomicClaim("c", "claim", 0, 5),
        state=VerificationState.CONFLICTING_EVIDENCE,
        evidence_assessments=(
            _assessment(
                "a", VerificationState.SUPPORTED, supported=0.92, contradicted=0.03, insufficient=0.05
            ),
            _assessment(
                "b", VerificationState.CONTRADICTED, supported=0.08, contradicted=0.81, insufficient=0.11
            ),
        ),
    )
    assert claim_state_strength(verdict) == pytest.approx(0.81)


def test_strength_rejects_evidence_less_not_checkable_or_insufficient():
    for state in (
        VerificationState.NOT_CHECKABLE,
        VerificationState.INSUFFICIENT_EVIDENCE,
    ):
        verdict = ClaimVerdict(
            claim=AtomicClaim(state.value, "claim", 0, 5),
            state=state,
            evidence_assessments=(),
        )
        with pytest.raises(DomainValidationError, match="at least one"):
            claim_state_strength(verdict)


def test_strength_rejects_logits_and_mixed_verifier_identity():
    logits = ClaimVerdict(
        claim=AtomicClaim("logits", "claim", 0, 5),
        state=VerificationState.SUPPORTED,
        evidence_assessments=(
            _assessment(
                "a", VerificationState.SUPPORTED, supported=2.0, contradicted=-1.0, insufficient=0.0, score_kind="logits"
            ),
        ),
    )
    with pytest.raises(DomainValidationError, match="probability"):
        claim_state_strength(logits)

    mixed = ClaimVerdict(
        claim=AtomicClaim("mixed", "claim", 0, 5),
        state=VerificationState.SUPPORTED,
        evidence_assessments=(
            _assessment(
                "a", VerificationState.SUPPORTED, supported=0.8, contradicted=0.1, insufficient=0.1, verifier_id="one"
            ),
            _assessment(
                "b", VerificationState.SUPPORTED, supported=0.7, contradicted=0.1, insufficient=0.2, verifier_id="two"
            ),
        ),
    )
    with pytest.raises(DomainValidationError, match="one verifier"):
        claim_state_strength(mixed)


def test_calibration_sample_derives_correctness_and_rejects_calibrated_input():
    sample = _sample("sample", 0.8, correct=True)
    assert sample.is_correct is True

    verdict = _supported_verdict(0.8)
    calibrated = ClaimVerdict(
        claim=verdict.claim,
        state=verdict.state,
        evidence_assessments=verdict.evidence_assessments,
        calibrated_confidence=0.9,
    )
    with pytest.raises(DomainValidationError, match="uncalibrated"):
        CalibrationSample("leak", calibrated, VerificationState.SUPPORTED)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_samples", 1),
        ("min_samples", True),
        ("max_iterations", 0),
        ("max_iterations", 1.5),
        ("tolerance", 0),
        ("tolerance", math.nan),
        ("l2", 0),
        ("l2", math.inf),
        ("probability_epsilon", 0),
        ("probability_epsilon", 0.5),
    ],
)
def test_fit_config_rejects_invalid_values(field, value):
    with pytest.raises(ConfigurationError):
        PlattFitConfig(**{field: value})


def test_fit_builds_content_addressed_provider_specific_artifact():
    calibrator = _fit_calibrator()
    artifact = calibrator.artifact

    assert artifact.schema_version == CALIBRATION_ARTIFACT_SCHEMA_VERSION
    assert artifact.feature_version == CLAIM_STATE_STRENGTH_VERSION
    assert artifact.calibrator_id == "claim-platt"
    assert artifact.verifier_id == "nli"
    assert artifact.verifier_revision == "rev-1"
    assert artifact.threshold_version == "threshold-v1"
    assert artifact.sample_count == 50
    assert artifact.correct_count == 32
    assert artifact.incorrect_count == 18
    assert artifact.slope >= 0.0
    assert artifact.fit_data_hash.startswith("sha256:")
    assert artifact.calibrator_revision.startswith("sha256:")


def test_fit_is_deterministic_and_independent_of_input_order():
    samples = _fit_samples()
    first = _fit_calibrator(samples)
    second = _fit_calibrator(list(reversed(samples)))

    assert first.artifact == second.artifact


def test_threshold_version_is_part_of_the_artifact_revision():
    samples = _fit_samples()
    first = _fit_calibrator(samples)
    second = PlattClaimCalibrator.fit(
        samples,
        calibrator_id="claim-platt",
        threshold_version="threshold-v2",
        config=PlattFitConfig(min_samples=10),
    )

    assert first.threshold_version == "threshold-v1"
    assert first.calibrator_revision != second.calibrator_revision


def test_fit_revision_changes_when_a_human_label_changes():
    samples = _fit_samples()
    first = _fit_calibrator(samples)
    changed = list(samples)
    original = changed[0]
    changed[0] = CalibrationSample(
        original.sample_id,
        original.verdict,
        VerificationState.CONTRADICTED,
    )
    second = _fit_calibrator(changed)

    assert first.artifact.fit_data_hash != second.artifact.fit_data_hash
    assert first.calibrator_revision != second.calibrator_revision


def test_calibrated_output_is_monotonic_and_not_an_identity_alias():
    calibrator = _fit_calibrator()
    low = _supported_verdict(0.55, claim_id="low")
    high = _supported_verdict(0.95, claim_id="high")

    low_confidence = calibrator.calibrate(low)
    high_confidence = calibrator.calibrate(high)

    assert 0.0 <= low_confidence < high_confidence <= 1.0
    assert low_confidence != pytest.approx(0.55)
    assert high_confidence != pytest.approx(0.95)


def test_calibrator_rejects_a_different_verifier_revision():
    calibrator = _fit_calibrator()
    other = _supported_verdict(
        0.8,
        verifier_id="nli",
        verifier_revision="rev-2",
    )
    with pytest.raises(DomainValidationError, match="does not match"):
        calibrator.calibrate(other)


def test_artifact_json_round_trip_is_exact():
    artifact = _fit_calibrator().artifact
    restored = PlattCalibrationArtifact.from_dict(artifact.to_dict())
    assert restored == artifact


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(slope=payload["slope"] + 0.1),
        lambda payload: payload.update(sample_count=payload["sample_count"] + 1),
        lambda payload: payload.update(calibrator_revision="sha256:" + "0" * 64),
        lambda payload: payload.update(fit_data_hash="not-a-hash"),
    ],
)
def test_artifact_tampering_is_rejected(mutation):
    payload = _fit_calibrator().artifact.to_dict()
    mutation(payload)
    with pytest.raises(CalibrationArtifactError):
        PlattCalibrationArtifact.from_dict(payload)


def test_artifact_rejects_missing_or_unknown_fields():
    payload = _fit_calibrator().artifact.to_dict()
    del payload["slope"]
    with pytest.raises(CalibrationArtifactError, match="fields mismatch"):
        PlattCalibrationArtifact.from_dict(payload)

    with pytest.raises(CalibrationArtifactError, match="mapping"):
        PlattCalibrationArtifact.from_dict([])

    payload = _fit_calibrator().artifact.to_dict()
    payload["surprise"] = 1
    with pytest.raises(CalibrationArtifactError, match="fields mismatch"):
        PlattCalibrationArtifact.from_dict(payload)


def test_fit_rejects_insufficient_duplicate_or_one_outcome_samples():
    config = PlattFitConfig(min_samples=3)
    with pytest.raises(CalibrationFitError, match="at least"):
        PlattClaimCalibrator.fit(
            [_sample("a", 0.6, True)],
            calibrator_id="c",
            threshold_version="t",
            config=config,
        )

    duplicate = [_sample("same", 0.6, True), _sample("same", 0.8, False), _sample("c", 0.9, True)]
    with pytest.raises(CalibrationFitError, match="duplicate"):
        PlattClaimCalibrator.fit(
            duplicate, calibrator_id="c", threshold_version="t", config=config
        )

    one_outcome = [_sample("a", 0.6, True), _sample("b", 0.8, True), _sample("c", 0.9, True)]
    with pytest.raises(CalibrationFitError, match="both correct and incorrect"):
        PlattClaimCalibrator.fit(
            one_outcome, calibrator_id="c", threshold_version="t", config=config
        )


def test_fit_rejects_one_feature_or_mixed_provider_samples():
    config = PlattFitConfig(min_samples=3)
    one_feature = [_sample("a", 0.7, True), _sample("b", 0.7, False), _sample("c", 0.7, True)]
    with pytest.raises(CalibrationFitError, match="distinct"):
        PlattClaimCalibrator.fit(
            one_feature, calibrator_id="c", threshold_version="t", config=config
        )

    mixed = [
        _sample("a", 0.6, True),
        _sample("b", 0.8, False),
        CalibrationSample(
            "c",
            _supported_verdict(0.9, claim_id="c", verifier_revision="other"),
            VerificationState.SUPPORTED,
        ),
    ]
    with pytest.raises(CalibrationFitError, match="one verifier"):
        PlattClaimCalibrator.fit(
            mixed, calibrator_id="c", threshold_version="t", config=config
        )


def test_fit_rejects_wrong_container_items_or_config():
    with pytest.raises(DomainValidationError, match="list"):
        PlattClaimCalibrator.fit((), calibrator_id="c", threshold_version="t")
    with pytest.raises(DomainValidationError, match="CalibrationSample"):
        PlattClaimCalibrator.fit(
            [object(), object()],
            calibrator_id="c",
            threshold_version="t",
            config=PlattFitConfig(min_samples=2),
        )
    with pytest.raises(ConfigurationError, match="PlattFitConfig"):
        PlattClaimCalibrator.fit(
            _fit_samples(),
            calibrator_id="c",
            threshold_version="t",
            config=object(),
        )
    with pytest.raises(ConfigurationError, match="threshold_version"):
        PlattClaimCalibrator.fit(
            _fit_samples(),
            calibrator_id="c",
            threshold_version="",
        )


class ConstantCalibrator(Calibrator):
    def __init__(self, value):
        self.value = value

    def calibrate(self, verdict):
        return self.value


def test_evaluation_reports_raw_and_calibrated_metrics():
    samples = [
        _sample("a", 0.9, True),
        _sample("b", 0.8, True),
        _sample("c", 0.7, False),
        _sample("d", 0.6, False),
    ]
    evaluation = evaluate_calibrator(ConstantCalibrator(0.5), samples, n_bins=2)

    assert evaluation.raw.count == 4
    assert evaluation.raw.accuracy == 0.5
    assert evaluation.calibrated.mean_confidence == 0.5
    assert evaluation.calibrated.brier_score == pytest.approx(0.25)
    assert evaluation.calibrated.log_loss == pytest.approx(-math.log(0.5))
    assert evaluation.calibrated.expected_calibration_error == pytest.approx(0.0)
    assert evaluation.to_dict()["n_bins"] == 2


def test_fitted_mapping_improves_synthetic_holdout_log_loss_and_brier():
    calibrator = _fit_calibrator()
    holdout = [
        _sample(
            f"holdout-{sample.sample_id}",
            claim_state_strength(sample.verdict),
            sample.is_correct,
        )
        for sample in _fit_samples()
    ]

    evaluation = evaluate_calibrator(calibrator, holdout, n_bins=5)

    assert evaluation.calibrated.log_loss < evaluation.raw.log_loss
    assert evaluation.calibrated.brier_score < evaluation.raw.brier_score


@pytest.mark.parametrize("n_bins", [0, True, 1.5])
def test_evaluation_rejects_invalid_bins(n_bins):
    with pytest.raises(ConfigurationError):
        evaluate_calibrator(ConstantCalibrator(0.5), [_sample("a", 0.7, True)], n_bins=n_bins)


@pytest.mark.parametrize("bad_probability", [True, -0.1, 1.1, math.nan, math.inf])
def test_evaluation_rejects_invalid_calibrator_output(bad_probability):
    with pytest.raises(DomainValidationError, match="invalid probability"):
        evaluate_calibrator(
            ConstantCalibrator(bad_probability),
            [_sample("a", 0.7, True)],
        )


class SupportedBackend(NliBackend):
    def predict_batch(self, pairs):
        return [LabelScores(8.0, -3.0, -2.0, "logits") for _ in pairs]


def test_full_local_pipeline_writes_calibrated_confidence_and_artifact_identity():
    calibrator = _fit_calibrator()
    verifier = NliVerifier(
        SupportedBackend(),
        NliVerifierConfig(
            verifier_id="nli",
            verifier_revision="rev-1",
            decision_threshold=0.8,
        ),
    )
    service = VerifyService(
        decomposer=RuleBasedClaimDecomposer(),
        selector=LexicalEvidenceSelector(
            LexicalSelectorConfig(top_k=1, min_score=0.1)
        ),
        verifier=verifier,
        calibrator=calibrator,
        calibrator_id=calibrator.calibrator_id,
        calibrator_revision=calibrator.calibrator_revision,
        config=VerifyConfig(),
        model_revision="nli@rev-1",
        threshold_version=calibrator.threshold_version,
        clock=lambda: "2026-09-02T00:00:00+00:00",
    )
    request = VerificationRequest(
        request_id="calibrated-request",
        answer="Paris is in France.",
        chunks=(Chunk("chunk-paris", "Paris is in France according to the source."),),
    )

    report = service.verify(request)

    assert report.verdicts[0].state is VerificationState.SUPPORTED
    assert 0.0 <= report.verdicts[0].calibrated_confidence <= 1.0
    assert report.calibrator_id == calibrator.calibrator_id
    assert report.calibrator_revision == calibrator.calibrator_revision
    assert report.to_dict()["verdicts"][0]["calibrated_confidence"] is not None
