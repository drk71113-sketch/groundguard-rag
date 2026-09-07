from __future__ import annotations

import pytest

from groundguard_rag.adapters.verification import RuleBasedCheckabilityVerifier
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import AtomicClaim, ClaimVerdict
from groundguard_rag.domain.ports import Verifier


class RecordingVerifier(Verifier):
    def __init__(self) -> None:
        self.calls = []

    def verify(self, claim, evidence):
        self.calls.append((claim, evidence))
        return ClaimVerdict(
            claim=claim,
            state=VerificationState.INSUFFICIENT_EVIDENCE,
            evidence_assessments=(),
        )


def claim(text: str) -> AtomicClaim:
    return AtomicClaim("c1", text, 0, len(text))


@pytest.mark.parametrize(
    "text",
    ["Hello!", "谢谢。", "Is Paris in France?", "我觉得蓝色更好。"],
)
def test_obvious_non_claims_are_not_checkable_without_delegate_call(text):
    delegate = RecordingVerifier()
    verdict = RuleBasedCheckabilityVerifier(delegate).verify(claim(text), [])

    assert verdict.state is VerificationState.NOT_CHECKABLE
    assert verdict.evidence_assessments == ()
    assert verdict.calibrated_confidence is None
    assert delegate.calls == []


def test_factual_claim_is_delegated_and_input_list_is_copied():
    delegate = RecordingVerifier()
    evidence = []
    verdict = RuleBasedCheckabilityVerifier(delegate).verify(
        claim("Paris is in France."), evidence
    )

    assert verdict.state is VerificationState.INSUFFICIENT_EVIDENCE
    assert delegate.calls[0][1] is not evidence


def test_constructor_and_direct_call_validate_contracts():
    with pytest.raises(ConfigurationError):
        RuleBasedCheckabilityVerifier(object())
    with pytest.raises(DomainValidationError):
        RuleBasedCheckabilityVerifier(RecordingVerifier()).verify("not-claim", [])
    with pytest.raises(DomainValidationError):
        RuleBasedCheckabilityVerifier(RecordingVerifier()).verify(claim("Fact."), ())
