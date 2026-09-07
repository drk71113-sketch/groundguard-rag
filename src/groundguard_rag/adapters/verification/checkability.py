"""High-precision checkability decoration for an existing verifier.

The rules intentionally cover only obvious non-assertive text.  Everything
else is delegated unchanged, so this adapter cannot silently replace a real
semantic verifier or turn uncertainty into ``NOT_CHECKABLE``.
"""

from __future__ import annotations

from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import AtomicClaim, ClaimVerdict, EvidenceCandidate
from groundguard_rag.domain.ports import Verifier

_TERMINAL_MARKS = " \t\r\n.!?。！？,，;；:："
_EXACT_NON_CHECKABLE = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "got it",
        "please continue",
        "你好",
        "您好",
        "谢谢",
        "多谢",
        "好的",
        "收到",
        "继续",
        "辛苦了",
    }
)
_SUBJECTIVE_PREFIXES = (
    "i think ",
    "i believe ",
    "i prefer ",
    "in my opinion ",
    "我认为",
    "我觉得",
    "我喜欢",
    "我更喜欢",
    "依我看",
)


class RuleBasedCheckabilityVerifier(Verifier):
    """Return ``NOT_CHECKABLE`` for obvious non-claims, else delegate.

    This is a transparent baseline, not a complete intent classifier.  It
    deliberately prefers false negatives over false positives because a
    false ``NOT_CHECKABLE`` result would suppress evidence evaluation.
    """

    def __init__(self, delegate: Verifier) -> None:
        if not isinstance(delegate, Verifier):
            raise ConfigurationError(
                "RuleBasedCheckabilityVerifier.delegate must implement Verifier"
            )
        self._delegate = delegate

    @property
    def delegate(self) -> Verifier:
        """The explicitly injected verifier used for checkable claims."""

        return self._delegate

    def verify(
        self, claim: AtomicClaim, evidence: list[EvidenceCandidate]
    ) -> ClaimVerdict:
        if not isinstance(claim, AtomicClaim):
            raise DomainValidationError(
                "RuleBasedCheckabilityVerifier.claim must be an AtomicClaim"
            )
        if not isinstance(evidence, list) or any(
            not isinstance(item, EvidenceCandidate) for item in evidence
        ):
            raise DomainValidationError(
                "RuleBasedCheckabilityVerifier.evidence must be a list of "
                "EvidenceCandidate instances"
            )

        reason = _non_checkable_reason(claim.text)
        if reason is not None:
            # Evidence is intentionally omitted: NOT_CHECKABLE is a whole-claim
            # classification made before evidence is judged.
            return ClaimVerdict(
                claim=claim,
                state=VerificationState.NOT_CHECKABLE,
                evidence_assessments=(),
                rationale=reason,
            )
        return self._delegate.verify(claim, list(evidence))


def _non_checkable_reason(text: str) -> str | None:
    stripped = text.strip()
    normalized = stripped.casefold().strip(_TERMINAL_MARKS)
    if normalized in _EXACT_NON_CHECKABLE:
        return "rule-based baseline: greeting, acknowledgement, or thanks"
    if stripped.endswith(("?", "？")):
        return "rule-based baseline: interrogative rather than an assertion"
    if normalized.startswith(_SUBJECTIVE_PREFIXES):
        return "rule-based baseline: explicitly subjective statement"
    return None
