"""Deterministic, zero-dependency claim segmentation baseline.

This adapter deliberately promises sentence/clause segmentation, not
semantic atomicity. It preserves exact source spans for the audit graph and
keeps non-factual material instead of silently filtering it, but a sentence
containing several coordinated facts can still remain one ``AtomicClaim``.
Callers that need semantic decomposition can replace it through the same
``ClaimDecomposer`` port without changing ``VerifyService``.
"""

from __future__ import annotations

import re

from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import AtomicClaim
from groundguard_rag.domain.ports import ClaimDecomposer


_TERMINAL_PUNCTUATION = frozenset(".?!。！？;；")
_CLOSING_MARKS = frozenset('"\'”’»）)]}】」』')
_BOUNDARY_TAIL = _TERMINAL_PUNCTUATION | _CLOSING_MARKS

# Intentionally small and documented: this is a deterministic baseline, not
# a general-purpose tokenizer. These cover high-frequency title/Latin
# abbreviations that would otherwise produce obviously broken audit spans.
_COMMON_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "vs.",
    "etc.",
    "fig.",
    "no.",
)
_ACRONYM_AT_END = re.compile(r"(?:[A-Za-z]\.){2,}$")
_SINGLE_INITIAL_AT_END = re.compile(r"(?:^|\s)[A-Za-z]\.$")


class RuleBasedClaimDecomposer(ClaimDecomposer):
    """Split English/Chinese text into traceable sentence-like claims.

    Boundaries are terminal punctuation, semicolons, and line breaks.
    Decimal points, token-internal dots (covering ordinary domains/emails),
    common abbreviations, and initialisms are protected from naive splitting.
    The rules are intentionally conservative and replaceable; they do not try
    to infer factuality or split every coordinated proposition semantically.
    """

    def decompose(self, answer: str) -> list[AtomicClaim]:
        if not isinstance(answer, str) or not answer.strip():
            raise DomainValidationError(
                "RuleBasedClaimDecomposer.answer must be a non-empty, "
                "non-whitespace string"
            )

        claims: list[AtomicClaim] = []
        segment_start = 0
        cursor = 0

        while cursor < len(answer):
            character = answer[cursor]

            if character in "\r\n":
                self._append_claim(answer, segment_start, cursor, claims)
                cursor = self._skip_line_breaks(answer, cursor)
                segment_start = cursor
                continue

            if character in _TERMINAL_PUNCTUATION:
                if character == "." and self._period_is_internal(answer, cursor):
                    cursor += 1
                    continue

                end = cursor + 1
                # Keep repeated punctuation and trailing closing quotes or
                # brackets attached to the same claim: ``Really?!"`` must not
                # become several punctuation-only claims.
                while end < len(answer) and answer[end] in _BOUNDARY_TAIL:
                    end += 1
                self._append_claim(answer, segment_start, end, claims)
                segment_start = end
                cursor = end
                continue

            cursor += 1

        self._append_claim(answer, segment_start, len(answer), claims)
        return claims

    @staticmethod
    def _append_claim(
        answer: str,
        raw_start: int,
        raw_end: int,
        claims: list[AtomicClaim],
    ) -> None:
        start = raw_start
        end = raw_end
        while start < end and answer[start].isspace():
            start += 1
        while end > start and answer[end - 1].isspace():
            end -= 1
        if start == end:
            return

        claims.append(
            AtomicClaim(
                claim_id=f"claim-{start}-{end}",
                text=answer[start:end],
                start_char=start,
                end_char=end,
            )
        )

    @staticmethod
    def _skip_line_breaks(answer: str, cursor: int) -> int:
        while cursor < len(answer) and answer[cursor] in "\r\n":
            cursor += 1
        return cursor

    @staticmethod
    def _period_is_internal(answer: str, index: int) -> bool:
        previous = answer[index - 1] if index > 0 else ""
        following = answer[index + 1] if index + 1 < len(answer) else ""

        # Covers decimals/version tokens (3.14) and ordinary domains/emails
        # (example.com, a.b@example.com). A no-space ``Hello.World`` is also
        # treated conservatively as one token; this is an explicit heuristic
        # limitation rather than an attempt at full natural-language parsing.
        if previous.isalnum() and following.isalnum():
            return True

        prefix = answer[: index + 1]
        lowered_prefix = prefix.lower()
        if any(lowered_prefix.endswith(item) for item in _COMMON_ABBREVIATIONS):
            return True
        if _ACRONYM_AT_END.search(prefix):
            return True

        # A single initial followed by another name token (``A. Smith``) is
        # not a sentence boundary. Do not protect it at end-of-input.
        if _SINGLE_INITIAL_AT_END.search(prefix):
            remainder = answer[index + 1 :].lstrip()
            if remainder and remainder[0].isupper():
                return True

        return False
