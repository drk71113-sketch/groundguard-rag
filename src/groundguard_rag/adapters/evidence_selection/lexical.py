"""Deterministic lexical candidate selection within request-local chunks.

The bounded overlap score in this module is a relevance heuristic only. It
must never be interpreted as entailment, factual support, or calibrated
confidence; those decisions belong to a ``Verifier`` and ``Calibrator``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import AtomicClaim, Chunk, EvidenceReference
from groundguard_rag.domain.ports import EvidenceSelector


_LATIN_OR_NUMBER_TOKEN = re.compile(r"[A-Za-z0-9]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")

# Small and intentionally static: enough to keep obvious English function
# words from dominating the baseline while avoiding language-model-like
# behavior or a hidden external resource dependency.
_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
_CJK_STOP_CHARS = frozenset("的是了在和与及或为把被")


@dataclass(frozen=True)
class LexicalSelectorConfig:
    """Bounds for the local lexical selector."""

    top_k: int = 5
    min_score: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ConfigurationError(
                "LexicalSelectorConfig.top_k must be a non-bool integer"
            )
        if self.top_k < 1:
            raise ConfigurationError("LexicalSelectorConfig.top_k must be >= 1")
        if isinstance(self.min_score, bool) or not isinstance(
            self.min_score, (int, float)
        ):
            raise ConfigurationError(
                "LexicalSelectorConfig.min_score must be a non-bool real number"
            )
        if not math.isfinite(self.min_score) or not (0.0 <= self.min_score <= 1.0):
            raise ConfigurationError(
                "LexicalSelectorConfig.min_score must be finite and within [0, 1]"
            )


class LexicalEvidenceSelector(EvidenceSelector):
    """Rank whole request-local chunks using transparent token overlap.

    Score = ``0.8 * claim coverage + 0.2 * Jaccard`` over token sets. English
    uses case-folded alphanumeric tokens with a small stopword list; CJK runs
    use character bigrams (or a unigram for a one-character run). This is a
    recall-oriented, explainable baseline, not full tokenization or semantic
    retrieval.
    """

    def __init__(self, config: LexicalSelectorConfig | None = None) -> None:
        if config is None:
            config = LexicalSelectorConfig()
        if not isinstance(config, LexicalSelectorConfig):
            raise ConfigurationError(
                "LexicalEvidenceSelector.config must be a LexicalSelectorConfig"
            )
        self._config = config

    @property
    def config(self) -> LexicalSelectorConfig:
        return self._config

    def select(self, claim: AtomicClaim, chunks: list[Chunk]) -> list[EvidenceReference]:
        self._validate_inputs(claim, chunks)
        claim_tokens = _tokenize(claim.text)
        if not claim_tokens:
            return []

        ranked: list[tuple[float, int, Chunk]] = []
        for index, chunk in enumerate(chunks):
            chunk_tokens = _tokenize(chunk.text)
            if not chunk_tokens:
                continue
            overlap = claim_tokens & chunk_tokens
            if not overlap:
                continue

            coverage = len(overlap) / len(claim_tokens)
            jaccard = len(overlap) / len(claim_tokens | chunk_tokens)
            score = 0.8 * coverage + 0.2 * jaccard
            if score >= self._config.min_score:
                ranked.append((score, index, chunk))

        # Python's sort is stable, but the explicit input index makes the
        # tie rule visible and robust to future key changes.
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            EvidenceReference(chunk_id=chunk.chunk_id, relevance_score=score)
            for score, _, chunk in ranked[: self._config.top_k]
        ]

    @staticmethod
    def _validate_inputs(claim: AtomicClaim, chunks: list[Chunk]) -> None:
        if not isinstance(claim, AtomicClaim):
            raise DomainValidationError(
                "LexicalEvidenceSelector.claim must be an AtomicClaim"
            )
        if not isinstance(chunks, list):
            raise DomainValidationError(
                "LexicalEvidenceSelector.chunks must be a list"
            )

        seen_chunk_ids: set[str] = set()
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise DomainValidationError(
                    "LexicalEvidenceSelector.chunks must contain only Chunk instances"
                )
            if chunk.chunk_id in seen_chunk_ids:
                raise DomainValidationError(
                    "LexicalEvidenceSelector.chunks contains duplicate chunk_id "
                    f"{chunk.chunk_id!r}"
                )
            seen_chunk_ids.add(chunk.chunk_id)


def _tokenize(text: str) -> frozenset[str]:
    tokens = {
        token
        for match in _LATIN_OR_NUMBER_TOKEN.finditer(text.casefold())
        if (token := match.group()) not in _ENGLISH_STOPWORDS
    }

    for match in _CJK_RUN.finditer(text):
        run = match.group()
        if len(run) == 1:
            if run not in _CJK_STOP_CHARS:
                tokens.add(run)
            continue
        for index in range(len(run) - 1):
            bigram = run[index : index + 2]
            if not all(character in _CJK_STOP_CHARS for character in bigram):
                tokens.add(bigram)

    return frozenset(tokens)
