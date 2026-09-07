"""Opt-in smoke test for one pinned, locally cached NLI checkpoint.

This is an acceptance fixture, not a product default or model recommendation.
The test always uses ``local_files_only=True`` and therefore can never download
weights.  Set ``GROUNDGUARD_RUN_REAL_NLI=1`` after separately caching the pinned
revision to run it.
"""

from __future__ import annotations

import os

import pytest

from groundguard_rag.adapters.verification import (
    NliLabelMapping,
    NliVerifier,
    NliVerifierConfig,
    TransformersNliBackend,
)
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.models import (
    AtomicClaim,
    EvidenceCandidate,
    EvidenceReference,
)


MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
MODEL_REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
# The pinned checkpoint config/model card declares contradiction=0,
# entailment=1, neutral=2.  Keep this explicit; never infer LABEL_n semantics.
LABEL_MAPPING = NliLabelMapping(
    entailment_index=1,
    contradiction_index=0,
    neutral_index=2,
)


@pytest.mark.real_model
@pytest.mark.filterwarnings(
    # Transformers 5.16.1 imports DeBERTa code decorated with torch.jit.script;
    # torch 2.13 emits this exact provider-internal deprecation warning.  Keep
    # the exception local and exact so all GroundGuard warnings remain errors.
    r"ignore:`torch\.jit\.script` is deprecated\..*:DeprecationWarning:torch\.jit\._script"
)
@pytest.mark.skipif(
    os.getenv("GROUNDGUARD_RUN_REAL_NLI") != "1",
    reason="set GROUNDGUARD_RUN_REAL_NLI=1 to use the cached checkpoint",
)
def test_pinned_checkpoint_handles_three_nli_states_without_confidence():
    backend = TransformersNliBackend.from_pretrained(
        MODEL_ID,
        label_mapping=LABEL_MAPPING,
        revision=MODEL_REVISION,
        # Hard boundary: the test is repeatable and must never access the Hub.
        local_files_only=True,
        trust_remote_code=False,
        batch_size=3,
        max_length=128,
        device="cpu",
    )
    verifier = NliVerifier(
        backend,
        NliVerifierConfig(
            verifier_id="transformers-nli-real-smoke",
            verifier_revision=f"{MODEL_ID}@{MODEL_REVISION}",
            decision_threshold=0.8,
        ),
    )
    premise = "A man is eating pizza."
    cases = (
        ("A man is eating food.", VerificationState.SUPPORTED),
        ("No person is eating anything.", VerificationState.CONTRADICTED),
        ("The man is wearing a blue hat.", VerificationState.INSUFFICIENT_EVIDENCE),
    )

    for index, (hypothesis, expected_state) in enumerate(cases):
        claim = AtomicClaim(
            claim_id=f"claim-{index}",
            text=hypothesis,
            start_char=0,
            end_char=len(hypothesis),
        )
        evidence = EvidenceCandidate(
            reference=EvidenceReference(chunk_id=f"chunk-{index}"),
            text=premise,
        )

        verdict = verifier.verify(claim, [evidence])

        assert verdict.state is expected_state
        assert verdict.raw_score is None
        assert verdict.calibrated_confidence is None
        assert len(verdict.evidence_assessments) == 1
        assert (
            verdict.evidence_assessments[0].label_scores.score_kind
            == "probabilities"
        )
