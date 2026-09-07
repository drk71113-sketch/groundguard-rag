# RAGTruth QA smoke v1

This is the first frozen, API-free evaluation of GroundGuard-RAG against human-annotated RAGTruth QA responses.

## Scope

- 50 deterministic `QA/test/good` responses.
- 25 responses with at least one RAGTruth hallucination span and 25 without one.
- Local `cross-encoder/nli-deberta-v3-xsmall` at pinned revision `a150876415327c80daeff35ca6f68f5ed8cf5c24`.
- Decision threshold fixed before evaluation at `0.8`.
- No external API, network access, healing, calibration artifact, training, or test-label tuning.

RAGTruth supplies hallucination spans rather than GroundGuard five-state gold labels. Metrics therefore describe binary grounding-risk detection only. A GroundGuard claim is gold-risk when its span intersects an annotated RAGTruth span; predicted-risk states are `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, and `CONFLICTING_EVIDENCE`.

## Results

| Unit | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Claim (495) | 0.1838 | 0.9577 | 0.3084 | 0.3838 |
| Response (50) | 0.5208 | 1.0000 | 0.6849 | 0.5400 |

Claim outcomes: 68 true positives, 302 false positives, 3 false negatives, and 122 true negatives.

## Honest interpretation

The frozen baseline detects nearly every annotated risk in this small subset, but it over-flags unsupported or contradictory claims. It is suitable for demonstrating the integration and exposing failure modes; it is not evidence of production-ready detection quality.

Of the 302 claim-level false positives, 238 were `INSUFFICIENT_EVIDENCE`, 49 were `CONTRADICTED`, and 15 were `CONFLICTING_EVIDENCE`. Forty-four false positives had zero selected evidence edges; the remainder require manual error analysis across decomposition, evidence selection, truncation, and NLI behavior. These are diagnostic hypotheses, not yet confirmed root causes.

## Files

- `summary.json`: aggregate configuration and metrics.
- `cases.jsonl`: all 50 requests, annotations, claim comparisons, views, and audit reports.
