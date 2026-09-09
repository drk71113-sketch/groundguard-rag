# GroundGuard-RAG

[![CI](https://github.com/drk71113-sketch/groundguard-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/drk71113-sketch/groundguard-rag/actions/workflows/ci.yml)

GroundGuard-RAG is a Python middleware library for checking a generated RAG
answer against the chunks supplied by its caller.  It decomposes the answer
into claims, records claim-evidence edges, produces a versioned audit report,
and can run an explicitly configured, bounded repair loop.

The project does **not** claim to determine world truth or eliminate
hallucinations.  `INSUFFICIENT_EVIDENCE` means only that the supplied chunks
did not settle a claim.

## What is included

- Five verification states: `SUPPORTED`, `CONTRADICTED`,
  `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, and `NOT_CHECKABLE`.
- Immutable claim/evidence/audit models and JSON Schema `1.1.0`.
- Local rule-based claim segmentation and lexical evidence candidate selection.
- A replaceable NLI verifier plus an optional Transformers/DeBERTa-compatible
  backend with explicit label mapping and offline-first loading.
- Claim-level Platt calibration artifacts. Raw logits, softmax values, and
  lexical relevance are never presented as calibrated confidence.
- A bounded heal state machine with round, attempt, timeout, estimated-cost,
  minimum-improvement, loop-hash, and final-reverification guards.
- Sidecar and inline answer views generated from the final audit graph.
- Explicit callback, LangChain, LlamaIndex, and MCP integration adapters.

LettuceDetect or another detector can be integrated through the stable
`Verifier` port.  GroundGuard's focus is the audit lifecycle, bounded repair,
and protocol-level integration rather than rebranding span detection.

## Installation

```powershell
python -m pip install -e ".[test]"
python -m pip install -e ".[nli-transformers]"  # optional local NLI runtime
python -m pip install -e ".[mcp]"               # optional MCP server
```

Python 3.10 or newer is required.  The core package has no runtime dependency.

For a lock-file-based contributor environment, install `uv` and sync only the
extras needed for the task. `--locked` fails instead of silently changing
`uv.lock`:

```powershell
python -m pip install uv
uv sync --locked --extra test --extra mcp
```

## Explicit assembly

GroundGuard never chooses a provider, downloads a checkpoint, or constructs a
retriever implicitly.  A normal application assembles the adapters itself:

```python
from datetime import datetime, timezone

from groundguard_rag import Chunk, GroundGuard, VerificationRequest, VerifyConfig
from groundguard_rag.adapters.decomposition import RuleBasedClaimDecomposer
from groundguard_rag.adapters.evidence_selection import LexicalEvidenceSelector
from groundguard_rag.adapters.verification import (
    NliLabelMapping,
    NliVerifier,
    NliVerifierConfig,
    RuleBasedCheckabilityVerifier,
    TransformersNliBackend,
)
from groundguard_rag.application.verify_service import VerifyService

backend = TransformersNliBackend.from_pretrained(
    "PATH_OR_PINNED_CHECKPOINT",
    label_mapping=NliLabelMapping(
        entailment_index=1,
        contradiction_index=0,
        neutral_index=2,
    ),
    local_files_only=True,
)
verifier = RuleBasedCheckabilityVerifier(
    NliVerifier(
        backend,
        NliVerifierConfig(
            verifier_id="my-nli",
            verifier_revision="PINNED_REVISION",
            decision_threshold=0.7,
        ),
    )
)
service = VerifyService(
    decomposer=RuleBasedClaimDecomposer(),
    selector=LexicalEvidenceSelector(),
    verifier=verifier,
    config=VerifyConfig(),
    model_revision="PINNED_REVISION",
    threshold_version="thresholds-v1",
    clock=lambda: datetime.now(timezone.utc).isoformat(),
)
guard = GroundGuard(verify_service=service)

result = guard.verify(
    VerificationRequest(
        request_id="request-1",
        answer="Paris is in France.",
        chunks=(Chunk("chunk-1", "Paris is the capital of France."),),
    )
)
print(result.views.inline)
print(result.audit_report.to_dict())
```

The label indexes above are only an example; every checkpoint's published
label semantics must be checked explicitly.

## Bounded healing

`HealService` requires an explicit `HealProgressEvaluator` whose output is a
calibrated `P(claim grounded/supported)` backed by independent labeled data.
GroundGuard intentionally provides no identity/softmax/enum shortcut for this
requirement. `CalibratedProgressCallback` lets an application attach its fitted
scorer together with mandatory artifact ID/revision metadata. Retriever and
Rewriter implementations are also explicit.  The
built-in `ConservativeRepairPolicy` prefers retrieval for missing evidence,
then constrained rewrite, and otherwise abstains; it never deletes an
unresolved claim automatically.

Healing returns a candidate answer and audit report.  It never writes the
candidate to the host RAG system, database, or vector store.

## MCP

Create an application factory returning a fully assembled `GroundGuard`, then
run:

```powershell
groundguard-rag-mcp --factory my_application:build_groundguard
```

The server exposes `groundguard_verify` and `groundguard_heal`.  Creating the
server does not start it, and the MCP adapter never creates provider clients.
`groundguard_heal` fails explicitly when the supplied guard has no heal service.

## Local demo

Run the dependency-free pipeline demo directly from a checkout:

```powershell
python examples/quickstart_local.py
```

The demo verifier is an exact-substring test double. It exists to demonstrate
assembly, the five-state audit shape, and answer views; it is not a production
hallucination detector and its output is not evidence of real-model quality.

## Offline JSONL evaluation

The evaluator accepts only a caller-supplied local file and calls `verify`; it
does not download a benchmark or perform open-web fact checking:

```powershell
python -m groundguard_rag.evaluation.cli `
  --factory examples.demo_factory:build_groundguard `
  --dataset examples/sample_benchmark.jsonl `
  --output benchmark-report.json
```

The module form above deliberately places the checkout root on Python's import
path so the repository-only `examples.demo_factory` can be loaded. Installed
applications should use the `groundguard-rag-eval` console script with a
factory module that is part of that application or otherwise importable.

Each non-empty JSONL line contains one request and the expected state of every
claim, in the exact order produced by the configured claim decomposer:

```json
{"case_id":"case-1","request":{"request_id":"request-1","answer":"Paris is in France.","chunks":[{"chunk_id":"chunk-1","text":"Paris is in France."}]},"expected_states":["SUPPORTED"]}
```

The report includes a complete five-state confusion matrix, accuracy, and
macro precision/recall/F1. Macro metrics always average across all five states,
so absent classes contribute zero; inspect per-state support alongside the
aggregate. If and only if verdicts carry traceable calibrated confidence, the
report also includes confidence coverage, Brier score, binary log loss, and
fixed-bin ECE for the event “the predicted verdict state is correct.” The
synthetic sample dataset validates plumbing only.

## Framework adapters

`chunks_from_langchain_documents` and `chunks_from_llamaindex_nodes` convert
host objects without importing either framework.  `LangChainRetrieverAdapter`
and `LlamaIndexRetrieverAdapter` wrap caller-owned retrievers; all external
timeouts, credentials, and provider policies remain the host's responsibility.

## Current baseline limitations

- Rule-based decomposition is sentence/clause segmentation, not guaranteed
  semantic atomicity.
- Lexical selection ranks candidate evidence; it is not entailment.
- The rule-based checkability decorator handles only obvious greetings,
  questions, and subjective statements.
- Calibration quality and end-to-end detection quality must be measured on an
  independently selected, labeled RAG dataset before making effectiveness
  claims.  The repository tests prove software contracts, not real-world model
  accuracy.

## Tests

```powershell
python -m pytest -q -W error
python -m pytest -q -W error --cov=groundguard_rag --cov-branch --cov-report=term-missing
ruff check src tests examples demo_rag
python -m pip check
```

The measured release baseline is enforced at 85% branch-aware coverage. GitHub
Actions runs these checks on Python 3.10, 3.11, and 3.12, checks `uv.lock`, runs
both local demos, and verifies wheel package data and CLI entry points. CI never
downloads model weights.

The cached real-model smoke test is opt-in and never downloads weights:

```powershell
$env:GROUNDGUARD_RUN_REAL_NLI = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m pytest -q -W error tests/integration/test_real_transformers_nli.py
```

## License status

No software license has been selected yet. Do not assume permission to copy,
redistribute, or modify the project until the repository owner adds one.
