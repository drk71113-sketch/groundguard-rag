# RAGTruth source manifest

## Upstream

- Repository: `https://github.com/ParticleMedia/RAGTruth`
- Pinned commit: `c103204b9ce28d6bbad859304bf30de72b8ed8fe`
- Retrieved: `2026-09-05`
- Repository license identifier reported by GitHub: `MIT`
- Local license copy: `LICENSE.RAGTruth`

Raw JSONL files are intentionally ignored by the parent repository. Reproduce them from the pinned raw URLs below rather than committing large third-party data files.

## Files

| File | Bytes | SHA-256 | Pinned raw URL |
| --- | ---: | --- | --- |
| `response.jsonl` | 21,458,735 | `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073` | `https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset/response.jsonl` |
| `source_info.jsonl` | 15,117,971 | `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b` | `https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset/source_info.jsonl` |
| `LICENSE.RAGTruth` | 1,071 | `b7fd7d6bdfe0cbba63c63a310914beb4a4acb8bf08da73849219f45385f5b244` | `https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/LICENSE` |

## Local integrity check

- Source rows: 2,965 (`QA`: 989, `Summary`: 943, `Data2txt`: 1,033).
- Response rows: 17,790 (`train`: 15,090, `test`: 2,700).
- QA responses: 5,934; QA responses in the official test split: 900.
- Missing response-to-source links: 0.
- Annotated hallucination spans: 14,289.
- Span text/offset mismatches: 0.

These checks establish file integrity and relational consistency only. They do not establish GroundGuard detection quality.
