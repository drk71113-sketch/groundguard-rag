# GroundGuard-RAG Reference Demo

这个目录是独立的参考 RAG 应用，用来证明 `groundguard-rag` 可以接入真实的检索—生成流程。

计划中的端到端链路：

```text
本地文档
  -> 文档切片与索引
  -> 根据问题检索 chunks
  -> 生成原始 RAG 回答
  -> GroundGuard verify/heal
  -> 原回答、标注视图、审计 JSON 和评测报告
```

## 目录

- `app/`：参考 RAG 与 GroundGuard 接线代码。
- `data/documents/`：允许公开、可复现的小型演示文档。
- `data/evaluation/`：带人工标签的演示问题和期望状态。
- `data/raw/ragtruth/`：锁定版本的 RAGTruth 原始数据；大型 JSONL 文件不提交 Git。
- `results/samples/`：可提交到仓库的脱敏样例输出。
- `tests/`：参考应用的单元测试和端到端测试。

## 当前状态

第一份验证数据选用 RAGTruth。第一阶段直接使用数据集已有的回答、来源资料和人工幻觉标注验证 GroundGuard，不调用生成 API，也不把这一步描述成重新运行了一套 RAG。

当前已完成锁定原始数据、完整性检查，以及 `QA/test/good` 数据到 `VerificationRequest` 的严格转换。下一步才会在固定的 50 条 smoke 子集上运行本地 GroundGuard verify。

## RAGTruth smoke 运行

前提是 pinned DeBERTa checkpoint 已经存在于本机 Hugging Face 缓存。运行时强制离线：

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m demo_rag.app.evaluation.ragtruth_smoke
```

固定 50 条 QA smoke 子集的结果写入 `results/samples/ragtruth_smoke_v1/`。这一步只评测二分类 grounding-risk 检出，不运行 heal，也不代表完整 RAGTruth test split 或生产效果。

首轮结果已经生成：claim-level recall 为 `0.9577`，precision 为 `0.1838`。这说明当前冻结基线召回高但误报严重，下一步是预注册的人工错误分析，不是在测试集上立即调参。
