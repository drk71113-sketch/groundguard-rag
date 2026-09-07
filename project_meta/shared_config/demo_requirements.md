# GroundGuard-RAG 参考应用与实证阶段

## 1. 目标

在不修改 GroundGuard-RAG 核心语义边界的前提下，建设一个独立的参考 RAG 应用，形成可复现的端到端证据链：

`本地文档 -> 检索 chunks -> 生成回答 -> GroundGuard verify/heal -> 展示结果 -> 离线评测报告`

该阶段用于证明中间件能够被真实 RAG 调用，不以演示数据或单次成功案例宣称生产效果。

## 2. 目录边界

- 根目录 `demo_rag/`：参考 RAG 应用、演示数据、端到端测试和示例结果。
- `src/groundguard_rag/`：中间件核心包；只有发现经过测试确认的核心缺陷时才修改。
- `demo_rag/results/samples/`：允许提交的脱敏、可复现样例结果。
- 本地 API Key、原始私有文档、模型缓存和临时运行结果不得提交到 Git。

## 3. 实施顺序

1. 先建立目录和数据契约，不写 RAG 逻辑。
2. 实现无需外部 API 的确定性离线演示，证明输入输出接线正确。
3. 再选择一个轻量参考 RAG 技术栈，并用本地小型公开语料跑通检索。
4. 最后通过环境变量接入用户选定的生成 API；不得把密钥写入代码或样例结果。
5. 建立五态测试集和端到端评测，记录准确率、宏平均 F1、各状态召回率、延迟及失败案例。
6. 生成一键运行说明、示例审计 JSON、sidecar/inline 展示和面试演示脚本。

## 4. 硬性规则

- 不直接复制未知 GitHub RAG 项目的整套代码；优先用最小参考实现或明确许可证的依赖。
- 不把“能运行”写成“效果优秀”；真实效果只能由隔离标注数据集评测支持。
- 不做开放网络事实核查；GroundGuard 只核验回答和本次检索 chunks 的一致性。
- verify/heal 保持无宿主写回副作用；修复结果仍由参考应用显式决定是否采用。
- 不引用未核实 NIST/43% 数据，不宣称“首个检测器”或“完全消除幻觉”。
- 每个小步骤完成后先运行单元测试和端到端检查，通过后再进入下一步。

## 5. 当前状态

- 2026-09-05：参考应用骨架和本需求文档已创建。
- 2026-09-05：数据阶段 1 已锁定 RAGTruth 官方仓库 commit `c103204b9ce28d6bbad859304bf30de72b8ed8fe`，下载 `response.jsonl`、`source_info.jsonl` 和许可证副本，并记录 SHA-256、规模与关联完整性检查结果。
- 当前尚未转换数据、运行 GroundGuard、实现 RAG、接入外部 API 或生成效果指标。下一步只设计并测试 RAGTruth QA test 子集到 GroundGuard 输入格式的转换器。

## 6. 数据阶段 2：RAGTruth QA 转换契约

- 只接收 `task_type == "QA"` 的来源，以及官方 `split == "test"`、`quality == "good"` 的回答；拒答质量问题和截断回答不进入首轮评测。
- RAGTruth QA 的 `source_info.passages` 必须解析为连续编号的 `passage N:` 三段文本；每段转换为一个不可变 `Chunk`，保留 upstream source ID、passage index 和来源数据集元数据。
- `response` 必须原样成为 `VerificationRequest.answer`，question 成为 `query`；人工 span 标注独立保留，不注入 GroundGuard 输入或提示，防止标签泄漏。
- 转换器严格校验 JSONL、唯一 ID、source 关联、字段类型、标注 offset/text 一致性；非法数据必须带文件行号失败，不静默修复。
- 提供不依赖输入顺序的确定性 smoke subset 选择：按固定 salt 和 response ID 哈希，在有/无人工幻觉两类中各取固定数量。该子集只用于接线和初步探索，不冒充完整 test split 指标。
- 本阶段只实现转换、选择和单元测试；不得加载 NLI 模型、调用 GroundGuard、接入 API 或计算效果指标。

### 6.1 完成状态

- 2026-09-05：数据阶段 2 已完成严格 RAGTruth 转换器、独立人工 span 数据模型和固定哈希 smoke subset 选择器。
- 完整原始文件转换得到 875 条 `QA/test/good` 样本：160 条含人工幻觉 span，715 条无人工幻觉 span；每条请求恰好包含 3 个 passage chunks。
- 固定 salt 的首轮 smoke subset 为 50 条（25 条含标注、25 条无标注），仅用于下一阶段接线验证，不作为官方 test split 效果结论。
- 阶段 2 专项测试 `12 passed`，全量严格回归 `799 passed, 1 skipped`，Ruff 检查通过。
- 已额外确认本机缓存 `cross-encoder/nli-deberta-v3-xsmall`；在禁止联网模式下真实 NLI 冒烟测试 `1 passed`。下一阶段才允许把 50 条 smoke 请求交给 GroundGuard verify，并设计 span 对齐评测，不运行 heal 或生成 API。

## 7. 数据阶段 3：RAGTruth 本地真实 verify smoke

- 输入固定为阶段 2 的 50 条 smoke subset（25 条含人工 span、25 条无人工 span）；禁止看结果后替换样本。
- 使用本机缓存并锁定 revision 的 `cross-encoder/nli-deberta-v3-xsmall`，强制 `local_files_only=True`、CPU、`trust_remote_code=False`，不访问外部 API。
- 使用现有规则式 claim decomposer、规则式 checkability decorator、词法 evidence selector和 NLI verifier；首轮 threshold 固定为既有真实模型验收值 `0.8`，不得用 RAGTruth test 标签调参。
- 本阶段没有独立 calibration artifact，因此输出的 calibrated confidence 必须保持未提供，不把 NLI softmax 冒充置信度。
- 二分类对齐单位为 GroundGuard claim：claim span 与任一人工 hallucination span 相交即为 gold risk；GroundGuard 的 `CONTRADICTED`、`INSUFFICIENT_EVIDENCE`、`CONFLICTING_EVIDENCE` 记为 predicted risk，`SUPPORTED`、`NOT_CHECKABLE` 记为 predicted non-risk。
- 同时报告 claim-level 和 response-level TP/FP/FN/TN、precision、recall、F1、accuracy；这些指标只描述固定 smoke subset 的风险检出，不代表完整 RAGTruth test split、五态准确率或生产效果。
- `implicit_true=True` 的 RAGTruth span 仍按 gold risk 处理，因为 GroundGuard 核验的是 supplied chunks grounding，而不是开放世界真假。
- 保存结构化 summary 和逐案例 JSONL，包含人工标注、claim 状态、对齐结果与审计报告；不得把异常栈或本地绝对模型缓存路径写入公开样例。
- 本阶段不运行 heal、不重新检索、不调用生成 API、不训练或调参。

### 7.1 完成状态

- 2026-09-05：固定 50 条 smoke subset 已在强制离线、本机 CPU 和 pinned DeBERTa revision 下真实运行两次，指标与五态计数完全一致。
- claim-level：TP=68、FP=302、FN=3、TN=122，precision=0.1838、recall=0.9577、F1=0.3084、accuracy=0.3838。
- response-level：TP=25、FP=23、FN=0、TN=2，precision=0.5208、recall=1.0000、F1=0.6849、accuracy=0.5400。
- 495 个 claim verdict 中：`SUPPORTED` 125、`INSUFFICIENT_EVIDENCE` 295、`CONTRADICTED` 59、`CONFLICTING_EVIDENCE` 16；无 `NOT_CHECKABLE`。
- 结果显示冻结基线具有高召回、低精度特征，不能作为生产可用性或完整五态质量证明；禁止隐藏误报或用该 test subset 直接调阈值。
- 已保存 `summary.json`、50 条完整 `cases.jsonl` 和诚实结果说明；产物检查确认输入回答/sidecar 未变、全部为 VERIFY、无伪 calibrated confidence、无 API Key 或本机绝对模型路径。
- 阶段 3 演示专项测试 `24 passed`；最终全量严格回归 `811 passed, 1 skipped`，Ruff 检查通过。唯一额外警告来自 PyTorch 导入 DeBERTa 的已知 `torch.jit.script` 弃用消息，严格验收仅按精确消息和模块局部豁免。
- 下一阶段必须先做固定规则的人工错误分析，区分 claim 拆解、证据选择、输入截断、NLI 和标签粒度不匹配；分析完成前不得优化实现。

## 8. 数据阶段 4：冻结结果的错误分析

- 输入只能是阶段 3 已冻结的 `ragtruth_smoke_v1/cases.jsonl`；本阶段不得重新抽样、运行推理、修改 threshold 或覆盖阶段 3 指标。
- 自动诊断只报告可观察信号，不把相关性写成已证实根因：零 evidence edge、NLI 各标签最高分与阈值距离、selected pair 是否超过模型 256-token 上限、claim 长度/多分句启发式、人工 span 占 claim 比例、RAGTruth label type/implicit_true/due_to_null。
- 固定人工复核队列按 response ID + claim ID 哈希选择：全部 FN；FP 按 `INSUFFICIENT_EVIDENCE` 6 条、`CONTRADICTED` 3 条、`CONFLICTING_EVIDENCE` 3 条；另取 TP/TN 各 4 条作对照。任何组数量不足则全部保留。
- 人工复核只允许记录：主要观察层、置信等级、可复核说明；`unknown` 是合法结论。不能仅凭一条样本宣称系统性根因。
- 输出 aggregate diagnostics、固定 review queue 和人工复核记录；报告各信号计数与交叉分布，明确区分自动信号和人工判断。
- 本阶段不实现任何优化，不新增模型或 API，不运行 heal；只有完成报告后才能提出下一阶段的改进假设和独立验证方案。
