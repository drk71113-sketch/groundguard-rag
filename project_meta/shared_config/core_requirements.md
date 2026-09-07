# GroundGuard-RAG 统一核心需求

## 1. 项目定位

GroundGuard-RAG 是一个可插拔的 RAG 事实校验与有界自愈中间件。它插入现有 RAG 的生成后阶段，在不绑定具体 RAG 框架、向量库或模型供应商的前提下，接收“RAG 生成回答 + 对应检索 chunks”，输出标准化审计报告和双视图回答。

核心接入形态：

- Python 库；
- MCP Server；
- 面向主流开源 RAG 模板的 adapter/plugin 接口。

## 2. LettuceDetect 2026 能力基线与差异化

LettuceDetect 2026 已具备以下能力，GroundGuard-RAG 不得把这些已有能力单独包装成自身创新：

- RAG、代码和工具输出的 span 级 grounding verification；
- 定位 unsupported、contradictory 或 fabricated 内容；
- 快速本地 encoder detector；
- 生成式 detector、typed spans，以及部分模型提供的错误类型和简短原因；
- 面向多语言和 agentic workflow 的检测能力及 API 接入。

GroundGuard-RAG 应将 LettuceDetect 视为竞品、评测基线或可插拔 `Verifier`，核心差异化聚焦于：

1. **Claim-evidence 审计图**：将回答拆成原子 claim，记录 claim 与证据 chunk/span、判定、分数和决策过程之间的可追踪关系。
2. **有界状态自愈**：使用显式状态机控制补检索、受约束改写、删除、拒答和重新验证；必须具备最大轮数、单 claim 尝试次数、超时、成本预算、最小改善量、状态哈希防回环和确定的停止原因。
3. **插件协议**：通过稳定的 ports/adapters 接口解耦 `ClaimDecomposer`、`EvidenceSelector`、`Verifier`、`Calibrator`、`Retriever`、`Rewriter`、`RepairPolicy` 和 `StopPolicy`。
4. **完整审计生命周期**：输出版本化、机器可读的审计 JSON，记录输入哈希、模型 revision、阈值版本、证据映射、自愈动作、停止原因和性能信息。

## 3. 内部状态基线

内部判定必须保留以下 5 种状态：

1. `SUPPORTED`：当前输入证据支持该 claim。
2. `CONTRADICTED`：当前输入证据明确与该 claim 冲突。
3. `INSUFFICIENT_EVIDENCE`：当前证据不足以支持或反驳该 claim。
4. `CONFLICTING_EVIDENCE`：不同输入证据之间相互冲突。
5. `NOT_CHECKABLE`：意见、寒暄、指令或其他不适合事实核查的内容。

必须明确：`INSUFFICIENT_EVIDENCE` 不等于 claim 在真实世界中为假。GroundGuard-RAG 校验的是相对于输入 chunks 的 groundedness，不是全网或全知世界事实核查。

## 4. `verify` / `heal` 副作用边界

### `verify`

- 只消费回答和 chunks；
- 默认不访问网络，不调用外部 Retriever 或生成模型；
- 不修改输入对象、宿主 RAG 状态、向量库或持久化存储；
- 返回校验结果和审计信息。

### `heal`

- 只有调用方显式注入 Retriever、Rewriter 等 adapter 后，才允许调用外部能力；
- 所有外部调用必须受 Policy、轮数、时间、成本和审计约束；
- 默认只返回修正候选、最终校验结果和审计报告，不直接写回宿主 RAG 或数据库；
- 没有 query 或 Retriever 时，只能依据现有 chunks 改写、删除、标注证据不足或拒答，不得声称已经重新检索。

## 5. 标准输出

1. **标准化审计报告**：版本化 JSON，至少包含 claim、状态、校准置信度、证据 chunk/span、模型与阈值版本、自愈轮次、动作及停止原因。
2. **双视图回答**：
   - `sidecar`：保持原回答不变，单独返回 claim 级标注；
   - `inline`：在展示视图中加入可信度、状态和证据引用。
3. **兼容性**：核心领域层不得直接依赖 LangChain、LlamaIndex、FAISS、Qdrant、具体 LLM SDK 或 MCP 类型；这些依赖只能存在于 adapter 层。

## 6. 禁止事项

- 不引用未经核实的 NIST 报告、所谓“NIST 43% RAG 回答含无根据断言”或“NIST 认证”等表述。
- 不虚构“首个检测器”“首个 RAG 校验器”“完全消除幻觉”等宣传。
- 不把证据不足解释为世界事实为假，也不越界执行全网事实核查。
- 不将原始 NLI softmax 直接包装成统计意义上的可信度；展示可信度前必须经过验证集校准。
- 不隐藏外部模型、检索或写入行为，不通过全局客户端绕过显式 adapter。
- 不用无限递归实现自愈，不允许没有预算、退出条件或最终复核的循环。
- 不重复实现 LettuceDetect 已有检测能力并将其包装成差异化成果；优先通过统一 `Verifier` 接口接入或对比。

## 7. 阶段化开发硬约束

- 严格按照阶段 0 到阶段 8 的顺序开发。
- 每次只完成当前阶段，必须编写单元测试和必要注释。
- 当前阶段完成后停止并等待明确批准，不提前实现、占位实现或顺手加入后续阶段的业务逻辑。

## 8. 阶段 0.3 领域契约收口决定

### 8.1 RunMode 跨字段不变量

- `RunMode.VERIFY`：`repair_rounds` 必须为 0；`repair_actions` 必须为空；`stop_reason` 必须为 `None`/`null`。
- `RunMode.HEAL`：`stop_reason` 必须是非空字符串；允许 `repair_rounds == 0`（例如输入已全部通过校验，未触发任何修复）；`repair_actions` 中每条记录的 `round` 仍必须 `<= repair_rounds`。
- 该不变量必须在领域模型（`AuditReport`）、JSON Schema 和 `semantic_validation` 三层保持一致。

### 8.2 Calibrator 接口收口

- `Calibrator.calibrate` 签名为 `calibrate(verdict: ClaimVerdict) -> float`，返回值是 claim 级校准置信度，必须位于 `[0, 1]`。
- `Verifier` 继续返回 `ClaimVerdict`，负责产出满足一致性约束的 `evidence_assessments` 和 claim 级 `state`；`Calibrator` 可读取 `verdict` 中全部 `EvidenceAssessment.label_scores`、`verifier_id`、`verifier_revision`。
- 本阶段不新增 `VerdictAggregator` port；是否拆分聚合器留待后续需求明确后再决定。
- 只收口接口契约，不实现真实校准算法。

### 8.3 有界自愈改善量量纲

- `RepairActionRecord.score_before`/`score_after` 更名为 `calibrated_confidence_before`/`calibrated_confidence_after`：允许 `None`；非 `None` 时必须是有限数且位于 `[0, 1]`。
- `HealConfig.min_improvement` 必须是有限数且位于 `[0, 1]`（不再是无界正数）。
- 原始 logits/softmax 等分数只保存在 `LabelScores` 或 `ClaimVerdict.raw_score` 中，不得用作跨轮次停止判断的量纲——跨轮次改善量比较必须使用校准后的置信度。

### 8.4 审计完整性收口

- `AuditReport` 构造期直接拒绝：报告内重复的 `claim_id`；`repair_action.claim_id` 引用不存在的 `claim_id`。`semantic_validation` 保留相同检查，用于校验未经 Python 领域构造器产生的外部 JSON 文档。
- `Chunk.metadata` 中数值出现 `NaN`/`Infinity`/`-Infinity` 时必须在领域边界拒绝。
- `LabelScores.score_kind == "probabilities"` 时，除三项均需位于 `[0, 1]` 外，三项之和必须在合理浮点容差内等于 1。

## 9. 阶段 1 verify 编排契约

### 9.1 VerifyService 定位与依赖边界

- 新增 `application` 层，`VerifyService` 把阶段 0 定义的 ports 串成纯本地 `verify` 编排流程，只做编排、契约检查和审计报告构造，不实现任何真实 NLP、检索或模型算法。
- 构造器通过关键字参数显式注入：`ClaimDecomposer`、`EvidenceSelector`、`Verifier`、可选 `Calibrator`、`VerifyConfig`、`model_revision`、`threshold_version`、可注入 `clock`（返回带时区 ISO-8601 字符串，用于生成可测试的 `created_at`）。
- 禁止持有或接收 `Retriever`、`Rewriter`、`RepairPolicy`、`StopPolicy`、`AuditStore`、网络客户端、LLM SDK 客户端、向量库客户端。
- `VerifyConfig.allow_network == True` 时，`VerifyService` 必须在构造期抛出 `ConfigurationError`，不得静默忽略——阶段 1 是严格本地模式。

### 9.2 verify() 编排顺序（严格按序，见 verify_service.py 注释）

1. `ClaimDecomposer.decompose(answer)`；
2. 校验 decomposer 输出：类型必须是 `AtomicClaim`；非空回答不得返回空 claim 集合；`claim_id` 唯一；`start_char`/`end_char` 在 answer 范围内；`answer[start_char:end_char]` 必须严格等于 `claim.text`；claim 顺序按 `start_char` 非递减；claim span 不得相互重叠；
3. 对每个 claim 调用 `EvidenceSelector.select`；
4. 校验 selector 输出：只能是 `EvidenceReference`；`chunk_id` 必须存在于本次 `request.chunks`；span 必须在对应 `chunk.text` 范围内；同一 claim 不允许重复 `chunk_id`/`start_char`/`end_char` 引用；
5. 只使用本次 `request.chunks`，通过 `EvidenceCandidate.from_chunk` 构造候选证据；
6. 调用 `Verifier.verify(claim, evidence_candidates)`；
7. 校验 verifier 输出：必须是 `ClaimVerdict`；`verdict.claim` 必须与当前 claim 完全一致（值相等）；`evidence_assessments` 不得引用未传给 verifier 的证据（不得虚构 chunk/span）；`ClaimVerdict` 自身五态和边一致性由领域模型保证；
8. 若注入了 `Calibrator`：仅在 `verdict.evidence_assessments` 非空时调用 `calibrator.calibrate(verdict)`，通过构造新的不可变 `ClaimVerdict` 写入 `calibrated_confidence`，让领域模型拒绝 bool/NaN/Infinity/超出 `[0,1]` 的返回值，不修改原 verdict；
9. 无 `Calibrator` 或 `evidence_assessments` 为空时：不得把 `raw_score`/softmax 当作置信度，保留已有合法 `calibrated_confidence` 或保持 `None`；
10. 按 claim 原顺序构造 `AuditReport`：`schema_version=SCHEMA_VERSION`；`request_id` 来自请求；`run_mode=RunMode.VERIFY`；`repair_rounds=0`；`repair_actions=()`；`stop_reason=None`；`metrics` 保持 `RunMetrics` 默认空值；`created_at` 来自注入 clock；`model_revision`/`threshold_version` 来自构造参数；
11. 对最终 `report.to_dict()` 执行 `semantic_validation.validate_audit_report_semantics` 作为输出前完整性断言；
12. 返回 `AuditReport`，不保存、不写回、不自动调用 `AuditStore`。

### 9.3 input_hash 规则

- `groundguard_rag.application.input_hashing.compute_input_hash(answer, chunks)` 用标准库 `hashlib.sha256` 计算确定性哈希，输出格式 `sha256:<64位十六进制>`。
- 哈希内容只包含 `verify` 实际消费的数据：`answer`；按输入顺序排列的 `chunks`，每个 chunk 的 `chunk_id`/`text`/`source`/`metadata`。不包含 `query`（`verify` 不消费）；不包含 `request_id`（追踪标识，非被校验内容）。
- `metadata` key 必须稳定排序；chunk 顺序必须保留；使用 UTF-8 和确定性 JSON 序列化（`sort_keys=True`）；严格禁止 NaN/Infinity（`allow_nan=False`）。

### 9.4 Adapter 契约违反异常

- 新增 `application.exceptions.AdapterContractError`，继承 `GroundGuardError`，表示 adapter **返回值**违反契约（返回错误类型、越界 span、引用不存在的 chunk、verdict 与 claim 不匹配、引用未提供的证据、非法置信度等）。
- adapter 自身抛出的业务异常不得被静默吞掉：`VerifyService` 默认不捕获 decomposer/selector/verifier 主动抛出的异常，让其原样传播；仅在校验 `Calibrator` 返回值触发领域校验失败时捕获并用 `raise AdapterContractError(...) from exc` 包装，保留原始 cause。

### 9.5 明确排除

阶段 1 不得持有 `Retriever`/`Rewriter`/`RepairPolicy`/`StopPolicy`/`AuditStore`，不访问网络，不修改输入对象，不写文件/数据库/向量库，不实现真实 claim 拆解、证据检索、NLI 推理或校准算法。`RepairActionRecord.evidence_ids_added`/`evidence_ids_removed` 的精确语义延期到 heal 阶段，本阶段不修改其定义。

### 9.6 完成状态

- 2026-09-02：用户明确授权 Codex 接管核心开发后，Codex 从 Claude Code 留下的阶段 1 半成品继续完成实现和验收。
- `VerifyService`、确定性输入哈希、`AdapterContractError` 及对应 application 测试已完成；另修复了 Verifier 通过修改传入 candidate list 扩大“已提供证据集合”的审计漏洞，允许证据集合现固定于调用前快照。
- 阶段 1 自检进一步收紧了输入和依赖边界：`VerificationRequest` 拒绝空白或非字符串回答；`VerifyService` 构造期校验 config、ports、可选 calibrator 和 clock，并对错误 request 类型给出领域异常。
- 自检修复后全量测试与严格警告测试均为 `467 passed`；阶段 1 验收通过。

## 10. 阶段 2 本地规则式 claim 拆解基线

### 10.1 定位与边界

- 阶段 2 只实现一个可替换的、零第三方依赖的 `RuleBasedClaimDecomposer` adapter，用于让 claim-evidence 审计图具备可运行的本地拆解基线；不实现证据选择、NLI、校准、双视图或 heal。
- 该 adapter 是句子/分句级启发式基线，不得宣传为语义级“真正原子化”，也不得作为 LettuceDetect 已有检测能力的替代或创新声明。包含多个并列事实的复杂句可能仍需要后续更强的可插拔 decomposer。
- 生产代码不得调用 LLM、网络、外部 NLP 包或隐藏全局客户端。

### 10.2 输出契约

- 输入必须是非空、非纯空白字符串；否则抛出 `DomainValidationError`。
- 识别英文和中文终止符 `.?!。！？`，以及分句边界 `;；` 和换行；连续终止符及其后的右引号/右括号必须留在同一个 claim 中。
- 前导、尾随和分隔空白不进入 claim；其他内容不得因“不可核查”而被过滤，意见、寒暄、指令和标点内容同样输出为 `AtomicClaim`。
- 每个 claim 的 `text` 必须严格等于原 answer 的 `answer[start_char:end_char]`；span 按原文顺序、互不重叠。
- `claim_id` 使用确定性的 `claim-{start_char}-{end_char}`，同一输入重复拆解必须得到完全一致的 ID、文本和 span。
- 英文句点至少要避免误切常见小数、域名/邮箱内部点、首字母缩写和内置常见缩写（如 `e.g.`、`i.e.`、`Mr.`、`Dr.`）。这些规则必须有测试，且在注释中诚实记录启发式局限。

### 10.3 阶段限制

- 新 adapter 放在 `groundguard_rag.adapters.decomposition`，不得写入 domain 或 application 编排内部。
- 阶段 1 的 `VerifyService` 只能通过 `ClaimDecomposer` port 使用它，不得依赖其具体类或私有方法。
- 本阶段不新增运行时依赖，不实现 EvidenceSelector、Verifier、Calibrator、Retriever、Rewriter、MCP 或任何自愈逻辑。

### 10.4 完成状态

- 2026-09-02：`RuleBasedClaimDecomposer` 已实现，专项测试覆盖中英文终止符、分号/换行、精确 span、确定性 ID、连续标点与右引号、小数、域名/邮箱、常见缩写、首字母缩写、非事实内容保留及 `VerifyService` port 集成。
- 阶段 2 专项测试为 `17 passed`；全量测试与严格警告测试均为 `484 passed`；无新增运行时依赖或禁止框架。
- 阶段 2 已停止，未进入阶段 3。

## 11. 阶段 3 本地词法 EvidenceSelector 基线

### 11.1 定位与禁止误用

- 阶段 3 只实现零第三方依赖的 `LexicalEvidenceSelector`，用于在本次 `VerificationRequest.chunks` 内为每条 claim 排序和选择候选证据；不访问网络、向量库或外部检索器。
- 词法重叠只代表“候选证据相关性”，绝不代表蕴含、事实支持或校准置信度。不得用该分数生成 `SUPPORTED`/`CONTRADICTED`，不得把关键词重叠包装成事实校验。
- 本阶段不实现 Verifier、NLI、Calibrator、Retriever、Rewriter、heal、双视图或 MCP。

### 11.2 配置和评分契约

- 新增不可变 `LexicalSelectorConfig(top_k, min_score)`：`top_k` 必须是非 bool 的正整数；`min_score` 必须是有限且位于 `[0,1]` 的实数。
- 英文使用大小写归一后的字母数字 token，并过滤一小组内置高频停用词；连续中日韩统一表意文字使用字符 bigram，单字符序列保留单字符 token。该 tokenizer 是可解释基线，不宣称完成词形还原或完整中文分词。
- 对每个 chunk 使用集合型有界分数：`0.8 * claim_token_coverage + 0.2 * Jaccard`。结果必须有限且位于 `[0,1]`；这是 selector relevance score，不是可信度。
- 只返回分数严格大于 0 且 `>= min_score` 的 chunk，按分数降序排列；同分时保持原始 chunk 输入顺序；最多返回 `top_k` 条。
- 阶段 3 只返回 whole-chunk `EvidenceReference`（`start_char/end_char=None`）并写入 `relevance_score`。span 级 evidence selection 留给后续可替换 adapter。

### 11.3 输入、输出与副作用边界

- `select` 直接调用时也必须校验：claim 是 `AtomicClaim`；chunks 是 list 且元素均为 `Chunk`；chunk_id 不重复。契约违反抛出 `DomainValidationError`。
- 标点或停用词导致 claim 无有效 token、chunks 为空、或所有 chunk 零重叠时返回空 list，不得为凑满 top_k 返回无关证据。
- 不修改 claim、chunks list、Chunk 或嵌套 metadata；不读取 chunk metadata/source 参与评分。
- adapter 位于 `groundguard_rag.adapters.evidence_selection`，`VerifyService` 只能通过 `EvidenceSelector` port 使用，不依赖具体实现。

### 11.4 完成状态

- 2026-09-02：`LexicalSelectorConfig` 与 `LexicalEvidenceSelector` 已实现，专项测试覆盖英文排序、top_k、阈值、零重叠、稳定同分顺序、大小写/停用词、CJK bigram、whole-chunk 引用、确定性、不可变输入、配置和输入反例及 `VerifyService` port 集成。
- 集成测试明确让测试 Verifier 对词法候选返回 `INSUFFICIENT_EVIDENCE`，保证 relevance score 没有被冒充成事实支持或校准置信度。
- 阶段 3 专项测试为 `36 passed`；全量测试与严格警告测试均为 `520 passed`；无新增运行时依赖或禁止框架。
- 阶段 3 已停止，未进入阶段 4。

## 12. 阶段 4 三分类 NLI Verifier 核心

### 12.1 定位与模型边界

- 阶段 4 实现 `NliVerifier`：对每条 `EvidenceCandidate` 执行 premise=`evidence.text`、hypothesis=`claim.text` 的三分类 NLI，形成 claim-evidence 审计边并聚合为五态中的四个证据相关状态。
- 模型推理由显式注入的 `NliBackend` 提供；本阶段不绑定或下载 Hugging Face/DeBERTa，不新增 torch/transformers 依赖。真实 DeBERTa backend、模型缓存和设备管理留给后续 provider adapter 阶段。
- `NliVerifier` 不负责 checkability 分类，因此不会自行输出 `NOT_CHECKABLE`；无证据或没有达到支持/反驳阈值时输出 `INSUFFICIENT_EVIDENCE`。
- NLI softmax/probability 是未校准模型分数，只写入 `EvidenceAssessment.label_scores`，绝不得写入 `ClaimVerdict.calibrated_confidence` 或宣传为可信度。

### 12.2 Backend 契约和数值处理

- `NliBackend.predict_batch(pairs: list[tuple[premise, hypothesis]]) -> list[LabelScores]`。`NliVerifier` 对一次 claim 的全部 evidence 只调用一次 backend，并校验返回 list 长度和顺序与输入完全对应，避免真实模型逐 edge 推理。backend 必须已把供应商标签明确映射为 domain 的 `supported`（entailment）、`contradicted`（contradiction）、`insufficient`（neutral）。禁止在 `NliVerifier` 中猜测 `LABEL_0/1/2` 含义。
- backend 只允许返回 `score_kind="logits"` 或 `score_kind="probabilities"`；错误类型或未知 score_kind 抛出 `NliBackendContractError`。backend 主动抛出的异常原样传播。
- logits 使用减最大值的稳定 softmax 转为三项和为 1 的 probabilities；probabilities 直接使用领域模型已经完成的有限性、范围和归一校验。
- `NliVerifierConfig` 必须包含非空 `verifier_id`、非空 `verifier_revision` 和 `[0.5,1]` 内有限的 `decision_threshold`。

### 12.3 Edge 判定与 claim 聚合

- 单 evidence edge：supported probability 达阈值且严格大于 contradicted 时为 `SUPPORTED`；contradicted probability 达阈值且严格大于 supported 时为 `CONTRADICTED`；其他情况为 `INSUFFICIENT_EVIDENCE`。并列最高不得武断选边，归入证据不足。
- 每个输入 candidate 必须恰好产生一个 `EvidenceAssessment`，保持输入顺序并原样保留其 `EvidenceReference`；记录完整三分类 probabilities、verifier ID/revision，不生成伪解释。
- claim 聚合：同时存在 supported 与 contradicted edge -> `CONFLICTING_EVIDENCE`；否则有 contradicted -> `CONTRADICTED`；否则有 supported -> `SUPPORTED`；否则 -> `INSUFFICIENT_EVIDENCE`。
- `ClaimVerdict.raw_score` 和 `calibrated_confidence` 本阶段均保持 `None`，避免定义没有验证依据的 claim 级分数。

### 12.4 输入与副作用边界

- `verify` 直接调用时校验 claim 是 `AtomicClaim`、evidence 是 list 且仅含 `EvidenceCandidate`、reference 不重复；违反时抛出 `DomainValidationError`。
- 不修改 claim、evidence list、candidate、reference 或文本；不访问网络、文件、数据库、向量库或全局模型。
- 实现位于 `groundguard_rag.adapters.verification`；`VerifyService` 只通过既有 `Verifier` port 使用它，不依赖具体类或 backend 私有接口。
- 本阶段不实现真实模型 backend、校准训练、checkability classifier、Retriever/Rewriter、heal、MCP 或双视图。

### 12.5 完成状态

- 2026-09-02：`NliBackend`、`NliBackendContractError`、`NliVerifierConfig` 和 `NliVerifier` 已实现；专项测试覆盖 premise/hypothesis 方向、稳定 softmax、阈值/并列、四种证据聚合状态、完整 edge 分数、顺序/引用保持、backend 契约和异常、输入不可变及本地完整流水线集成。
- 额外使用 200 组范围在 `[-10000,10000]` 的随机极端 logits 验证 softmax 输出有限、三项位于 `[0,1]` 且和为 1；所有 claim 级 calibrated confidence 均保持 `None`。
- 阶段 4 首轮专项测试为 `45 passed`；自检发现原 backend 逐 evidence 调用会让真实模型重复推理，随后把接口收口为每个 claim 一次 `predict_batch`，并补齐返回容器与长度契约测试。最终阶段 4 专项测试为 `52 passed`，全量严格警告回归为 `572 passed`。
- 阶段 4 验收通过后已停止；阶段 5 仅在下一次用户明确要求“检查通过再往下”后开始。

## 13. 阶段 5 可选 Transformers / DeBERTa 兼容 Backend

### 13.1 定位与依赖隔离

- 阶段 5 只实现 `TransformersNliBackend`，把 Hugging Face 三分类 sequence-classification checkpoint 适配到阶段 4 的 `NliBackend.predict_batch`；它兼容 DeBERTa 类 checkpoint，但不硬编码或自动下载某个具体模型。
- `torch`/`transformers` 只存在于 `nli-transformers` 可选依赖组；核心 `dependencies` 继续为空。导入 `groundguard_rag` 或 verification adapter 不得隐式导入这两个包。
- 已显式注入 tokenizer/model 时，推理路径只延迟加载 torch，不得为此额外加载 Transformers checkpoint loader。默认测试不安装 provider 依赖、不下载权重、不访问网络。

### 13.2 模型加载与标签契约

- `from_pretrained` 必须由调用方显式提供非空 `model_name_or_path` 和 `NliLabelMapping`；映射必须把模型列 `0/1/2` 各使用一次，分别声明 entailment、contradiction、neutral，禁止根据 `LABEL_0/1/2` 猜语义。
- 模型必须声明 `config.num_labels == 3`；二分类、四分类、缺失或非法标签数在构造期拒绝。
- 默认 `local_files_only=True`、`trust_remote_code=False`、`device="cpu"`。允许网络解析或执行 checkpoint 自定义代码都必须分别显式 opt-in；`revision` 非空时原样传给 tokenizer/model loader。
- 所有本地策略参数必须在导入重型运行时或解析 checkpoint 前校验，防止非法 batch/device/mapping 配置触发本可避免的加载或外部访问。

### 13.3 批量推理与职责边界

- tokenizer 必须以 `text=premises`、`text_pair=hypotheses` 的成对批量语义接收 premise=`evidence.text`、hypothesis=`claim.text`，启用 padding/truncation，并使用显式 `max_length` 和 PyTorch tensor 输出。
- adapter 按正整数 `batch_size` 分批，使用 `model.eval()` 和 `torch.inference_mode()`，把 tensor 显式移动到调用方选定的 device；输出顺序必须与输入 pair 顺序一致。
- 每个模型输出必须恰好一行三个有限实数。后端只按显式 label mapping 返回 `score_kind="logits"` 的 `LabelScores`；不做 softmax、阈值判定、claim 聚合或校准，不生成 `calibrated_confidence`。

### 13.4 失败与副作用边界

- 直接输入不是 `list[tuple[str, str]]` 时抛 `DomainValidationError`；tokenizer/runtime/model 的返回 shape 或能力违反契约时抛 `NliBackendContractError`；静态配置错误抛 `ConfigurationError`。
- 空 batch 直接返回空 list，不导入 runtime、不调用 tokenizer/model。初始化只把显式注入且由 adapter 持有的模型切到指定 device/eval；不修改 claim、evidence 或宿主 RAG 数据。
- 阶段 5 不实现真实模型选择/下载脚本、校准器、checkability classifier、heal、Retriever/Rewriter、双视图或 MCP。

### 13.5 完成状态

- 2026-09-02：`NliLabelMapping`、`TransformersNliBackend`、延迟可选依赖加载和 `nli-transformers` extra 已实现。
- 自检首轮发现并修复两项边界问题：注入 provider 时不再连带加载 Transformers loader；`from_pretrained` 在任何重型导入/checkpoint 解析前验证 mapping、batch、max length 和 device。
- 阶段 5 专项严格警告测试为 `58 passed`；全量严格警告回归为 `630 passed`；独立进程确认导入核心/verification adapter 后 `torch` 和 `transformers` 均不在 `sys.modules`。
- 首轮完成时环境尚未安装 torch/transformers；用户随后于 2026-09-02 明确批准本机安装和真实模型复验。项目专用 `.venv` 已通过 `pip install -e ".[test,nli-transformers]"` 建立，实装 `torch 2.13.0+cpu`、`transformers 5.16.1`，`pip check` 无破损依赖，原全量测试仍为 `630 passed`。
- 真实验收夹具固定为 `cross-encoder/nli-deberta-v3-xsmall@a150876415327c80daeff35ca6f68f5ed8cf5c24`，Apache-2.0，权重使用 safetensors；模型声明的标签映射为 contradiction=0、entailment=1、neutral=2。该夹具只用于验收，不是产品默认模型或“最佳模型”声明。
- 显式 `local_files_only=False` 的首次加载完成真实 CPU 推理，支持、反驳、中立三个样例的 argmax 分别映射为 `SUPPORTED`、`CONTRADICTED`、`INSUFFICIENT_EVIDENCE`；随后在 `HF_HUB_OFFLINE=1` 且默认 `local_files_only=True` 下从缓存复验通过。
- 新增默认跳过的 `real_model` 集成测试：普通无 provider 环境仍为 `630 passed, 1 skipped`；显式设置 `GROUNDGUARD_RUN_REAL_NLI=1`、`HF_HUB_OFFLINE=1` 后为 `1 passed`，测试本身绝不下载权重。
- Transformers 5.16.1 加载 DeBERTa 时会触发 PyTorch 2.13 对 provider 内部 `torch.jit.script` 的精确弃用警告；只在该 opt-in 集成测试上按消息和模块局部过滤，GroundGuard 自身及其余测试仍使用 `-W error`，未做全局警告屏蔽。
- 阶段 5 已停止，未进入阶段 6。

## 14. 阶段 6 Claim 级置信度校准

### 14.1 定位与方法边界

- 阶段 6 实现一个纯本地、零新增运行时依赖的 `PlattClaimCalibrator`：根据独立、人工标注的 claim 级校准样本，把“当前 verdict 状态的原始强度特征”映射为该完整 claim verdict 正确的经验概率。
- 采用单变量 sigmoid/Platt 后处理作为轻量基线，不实现或宣称“最佳校准器”。非参数 isotonic 需要更多数据且更易在小样本中过拟合，留作可替换后续 adapter；阶段 6 不增加 scikit-learn。
- 校准集必须与 verifier/model 的训练集分离；拟合数据和最终评测数据也应分离。代码只能验证结构、样本 ID 和 provider 一致性，无法自动证明数据集之间不存在语义泄漏，调用方对此负责。
- 原始 softmax/概率只用于提取拟合特征；没有成功拟合的校准 artifact 时不得输出 `calibrated_confidence`，不得提供 identity calibrator 把原始分数改名为可信度。

### 14.2 `claim-state-strength.v1` 特征

- 输入必须是有至少一条 `EvidenceAssessment` 的 `ClaimVerdict`；所有 edge 必须是三分类 `probabilities`，且共享同一个非空 `verifier_id`/`verifier_revision`。logits、混合 provider、无 evidence 的 `NOT_CHECKABLE`/`INSUFFICIENT_EVIDENCE` 均拒绝校准。
- `SUPPORTED`：取所有 supported edge 的最大 `supported` probability。
- `CONTRADICTED`：取所有 contradicted edge 的最大 `contradicted` probability。
- `INSUFFICIENT_EVIDENCE`：取 `1 - max(all edge supported/contradicted probability)`，表达所有 evidence 都未形成强支持/反驳的最弱边界。
- `CONFLICTING_EVIDENCE`：取 `min(max supported-edge supported probability, max contradicted-edge contradicted probability)`，表达冲突两侧中较弱一侧的强度。
- 该标量仍是未校准 feature，不得直接展示为 claim confidence。拟合时只在数值 logit 变换中按显式 epsilon 裁剪，最终 sigmoid 输出必须有限且位于 `[0,1]`。

### 14.3 校准样本、拟合与 artifact

- `CalibrationSample` 至少包含全局唯一的非空 `sample_id`、未校准 `ClaimVerdict`、人工 `expected_state`；`is_correct` 由 `verdict.state == expected_state` 得出，不接受调用方另传易矛盾的布尔标签。
- `PlattFitConfig` 显式限制 `min_samples`、`max_iterations`、`tolerance`、`l2` 和 `probability_epsilon`；校准集必须达到最小样本数、同时含正确/错误 verdict、至少两个不同 feature，并且 sample_id 不重复。
- 拟合最小化二元 log loss，slope 强制非负以保持“原始状态强度越高，校准置信度不应越低”的单调性；使用确定性 Newton/回溯求解，未收敛或数值退化必须显式失败，不静默产出 artifact。
- `PlattCalibrationArtifact` 必须记录 schema/feature version、calibrator ID、verifier ID/revision、`threshold_version`、slope/intercept、样本计数与正确/错误计数；同一 verifier 改变判定阈值后不得复用旧 artifact。revision 由规范化 artifact 内容和拟合样本特征/标签确定性 SHA-256 计算，不使用时间戳冒充可复现 revision。
- artifact 支持标准 JSON-compatible `to_dict`/严格 `from_dict`；未知字段、非法有限性/范围、revision 与内容不一致都拒绝。

### 14.4 评测与审计

- 提供显式 `evaluate_calibrator`，对调用方提供的带标签评测样本分别计算 raw/calibrated 的 count、accuracy、mean confidence、binary log loss、Brier score 和固定 bins ECE。Brier/log loss 同时受区分能力和不确定性影响，不得只凭单一指标宣称校准改善。
- `AuditReport` 增加可空、成对出现的 `calibrator_id`/`calibrator_revision`；任何 verdict 含非空 `calibrated_confidence` 时两者必须存在。JSON Schema、Python 领域模型和 foreign-JSON semantic validation 保持一致。
- `VerifyService` 注入 Calibrator 时必须同时显式注入非空 calibration ID/revision。未注入 Calibrator 时通常保持两者为空；但为兼容阶段 1 已锁定的“Verifier 可返回合法的预校准 verdict”路径，调用方可成对提供上游 calibrator ID/revision，报告必须据此追踪已有 confidence。`Calibrator` port 提供向后兼容的可选自描述属性；具体 `PlattClaimCalibrator` 暴露 artifact ID/revision/threshold version，`VerifyService` 必须拒绝显式 wiring 与自描述值不一致。

### 14.5 阶段限制

- 阶段 6 不修改 verifier 判定状态，不训练/微调 NLI 模型，不下载数据，不访问网络或文件，不实现 checkability classifier、Retriever/Rewriter、heal、MCP 或双视图。
- 本阶段只完成 calibration adapter、artifact、评测函数、审计可追踪字段及单元测试；真实校准效果必须等用户选定并标注独立数据集后才能报告。

### 14.6 完成状态

- 2026-09-02：`CalibrationSample`、`claim-state-strength.v1`、`PlattFitConfig`、确定性单调 sigmoid 拟合、内容寻址 `PlattCalibrationArtifact`、`PlattClaimCalibrator`、raw/calibrated 评测指标和 JSON round-trip 已实现；未新增运行时依赖。
- `AuditReport`、audit JSON Schema、foreign-JSON semantic validation 和 `VerifyService` 已加入成对 calibration ID/revision 追踪；保留阶段 1 的上游预校准 verdict 兼容路径，但没有元数据的 calibrated confidence 会被拒绝。
- 自检发现并修复两个设计问题：一是不能因新元数据规则破坏合法的上游预校准 verdict；二是 artifact 必须绑定 `threshold_version`，并通过 Calibrator 自描述属性拒绝 ID/revision/threshold wiring 不一致。
- 阶段 6 专项严格警告测试为 `43 passed`；默认全量严格警告回归为 `683 passed, 1 skipped`，其中唯一 skip 是显式 opt-in 的真实模型测试；在 `GROUNDGUARD_RUN_REAL_NLI=1`、`HF_HUB_OFFLINE=1` 下该真实 DeBERTa 测试仍为 `1 passed`。
- `pip check`、compileall、核心导入不加载 torch/transformers、阶段边界/禁止依赖扫描均通过。合成数据测试只证明拟合、单调性、指标和审计 wiring 实现正确，不作为真实 RAG 数据上的校准效果声明。
- 阶段 6 已停止，未进入阶段 7。

## 15. 阶段 7 有界自愈状态机

### 15.1 先修正改善量语义

- 阶段 6 的 `ClaimVerdict.calibrated_confidence` 表示“当前五态 verdict 正确”的经验概率；它不是 `SUPPORTED` 的概率。比如高置信度的 `CONTRADICTED` 是可靠的反驳结论，不能因为数值高就解释为更接近修复目标。
- 因此阶段 7 不直接用 `ClaimVerdict.calibrated_confidence` 做自愈改善量。新增可插拔 `HealProgressEvaluator`，输出经独立验证集校准的 `P(claim grounded/supported)`；`RepairActionRecord.calibrated_confidence_before/after` 在 HEAL 记录中明确承载这个自愈进度概率，并用独立的 progress calibrator ID/revision 追踪来源。
- 禁止把 NLI raw logits/softmax、词法相关性、五态枚举序号或阶段 6 的“verdict correctness confidence”冒充自愈进度。没有合法 progress evaluator 时不得运行阶段 7 `HealService`。

### 15.2 状态机与硬边界

- `HealService` 先调用纯本地 `VerifyService` 得到初始快照；每轮最多处理一个 claim，动作完成后重新校验候选状态，只有满足提交条件才替换当前候选。最终输出所对应的 answer/chunks 必须已有一次完整复核，不允许返回未校验候选。
- 内建硬停止条件不可被插件放宽：`max_rounds`、`max_attempts_per_claim`、`timeout_seconds`、`cost_budget`、`min_improvement`、状态哈希回环检测；`StopPolicy` 只能增加提前停止条件。
- 超时为协作式边界：在策略调用和外部动作前后检查；Python 无法安全强杀已经进入的任意第三方同步调用，因此 adapter 自身仍应配置供应商超时。
- 阶段 0 为延期决策而保留的 `Mapping[str, Any]` policy state 在本阶段收口为不可变 `HealPolicyState`；轮次、单 claim 尝试数、预算、能力可用性、当前五态和硬限制都有稳定类型，插件不得依赖拼写脆弱的自由字典。
- `RepairPolicy` 返回结构化 `RepairDecision`，必须显式给出动作和该轮预计成本。成本在调用 Retriever/Rewriter 前预检；预计超预算时不得发起外部调用。审计必须明确这是 caller/plugin 提供的 estimated cost，不冒充实际账单。
- 每轮只允许 `retrieve`、`rewrite`、`delete`、`abstain`、`accept` 五种动作。`retrieve` 只有请求包含非空 query 且显式注入 Retriever 时可执行；`rewrite` 只有显式注入 Rewriter 时可执行；无 adapter 时不得伪造已检索或已改写。

### 15.3 候选提交、回滚与 claim 血缘

- `retrieve` 返回的 chunks 必须类型正确、批内 ID 唯一且不得与当前 chunks 冲突；`rewrite` 必须返回非空、无首尾空白且重新拆解后仍恰好对应一个原 claim；`delete` 不得生成空白回答。违反时候选不提交，并以确定停止原因结束。
- `retrieve`/`rewrite` 候选只有在 progress evaluator 的 `after - before >= min_improvement` 时提交；否则回滚到此前已验证状态并停止。`delete` 是显式移除目标 claim，不伪造删除后的 confidence；`abstain`/`accept` 是有审计的终端策略决定，不修改宿主回答。
- 状态哈希覆盖 candidate answer 和有序 chunks。命中任何已提交历史状态时，候选不提交并以 `state_loop` 停止。
- HEAL 审计同时保存初始 verdict 快照和最终 verdict；存活 claim 的稳定 ID 跨轮保留，删除动作允许最终 verdict 中不存在原 claim，但必须能在初始快照中解析。每个 action 记录是否 committed、动作前后 claim ID/text、answer hash、context chunk 增删、进度概率、耗时、预计成本及失败/停止原因；claim-evidence 边变化由初始/最终 verdict 图表达，不能把“claim 改写后 selector 不再选某 chunk”误记成 context 已删除该 chunk。

### 15.4 副作用与阶段限制

- 输入 `VerificationRequest`、Chunk、verdict 和宿主状态始终不修改；所有候选均由新的不可变对象构造。默认只返回 `HealResult(candidate_answer, candidate_chunks, final_audit_report, terminal claim IDs)`，不写文件、数据库、向量库或宿主 RAG。
- 初始 verify 失败继续原样抛出；进入 heal 后的策略/进度/外部 adapter 失败必须转为可审计的确定停止结果，不得提交部分候选，也不得记录异常消息中的潜在敏感数据，只记录异常类型。
- 阶段 7 只实现状态机、领域/审计契约和 fake-adapter 单元测试；不实现真实 Retriever、真实生成式 Rewriter、双视图、MCP Server、LangChain/LlamaIndex adapter 或写回能力，这些不得提前进入阶段 8。

### 15.5 完成状态

- 2026-09-02：`RepairAction`、`HealStopReason`、`RepairDecision`、不可变 `HealPolicyState`、独立 `HealProgressEvaluator` port、`HealResult` 与 `HealService` 已实现；每轮最多一个 claim/action，内建轮数、单 claim 尝试、协作式超时、预计成本、最小 groundedness 改善量、状态哈希和最终已验证状态约束。
- `retrieve`、`rewrite`、`delete`、`abstain`、`accept` 五种路径均有 fake-adapter 单元测试；覆盖无 query/无 adapter、调用前成本拒绝、累计成本、调用后超时回滚、回环、改善不足回滚、claim 分裂、chunk ID 冲突、adapter 异常脱敏、删除血缘、终端策略、最大轮次/尝试数、只读 policy state 和输入无副作用。
- 审计报告加入初始 verdict 图、最终 verdict 图、progress calibrator ID/revision、commit/rollback 与 claim before/after 血缘；由于动作记录形状发生不兼容变化，audit `schema_version` 从 `1.0.0` 正确升级到 `1.1.0`（仍在 `audit_report.v1` 主版本文件内）。
- 自检修复四项设计问题：不得用“当前五态 verdict 正确概率”替代 groundedness 进度；不得继续用自由字典作为稳定插件协议；rewrite/delete 不得把 selector 边变化误报为 context chunk 已删除；未验证或失败候选不得伪造 `state_after`。
- 阶段 7 新增核心专项测试为 `49 passed`；默认全量严格警告回归为 `741 passed, 1 skipped`，唯一 skip 仍是 opt-in 真实模型测试。`pip check`、compileall、禁止宣传/禁止核心框架依赖扫描、核心导入不加载 torch/transformers 均通过。
- 在 `GROUNDGUARD_RUN_REAL_NLI=1`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 下，缓存的真实 DeBERTa NLI 集成回归为 `1 passed`，未访问网络。阶段 7 已停止，未进入阶段 8。

## 16. 阶段 8 公开集成、双视图与可发布交付

### 16.1 Python 公开 API

- 包根导出稳定的 `GroundGuard`、`verify`、`heal` 和输出类型；底层 `VerifyService`/`HealService` 仍可用，但普通接入方不必自己拼接展示层。
- 公开门面只接收显式注入的 service/adapter，不创建隐藏的网络、LLM、向量库或全局客户端。
- `verify` 仍为纯本地、无写回操作；`heal` 未配置 `HealService` 时必须明确拒绝，不得情况不明地降级为 verify。

### 16.2 双视图与审计图

- `sidecar` 保留传入回答的字符串原样，并输出 claim ID/span/五态/校准置信度/证据引用。
- `inline` 只在展示副本中按 claim span 插入状态标注；不修改原回答、`AuditReport` 或宿主状态。
- 没有 `calibrated_confidence` 时必须显示为未提供，不得退回使用 raw score、NLI softmax 或词法相关性。
- 审计图是对既有版本化 `AuditReport` 中 claim-evidence edge 的无损视图，不发明第二份不兼容的事实源。

### 16.3 默认安全插件与自愈组装

- 提供保守、确定性的 `RepairPolicy` 和“仅依赖内建硬边界”的 `StopPolicy`；默认政策不自动删除 claim，无可用能力时选择 abstain。
- 提供显式 callback/host adapter，用于把调用方已有 Retriever/Rewriter 接入稳定 port；adapter 必须做输入输出契约校验，不得隐藏客户端或写回。
- groundedness progress 仍必须由调用方注入带稳定 ID/revision、基于独立标注数据的 `HealProgressEvaluator`；不提供伪校准的 identity/enum/softmax 默认实现。
- 增加高精度、可替换的 checkability 装饰 Verifier 基线，只对明确的寒暄、致谢、主观偏好和问句输出 `NOT_CHECKABLE`；其余内容交给已注入 Verifier，不宣称语义级完备性。

### 16.4 MCP 与 RAG 框架边界

- MCP Server 是可选 adapter，延迟导入 MCP SDK；未安装 extra 时，导入 GroundGuard 核心包不得失败。
- MCP 工具只调用已显式组装的 `GroundGuard`，不自动下载模型、创建外部 Retriever 或写回宿主。
- LangChain/LlamaIndex adapter 仅负责将宿主 Document/Node 转换为不可变 `Chunk`，以及显式包装宿主 retriever；不在 domain/application 层引入对方类型。

### 16.5 发布与验收

- `pyproject.toml` 必须具备可复现的 build backend、可选 MCP extra 和包数据声明；发布 wheel 必须包含 audit JSON Schema。
- README 必须说明 verify/heal 副作用边界、五态语义、校准前提、MCP 组装方式和当前基线局限。
- 测试覆盖公开 API、双视图、默认 policy、callback 契约、MCP 工具注册/调用、LangChain/LlamaIndex 转换、wheel 内 schema 和端到端的 verify/heal 流程。
- 阶段 8 不引入未核实 NIST/43% 数据、“首个”或“完全消除幻觉”表述；不以合成测试代替真实 RAG 数据集效果声明。

### 16.6 完成状态

- 2026-09-03：阶段 8 已完成 `GroundGuard`/`verify`/`heal` 公开 API、sidecar/inline 双视图、规则式 checkability Verifier 装饰器、保守默认 Repair/Stop policy、显式 Retriever/Rewriter/校准 groundedness callback adapter、LangChain/LlamaIndex 转换与 retriever adapter、MCP Server 及 `groundguard-rag-mcp` 入口。
- 包版本升为 `0.1.0`，加入 setuptools build backend、MCP 可选 extra、schema package-data 与 README；发布 wheel 已成功构建并在隔离 target 中导入，确认包含 `audit_report.v1.schema.json` 和 MCP entry point。
- 阶段 8 新增专项测试 `36 passed`；默认全量严格警告回归为 `777 passed, 1 skipped`（唯一 skip 是显式 opt-in 的真实模型测试）；缓存 DeBERTa 离线集成测试为 `1 passed`。
- 本机实装 MCP Python SDK `2.1.1`，完成真实工具 schema 注册和 `call_tool` 调用；`pip check`、compileall、阶段 8 新代码 Ruff 检查、可选依赖延迟导入、核心依赖边界和禁止宣传扫描均通过。
- 阶段 0 至阶段 8 的工程交付已全部完成。真实 RAG 数据集上的效果和 groundedness 校准质量仍必须在选定、隔离的标注数据上另行评测；未完成该评测前不做准确率、校准改善或“生产有效”声明。
