# GroundGuard-RAG 老师审查指南

## 1. 审查目标与边界

本次希望审查的是 GroundGuard-RAG `0.1.0` 工程候选版的架构、领域契约、代码质量、测试充分性和可发布性。

请不要仅凭仓库内的合成数据评价真实检测准确率。当前 demo verifier 是管道演示用 test double；项目尚未在独立真实业务数据集上形成可对外宣传的效果结论，也不声称执行全网事实核查、完全消除幻觉或属于“首个”检测器。

老师可以直接在独立审查副本中运行测试或写评语，但建议不要直接修改学生的原始工作目录。请优先把意见写入 `TEACHER_FEEDBACK_TEMPLATE.md` 的副本中，保留文件路径和行号。

## 2. 建议阅读顺序

### 第一步：先理解项目要解决什么

1. `README.md`：产品边界、五态、安装、示例、MCP、评测方式和局限。
2. `project_meta/shared_config/core_requirements.md`：完整需求基线、LettuceDetect 差异化、阶段 0 至阶段 8 的契约和完成记录。
3. `project_meta/shared_config/release_hardening.md`：发布前工程验收范围和已有结果。

重点判断：项目是否真正聚焦 claim-evidence 审计图、有界自愈和插件协议，而不是重复包装已有 span 检测功能。

### 第二步：检查最关键的领域契约

1. `src/groundguard_rag/domain/enums.py`：五种验证状态和自愈动作/停止原因。
2. `src/groundguard_rag/domain/models.py`：不可变对象、claim-evidence 边、跨字段不变量、审计记录和自愈血缘。
3. `src/groundguard_rag/domain/ports.py`：可替换插件接口及其输入输出边界。
4. `src/groundguard_rag/domain/config.py`：verify/heal 的硬限制。
5. `src/groundguard_rag/schema/audit_report.v1.schema.json` 与 `schema/semantic_validation.py`：Python 对象、JSON Schema、外部 JSON 语义校验是否一致。

重点判断：五态是否清晰；`INSUFFICIENT_EVIDENCE` 是否被正确限制为“输入证据不足”；输入、输出和审计对象是否能拒绝不一致数据。

### 第三步：检查 verify 主链路

1. `src/groundguard_rag/application/verify_service.py`：拆 claim、选证据、调用 verifier、校准、构建审计报告的顺序和 adapter 返回值检查。
2. `src/groundguard_rag/application/input_hashing.py`：输入哈希是否确定、是否只覆盖真正消费的数据。
3. `src/groundguard_rag/adapters/decomposition/rule_based.py`：规则式拆解的适用范围和诚实局限。
4. `src/groundguard_rag/adapters/evidence_selection/lexical.py`：相关性候选排序是否与事实判定、置信度严格分离。
5. `src/groundguard_rag/adapters/verification/nli.py`、`checkability.py`、`transformers_backend.py`：NLI 标签映射、模型分数语义、延迟加载和离线边界。
6. `src/groundguard_rag/adapters/calibration/platt.py`：校准数据隔离、artifact 完整性和 `P(当前 verdict 正确)` 的语义。

重点判断：`verify` 是否严格无网络、无写回、无输入修改；raw logits、softmax、词法分数是否从未被冒充为校准可信度。

### 第四步：检查 heal 状态机

1. `src/groundguard_rag/application/heal_service.py`：候选生成、重新验证、commit/rollback、claim 血缘和所有硬停止条件。
2. `src/groundguard_rag/adapters/heal/policies.py`：默认策略是否保守，是否避免自动删除未解决 claim。
3. `src/groundguard_rag/adapters/heal/callbacks.py`：调用方 Retriever/Rewriter/进度校准器的显式接入边界。

重点判断：最大轮数、单 claim 尝试次数、超时、成本、最小改善量、状态哈希防回环和最终复核能否共同阻止无限循环；失败候选是否一定不会被提交。特别检查阶段 6 的 `P(verdict 正确)` 与 heal 使用的 `P(claim grounded/supported)` 是否始终分离。

### 第五步：检查公开 API 与插件接入

1. `src/groundguard_rag/api.py`：公开门面是否保持显式依赖注入和无隐藏客户端。
2. `src/groundguard_rag/presentation/views.py`：sidecar/inline 是否只是审计图的无损视图。
3. `src/groundguard_rag/integrations/`：MCP JSON 边界、factory 加载和可选依赖延迟导入。
4. `src/groundguard_rag/adapters/frameworks/`：LangChain/LlamaIndex 接口是否停留在 adapter 层。
5. `src/groundguard_rag/evaluation/benchmark.py`：五态评测、混淆矩阵、宏平均和校准指标定义。

### 第六步：用测试验证上述判断

建议先看与待审代码同名的测试，再重点查看：

- `tests/application/test_verify_service.py`
- `tests/application/test_heal_service.py`
- `tests/adapters/calibration/test_platt.py`
- `tests/adapters/verification/test_nli.py`
- `tests/integration/test_real_transformers_nli.py`
- `tests/stage8/test_mcp_integration.py`
- `tests/test_schema.py` 与 `tests/test_semantic_validation.py`
- `tests/evaluation/test_benchmark.py`

## 3. 本地复现命令

在审查副本根目录执行：

```powershell
python -m pip install uv
uv sync --locked --extra test --extra mcp
uv run python -m pytest -q -W error
uv run python -m pytest -q -W error --cov=groundguard_rag --cov-branch --cov-report=term-missing
uv run ruff check src tests examples
uv pip check
uv run python examples/quickstart_local.py
uv run python -m groundguard_rag.evaluation.cli --factory examples.demo_factory:build_groundguard --dataset examples/sample_benchmark.jsonl
uv run python -m build
```

默认测试会跳过需要本地模型缓存的真实 DeBERTa smoke test，这是预期行为。不要为了普通代码审查临时下载大模型。若本机已经具有测试文件中指定 revision 的缓存，才设置 `GROUNDGUARD_RUN_REAL_NLI=1`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 后单独运行该测试。

## 4. 希望老师重点回答的问题

1. 五态及 claim-evidence 审计图是否足够清晰、完整、可扩展？
2. `VerifyService` 和 `HealService` 的职责是否合理，是否存在隐藏副作用或越权调用？
3. 校准可信度的两个语义是否分离正确，是否还有误导用户的风险？
4. 自愈状态机是否存在未覆盖的无限循环、错误提交、预算绕过或异常泄露路径？
5. ports/adapters 是否足够稳定，接入 LettuceDetect、其他 NLI 模型或现有 RAG 框架时是否需要改核心层？
6. JSON Schema、Python 模型和 semantic validation 是否存在不一致？
7. 当前测试是否过度依赖 fake，最值得补的真实集成/性质测试是什么？
8. 离线评测工具的 claim 对齐、五态宏平均和校准指标设计是否合理？
9. 哪些问题必须在公开 GitHub/PyPI 前修复，哪些可放入后续版本？
10. 在不夸大效果的前提下，最适合作为毕业/求职项目展示的技术亮点是什么？

## 5. 当前已知但尚未完成的事项

- 尚未选择软件许可证。
- 尚未执行 Git commit、tag 或 push，远程 GitHub Actions 尚未实际运行。
- 尚未在独立真实 RAG 标注数据上报告检测效果和校准质量。
- 规则式 claim 拆解、词法候选选择和 demo verifier 都是可替换基线，不是最终生产模型。

## 6. 老师审查后的交接方式

请按 `TEACHER_FEEDBACK_TEMPLATE.md` 记录意见，尤其写清：优先级、文件路径/行号、风险、建议修改方式和验收测试。学生把该反馈文件带回后，再按 P0 → P1 → P2 顺序处理；任何改变五态、置信度语义、审计 Schema 或 heal 状态机的建议，应先更新 `project_meta/shared_config/` 的需求，再修改代码。
