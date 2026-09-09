# Claude 审查后发布硬化

## 背景与授权

- 2026-09-08 的 Claude 只读审查未发现 P0 问题，并确认五态、verify 无副作用边界、有界 heal 状态机、审计血缘和校准语义未被破坏。
- 用户随后明确要求 Codex 按审查建议修复。该工作属于现有 `0.1.0` 发布硬化，不新增“阶段 9”，不提前增加新业务功能。

## 本轮修复范围

1. 将 `HealService` 中实际存在的五处生产代码 `assert`（审查报告漏列了循环结束处一处）改为优化模式下也生效的显式异常；不得改变正常控制流、提交/回滚条件或停止原因。
2. 将 CI、README 和老师审查指南的 Ruff 范围统一为 `src tests examples demo_rag`。
3. 在 `test` extra 中增加 Hypothesis，并为规则式 claim 拆解的 span/顺序不变量和词法证据选择的分数/top-k 不变量增加性质测试。
4. 增加同一个 `VerifyService`/`HealService` 实例的并发重入测试。保证范围仅为：核心 service 不保存跨调用可变状态；注入 adapter 仍由调用方负责提供线程安全或调用级实例。
5. 增加 HEAL 模式覆盖五种 verdict 状态与 committed DELETE action 的 `to_dict -> JSON Schema -> semantic validation` 组合往返测试。仓库已有 HEAL+REWRITE 三层测试，本项是组合补强而非从零补齐。

## 硬性边界

- 不改变五态枚举、AuditReport 字段、校准含义、verify/heal 副作用边界或有界自愈策略。
- 不把性质测试扩展成新的生产检测算法，不引入运行时依赖；Hypothesis 仅属于测试 extra。
- 不引用未核实 NIST/43% 数据，不声称“首个”或“完全消除幻觉”。
- 不下载真实模型，不在默认 CI 中启用真实模型测试。
- 本轮先完成代码与测试验收；未经用户进一步明确授权，不创建 Git commit、不 push、不改 GitHub 权限或发布 PyPI。

## 验收标准

1. `src/groundguard_rag/application/` 不再包含生产代码裸 `assert`。
2. 新增测试在普通 Python 与 `python -O` 场景下验证显式不变量保护。
3. 全量 `pytest -W error`、branch coverage、Ruff、`uv lock --check`、`uv pip check`、本地 demo/评测和 wheel 构建通过。
4. 核心运行时依赖仍为空，可选 provider/MCP 保持延迟导入；敏感信息与禁止宣传扫描通过。

## 完成记录（2026-09-08，Codex）

- P1-1：已用 `ApplicationInvariantError` 替换 `HealService` 的五处生产代码 `assert`；该内部异常不会被误记为 adapter failure。AST 回归和 `python -O` 子进程回归均已加入。
- P1-2：CI、README、老师审查指南的 Ruff 范围均已统一为 `src tests examples demo_rag`，并增加 CI 配置契约测试。
- P1-3：Hypothesis 仅加入 `test` extra；claim span/顺序/确定性与 lexical score/top-k/输入不可变性质测试已加入，锁文件已更新。
- P1-4：已增加同一 `VerifyService`、`HealService` 实例在八线程重叠调用下的无串扰回归；文档明确 adapter/clock 的线程安全仍由注入方负责。
- P2：已补充 `AuditReport` 与 `HealService` 的血缘校验职责注释，以及 HEAL 五态 + committed DELETE 的 dataclass → JSON → JSON Schema → 语义校验组合测试。聚合耗时进一步拆分和中文缩写启发式属于后续功能优化，本轮未改变现有行为。
- 最终本机验收：`831 passed, 1 skipped`；多次 property-based 回归的 branch coverage 为 `88.94%–88.99%`（门槛 `85%`）；Ruff、`uv lock --check`、`uv pip check`、compileall、本地 demo、评测、wheel 构建与 wheel 内容核对均通过。
- 发布边界：未创建 commit、未 push、未发布 PyPI；软件许可证仍待项目方决定。
