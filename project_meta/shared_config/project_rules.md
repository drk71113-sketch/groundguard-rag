# GroundGuard-RAG 协作规则

## 1. 代码权责

- 阶段 0 至阶段 0.3 的核心业务逻辑由 Claude Code 完成。因 Claude Code 当前不可用，用户于 2026-09-02 明确授权 Codex 从阶段 1 起接管核心代码和阶段化开发，直至用户再次变更授权。
- Codex 接管后仍须严格按阶段 0 到阶段 8 的顺序推进；每次只完成当前阶段，配套单元测试和必要注释，不提前实现后续逻辑。
- Codex 可以审查、续写和修正 Claude Code 已留在根目录正常项目结构中的未完成阶段代码，但不得进入、读取或改写 `claude_storage/` 中的私有产物。
- 未经用户明确授权，任何一方不得再变更代码权责或并行修改同一阶段的核心文件。

## 2. 存储边界

- Codex 的会话产物、自动生成文件、快照、备份和 checkpoint 只能存放在 `project_meta/codex_storage/`。
- Claude Code 的会话记录、中间文件、快照和私有工作产物只能存放在 `project_meta/claude_storage/`。
- Codex 不进入、不读取、不写入 `claude_storage/`；Claude Code 不进入、不读取、不写入 `codex_storage/`。
- 双方仅通过 `project_meta/shared_config/` 同步需求、约束、变更说明和交接信息。
- 除非用户明确要求，不在项目根目录散落会话、快照、备份或中间文件。

## 3. 版本控制

- 双方共用项目根目录已有的 `.git`，不得争抢、替换、删除或另行初始化根目录 `.git`。
- 不强制自动提交；是否创建 Git commit 由用户明确决定。
- Codex 只在 `codex_storage/` 内维护自己的文件级 checkpoint、快照和清单，不借此修改根目录 Git 配置。
- Claude Code 不随意修改 Git 配置、hooks、remote、branch 策略或用户已有提交历史。

## 4. 同步方式

- 新需求、需求变更、接口约定、阶段状态和交接说明必须先写入 `shared_config/` 中的共享文档。
- 不以口头约定、会话隐含信息或某一方私有 storage 中的内容作为另一方必须遵守的项目要求。
- 不私下修改根目录文件来传递需求；需要修改核心项目文件时，由对应责任方根据 `shared_config/` 中已同步的要求执行。
- 共享文档发生变更时，应记录变更内容和影响范围，避免双方基于不同版本工作。
