# GroundGuard-RAG v0.1.0 发布前硬化

## 定位

阶段 0 至阶段 8 已结束。本轮不新建“阶段 9”，不改变五态、verify/heal 副作用边界、校准语义或有界自愈状态机；只做 `0.1.0` 发布、演示和真实数据评测前的工程硬化。

## 本轮范围

1. 增加 CI：在支持的 Python 版本上安装核心测试依赖，执行严格警告测试、基础 lint、wheel 构建和 schema 包数据检查。CI 不自动下载模型权重。
2. 增加覆盖率验收：使用 branch coverage，门槛必须基于实际测量结果设定，不伪造 100% 声明。
3. 增加可复现依赖锁文件；核心 `dependencies=[]` 和可选 provider/MCP 边界保持不变。
4. 增加可直接运行的纯本地 demo 与 MCP factory demo。Demo verifier 必须明确标注为管道演示用 test double，不宣称具有生产检测能力。
5. 增加纯本地 JSONL 评测工具：读取调用方已标注的 answer/chunks/claim expected states，输出五态 confusion matrix、accuracy、macro precision/recall/F1，以及在存在校准置信度时的 Brier/log loss/ECE。评测工具不联网、不自动下载数据。
6. 扩充 README，写明 demo、评测数据格式、CI/覆盖率命令和发布前限制。

## 仍然禁止

- 不引入未核实 NIST/43% 数据、“首个”、“完全消除幻觉”或任何未经真实数据支持的效果声明。
- 不把 demo 精确匹配规则作为新的生产 Verifier。
- 不在 CI 中开启网络模型下载或把真实模型写成必选依赖。
- 不在未经用户明确选择时擅自添加 MIT、Apache-2.0 或其他软件许可证。
- 不执行 Git add/commit/tag/push，除非用户另行明确授权。

## 验收标准

- 新增评测与 demo 模块有单元测试和必要注释。
- 全量 `pytest -W error` 通过，覆盖率达到实测门槛，新增代码 lint 通过。
- wheel 可构建、可隔离导入，且包含审计 JSON Schema 和两个 CLI entry point（MCP/评测）。
- 禁止宣传、核心依赖边界和可选依赖延迟导入扫描通过。

## 完成记录（2026-09-03）

- 已增加 GitHub Actions：Python 3.10/3.11/3.12 使用 `uv.lock` 同步测试与 MCP extras，执行依赖检查、ruff、带分支覆盖率的全量测试、两个本地 demo；独立任务构建并检查 wheel。
- 已生成并校验 `uv.lock`；从锁文件创建 Python 3.10.21、3.11.2、3.12.14 三套隔离环境，依赖兼容检查全部通过，每套全量结果均为 `787 passed, 1 skipped`。环境均位于 `project_meta/codex_storage/`，未改变系统默认 Python。
- 默认跳过项是明确 opt-in 的真实模型测试。Python 3.11 主环境的分支感知总覆盖率为 `88.84%`，已在配置中设定 `85%` 门槛。
- 缓存模型在 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 下单独验证，结果为 `1 passed`，未联网下载权重。
- `ruff check`、`pip check`、`compileall`、禁止宣传词扫描与核心层可选依赖导入扫描均通过。
- 纯本地 quickstart 和 JSONL 评测 CLI 均运行成功；示例明确标注为管道 test double，不能作为检测效果证据。
- 最终 sdist/wheel 位于 `project_meta/codex_storage/release_hardening_final/dist/`。wheel 已在独立环境安装并确认版本、Schema 包数据、`groundguard-rag-eval` 和 `groundguard-rag-mcp` 两个入口。

## 仍需用户决定

- 尚未选择软件许可证；README 已明确当前无许可证，未擅自添加 MIT、Apache-2.0 或其他许可证。
- 尚未执行 Git add/commit/tag/push；是否建立 `v0.1.0` 基线由用户明确决定。
