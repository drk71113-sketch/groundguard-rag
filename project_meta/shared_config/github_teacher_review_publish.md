# GitHub 老师审查版发布约定

## 授权与目标

- 用户于 2026-09-07 明确授权创建首次 Git commit，并把当前 GroundGuard-RAG 项目推送到用户指定的 GitHub 仓库，供老师审查。
- 用户于 2026-09-08 确认新建私有仓库 `drk71113-sketch/groundguard-rag` 作为老师审查仓库；不覆盖账号中已有的其他项目。
- 本次只建立老师审查基线，不创建发布标签、不发布 GitHub Release、不发布 PyPI 包，也不擅自添加软件许可证。

## 纳入仓库的内容

- 核心源码、单元/集成测试、示例、CI 工作流、`pyproject.toml`、`uv.lock` 和项目 README。
- `demo_rag` 的代码、测试、体积可控的示例评测结果，以及 RAGTruth 的许可证和来源说明。
- `project_meta/shared_config/` 中的统一需求、阶段记录、发布硬化和老师审查文档。

## 永不上传的内容

- `project_meta/codex_storage/` 与 `project_meta/claude_storage/` 中的会话、快照、备份、构建产物和审查压缩包。
- `.venv/`、测试/静态检查缓存、覆盖率临时文件、构建目录和本机配置。
- `.env`、访问令牌、API 密钥、密码或其他凭据。
- 本地下载的原始 RAGTruth JSONL、模型权重、模型缓存和 checkpoint。

## 推送前验收

1. 只按明确白名单暂存文件，不使用未经检查的整目录盲目上传。
2. 检查暂存路径、文件体积、敏感信息模式和 Git diff。
3. 运行全量测试、lint 和必要的本地演示/构建检查。
4. 首次提交推送后验证远端分支和 GitHub Actions；若远端已有历史，禁止强制覆盖。
