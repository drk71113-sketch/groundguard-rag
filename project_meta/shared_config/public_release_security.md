# 公开发布与信息安全基线

## 用户目标

- GitHub 仓库用于让公众查看和使用 GroundGuard-RAG，而不只是老师私下审核。
- 公开发布必须同时保护凭据、私有文档、本机信息和协作会话。

## 本轮处理

- 删除已有真实内容目录中的多余 `.gitkeep`；空的本地文档与评测目录改用说明性 README，避免外部读者误解。
- `.gitignore` 默认排除本地 RAG 文档与评测输入，只允许说明文件进入公开仓库。
- 新增根目录 `SECURITY.md`，要求漏洞通过 GitHub Private Vulnerability Reporting 提交，禁止在公开 issue 粘贴凭据或私有数据。
- 启用 GitHub Secret Scanning、Push Protection、Dependabot 漏洞告警/安全更新及 Private Vulnerability Reporting。
- 后续提交使用 GitHub noreply 邮箱。既有四条提交包含真实 Gmail；是否重写公开历史必须由用户再次明确授权。

## 待用户决定

- 软件许可证：没有 LICENSE 时公众可以查看，但不能明确获得复制、修改和分发授权。用户必须在 MIT 与 Apache-2.0 等许可证之间明确选择，Codex 不代替用户作法律授权决定。
