# Security Policy

## Supported versions

Security fixes are currently applied to the latest `0.1.x` revision on the
default branch.

## Reporting a vulnerability

Please use the repository's private vulnerability reporting flow:

https://github.com/drk71113-sketch/groundguard-rag/security/advisories/new

Do not open a public issue containing credentials, private documents, provider
responses, exploit details, or personal information. Include the affected
version, a minimal reproduction, potential impact, and any suggested mitigation
in the private report.

## Secret-handling boundary

GroundGuard-RAG does not require secrets in its core package. Provider adapters
must receive credentials from the host application's environment or secret
manager. Never commit `.env` files, API keys, access tokens, private RAG source
documents, model caches, or generated traces containing private prompts.
