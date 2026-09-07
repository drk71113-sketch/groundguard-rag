"""Application layer: orchestrates domain ports into runnable services.

``VerifyService`` is the purely local verify orchestrator. Stage 7 adds
``HealService``, which may invoke explicitly injected Retriever/Rewriter
ports inside hard round/attempt/time/cost/improvement/loop bounds. Neither
service writes to the host, an AuditStore, a file, or a database; heal only
returns an immutable correction candidate and its audit report.
"""
