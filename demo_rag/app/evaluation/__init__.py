"""Evaluation helpers for the independent reference application.

Keep this package initializer lazy: eager re-exports would import the smoke
module before ``python -m ...ragtruth_smoke`` executes it and trigger a runpy
warning.  Callers should import the concrete module explicitly.
"""
