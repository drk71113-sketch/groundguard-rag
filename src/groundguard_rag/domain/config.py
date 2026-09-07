"""Structural configuration models.

These hold *values* (limits, budgets, toggles) that application services read.
They intentionally contain no aggregation, threshold-decision, or loop logic
themselves -- only the range/consistency validation needed so invalid config
fails at construction instead of misbehaving inside orchestration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from groundguard_rag.domain.exceptions import ConfigurationError


def _require_int(value: int, field_name: str) -> None:
    # bool is a subclass of int in Python, so isinstance(True, int) is True
    # and True/False would otherwise silently pass as 1/0 -- reject that
    # explicitly rather than accepting a bool where a round/attempt count
    # is expected.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{field_name} must be an int, not a bool")


def _require_finite_real(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field_name} must be a real number, not a bool")
    if not math.isfinite(value):
        raise ConfigurationError(f"{field_name} must be finite (no NaN/Infinity)")


@dataclass(frozen=True)
class VerifyConfig:
    """Configuration for verify-only runs.

    ``allow_network`` defaults to ``False`` per core_requirements #4
    ("verify 默认不访问网络"). ``VerifyService`` enforces this boundary and
    rejects ``allow_network=True`` rather than silently ignoring it.
    """

    allow_network: bool = False

    def __post_init__(self) -> None:
        if self.allow_network is not False and self.allow_network is not True:
            raise ConfigurationError("VerifyConfig.allow_network must be a bool")


@dataclass(frozen=True)
class HealConfig:
    """Hard bounds consumed by the stage-7 self-healing state machine.

    Every bound here is mandatory and validated to be a sane positive
    value, in line with core_requirements #6 ("不允许没有预算、退出条件或
    最终复核的循环") -- there is deliberately no default that would let a
    caller construct an unbounded heal configuration by omission.

    ``min_improvement`` is bounded to [0, 1] and stage 7 compares it only
    against deltas from ``HealProgressEvaluator``'s independently calibrated
    P(claim grounded/supported). It never compares raw NLI scores or the
    different stage-6 probability that the current five-state verdict is
    correct.
    """

    max_rounds: int
    max_attempts_per_claim: int
    timeout_seconds: float
    cost_budget: float
    min_improvement: float

    def __post_init__(self) -> None:
        _require_int(self.max_rounds, "HealConfig.max_rounds")
        if self.max_rounds < 1:
            raise ConfigurationError("HealConfig.max_rounds must be >= 1")

        _require_int(self.max_attempts_per_claim, "HealConfig.max_attempts_per_claim")
        if self.max_attempts_per_claim < 1:
            raise ConfigurationError(
                "HealConfig.max_attempts_per_claim must be >= 1"
            )

        _require_finite_real(self.timeout_seconds, "HealConfig.timeout_seconds")
        if self.timeout_seconds <= 0:
            raise ConfigurationError("HealConfig.timeout_seconds must be > 0")

        _require_finite_real(self.cost_budget, "HealConfig.cost_budget")
        if self.cost_budget <= 0:
            raise ConfigurationError("HealConfig.cost_budget must be > 0")

        _require_finite_real(self.min_improvement, "HealConfig.min_improvement")
        if not (0.0 <= self.min_improvement <= 1.0):
            raise ConfigurationError(
                "HealConfig.min_improvement must be within [0, 1]"
            )
