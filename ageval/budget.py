from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ageval.core import Prediction


@dataclass
class Budget:
    """Resource limits for a run.

    A limit of ``None`` means "unlimited". A non-``None`` value must be
    non-negative. Any limit that is configured (not ``None``) is enforced:
    when cumulative consumption exceeds it, :meth:`BudgetManager.should_stop`
    returns ``True``.
    """

    max_cost_usd: float | None = None
    max_tokens: int | None = None
    max_latency_ms: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_cost_usd", self.max_cost_usd),
            ("max_tokens", self.max_tokens),
            ("max_latency_ms", self.max_latency_ms),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass
class BudgetStatus:
    """Immutable snapshot of cumulative consumption.

    Attributes:
        total_cost_usd: Sum of ``cost_usd`` across all consumed predictions.
        total_tokens: Sum of ``tokens_in + tokens_out`` across all predictions.
        total_latency_ms: Sum of ``latency_ms`` across all predictions.
        attempts: Number of predictions consumed.
        stopped: Whether any configured limit has been exceeded.
    """

    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    attempts: int = 0
    stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "attempts": self.attempts,
            "stopped": self.stopped,
        }


class BudgetManager:
    """Thread-safe accumulator that enforces a :class:`Budget`.

    Every :meth:`consume` call counts as one attempt. Limits are checked
    against cumulative totals; when any configured limit is exceeded
    ``should_stop`` returns ``True``. Missing metrics (``None``) in a
    prediction are treated as zero.
    """

    def __init__(self, budget: Budget | None = None) -> None:
        self._budget = budget if budget is not None else Budget()
        self._lock = threading.Lock()
        self._total_cost_usd: float = 0.0
        self._total_tokens: int = 0
        self._total_latency_ms: float = 0.0
        self._attempts: int = 0
        self._stopped: bool = False

    def check(self, prediction: Prediction) -> bool:
        """Return ``True`` if *prediction* alone is within every configured limit.

        Does not mutate state. Missing metrics are treated as zero.
        """
        cost = float(prediction.cost_usd) if prediction.cost_usd is not None else 0.0
        tokens = int(prediction.tokens_in or 0) + int(prediction.tokens_out or 0)
        latency = float(prediction.latency_ms) if prediction.latency_ms is not None else 0.0

        if self._budget.max_cost_usd is not None and cost > self._budget.max_cost_usd:
            return False
        if self._budget.max_tokens is not None and tokens > self._budget.max_tokens:
            return False
        if self._budget.max_latency_ms is not None and latency > self._budget.max_latency_ms:
            return False
        return True

    def consume(self, prediction: Prediction) -> BudgetStatus:
        """Record one prediction attempt and update cumulative totals.

        Returns the resulting :class:`BudgetStatus`.
        """
        cost = float(prediction.cost_usd) if prediction.cost_usd is not None else 0.0
        tokens = int(prediction.tokens_in or 0) + int(prediction.tokens_out or 0)
        latency = float(prediction.latency_ms) if prediction.latency_ms is not None else 0.0

        with self._lock:
            self._total_cost_usd += cost
            self._total_tokens += tokens
            self._total_latency_ms += latency
            self._attempts += 1

            if self._stopped:
                return self._snapshot()

            if (
                (self._budget.max_cost_usd is not None and self._total_cost_usd > self._budget.max_cost_usd)
                or (self._budget.max_tokens is not None and self._total_tokens > self._budget.max_tokens)
                or (self._budget.max_latency_ms is not None and self._total_latency_ms > self._budget.max_latency_ms)
            ):
                self._stopped = True

            return self._snapshot()

    def should_stop(self) -> bool:
        """Return ``True`` if any configured limit has been exceeded."""
        with self._lock:
            return self._stopped

    def status(self) -> BudgetStatus:
        """Return a snapshot of current consumption."""
        with self._lock:
            return self._snapshot()

    def reset(self) -> None:
        """Zero out all counters and clear the stopped flag."""
        with self._lock:
            self._total_cost_usd = 0.0
            self._total_tokens = 0
            self._total_latency_ms = 0.0
            self._attempts = 0
            self._stopped = False

    def _snapshot(self) -> BudgetStatus:
        return BudgetStatus(
            total_cost_usd=self._total_cost_usd,
            total_tokens=self._total_tokens,
            total_latency_ms=self._total_latency_ms,
            attempts=self._attempts,
            stopped=self._stopped,
        )
