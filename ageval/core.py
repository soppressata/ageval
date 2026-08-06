from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Protocol, runtime_checkable

from ageval.errors import UnknownComponentError


def clamp01(x: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    return max(0.0, min(1.0, x))


@dataclass
class Task:
    id: str
    input: str
    expected: Any | None = None
    scorer: str = "exact"
    scorer_args: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        return cls(
            id=d["id"],
            input=d["input"],
            expected=d.get("expected"),
            scorer=d.get("scorer", "exact"),
            scorer_args=d.get("scorer_args", {}),
            tags=list(d.get("tags", [])),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class Prediction:
    output: str = ""
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Prediction:
        return cls(
            output=d.get("output", ""),
            latency_ms=d.get("latency_ms", 0.0),
            tokens_in=d.get("tokens_in", 0),
            tokens_out=d.get("tokens_out", 0),
            cost_usd=d.get("cost_usd", 0.0),
            error=d.get("error"),
            raw=dict(d.get("raw", {})),
        )


@dataclass(frozen=True)
class Score:
    value: float
    passed: bool
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", clamp01(self.value))

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "passed": self.passed,
            "detail": self.detail,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Score:
        return cls(
            value=d["value"],
            passed=d["passed"],
            detail=d.get("detail", ""),
            extra=dict(d.get("extra", {})),
        )


@dataclass
class TaskResult:
    task: Task
    prediction: Prediction
    score: Score | None = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return (
            self.prediction.error is None
            and self.score is not None
            and self.score.passed
        )

    def to_dict(self) -> dict:
        return {
            "task": self.task.to_dict(),
            "prediction": self.prediction.to_dict(),
            "score": self.score.to_dict() if self.score is not None else None,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskResult:
        return cls(
            task=Task.from_dict(d["task"]),
            prediction=Prediction.from_dict(d["prediction"]),
            score=Score.from_dict(d["score"]) if d.get("score") is not None else None,
            attempts=d.get("attempts", 1),
        )


@dataclass
class RunReport:
    run_id: str
    suite_name: str
    agent_name: str
    started_at: str
    finished_at: str = ""
    results: list[TaskResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.ok)
        errored = sum(1 for r in self.results if r.prediction.error is not None)
        failed = total - passed - errored

        if total == 0:
            pass_rate = 0.0
            mean_score = 0.0
        else:
            pass_rate = passed / total
            mean_score = sum(
                r.score.value if r.score is not None else 0.0 for r in self.results
            ) / total

        total_cost_usd = sum(r.prediction.cost_usd for r in self.results)

        if total == 0:
            mean_latency_ms = 0.0
            p95_latency_ms = 0.0
        else:
            latencies = sorted(r.prediction.latency_ms for r in self.results)
            mean_latency_ms = sum(latencies) / total
            idx = min(len(latencies) - 1, int(math.ceil(0.95 * len(latencies))) - 1)
            p95_latency_ms = latencies[idx]

        by_tag: dict[str, dict[str, Any]] = {}
        for r in self.results:
            for tag in r.task.tags:
                if tag not in by_tag:
                    by_tag[tag] = {"total": 0, "passed": 0}
                by_tag[tag]["total"] += 1
                if r.ok:
                    by_tag[tag]["passed"] += 1
        for tag in by_tag:
            t = by_tag[tag]["total"]
            p = by_tag[tag]["passed"]
            by_tag[tag]["pass_rate"] = p / t if t > 0 else 0.0

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errored": errored,
            "pass_rate": pass_rate,
            "mean_score": mean_score,
            "total_cost_usd": total_cost_usd,
            "mean_latency_ms": mean_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "by_tag": by_tag,
        }

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "suite_name": self.suite_name,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [r.to_dict() for r in self.results],
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunReport:
        return cls(
            run_id=d["run_id"],
            suite_name=d["suite_name"],
            agent_name=d["agent_name"],
            started_at=d["started_at"],
            finished_at=d.get("finished_at", ""),
            results=[TaskResult.from_dict(r) for r in d.get("results", [])],
            config=dict(d.get("config", {})),
        )


@runtime_checkable
class Agent(Protocol):
    name: str

    def predict(self, task: Task) -> Prediction: ...


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, task: Task, prediction: Prediction) -> Score: ...


_agent_registry: dict[str, type] = {}
_scorer_registry: dict[str, type] = {}


def register_agent(name: str) -> Callable[[type], type]:
    """Class decorator that registers an agent class under the given name."""

    def decorator(cls: type) -> type:
        _agent_registry[name] = cls
        if not hasattr(cls, "name"):
            cls.name = name
        return cls

    return decorator


def register_scorer(name: str) -> Callable[[type], type]:
    """Class decorator that registers a scorer class under the given name."""

    def decorator(cls: type) -> type:
        _scorer_registry[name] = cls
        if not hasattr(cls, "name"):
            cls.name = name
        return cls

    return decorator


def get_agent(name: str, **kwargs: Any) -> Agent:
    """Instantiate a registered agent by name, forwarding kwargs."""
    if name not in _agent_registry:
        raise UnknownComponentError(name, list(_agent_registry.keys()), kind="agent")
    return _agent_registry[name](**kwargs)


def get_scorer(name: str, **kwargs: Any) -> Scorer:
    """Instantiate a registered scorer by name, forwarding kwargs."""
    if name not in _scorer_registry:
        raise UnknownComponentError(name, list(_scorer_registry.keys()), kind="scorer")
    return _scorer_registry[name](**kwargs)


def list_agents() -> list[str]:
    """Return sorted list of registered agent names."""
    return sorted(_agent_registry.keys())


def list_scorers() -> list[str]:
    """Return sorted list of registered scorer names."""
    return sorted(_scorer_registry.keys())
