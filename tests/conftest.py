from __future__ import annotations

import copy

import pytest

from ageval.core import (
    Task,
    Prediction,
    Score,
    TaskResult,
    RunReport,
    register_agent,
    _agent_registry,
    _scorer_registry,
)


@register_agent("_test_fake")
class FakeAgent:
    """Agent that echoes task input back, used only for tests."""

    name = "_test_fake"

    def predict(self, task: Task) -> Prediction:
        return Prediction(output=task.input, latency_ms=1.0, cost_usd=0.0)


@pytest.fixture
def sample_tasks() -> list[Task]:
    return [
        Task(id="t1", input="hello", expected="hello", scorer="exact", tags=["smoke"]),
        Task(id="t2", input="calc 2+2", expected=4, scorer="numeric", tags=["math"]),
        Task(id="t3", input="say hi", expected="hi", scorer="includes", tags=["format"]),
    ]


@pytest.fixture
def fake_agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture
def sample_report(sample_tasks: list[Task]) -> RunReport:
    t1, t2, t3 = sample_tasks
    return RunReport(
        run_id="test-run",
        suite_name="demo",
        agent_name="_test_fake",
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:01:00+00:00",
        results=[
            TaskResult(
                task=t1,
                prediction=Prediction(output="hello", latency_ms=10),
                score=Score(1.0, True),
                attempts=1,
            ),
            TaskResult(
                task=t2,
                prediction=Prediction(output="wrong", latency_ms=20),
                score=Score(0.0, False, "nope"),
                attempts=1,
            ),
            TaskResult(
                task=t3,
                prediction=Prediction(output="", error="boom", latency_ms=5),
                score=Score(0.0, False, "agent error"),
                attempts=2,
            ),
        ],
    )


@pytest.fixture(autouse=False)
def registry_cleanup():
    """Snapshot and restore the global registries around each test."""
    saved_agents = copy.copy(_agent_registry)
    saved_scorers = copy.copy(_scorer_registry)
    yield
    _agent_registry.clear()
    _agent_registry.update(saved_agents)
    _scorer_registry.clear()
    _scorer_registry.update(saved_scorers)
