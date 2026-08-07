from __future__ import annotations

import time
from typing import Any

import pytest

from ageval.core import Agent, Prediction, Task
from ageval.sandbox import Sandbox, SubprocessSandbox


class _EchoAgent:
    name = "_echo"

    def predict(self, task: Task) -> Prediction:
        return Prediction(output=f"echo:{task.input}", latency_ms=1.0, cost_usd=0.0)


class _SlowAgent:
    name = "_slow"

    def predict(self, task: Task) -> Prediction:
        time.sleep(task.metadata.get("sleep", 10.0))
        return Prediction(output="done")


class _ExplodingAgent:
    name = "_exploding"

    def predict(self, task: Task) -> Prediction:
        raise RuntimeError("child process exploded")


class _UnpicklableAgent:
    name = "_unpicklable"

    def __init__(self) -> None:
        self._fn = lambda x: x  # lambdas are not picklable

    def predict(self, task: Task) -> Prediction:
        return Prediction(output="never")


def _make_task(input_str: str = "hi", **metadata: Any) -> Task:
    return Task(id="t-sandbox", input=input_str, metadata=metadata)


def test_successful_prediction() -> None:
    agent = _EchoAgent()
    task = _make_task("hello")
    sandbox = SubprocessSandbox()

    result = sandbox.run(agent, task, timeout=5.0)

    assert result.error is None
    assert result.output == "echo:hello"
    assert result.latency_ms == 1.0
    assert result.cost_usd == 0.0


def test_prediction_is_serializable() -> None:
    agent = _EchoAgent()
    task = _make_task("serial")
    sandbox = SubprocessSandbox()

    result = sandbox.run(agent, task, timeout=5.0)

    as_dict = result.to_dict()
    restored = Prediction.from_dict(as_dict)
    assert restored.output == "echo:serial"
    assert restored.error is None


def test_timeout_termination() -> None:
    agent = _SlowAgent()
    task = _make_task("slow", sleep=10.0)
    sandbox = SubprocessSandbox()

    start = time.perf_counter()
    result = sandbox.run(agent, task, timeout=1.0)
    elapsed = time.perf_counter() - start

    assert result.error is not None
    assert "timeout" in result.error
    assert elapsed < 5.0  # must not actually wait the full 10s


def test_child_exception_conversion() -> None:
    agent = _ExplodingAgent()
    task = _make_task("boom")
    sandbox = SubprocessSandbox()

    result = sandbox.run(agent, task, timeout=5.0)

    assert result.error is not None
    assert "child process exploded" in result.error


def test_unpicklable_agent_returns_error() -> None:
    agent = _UnpicklableAgent()
    task = _make_task("nope")
    sandbox = SubprocessSandbox(context="spawn")

    result = sandbox.run(agent, task, timeout=5.0)

    assert result.error is not None


def test_import_exposes_sandbox_protocol() -> None:
    from ageval import sandbox as sb

    assert hasattr(sb, "Sandbox")
    assert hasattr(sb, "SubprocessSandbox")
    assert issubclass(SubprocessSandbox, sb.Sandbox)
