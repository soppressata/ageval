from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ageval.core import (
    Prediction,
    RunReport,
    Score,
    Task,
    TaskResult,
    register_scorer,
)
from ageval.runner import RunConfig, run_suite


@register_scorer("_eq_run")
class _EqRunScorer:
    """Scorer that passes when prediction.output == str(task.expected)."""

    name = "_eq_run"

    def score(self, task: Task, prediction: Prediction) -> Score:
        if prediction.error:
            return Score(0.0, False, "agent error")
        if prediction.output == str(task.expected):
            return Score(1.0, True, "")
        return Score(0.0, False, "no")


# ---- Agent helpers -----------------------------------------------------------


class _RightAgent:
    name = "right"

    def predict(self, task: Task) -> Prediction:
        return Prediction(output=str(task.expected))


class _OrderAgent:
    """Returns correct output, sleeping a per-task duration so completions
    reorder relative to the input sequence."""

    name = "order"

    def predict(self, task: Task) -> Prediction:
        time.sleep(0.005 * (8 - int(task.id[1:])))
        return Prediction(output=str(task.expected))


class _FlakyAgent:
    """Returns error for the first `fail_times` calls, then succeeds."""

    name = "flaky"

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def predict(self, task: Task) -> Prediction:
        self.calls += 1
        if self.calls <= self.fail_times:
            return Prediction(error=f"flaky #{self.calls}")
        return Prediction(output=str(task.expected))


class _FlakyExceptionAgent:
    """Raises for the first `fail_times` calls, then succeeds."""

    name = "flaky_exc"

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def predict(self, task: Task) -> Prediction:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"boom #{self.calls}")
        return Prediction(output=str(task.expected))


class _SleepingAgent:
    name = "sleeper"

    def __init__(self, sleep: float) -> None:
        self.sleep = sleep

    def predict(self, task: Task) -> Prediction:
        time.sleep(self.sleep)
        return Prediction(output=str(task.expected))


class _WrongFirstAgent:
    name = "wrongfirst"

    def predict(self, task: Task) -> Prediction:
        if task.id == "t0":
            return Prediction(output="WRONG")
        return Prediction(output=str(task.expected))


class _CountingSlowAgent:
    name = "countslow"

    def __init__(self, sleep: float = 0.05) -> None:
        self.sleep = sleep
        self.calls = 0

    def predict(self, task: Task) -> Prediction:
        self.calls += 1
        time.sleep(self.sleep)
        return Prediction(output=str(task.expected))


# ---- A. Input-order preservation --------------------------------------------


def test_input_order_preserved_under_concurrency() -> None:
    """8 tasks via ThreadPool(concurrency=4) with jitter keep input order."""
    agent = _OrderAgent()
    tasks = [
        Task(id=f"t{i}", input=f"q{i}", expected=f"a{i}", scorer="_eq_run")
        for i in range(8)
    ]
    report = run_suite(tasks, agent, RunConfig(concurrency=4))
    assert [r.task.id for r in report.results] == [f"t{i}" for i in range(8)]


# ---- B. Retry counting -------------------------------------------------------


def test_retry_flaky_succeeds_after_two_failures() -> None:
    """Flaky agent fails twice then succeeds; max_retries=2 -> ok, attempts==3."""
    agent = _FlakyAgent(fail_times=2)
    tasks = [Task(id="t0", input="q", expected="a", scorer="_eq_run")]
    report = run_suite(
        tasks, agent, RunConfig(max_retries=2, retry_backoff=0.01)
    )
    result = report.results[0]
    assert result.ok
    assert result.attempts == 3


def test_retry_zero_fails_after_one_attempt() -> None:
    """max_retries=0 -> single attempt, not ok, attempts==1."""
    agent = _FlakyAgent(fail_times=5)
    tasks = [Task(id="t0", input="q", expected="a", scorer="_eq_run")]
    report = run_suite(tasks, agent, RunConfig(max_retries=0))
    result = report.results[0]
    assert not result.ok
    assert result.attempts == 1


def test_retry_predict_exception_counts_as_attempt() -> None:
    """predict() raising still counts as an attempt and triggers retry."""
    agent = _FlakyExceptionAgent(fail_times=2)
    tasks = [Task(id="t0", input="q", expected="a", scorer="_eq_run")]
    report = run_suite(
        tasks, agent, RunConfig(max_retries=2, retry_backoff=0.01)
    )
    result = report.results[0]
    assert result.ok
    assert result.attempts == 3


# ---- C. Timeout --------------------------------------------------------------


def test_timeout_marks_task_errored() -> None:
    """Agent sleeping longer than task_timeout -> timeout error, errored==1."""
    agent = _SleepingAgent(sleep=0.5)
    tasks = [Task(id="t0", input="q", expected="a", scorer="_eq_run")]
    report = run_suite(
        tasks, agent, RunConfig(task_timeout=0.3, concurrency=1)
    )
    result = report.results[0]
    assert "timeout" in result.prediction.error.lower()
    assert report.summary()["errored"] == 1


# ---- D. Tag filter, limit, ordering ------------------------------------------


def test_tag_filter_keeps_only_matching() -> None:
    agent = _RightAgent()
    tasks = [
        Task(id="t0", input="q", expected="a", scorer="_eq_run", tags=["x"]),
        Task(id="t1", input="q", expected="a", scorer="_eq_run", tags=["y"]),
        Task(id="t2", input="q", expected="a", scorer="_eq_run", tags=["y"]),
        Task(id="t3", input="q", expected="a", scorer="_eq_run", tags=["x"]),
    ]
    report = run_suite(tasks, agent, RunConfig(tags=["y"]))
    assert [r.task.id for r in report.results] == ["t1", "t2"]


def test_limit_keeps_first_n() -> None:
    agent = _RightAgent()
    tasks = [
        Task(id=f"t{i}", input="q", expected="a", scorer="_eq_run")
        for i in range(5)
    ]
    report = run_suite(tasks, agent, RunConfig(limit=2))
    assert [r.task.id for r in report.results] == ["t0", "t1"]


def test_tags_then_limit_ordering() -> None:
    """Filter by tags first, then apply limit."""
    agent = _RightAgent()
    tasks = [
        Task(id="t0", input="q", expected="a", scorer="_eq_run", tags=["x"]),
        Task(id="t1", input="q", expected="a", scorer="_eq_run", tags=["y"]),
        Task(id="t2", input="q", expected="a", scorer="_eq_run", tags=["y"]),
        Task(id="t3", input="q", expected="a", scorer="_eq_run", tags=["y"]),
    ]
    report = run_suite(tasks, agent, RunConfig(tags=["y"], limit=1))
    assert [r.task.id for r in report.results] == ["t1"]


# ---- E. Progress callback ----------------------------------------------------


def test_progress_callback_count_equals_total() -> None:
    agent = _RightAgent()
    tasks = [
        Task(id=f"t{i}", input="q", expected="a", scorer="_eq_run")
        for i in range(5)
    ]
    count = 0

    def progress(_: TaskResult) -> None:
        nonlocal count
        count += 1

    run_suite(tasks, agent, progress=progress)
    assert count == len(tasks)


def test_progress_callback_exception_does_not_propagate() -> None:
    agent = _RightAgent()
    tasks = [
        Task(id=f"t{i}", input="q", expected="a", scorer="_eq_run")
        for i in range(3)
    ]

    def bad_progress(_: TaskResult) -> None:
        raise RuntimeError("boom")

    report = run_suite(tasks, agent, progress=bad_progress)
    assert len(report.results) == len(tasks)
    assert all(r.ok for r in report.results)


# ---- F. fail_fast ------------------------------------------------------------


def test_fail_fast_stops_early() -> None:
    """First task fails -> remaining cancelled, results shorter than total."""
    agent = _WrongFirstAgent()
    tasks = [
        Task(id=f"t{i}", input="q", expected="a", scorer="_eq_run")
        for i in range(4)
    ]
    report = run_suite(
        tasks, agent, RunConfig(fail_fast=True, concurrency=1)
    )
    assert len(report.results) < len(tasks)
    assert all(r is not None for r in report.results)


# ---- G. Disk cache -----------------------------------------------------------


def test_cache_second_run_skips_agent(tmp_path: Path) -> None:
    agent = _CountingSlowAgent(sleep=0.05)
    tasks = [
        Task(id=f"t{i}", input=f"q{i}", expected=f"a{i}", scorer="_eq_run")
        for i in range(3)
    ]
    report1 = run_suite(
        tasks, agent, RunConfig(cache_dir=str(tmp_path), concurrency=2)
    )
    assert agent.calls == 3
    assert report1.summary()["passed"] == 3

    report2 = run_suite(
        tasks, agent, RunConfig(cache_dir=str(tmp_path), concurrency=2)
    )
    assert agent.calls == 3
    assert report2.summary()["passed"] == 3


def test_cache_corrupt_file_still_passes(tmp_path: Path) -> None:
    agent = _CountingSlowAgent(sleep=0.05)
    tasks = [Task(id="t0", input="q0", expected="a0", scorer="_eq_run")]
    run_suite(tasks, agent, RunConfig(cache_dir=str(tmp_path)))
    assert agent.calls == 1

    for f in tmp_path.glob("*.json"):
        f.write_text("not json {{{")

    agent2 = _CountingSlowAgent(sleep=0.05)
    report = run_suite(
        tasks, agent2, RunConfig(cache_dir=str(tmp_path))
    )
    assert agent2.calls == 1
    assert report.results[0].ok


def test_empty_tasks_summary_zero() -> None:
    agent = _RightAgent()
    report = run_suite([], agent)
    summary = report.summary()
    assert summary["total"] == 0
    assert summary["pass_rate"] == 0.0


# ---- H. Unknown scorer, None score, summary, run_id -------------------------


def test_unknown_scorer_degrades_to_failed() -> None:
    agent = _RightAgent()
    tasks = [Task(id="t0", input="q", expected="a", scorer="does_not_exist")]
    report = run_suite(tasks, agent)
    result = report.results[0]
    assert result.score is not None
    assert not result.score.passed
    assert not result.ok


def test_summary_handles_none_score() -> None:
    task = Task(id="t0", input="q", expected="a")
    result = TaskResult(task=task, prediction=Prediction(output="a"), score=None)
    report = RunReport(
        run_id="s-a-20260101-000000",
        suite_name="s",
        agent_name="a",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        results=[result],
    )
    summary = report.summary()
    assert summary["mean_score"] == 0.0
    assert summary["passed"] == 0


def test_summary_keys_and_by_tag() -> None:
    agent = _RightAgent()
    tasks = [
        Task(
            id="t0", input="q", expected="a", scorer="_eq_run", tags=["x", "y"]
        ),
        Task(id="t1", input="q", expected="a", scorer="_eq_run", tags=["y"]),
    ]
    report = run_suite(tasks, agent)
    summary = report.summary()
    expected_keys = {
        "total",
        "passed",
        "failed",
        "errored",
        "pass_rate",
        "mean_score",
        "total_cost_usd",
        "mean_latency_ms",
        "p95_latency_ms",
        "by_tag",
    }
    assert set(summary.keys()) == expected_keys
    assert set(summary["by_tag"].keys()) == {"x", "y"}
    assert summary["by_tag"]["y"]["total"] == 2
    assert summary["by_tag"]["y"]["passed"] == 2


def test_run_id_format_and_timestamps() -> None:
    agent = _RightAgent()
    tasks = [Task(id="t0", input="q", expected="a", scorer="_eq_run")]
    report = run_suite(tasks, agent, suite_name="mysuite")
    assert report.run_id.startswith("mysuite-right-")
    suffix = report.run_id[len("mysuite-right-"):]
    assert len(suffix) == 15
    assert suffix[8] == "-"
    assert suffix[:8].isdigit()
    assert suffix[9:].isdigit()
    assert report.started_at
    assert report.finished_at
