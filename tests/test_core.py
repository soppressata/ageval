from __future__ import annotations

import json

import pytest

from ageval.core import (
    Task,
    Prediction,
    Score,
    TaskResult,
    RunReport,
    clamp01,
    register_agent,
    register_scorer,
    get_agent,
    get_scorer,
    list_agents,
    list_scorers,
    _agent_registry,
    _scorer_registry,
)
from ageval.errors import UnknownComponentError


# ---------------------------------------------------------------------------
# A. Dataclass defaults & no shared mutable defaults
# ---------------------------------------------------------------------------


def test_task_defaults() -> None:
    t = Task(id="a", input="hi")
    assert t.expected is None
    assert t.scorer == "exact"
    assert t.scorer_args == {}
    assert t.tags == []
    assert t.metadata == {}


def test_prediction_defaults() -> None:
    p = Prediction()
    assert p.output == ""
    assert p.latency_ms == 0.0
    assert p.tokens_in == 0
    assert p.tokens_out == 0
    assert p.cost_usd == 0.0
    assert p.error is None
    assert p.raw == {}


def test_score_frozen() -> None:
    s = Score(value=0.5, passed=False)
    with pytest.raises(AttributeError):
        s.value = 1.0  # type: ignore[misc]


def test_taskresult_ok_true() -> None:
    tr = TaskResult(
        task=Task(id="a", input="x"),
        prediction=Prediction(output="y"),
        score=Score(value=1.0, passed=True),
    )
    assert tr.ok is True


def test_taskresult_ok_false_when_error() -> None:
    tr = TaskResult(
        task=Task(id="a", input="x"),
        prediction=Prediction(error="boom"),
        score=Score(value=1.0, passed=True),
    )
    assert tr.ok is False


def test_taskresult_ok_false_when_score_none() -> None:
    tr = TaskResult(
        task=Task(id="a", input="x"),
        prediction=Prediction(output="y"),
        score=None,
    )
    assert tr.ok is False


def test_taskresult_ok_false_when_failed() -> None:
    tr = TaskResult(
        task=Task(id="a", input="x"),
        prediction=Prediction(output="y"),
        score=Score(value=0.0, passed=False),
    )
    assert tr.ok is False


def test_runreport_defaults() -> None:
    r = RunReport(run_id="r", suite_name="s", agent_name="a", started_at="2024-01-01T00:00:00+00:00")
    assert r.finished_at == ""
    assert r.results == []
    assert r.config == {}


def test_task_mutable_defaults_not_shared() -> None:
    t1 = Task(id="a", input="x")
    t1.tags.append("added")
    t2 = Task(id="b", input="y")
    assert t2.tags == []


def test_prediction_mutable_defaults_not_shared() -> None:
    p1 = Prediction()
    p1.raw["k"] = "v"
    p2 = Prediction()
    assert p2.raw == {}


# ---------------------------------------------------------------------------
# B. JSON round-trips
# ---------------------------------------------------------------------------


def test_task_roundtrip() -> None:
    t = Task(id="a", input="hi", expected="hi", scorer="exact", tags=["t1"], metadata={"k": "v"})
    d = json.loads(json.dumps(t.to_dict()))
    t2 = Task.from_dict(d)
    assert t2 == t
    assert isinstance(t2, Task)


def test_prediction_roundtrip() -> None:
    p = Prediction(output="x", latency_ms=1.5, tokens_in=2, tokens_out=3, cost_usd=0.01, error=None, raw={"a": 1})
    d = json.loads(json.dumps(p.to_dict()))
    p2 = Prediction.from_dict(d)
    assert p2 == p


def test_score_roundtrip() -> None:
    s = Score(value=0.7, passed=True, detail="ok", extra={"n": 1})
    d = json.loads(json.dumps(s.to_dict()))
    s2 = Score.from_dict(d)
    assert s2 == s
    assert isinstance(s2, Score)


def test_taskresult_roundtrip_nested_types() -> None:
    tr = TaskResult(
        task=Task(id="a", input="x", tags=["t"]),
        prediction=Prediction(output="y", latency_ms=3.0),
        score=Score(value=1.0, passed=True),
        attempts=2,
    )
    d = json.loads(json.dumps(tr.to_dict()))
    tr2 = TaskResult.from_dict(d)
    assert isinstance(tr2.task, Task)
    assert isinstance(tr2.prediction, Prediction)
    assert isinstance(tr2.score, Score)
    assert tr2 == tr


def test_runreport_roundtrip_nested_types() -> None:
    tr = TaskResult(
        task=Task(id="a", input="x"),
        prediction=Prediction(output="y"),
        score=Score(value=0.5, passed=False),
    )
    r = RunReport(
        run_id="r1",
        suite_name="s",
        agent_name="a",
        started_at="2024-01-01T00:00:00+00:00",
        results=[tr],
        config={"k": "v"},
    )
    d = json.loads(json.dumps(r.to_dict()))
    r2 = RunReport.from_dict(d)
    assert isinstance(r2.results[0], TaskResult)
    assert isinstance(r2.results[0].task, Task)
    assert r2 == r


def test_from_dict_tolerates_missing_optional_keys() -> None:
    t = Task.from_dict({"id": "a", "input": "x"})
    assert t.expected is None
    assert t.scorer == "exact"
    assert t.tags == []

    p = Prediction.from_dict({})
    assert p.output == ""
    assert p.error is None


def test_from_dict_ignores_unknown_keys() -> None:
    d = {"id": "a", "input": "x", "bogus": 1, "nonsense": [1, 2]}
    t = Task.from_dict(d)
    assert t.id == "a"
    assert t.input == "x"


# ---------------------------------------------------------------------------
# C. clamp01
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inp,exp",
    [
        (-1.0, 0.0),
        (-0.001, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.5, 1.0),
        (100.0, 1.0),
    ],
)
def test_clamp01(inp: float, exp: float) -> None:
    assert clamp01(inp) == exp


# ---------------------------------------------------------------------------
# D. summary()
# ---------------------------------------------------------------------------


def test_summary_empty() -> None:
    r = RunReport(run_id="r", suite_name="s", agent_name="a", started_at="2024-01-01T00:00:00+00:00")
    s = r.summary()
    assert s["total"] == 0
    assert s["passed"] == 0
    assert s["failed"] == 0
    assert s["errored"] == 0
    assert s["pass_rate"] == 0.0
    assert s["mean_score"] == 0.0
    assert s["total_cost_usd"] == 0.0
    assert s["mean_latency_ms"] == 0.0
    assert s["p95_latency_ms"] == 0.0
    assert s["by_tag"] == {}


def test_summary_full_pass() -> None:
    r = RunReport(
        run_id="r",
        suite_name="s",
        agent_name="a",
        started_at="2024-01-01T00:00:00+00:00",
        results=[
            TaskResult(
                task=Task(id="a", input="x"),
                prediction=Prediction(output="y", latency_ms=10.0, cost_usd=0.01),
                score=Score(value=1.0, passed=True),
            ),
            TaskResult(
                task=Task(id="b", input="y"),
                prediction=Prediction(output="z", latency_ms=30.0, cost_usd=0.02),
                score=Score(value=0.8, passed=True),
            ),
        ],
    )
    s = r.summary()
    assert s["total"] == 2
    assert s["passed"] == 2
    assert s["failed"] == 0
    assert s["errored"] == 0
    assert s["pass_rate"] == 1.0
    assert s["mean_score"] == pytest.approx(0.9)
    assert s["total_cost_usd"] == pytest.approx(0.03)
    assert s["mean_latency_ms"] == pytest.approx(20.0)


def test_summary_all_error() -> None:
    r = RunReport(
        run_id="r",
        suite_name="s",
        agent_name="a",
        started_at="2024-01-01T00:00:00+00:00",
        results=[
            TaskResult(
                task=Task(id="a", input="x"),
                prediction=Prediction(error="boom", latency_ms=5.0),
                score=Score(value=0.0, passed=False),
            ),
            TaskResult(
                task=Task(id="b", input="y"),
                prediction=Prediction(error="bang", latency_ms=7.0),
                score=Score(value=0.0, passed=False),
            ),
        ],
    )
    s = r.summary()
    assert s["total"] == 2
    assert s["passed"] == 0
    assert s["errored"] == 2
    assert s["failed"] == 0
    assert s["pass_rate"] == 0.0


def test_summary_by_tag_counts_per_tag() -> None:
    r = RunReport(
        run_id="r",
        suite_name="s",
        agent_name="a",
        started_at="2024-01-01T00:00:00+00:00",
        results=[
            TaskResult(
                task=Task(id="a", input="x", tags=["math", "easy"]),
                prediction=Prediction(output="1"),
                score=Score(value=1.0, passed=True),
            ),
            TaskResult(
                task=Task(id="b", input="y", tags=["math"]),
                prediction=Prediction(output="2"),
                score=Score(value=0.0, passed=False),
            ),
            TaskResult(
                task=Task(id="c", input="z", tags=["easy"]),
                prediction=Prediction(output="3"),
                score=Score(value=1.0, passed=True),
            ),
        ],
    )
    s = r.summary()
    assert s["by_tag"]["math"]["total"] == 2
    assert s["by_tag"]["math"]["passed"] == 1
    assert s["by_tag"]["math"]["pass_rate"] == pytest.approx(0.5)
    assert s["by_tag"]["easy"]["total"] == 2
    assert s["by_tag"]["easy"]["passed"] == 2
    assert s["by_tag"]["easy"]["pass_rate"] == 1.0


def test_summary_p95_one_sample() -> None:
    r = RunReport(
        run_id="r",
        suite_name="s",
        agent_name="a",
        started_at="2024-01-01T00:00:00+00:00",
        results=[
            TaskResult(
                task=Task(id="a", input="x"),
                prediction=Prediction(latency_ms=42.0),
                score=Score(value=1.0, passed=True),
            ),
        ],
    )
    assert r.summary()["p95_latency_ms"] == 42.0


def test_summary_p95_two_samples() -> None:
    r = RunReport(
        run_id="r",
        suite_name="s",
        agent_name="a",
        started_at="2024-01-01T00:00:00+00:00",
        results=[
            TaskResult(task=Task(id="a", input="x"), prediction=Prediction(latency_ms=10.0), score=Score(1.0, True)),
            TaskResult(task=Task(id="b", input="y"), prediction=Prediction(latency_ms=20.0), score=Score(1.0, True)),
        ],
    )
    assert r.summary()["p95_latency_ms"] == 20.0


def test_summary_p95_twenty_samples() -> None:
    results = [
        TaskResult(
            task=Task(id=f"t{i}", input=f"x{i}"),
            prediction=Prediction(latency_ms=float(i + 1)),
            score=Score(value=1.0, passed=True),
        )
        for i in range(20)
    ]
    r = RunReport(run_id="r", suite_name="s", agent_name="a", started_at="2024-01-01T00:00:00+00:00", results=results)
    assert r.summary()["p95_latency_ms"] == 19.0


def test_summary_pass_rate_empty_is_zero() -> None:
    r = RunReport(run_id="r", suite_name="s", agent_name="a", started_at="2024-01-01T00:00:00+00:00")
    assert r.summary()["pass_rate"] == 0.0


def test_summary_mean_score_treats_none_as_zero() -> None:
    r = RunReport(
        run_id="r",
        suite_name="s",
        agent_name="a",
        started_at="2024-01-01T00:00:00+00:00",
        results=[
            TaskResult(
                task=Task(id="a", input="x"),
                prediction=Prediction(output="y"),
                score=Score(value=1.0, passed=True),
            ),
            TaskResult(
                task=Task(id="b", input="y"),
                prediction=Prediction(output="z"),
                score=None,
            ),
        ],
    )
    assert r.summary()["mean_score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# E. Registry decorators
# ---------------------------------------------------------------------------


def test_register_agent_sets_name_if_missing(registry_cleanup) -> None:
    @register_agent("name_default_agent")
    class MyAgent:
        def predict(self, task: Task) -> Prediction:
            return Prediction()

    assert MyAgent.name == "name_default_agent"  # type: ignore[attr-defined]


def test_register_agent_does_not_overwrite_existing_name(registry_cleanup) -> None:
    @register_agent("some_agent")
    class MyAgent:
        name = "preset"

        def predict(self, task: Task) -> Prediction:
            return Prediction()

    assert MyAgent.name == "preset"


def test_register_scorer_sets_name_if_missing(registry_cleanup) -> None:
    @register_scorer("name_default_scorer")
    class MyScorer:
        def score(self, task: Task, prediction: Prediction) -> Score:
            return Score(value=1.0, passed=True)

    assert MyScorer.name == "name_default_scorer"  # type: ignore[attr-defined]


def test_register_scorer_does_not_overwrite_existing_name(registry_cleanup) -> None:
    @register_scorer("some_scorer")
    class MyScorer:
        name = "preset"

        def score(self, task: Task, prediction: Prediction) -> Score:
            return Score(value=1.0, passed=True)

    assert MyScorer.name == "preset"


def test_get_agent_unknown_raises_with_available(registry_cleanup) -> None:
    @register_agent("alpha")
    class A:
        def predict(self, task: Task) -> Prediction:
            return Prediction()

    @register_agent("beta")
    class B:
        def predict(self, task: Task) -> Prediction:
            return Prediction()

    with pytest.raises(UnknownComponentError) as excinfo:
        get_agent("nonexistent")
    msg = str(excinfo.value)
    assert "alpha" in msg
    assert "beta" in msg


def test_get_scorer_unknown_raises_with_available(registry_cleanup) -> None:
    @register_scorer("s1")
    class S1:
        def score(self, task: Task, prediction: Prediction) -> Score:
            return Score(value=1.0, passed=True)

    @register_scorer("s2")
    class S2:
        def score(self, task: Task, prediction: Prediction) -> Score:
            return Score(value=1.0, passed=True)

    with pytest.raises(UnknownComponentError) as excinfo:
        get_scorer("missing")
    msg = str(excinfo.value)
    assert "s1" in msg
    assert "s2" in msg


def test_list_agents_sorted(registry_cleanup) -> None:
    @register_agent("zebra")
    class Z:
        def predict(self, task: Task) -> Prediction:
            return Prediction()

    @register_agent("apple")
    class A:
        def predict(self, task: Task) -> Prediction:
            return Prediction()

    assert list_agents() == sorted(list_agents())
    assert "apple" in list_agents()
    assert "zebra" in list_agents()


def test_list_scorers_sorted(registry_cleanup) -> None:
    @register_scorer("two")
    class T:
        def score(self, task: Task, prediction: Prediction) -> Score:
            return Score(value=1.0, passed=True)

    @register_scorer("one")
    class O:
        def score(self, task: Task, prediction: Prediction) -> Score:
            return Score(value=1.0, passed=True)

    assert list_scorers() == sorted(list_scorers())
    assert "one" in list_scorers()
    assert "two" in list_scorers()


def test_reregister_same_name_overwrites_silently(registry_cleanup) -> None:
    @register_agent("dup")
    class First:
        def predict(self, task: Task) -> Prediction:
            return Prediction(output="first")

    @register_agent("dup")
    class Second:
        def predict(self, task: Task) -> Prediction:
            return Prediction(output="second")

    agent = get_agent("dup")
    assert agent.predict(Task(id="a", input="x")).output == "second"
