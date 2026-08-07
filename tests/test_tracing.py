from __future__ import annotations

import json

import pytest

from ageval.conversation import Turn
from ageval.core import Prediction
from ageval.tracing import (
    ToolCall,
    Trace,
    _make_json_safe,
    _tool_calls_match,
    record_tool_call,
    tool_call_summary,
    trace_matches_expected,
    validate_trace,
)


class _NotJsonSafe:
    """An object that json.dumps cannot serialize natively."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


# ---------------------------------------------------------------------------
# A. ToolCall to_dict / from_dict JSON-safe round trips
# ---------------------------------------------------------------------------


def test_toolcall_defaults() -> None:
    tc = ToolCall(name="search")
    assert tc.arguments == {}
    assert tc.result is None
    assert tc.latency_ms == 0.0
    assert tc.error is None


def test_toolcall_roundtrip_full() -> None:
    tc = ToolCall(
        name="search",
        arguments={"q": "weather", "filters": {"lang": "en"}},
        result={"hits": [1, 2, 3], "ok": True},
        latency_ms=42.5,
        error=None,
    )
    d = json.loads(json.dumps(tc.to_dict()))
    tc2 = ToolCall.from_dict(d)
    assert tc2 == tc
    assert isinstance(tc2, ToolCall)
    assert tc2.latency_ms == 42.5
    assert tc2.result == {"hits": [1, 2, 3], "ok": True}


def test_toolcall_roundtrip_minimal() -> None:
    tc = ToolCall(name="noop", arguments={})
    d = json.loads(json.dumps(tc.to_dict()))
    tc2 = ToolCall.from_dict(d)
    assert tc2 == tc
    assert tc2.arguments == {}
    assert tc2.result is None


def test_toolcall_roundtrip_with_error() -> None:
    tc = ToolCall(
        name="fetch",
        arguments={"url": "/x"},
        result=None,
        latency_ms=1.0,
        error="connection refused",
    )
    tc2 = ToolCall.from_dict(json.loads(json.dumps(tc.to_dict())))
    assert tc2 == tc
    assert tc2.error == "connection refused"


def test_toolcall_to_dict_makes_non_json_safe_repr() -> None:
    tc = ToolCall(name="weird", arguments={"obj": _NotJsonSafe("z")}, result=_NotJsonSafe("r"))
    safe = tc.to_dict()
    assert isinstance(safe["arguments"]["obj"], str)
    assert isinstance(safe["result"], str)
    # And the safe repr is itself JSON serializable.
    json.dumps(safe)


def test_make_json_safe_passes_through_primitives() -> None:
    assert _make_json_safe(None) is None
    assert _make_json_safe(True) is True
    assert _make_json_safe(3) == 3
    assert _make_json_safe(1.5) == 1.5
    assert _make_json_safe("s") == "s"


def test_make_json_safe_handles_collections() -> None:
    out = _make_json_safe({"a": [1, 2, (3, 4)]})
    # Tuples are converted to lists recursively.
    assert out == {"a": [1, 2, [3, 4]]}


def test_make_json_safe_dict_coerces_keys_to_str() -> None:
    out = _make_json_safe({1: "v", 2.0: "w"})
    assert out == {"1": "v", "2.0": "w"}


def test_make_json_safe_falls_back_to_repr() -> None:
    out = _make_json_safe(_NotJsonSafe("p"))
    assert out == repr(_NotJsonSafe("p"))


# ---------------------------------------------------------------------------
# B. ToolCall malformed optional data handling
# ---------------------------------------------------------------------------


def test_toolcall_from_dict_non_dict_input() -> None:
    tc = ToolCall.from_dict("not a dict")  # type: ignore[arg-type]
    assert tc.name == ""
    assert tc.arguments == {}
    assert tc.result is None
    assert tc.latency_ms == 0.0
    assert tc.error is None


def test_toolcall_from_dict_missing_name() -> None:
    tc = ToolCall.from_dict({"arguments": {"a": 1}})
    assert tc.name == ""


def test_toolcall_from_dict_name_non_str_coerced() -> None:
    tc = ToolCall.from_dict({"name": 5})
    assert tc.name == "5"
    assert isinstance(tc.name, str)


def test_toolcall_from_dict_name_none() -> None:
    tc = ToolCall.from_dict({"name": None})
    assert tc.name == ""


def test_toolcall_from_dict_arguments_not_dict() -> None:
    tc = ToolCall.from_dict({"arguments": ["x", "y"], "result": "z"})
    assert tc.arguments == {}
    assert tc.result == "z"


def test_toolcall_from_dict_missing_arguments() -> None:
    tc = ToolCall.from_dict({"name": "f"})
    assert tc.arguments == {}


def test_toolcall_from_dict_missing_latency_defaults_zero() -> None:
    assert ToolCall.from_dict({"name": "f"}).latency_ms == 0.0


def test_toolcall_from_dict_latency_coerced_to_float() -> None:
    assert ToolCall.from_dict({"name": "f", "latency_ms": 7}).latency_ms == 7.0
    assert isinstance(ToolCall.from_dict({"name": "f", "latency_ms": 7}).latency_ms, float)


def test_toolcall_from_dict_latency_bad_string_becomes_zero() -> None:
    tc = ToolCall.from_dict({"name": "f", "latency_ms": "abc"})
    assert tc.latency_ms == 0.0


def test_toolcall_from_dict_latency_none_becomes_zero() -> None:
    assert ToolCall.from_dict({"name": "f", "latency_ms": None}).latency_ms == 0.0


def test_toolcall_from_dict_error_non_str_coerced() -> None:
    tc = ToolCall.from_dict({"name": "f", "error": 404})
    assert tc.error == "404"
    assert isinstance(tc.error, str)


def test_toolcall_from_dict_error_none_preserved() -> None:
    assert ToolCall.from_dict({"name": "f", "error": None}).error is None


def test_toolcall_from_dict_result_passthrough() -> None:
    tc = ToolCall.from_dict({"name": "f", "result": {"k": [1, 2]}})
    assert tc.result == {"k": [1, 2]}


# ---------------------------------------------------------------------------
# C. Trace to_dict / from_dict JSON-safe round trips
# ---------------------------------------------------------------------------


def test_trace_defaults() -> None:
    tr = Trace(task_id="t1")
    assert tr.turns == []
    assert tr.tool_calls == []
    assert tr.final_prediction is None


def test_trace_roundtrip_empty() -> None:
    tr = Trace(task_id="t1")
    tr2 = Trace.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert tr2 == tr
    assert tr2.final_prediction is None


def test_trace_roundtrip_with_turns() -> None:
    turns = [
        Turn(role="user", content="hi", latency_ms=5.0),
        Turn(
            role="assistant",
            content="hello",
            tool_call={"name": "greet", "arguments": {"who": "world"}},
            latency_ms=8.0,
        ),
    ]
    tr = Trace(task_id="t1", turns=turns)
    tr2 = Trace.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert tr2 == tr
    assert len(tr2.turns) == 2
    assert all(isinstance(t, Turn) for t in tr2.turns)
    assert tr2.turns[1].tool_call == {"name": "greet", "arguments": {"who": "world"}}


def test_trace_roundtrip_with_tool_calls() -> None:
    calls = [
        ToolCall(name="search", arguments={"q": "x"}, result={"n": 1}, latency_ms=1.0),
        ToolCall(name="fetch", arguments={"u": "y"}, error="timeout", latency_ms=2.0),
    ]
    tr = Trace(task_id="t1", tool_calls=calls)
    tr2 = Trace.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert tr2 == tr
    assert len(tr2.tool_calls) == 2
    assert all(isinstance(c, ToolCall) for c in tr2.tool_calls)


def test_trace_roundtrip_with_final_prediction() -> None:
    pred = Prediction(
        output="42",
        latency_ms=12.0,
        tokens_in=3,
        tokens_out=1,
        cost_usd=0.001,
        raw={"model": "gpt-4"},
    )
    tr = Trace(task_id="t1", final_prediction=pred)
    tr2 = Trace.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert tr2 == tr
    assert isinstance(tr2.final_prediction, Prediction)
    assert tr2.final_prediction.output == "42"
    assert tr2.final_prediction.raw == {"model": "gpt-4"}


def test_trace_roundtrip_full_composite() -> None:
    tr = Trace(
        task_id="t99",
        turns=[
            Turn(role="user", content="compute", latency_ms=1.0),
            Turn(role="assistant", content="ok", latency_ms=2.0),
        ],
        tool_calls=[
            ToolCall(name="math", arguments={"expr": "2+2"}, result=4, latency_ms=0.5),
        ],
        final_prediction=Prediction(output="4", error=None, latency_ms=3.0),
    )
    tr2 = Trace.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert tr2 == tr


# ---------------------------------------------------------------------------
# D. Trace malformed data handling
# ---------------------------------------------------------------------------


def test_trace_from_dict_non_dict_input() -> None:
    tr = Trace.from_dict(None)  # type: ignore[arg-type]
    assert tr.task_id == ""
    assert tr.turns == []
    assert tr.tool_calls == []
    assert tr.final_prediction is None


def test_trace_from_dict_missing_task_id() -> None:
    assert Trace.from_dict({}).task_id == ""


def test_trace_from_dict_task_id_non_str_coerced() -> None:
    assert Trace.from_dict({"task_id": 7}).task_id == "7"


def test_trace_from_dict_task_id_none() -> None:
    assert Trace.from_dict({"task_id": None}).task_id == ""


def test_trace_from_dict_no_turns_key() -> None:
    tr = Trace.from_dict({"task_id": "t1"})
    assert tr.turns == []


def test_trace_from_dict_skips_bad_turns() -> None:
    # First turn valid, second has invalid role -> skipped, third valid.
    tr = Trace.from_dict(
        {
            "task_id": "t1",
            "turns": [
                {"role": "user", "content": "a"},
                {"role": "narrator", "content": "bad"},
                {"role": "assistant", "content": "b"},
            ],
        }
    )
    assert len(tr.turns) == 2
    assert [t.role for t in tr.turns] == ["user", "assistant"]


def test_trace_from_dict_handles_malformed_tool_call_entries() -> None:
    # ToolCall.from_dict is fully defensive: it never raises, so every entry
    # in the list yields a ToolCall (coercing bad fields to defaults).
    tr = Trace.from_dict(
        {
            "task_id": "t1",
            "tool_calls": [
                {"name": "ok", "arguments": {}},
                {"latency_ms": "x", "name": "tc2"},
                5,
                {"arguments": {"k": 1}},
            ],
        }
    )
    assert len(tr.tool_calls) == 4
    assert tr.tool_calls[0].name == "ok"
    assert tr.tool_calls[1].latency_ms == 0.0
    assert tr.tool_calls[2].name == ""
    assert tr.tool_calls[3].name == ""
    assert tr.tool_calls[3].arguments == {"k": 1}


def test_trace_from_dict_bad_final_prediction_becomes_none() -> None:
    tr = Trace.from_dict(
        {"task_id": "t1", "final_prediction": "a string is not a dict"}
    )
    assert tr.final_prediction is None


def test_trace_from_dict_final_prediction_missing_raw_defaults_empty() -> None:
    tr = Trace.from_dict({"task_id": "t1", "final_prediction": {"output": "x"}})
    assert tr.final_prediction is not None
    assert tr.final_prediction.output == "x"
    assert tr.final_prediction.raw == {}


def test_trace_from_dict_final_prediction_unparseable_becomes_none() -> None:
    # raw=[1, 2] cannot be coerced by dict(), so Prediction.from_dict raises
    # and the tolerant Trace loader catches it -> final_prediction is None.
    tr = Trace.from_dict(
        {"task_id": "t1", "final_prediction": {"output": "x", "raw": [1, 2]}}
    )
    assert tr.final_prediction is None


def test_trace_from_dict_final_prediction_none_preserved() -> None:
    tr = Trace.from_dict({"task_id": "t1", "final_prediction": None})
    assert tr.final_prediction is None


# ---------------------------------------------------------------------------
# E. tool_call_summary metrics
# ---------------------------------------------------------------------------


def test_tool_call_summary_empty() -> None:
    tr = Trace(task_id="t1")
    s = tool_call_summary(tr)
    assert s == {
        "call_count": 0,
        "error_count": 0,
        "total_latency_ms": 0.0,
        "names": [],
        "unique_names": [],
    }


def test_tool_call_summary_counts_and_names() -> None:
    tr = Trace(
        task_id="t1",
        tool_calls=[
            ToolCall(name="search", latency_ms=10.0),
            ToolCall(name="fetch", latency_ms=20.0, error="boom"),
            ToolCall(name="search", latency_ms=5.0),
            ToolCall(name="search", latency_ms=3.0, error="again"),
        ],
    )
    s = tool_call_summary(tr)
    assert s["call_count"] == 4
    assert s["error_count"] == 2
    assert s["total_latency_ms"] == 38.0
    assert s["names"] == ["search", "fetch", "search", "search"]
    assert s["unique_names"] == ["fetch", "search"]


def test_tool_call_summary_no_errors() -> None:
    tr = Trace(
        task_id="t1",
        tool_calls=[ToolCall(name="a", latency_ms=1.0), ToolCall(name="b", latency_ms=2.0)],
    )
    s = tool_call_summary(tr)
    assert s["error_count"] == 0
    assert s["total_latency_ms"] == 3.0


def test_tool_call_summary_error_none_not_counted() -> None:
    tr = Trace(
        task_id="t1",
        tool_calls=[ToolCall(name="a", error=None), ToolCall(name="b", error="x")],
    )
    s = tool_call_summary(tr)
    assert s["error_count"] == 1


def test_tool_call_summary_unique_names_sorted() -> None:
    tr = Trace(
        task_id="t1",
        tool_calls=[
            ToolCall(name="zeta"),
            ToolCall(name="alpha"),
            ToolCall(name="zeta"),
            ToolCall(name="beta"),
        ],
    )
    s = tool_call_summary(tr)
    assert s["unique_names"] == ["alpha", "beta", "zeta"]


# ---------------------------------------------------------------------------
# F. trace_matches_expected / _tool_calls_match
# ---------------------------------------------------------------------------


def test_tool_calls_match_identical() -> None:
    a = ToolCall(name="f", arguments={"k": 1}, result={"r": 2}, latency_ms=5.0, error=None)
    b = ToolCall(name="f", arguments={"k": 1}, result={"r": 2}, latency_ms=9.0, error="e")
    assert _tool_calls_match(a, b)


def test_tool_calls_match_different_name() -> None:
    a = ToolCall(name="f")
    b = ToolCall(name="g")
    assert not _tool_calls_match(a, b)


def test_tool_calls_match_different_arguments() -> None:
    a = ToolCall(name="f", arguments={"k": 1})
    b = ToolCall(name="f", arguments={"k": 2})
    assert not _tool_calls_match(a, b)


def test_tool_calls_match_different_result() -> None:
    a = ToolCall(name="f", arguments={}, result={"x": 1})
    b = ToolCall(name="f", arguments={}, result={"x": 2})
    assert not _tool_calls_match(a, b)


def test_trace_matches_expected_empty() -> None:
    assert trace_matches_expected(Trace(task_id="t"), Trace(task_id="t"))


def test_trace_matches_expected_equal_length() -> None:
    t1 = Trace(
        task_id="t",
        tool_calls=[ToolCall(name="a", arguments={"q": 1}, result={"n": 2})],
    )
    t2 = Trace(
        task_id="t",
        tool_calls=[ToolCall(name="a", arguments={"q": 1}, result={"n": 2})],
    )
    assert trace_matches_expected(t1, t2)


def test_trace_matches_expected_different_length() -> None:
    t1 = Trace(
        task_id="t",
        tool_calls=[ToolCall(name="a"), ToolCall(name="b")],
    )
    t2 = Trace(task_id="t", tool_calls=[ToolCall(name="a")])
    assert not trace_matches_expected(t1, t2)


def test_trace_matches_expected_name_mismatch() -> None:
    t1 = Trace(task_id="t", tool_calls=[ToolCall(name="a", arguments={})])
    t2 = Trace(task_id="t", tool_calls=[ToolCall(name="b", arguments={})])
    assert not trace_matches_expected(t1, t2)


def test_trace_matches_expected_ignore_latency_default_true() -> None:
    t1 = Trace(task_id="t", tool_calls=[ToolCall(name="a", latency_ms=1.0)])
    t2 = Trace(task_id="t", tool_calls=[ToolCall(name="a", latency_ms=99.0)])
    assert trace_matches_expected(t1, t2)


def test_trace_matches_expected_respect_latency_when_flag() -> None:
    t1 = Trace(task_id="t", tool_calls=[ToolCall(name="a", latency_ms=1.0)])
    t2 = Trace(task_id="t", tool_calls=[ToolCall(name="a", latency_ms=99.0)])
    assert not trace_matches_expected(t1, t2, ignore_latency=False)


def test_trace_matches_expected_ignore_errors_default_true() -> None:
    t1 = Trace(task_id="t", tool_calls=[ToolCall(name="a", error="e1")])
    t2 = Trace(task_id="t", tool_calls=[ToolCall(name="a", error="e2")])
    assert trace_matches_expected(t1, t2)


def test_trace_matches_expected_respect_errors_when_flag() -> None:
    t1 = Trace(task_id="t", tool_calls=[ToolCall(name="a", error="e1")])
    t2 = Trace(task_id="t", tool_calls=[ToolCall(name="a", error="e2")])
    assert not trace_matches_expected(t1, t2, ignore_errors=False)


def test_trace_matches_expected_respects_args_and_results() -> None:
    t1 = Trace(
        task_id="t",
        tool_calls=[ToolCall(name="a", arguments={"k": 1}, result={"r": 2})],
    )
    t2 = Trace(
        task_id="t",
        tool_calls=[ToolCall(name="a", arguments={"k": 9}, result={"r": 2})],
    )
    assert not trace_matches_expected(t1, t2)


# ---------------------------------------------------------------------------
# G. record_tool_call
# ---------------------------------------------------------------------------


def test_record_tool_call_appends_and_returns() -> None:
    tr = Trace(task_id="t1")
    tc = record_tool_call(tr, "search", arguments={"q": "x"}, result={"n": 1}, latency_ms=5.0)
    assert tc is not None
    assert tr.tool_calls[-1] is tc
    assert tc.name == "search"
    assert tc.arguments == {"q": "x"}
    assert tc.result == {"n": 1}
    assert tc.latency_ms == 5.0
    assert tc.error is None


def test_record_tool_call_defaults_arguments() -> None:
    tr = Trace(task_id="t1")
    tc = record_tool_call(tr, "noop")
    assert tc.arguments == {}
    assert len(tr.tool_calls) == 1


def test_record_tool_call_with_error() -> None:
    tr = Trace(task_id="t1")
    tc = record_tool_call(tr, "fail", error="bad", latency_ms=2.0)
    assert tc.error == "bad"
    assert tr.tool_calls[0].error == "bad"


# ---------------------------------------------------------------------------
# H. validate_trace
# ---------------------------------------------------------------------------


def test_validate_trace_clean() -> None:
    tr = Trace(
        task_id="t1",
        tool_calls=[ToolCall(name="a", arguments={}, latency_ms=1.0)],
    )
    assert validate_trace(tr) == []


def test_validate_trace_empty_task_id() -> None:
    tr = Trace(task_id="", tool_calls=[ToolCall(name="a")])
    assert "task_id is empty" in validate_trace(tr)


def test_validate_trace_empty_tool_name() -> None:
    tr = Trace(task_id="t1", tool_calls=[ToolCall(name="", arguments={})])
    issues = validate_trace(tr)
    assert any("name is empty" in i for i in issues)


def test_validate_trace_arguments_not_dict() -> None:
    tc = ToolCall(name="a")
    tc.arguments = ["not", "a", "dict"]  # type: ignore[assignment]
    tr = Trace(task_id="t1", tool_calls=[tc])
    issues = validate_trace(tr)
    assert any("arguments is not a dict" in i for i in issues)


def test_validate_trace_negative_latency() -> None:
    tr = Trace(task_id="t1", tool_calls=[ToolCall(name="a", arguments={}, latency_ms=-1.0)])
    issues = validate_trace(tr)
    assert any("latency_ms is negative" in i for i in issues)


def test_validate_trace_reports_multiple_issues() -> None:
    tr = Trace(
        task_id="",
        tool_calls=[ToolCall(name="", arguments="bad", latency_ms=-5.0)],
    )
    issues = validate_trace(tr)
    assert len(issues) == 4
    assert any("task_id" in i for i in issues)
    assert any("not a dict" in i for i in issues)
    assert any("negative" in i for i in issues)


# ---------------------------------------------------------------------------
# I. Final Prediction serialization through Trace
# ---------------------------------------------------------------------------


def test_trace_final_prediction_roundtrip_error_prediction() -> None:
    pred = Prediction(output="", error="failed", latency_ms=0.0, cost_usd=0.5)
    tr = Trace(task_id="t", final_prediction=pred)
    tr2 = Trace.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert tr2 == tr
    assert tr2.final_prediction.error == "failed"
    assert tr2.final_prediction.cost_usd == 0.5


def test_trace_final_prediction_none_serializes_as_null() -> None:
    tr = Trace(task_id="t", final_prediction=None)
    assert tr.to_dict()["final_prediction"] is None
    tr2 = Trace.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert tr2.final_prediction is None
