from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ageval.conversation import Turn
from ageval.core import Prediction


def _make_json_safe(value: Any) -> Any:
    """Recursively convert a value to a JSON-safe representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class ToolCall:
    """A single tool invocation recorded during a trace.

    Attributes:
        name: Identifier of the tool that was called.
        arguments: Key-value arguments passed to the tool.
        result: Return value from the tool, if any.
        latency_ms: Wall-clock duration of the call in milliseconds.
        error: Error message if the call failed, otherwise None.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool call to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "arguments": _make_json_safe(self.arguments),
            "result": _make_json_safe(self.result),
            "latency_ms": float(self.latency_ms),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """Deserialize a tool call from a dictionary.

        Malformed or missing fields fall back to defaults rather than
        raising, so that partial trace data can still be loaded.
        """
        if not isinstance(data, dict):
            data = {}
        name = data.get("name", "")
        if not isinstance(name, str):
            name = str(name) if name is not None else ""
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        result = data.get("result", None)
        latency_ms = data.get("latency_ms", 0.0)
        try:
            latency_ms = float(latency_ms)
        except (TypeError, ValueError):
            latency_ms = 0.0
        error = data.get("error")
        if error is not None and not isinstance(error, str):
            error = str(error)
        return cls(
            name=name,
            arguments=arguments,
            result=result,
            latency_ms=latency_ms,
            error=error,
        )


@dataclass
class Trace:
    """A recorded trace of a single task execution.

    Attributes:
        task_id: Identifier of the task this trace belongs to.
        turns: Ordered list of conversation turns.
        tool_calls: Tool invocations recorded during the run.
        final_prediction: The agent's final output, if any.
    """

    task_id: str
    turns: list[Turn] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_prediction: Prediction | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this trace to a JSON-safe dictionary."""
        return {
            "task_id": self.task_id,
            "turns": [t.to_dict() for t in self.turns],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "final_prediction": (
                self.final_prediction.to_dict()
                if self.final_prediction is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        """Deserialize a trace from a dictionary.

        Malformed sections fall back to empty lists / None rather than
        raising, so that partial trace data can still be loaded.
        """
        if not isinstance(data, dict):
            data = {}
        task_id = data.get("task_id", "")
        if not isinstance(task_id, str):
            task_id = str(task_id) if task_id is not None else ""

        turns_raw = data.get("turns", [])
        turns: list[Turn] = []
        if isinstance(turns_raw, list):
            for t in turns_raw:
                try:
                    turns.append(Turn.from_dict(t))
                except Exception:
                    continue

        tool_calls_raw = data.get("tool_calls", [])
        tool_calls: list[ToolCall] = []
        if isinstance(tool_calls_raw, list):
            for tc in tool_calls_raw:
                try:
                    tool_calls.append(ToolCall.from_dict(tc))
                except Exception:
                    continue

        final_prediction: Prediction | None = None
        fp_raw = data.get("final_prediction")
        if fp_raw is not None:
            try:
                final_prediction = Prediction.from_dict(fp_raw)
            except Exception:
                final_prediction = None

        return cls(
            task_id=task_id,
            turns=turns,
            tool_calls=tool_calls,
            final_prediction=final_prediction,
        )


def record_tool_call(
    trace: Trace,
    name: str,
    arguments: dict[str, Any] | None = None,
    result: Any = None,
    latency_ms: float = 0.0,
    error: str | None = None,
) -> ToolCall:
    """Create and append a ToolCall to *trace*, returning it for chaining."""
    call = ToolCall(
        name=name,
        arguments=arguments if arguments is not None else {},
        result=result,
        latency_ms=latency_ms,
        error=error,
    )
    trace.tool_calls.append(call)
    return call


def validate_trace(trace: Trace) -> list[str]:
    """Return a list of human-readable issues found in *trace*.

    An empty list indicates the trace is well-formed.  This never raises;
    it reports problems as strings so callers can decide how to handle them.
    """
    issues: list[str] = []
    if not trace.task_id:
        issues.append("task_id is empty")
    for i, tc in enumerate(trace.tool_calls):
        if not tc.name:
            issues.append(f"tool_calls[{i}].name is empty")
        if not isinstance(tc.arguments, dict):
            issues.append(f"tool_calls[{i}].arguments is not a dict")
        if tc.latency_ms < 0:
            issues.append(f"tool_calls[{i}].latency_ms is negative")
    return issues


def tool_call_summary(trace: Trace) -> dict[str, Any]:
    """Compute tool-use summary metrics for *trace*.

    Returns a dict with:
        - ``call_count``: total number of tool calls.
        - ``error_count``: number of calls with an error set.
        - ``total_latency_ms``: sum of all call latencies.
        - ``names``: ordered list of tool names called.
        - ``unique_names``: sorted list of distinct tool names.
    """
    call_count = len(trace.tool_calls)
    error_count = sum(1 for tc in trace.tool_calls if tc.error is not None)
    total_latency_ms = sum(tc.latency_ms for tc in trace.tool_calls)
    names = [tc.name for tc in trace.tool_calls]
    unique_names = sorted(set(names))
    return {
        "call_count": call_count,
        "error_count": error_count,
        "total_latency_ms": total_latency_ms,
        "names": names,
        "unique_names": unique_names,
    }


def _tool_calls_match(actual: ToolCall, expected: ToolCall) -> bool:
    """Check whether two tool calls match on name, arguments, and result."""
    if actual.name != expected.name:
        return False
    if _make_json_safe(actual.arguments) != _make_json_safe(expected.arguments):
        return False
    if _make_json_safe(actual.result) != _make_json_safe(expected.result):
        return False
    return True


def trace_matches_expected(
    trace: Trace,
    expected: Trace,
    *,
    ignore_latency: bool = True,
    ignore_errors: bool = True,
) -> bool:
    """Compare *trace* against *expected* on tool-call arguments and results.

    Args:
        ignore_latency: If True (default), ``latency_ms`` is not compared.
        ignore_errors: If True (default), ``error`` is not compared.

    Returns True only when the tool-call sequences match in length and
    content (name, arguments, result).  When *ignore_latency* or
    *ignore_errors* is False, those fields are also compared.
    """
    actual_calls = trace.tool_calls
    expected_calls = expected.tool_calls
    if len(actual_calls) != len(expected_calls):
        return False
    for act, exp in zip(actual_calls, expected_calls):
        if not _tool_calls_match(act, exp):
            return False
        if not ignore_latency and act.latency_ms != exp.latency_ms:
            return False
        if not ignore_errors and act.error != exp.error:
            return False
    return True
