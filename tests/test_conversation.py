from __future__ import annotations

import json
from typing import Any

import pytest

from ageval.core import Task
from ageval.conversation import (
    ALLOWED_ROLES,
    Turn,
    ConversationResult,
    ConversationalAgent,
)


# ---------------------------------------------------------------------------
# A. Turn validation
# ---------------------------------------------------------------------------


def test_allowed_roles_contains_expected() -> None:
    assert ALLOWED_ROLES == frozenset({"user", "assistant", "system", "tool"})


@pytest.mark.parametrize("role", ["user", "assistant", "system", "tool"])
def test_turn_accepts_valid_roles(role: str) -> None:
    t = Turn(role=role, content="hello")
    assert t.role == role
    assert t.content == "hello"


def test_turn_invalid_role_raises_in_constructor() -> None:
    with pytest.raises(ValueError, match="invalid role"):
        Turn(role="admin", content="x")


def test_turn_invalid_role_message_lists_allowed() -> None:
    with pytest.raises(ValueError) as excinfo:
        Turn(role="narrator", content="x")
    msg = str(excinfo.value)
    for role in ("user", "assistant", "system", "tool"):
        assert role in msg


# ---------------------------------------------------------------------------
# B. Turn to_dict / from_dict round trips
# ---------------------------------------------------------------------------


def test_turn_defaults() -> None:
    t = Turn(role="user", content="hi")
    assert t.tool_call is None
    assert t.tool_result is None
    assert t.latency_ms == 0.0


def test_turn_mutable_defaults_not_shared() -> None:
    t = Turn(role="assistant", content="ok")
    d = t.to_dict()
    assert d["tool_call"] is None
    assert d["tool_result"] is None


def test_turn_roundtrip_full() -> None:
    t = Turn(
        role="user",
        content="do something",
        tool_call={"name": "search", "arguments": {"q": "weather"}},
        tool_result={"temperature": 21},
        latency_ms=12.5,
    )
    d = json.loads(json.dumps(t.to_dict()))
    t2 = Turn.from_dict(d)
    assert t2 == t
    assert isinstance(t2, Turn)


def test_turn_roundtrip_minimal() -> None:
    t = Turn(role="system", content="you are helpful")
    d = json.loads(json.dumps(t.to_dict()))
    t2 = Turn.from_dict(d)
    assert t2 == t


def test_turn_from_dict_tolerates_missing_optionals() -> None:
    t = Turn.from_dict({"role": "user"})
    assert t.role == "user"
    assert t.content == ""
    assert t.tool_call is None
    assert t.tool_result is None
    assert t.latency_ms == 0.0


def test_turn_from_dict_ignores_unknown_keys() -> None:
    t = Turn.from_dict({"role": "user", "content": "x", "bogus": 1, "extra": [1, 2]})
    assert t.role == "user"
    assert t.content == "x"


def test_turn_from_dict_coerces_latency_to_float() -> None:
    t = Turn.from_dict({"role": "assistant", "latency_ms": 7})
    assert isinstance(t.latency_ms, float)
    assert t.latency_ms == 7.0


# ---------------------------------------------------------------------------
# C. Turn malformed / invalid data
# ---------------------------------------------------------------------------


def test_turn_from_dict_missing_role_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        Turn.from_dict({"content": "no role here"})


def test_turn_from_dict_invalid_role_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="invalid role"):
        Turn.from_dict({"role": "root", "content": "x"})


def test_turn_from_dict_non_numeric_latency_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        Turn.from_dict({"role": "user", "latency_ms": "not-a-number"})


# ---------------------------------------------------------------------------
# D. ConversationResult round trip and totals
# ---------------------------------------------------------------------------


def test_conversationresult_defaults() -> None:
    cr = ConversationResult()
    assert cr.turns == []
    assert cr.final_output == ""
    assert cr.total_latency_ms == 0.0
    assert cr.total_cost_usd == 0.0
    assert cr.raw == {}


def test_conversationresult_roundtrip_preserves_totals() -> None:
    turns = [
        Turn(role="user", content="hi", latency_ms=10.0),
        Turn(role="assistant", content="hello", latency_ms=20.0),
    ]
    cr = ConversationResult(
        turns=turns,
        final_output="hello",
        total_latency_ms=30.0,
        total_cost_usd=0.05,
        raw={"model": "gpt-4", "id": "chat-1"},
    )
    d = json.loads(json.dumps(cr.to_dict()))
    cr2 = ConversationResult.from_dict(d)
    assert cr2 == cr
    assert isinstance(cr2, ConversationResult)
    assert cr2.total_latency_ms == 30.0
    assert cr2.total_cost_usd == pytest.approx(0.05)
    assert cr2.final_output == "hello"
    assert cr2.raw == {"model": "gpt-4", "id": "chat-1"}


def test_conversationresult_roundtrip_rebuilds_turn_instances() -> None:
    turns = [
        Turn(role="system", content="sys"),
        Turn(role="user", content="ask", tool_call={"name": "f", "arguments": {}}),
    ]
    cr = ConversationResult(turns=turns, final_output="ans", total_cost_usd=0.01)
    d = json.loads(json.dumps(cr.to_dict()))
    cr2 = ConversationResult.from_dict(d)
    assert len(cr2.turns) == 2
    assert all(isinstance(t, Turn) for t in cr2.turns)
    assert cr2.turns[0].role == "system"
    assert cr2.turns[1].tool_call == {"name": "f", "arguments": {}}
    assert cr2 == cr


def test_conversationresult_from_dict_tolerates_empty() -> None:
    cr = ConversationResult.from_dict({})
    assert cr.turns == []
    assert cr.final_output == ""
    assert cr.total_latency_ms == 0.0
    assert cr.total_cost_usd == 0.0
    assert cr.raw == {}


def test_conversationresult_from_dict_tolerates_no_turns_key() -> None:
    cr = ConversationResult.from_dict({"final_output": "ok", "total_cost_usd": 0.2})
    assert cr.turns == []
    assert cr.final_output == "ok"
    assert cr.total_cost_usd == pytest.approx(0.2)


def test_conversationresult_from_dict_invalid_turn_role_propagates() -> None:
    d: dict[str, Any] = {
        "turns": [{"role": "hacker", "content": "x"}],
        "final_output": "",
        "total_latency_ms": 0.0,
        "total_cost_usd": 0.0,
        "raw": {},
    }
    with pytest.raises(ValueError, match="invalid role"):
        ConversationResult.from_dict(d)


def test_conversationresult_mutable_defaults_not_shared() -> None:
    cr1 = ConversationResult()
    cr1.turns.append(Turn(role="user", content="a"))
    cr1.raw["k"] = "v"
    cr2 = ConversationResult()
    assert cr2.turns == []
    assert cr2.raw == {}


# ---------------------------------------------------------------------------
# E. ConversationalAgent runtime_checkable protocol behavior
# ---------------------------------------------------------------------------


def test_protocol_is_runtime_checkable() -> None:
    assert getattr(ConversationalAgent, "_is_runtime_protocol", False) is True


def test_protocol_accepts_structurally_conforming_instance() -> None:
    class GoodAgent:
        def converse(
            self,
            task: Task,
            history: list[Turn] | None = None,
        ) -> ConversationResult:
            return ConversationResult()

    assert isinstance(GoodAgent(), ConversationalAgent)


def test_protocol_rejects_non_conforming_instance() -> None:
    class NoConverse:
        pass

    assert not isinstance(NoConverse(), ConversationalAgent)


def test_protocol_rejects_object_with_wrong_method() -> None:
    class PredictOnly:
        def predict(self, task: Task) -> Any:
            return None

    assert not isinstance(PredictOnly(), ConversationalAgent)


def test_protocol_rejects_primitive() -> None:
    assert not isinstance(42, ConversationalAgent)
    assert not isinstance("string", ConversationalAgent)
    assert not isinstance(None, ConversationalAgent)  # type: ignore[arg-type]
