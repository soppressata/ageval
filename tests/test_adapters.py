from __future__ import annotations

import http.client
import sys
import urllib.error

import pytest

from ageval.core import Prediction, Task, list_agents
from ageval.adapters import EchoAgent, OpenRouterAgent, SubprocessAgent
from ageval.errors import ConfigError


# --------------------------------------------------------------------------- #
# A. EchoAgent
# --------------------------------------------------------------------------- #

def _task(text: str = "hello world") -> Task:
    return Task(id="t1", input=text)


def test_echo_returns_input():
    pred = EchoAgent().predict(_task("hello world"))
    assert pred.error is None
    assert pred.output == "hello world"
    assert pred.latency_ms >= 0
    assert pred.tokens_in == 2
    assert pred.tokens_out == 2


def test_echo_prefix():
    pred = EchoAgent(prefix=">> ").predict(_task("hi there"))
    assert pred.error is None
    assert pred.output == ">> hi there"
    assert pred.tokens_in == len(">> hi there".split())
    assert pred.tokens_out == len(">> hi there".split())


def test_echo_uppercase():
    pred = EchoAgent(uppercase=True).predict(_task("hello world"))
    assert pred.error is None
    assert pred.output == "HELLO WORLD"


def test_echo_prefix_and_uppercase_together():
    pred = EchoAgent(prefix=">> ", uppercase=True).predict(_task("hello world"))
    assert pred.error is None
    expected = ">> HELLO WORLD"
    assert pred.output == expected
    assert pred.tokens_in == len(expected.split())
    assert pred.tokens_out == len(expected.split())


class _HostileTask:
    """Object whose `.input` raises — predict must not raise."""

    @property
    def input(self) -> str:
        raise RuntimeError("cannot read input")


def test_echo_predict_never_raises_on_hostile_task():
    pred = EchoAgent().predict(_HostileTask())  # type: ignore[arg-type]
    assert pred.error is not None
    assert pred.output == ""
    assert pred.latency_ms >= 0


def test_echo_empty_input():
    pred = EchoAgent().predict(_task(""))
    assert pred.error is None
    assert pred.output == ""
    assert pred.tokens_in == 0
    assert pred.tokens_out == 0


def test_list_agents_contains_all_adapters():
    names = list_agents()
    assert "echo" in names
    assert "openrouter" in names
    assert "subprocess" in names


# --------------------------------------------------------------------------- #
# B. OpenRouterAgent
# --------------------------------------------------------------------------- #

def test_openrouter_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        OpenRouterAgent(model="x")


def test_openrouter_explicit_api_key_works():
    agent = OpenRouterAgent(model="gpt-4", api_key="secret")
    assert agent.model == "gpt-4"
    assert agent.api_key == "secret"


def test_openrouter_payload_construction_and_usage(monkeypatch):
    agent = OpenRouterAgent(
        model="gpt-4", api_key="k", system="be brief", temperature=0.2, max_tokens=8
    )
    captured: dict = {}

    def fake_post(payload: dict) -> dict:
        captured["payload"] = payload
        return {
            "choices": [{"message": {"content": "42"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        }

    monkeypatch.setattr(agent, "_post", fake_post)
    pred = agent.predict(_task("hello"))

    payload = captured["payload"]
    assert payload["model"] == "gpt-4"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 8
    assert "messages" in payload
    assert pred.error is None
    assert pred.output == "42"
    assert pred.tokens_in == 7
    assert pred.tokens_out == 2
    assert pred.raw["usage"] == {"prompt_tokens": 7, "completion_tokens": 2}


def test_openrouter_system_prompt_is_first_message(monkeypatch):
    agent = OpenRouterAgent(model="m", api_key="k", system="you are terse")
    captured: dict = {}

    def fake_post(payload: dict) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(agent, "_post", fake_post)
    agent.predict(_task("hi"))

    messages = captured["payload"]["messages"]
    assert messages[0] == {"role": "system", "content": "you are terse"}
    assert messages[1] == {"role": "user", "content": "hi"}


def test_openrouter_system_absent_no_system_message(monkeypatch):
    agent = OpenRouterAgent(model="m", api_key="k")
    captured: dict = {}

    def fake_post(payload: dict) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(agent, "_post", fake_post)
    agent.predict(_task("hi"))

    messages = captured["payload"]["messages"]
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "hi"}


def test_openrouter_predict_never_raises_when_post_raises(monkeypatch):
    agent = OpenRouterAgent(model="m", api_key="k")

    def boom(payload: dict) -> dict:
        raise RuntimeError("network down")

    monkeypatch.setattr(agent, "_post", boom)
    pred = agent.predict(_task("hi"))
    assert pred.error is not None
    assert pred.output == ""


def test_openrouter_retry_exhaustion_returns_error(monkeypatch):
    import ageval.adapters.openrouter as or_mod

    monkeypatch.setattr(or_mod.time, "sleep", lambda *a, **k: None)
    agent = OpenRouterAgent(model="m", api_key="k")
    calls: list = []

    def boom(payload: dict) -> dict:
        calls.append(payload)
        raise urllib.error.HTTPError("http://x", 429, "rate", http.client.HTTPMessage(), None)

    monkeypatch.setattr(agent, "_post", boom)
    pred = agent.predict(_task("hi"))
    assert pred.error is not None
    assert pred.output == ""
    assert len(calls) == 3


# --------------------------------------------------------------------------- #
# D. SubprocessAgent
# --------------------------------------------------------------------------- #

def test_subprocess_success():
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"]
    agent = SubprocessAgent(command=cmd)
    pred = agent.predict(_task("hello"))
    assert pred.error is None
    assert pred.output == "HELLO"


def test_subprocess_nonzero_exit_errors(monkeypatch):
    cmd = [
        sys.executable, "-c",
        "import sys; sys.stderr.write('boom'); sys.exit(3)",
    ]
    agent = SubprocessAgent(command=cmd)
    pred = agent.predict(_task("ignored"))
    assert pred.error is not None
    assert "3" in pred.error
    assert "boom" in pred.error


def test_subprocess_timeout():
    cmd = [sys.executable, "-c", "import time; time.sleep(5)"]
    agent = SubprocessAgent(command=cmd, timeout=0.2)
    pred = agent.predict(_task("ignored"))
    assert pred.error is not None
    assert "timeout" in pred.error.lower()


def test_subprocess_empty_command_raises_config_error():
    with pytest.raises(ConfigError):
        SubprocessAgent(command=[])
