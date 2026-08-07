from __future__ import annotations

import http.client
import urllib.error

import pytest

from ageval.core import Prediction, Task
from ageval.adapters.openai_native import OpenAiNativeAgent
from ageval.errors import ConfigError


def _task(text: str = "hello world") -> Task:
    return Task(id="t1", input=text)


# --------------------------------------------------------------------------- #
# A. Constructor + API-key fallback
# --------------------------------------------------------------------------- #

def test_openai_native_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        OpenAiNativeAgent(model="x")


def test_openai_native_env_api_key_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
    agent = OpenAiNativeAgent(model="gpt-4")
    assert agent.api_key == "env-secret"
    assert agent.model == "gpt-4"


def test_openai_native_explicit_api_key_works():
    agent = OpenAiNativeAgent(model="gpt-4", api_key="explicit-secret")
    assert agent.api_key == "explicit-secret"


def test_openai_native_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
    agent = OpenAiNativeAgent(model="gpt-4", api_key="explicit-secret")
    assert agent.api_key == "explicit-secret"


# --------------------------------------------------------------------------- #
# B. Payload construction + system messages
# --------------------------------------------------------------------------- #

def test_openai_native_payload_construction_and_usage(monkeypatch):
    agent = OpenAiNativeAgent(
        model="gpt-4",
        api_key="k",
        system="be brief",
        temperature=0.2,
        max_tokens=8,
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


def test_openai_native_system_prompt_is_first_message(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k", system="you are terse")
    captured: dict = {}

    def fake_post(payload: dict) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(agent, "_post", fake_post)
    agent.predict(_task("hi"))

    messages = captured["payload"]["messages"]
    assert messages[0] == {"role": "system", "content": "you are terse"}
    assert messages[1] == {"role": "user", "content": "hi"}


def test_openai_native_system_absent_no_system_message(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k")
    captured: dict = {}

    def fake_post(payload: dict) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(agent, "_post", fake_post)
    agent.predict(_task("hi"))

    messages = captured["payload"]["messages"]
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "hi"}


def test_openai_native_user_input_reflects_task(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k")
    captured: dict = {}

    def fake_post(payload: dict) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(agent, "_post", fake_post)
    agent.predict(_task("specific prompt text"))
    assert captured["payload"]["messages"][0] == {
        "role": "user",
        "content": "specific prompt text",
    }


# --------------------------------------------------------------------------- #
# C. Usage / raw parsing
# --------------------------------------------------------------------------- #

def test_openai_native_usage_parsed_into_tokens(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k")
    monkeypatch.setattr(
        agent,
        "_post",
        lambda payload: {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
        },
    )
    pred = agent.predict(_task("hi"))
    assert pred.tokens_in == 13
    assert pred.tokens_out == 5


def test_openai_native_missing_usage_defaults_to_zero(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k")
    monkeypatch.setattr(
        agent,
        "_post",
        lambda payload: {"choices": [{"message": {"content": "done"}}]},
    )
    pred = agent.predict(_task("hi"))
    assert pred.tokens_in == 0
    assert pred.tokens_out == 0


def test_openai_native_raw_carries_full_response(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k")
    response = {
        "choices": [{"message": {"content": "done", "role": "assistant"}}],
        "usage": {"prompt_tokens": 1},
        "model": "gpt-4",
        "id": "chatcmpl-123",
    }
    monkeypatch.setattr(agent, "_post", lambda payload: response)
    pred = agent.predict(_task("hi"))
    assert pred.raw is response


# --------------------------------------------------------------------------- #
# D. Success latency
# --------------------------------------------------------------------------- #

def test_openai_native_success_latency_recorded(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k")
    monkeypatch.setattr(
        agent,
        "_post",
        lambda payload: {"choices": [{"message": {"content": "ok"}}]},
    )
    pred = agent.predict(_task("hi"))
    assert pred.error is None
    assert isinstance(pred.latency_ms, float)
    assert pred.latency_ms >= 0.0


# --------------------------------------------------------------------------- #
# E. 400 does not retry
# --------------------------------------------------------------------------- #

def test_openai_native_400_no_retry(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k")
    calls: list = []
    err = urllib.error.HTTPError(
        "http://x", 400, "bad request", http.client.HTTPMessage(), None
    )

    def fake_post(payload: dict) -> dict:
        calls.append(payload)
        raise err

    monkeypatch.setattr(agent, "_post", fake_post)
    pred = agent.predict(_task("hi"))
    assert pred.error is not None
    assert pred.output == ""
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# F. 429 / 5xx retry count (via monkeypatch)
# --------------------------------------------------------------------------- #

def test_openai_native_429_retries_three_times(monkeypatch):
    import ageval.adapters.openai_native as mod

    monkeypatch.setattr(mod.time, "sleep", lambda *a, **k: None)
    agent = OpenAiNativeAgent(model="m", api_key="k")
    calls: list = []
    err = urllib.error.HTTPError(
        "http://x", 429, "rate limited", http.client.HTTPMessage(), None
    )

    def fake_post(payload: dict) -> dict:
        calls.append(payload)
        raise err

    monkeypatch.setattr(agent, "_post", fake_post)
    pred = agent.predict(_task("hi"))
    assert pred.error is not None
    assert pred.output == ""
    assert len(calls) == 3


def test_openai_native_5xx_retries_three_times(monkeypatch):
    import ageval.adapters.openai_native as mod

    monkeypatch.setattr(mod.time, "sleep", lambda *a, **k: None)
    agent = OpenAiNativeAgent(model="m", api_key="k")
    calls: list = []
    err = urllib.error.HTTPError(
        "http://x", 503, "service unavailable", http.client.HTTPMessage(), None
    )

    def fake_post(payload: dict) -> dict:
        calls.append(payload)
        raise err

    monkeypatch.setattr(agent, "_post", fake_post)
    pred = agent.predict(_task("hi"))
    assert pred.error is not None
    assert pred.output == ""
    assert len(calls) == 3


def test_openai_native_429_recovers_after_transient(monkeypatch):
    import ageval.adapters.openai_native as mod

    monkeypatch.setattr(mod.time, "sleep", lambda *a, **k: None)
    agent = OpenAiNativeAgent(model="m", api_key="k")
    calls: list = []
    err = urllib.error.HTTPError(
        "http://x", 429, "rate limited", http.client.HTTPMessage(), None
    )

    def fake_post(payload: dict) -> dict:
        calls.append(payload)
        if len(calls) < 2:
            raise err
        return {"choices": [{"message": {"content": "survived"}}]}

    monkeypatch.setattr(agent, "_post", fake_post)
    pred = agent.predict(_task("hi"))
    assert pred.error is None
    assert pred.output == "survived"
    assert len(calls) == 2


def test_openai_native_507_no_retry_after_exhaustion_still_400(monkeypatch):
    # 4xx (other than 429) should not retry; verify a distinct 4xx path.
    agent = OpenAiNativeAgent(model="m", api_key="k")
    calls: list = []
    err = urllib.error.HTTPError(
        "http://x", 401, "unauthorized", http.client.HTTPMessage(), None
    )

    def fake_post(payload: dict) -> dict:
        calls.append(payload)
        raise err

    monkeypatch.setattr(agent, "_post", fake_post)
    pred = agent.predict(_task("hi"))
    assert pred.error is not None
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# G. Predict error handling without network
# --------------------------------------------------------------------------- #

def test_openai_native_predict_never_raises_when_post_raises(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k")

    def boom(payload: dict) -> dict:
        raise RuntimeError("network down")

    monkeypatch.setattr(agent, "_post", boom)
    pred = agent.predict(_task("hi"))
    assert pred.error is not None
    assert pred.error == "network down"
    assert pred.output == ""


def test_openai_native_predict_handles_malformed_response(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k")

    def missing_choices(payload: dict) -> dict:
        return {"unexpected": "shape"}

    monkeypatch.setattr(agent, "_post", missing_choices)
    pred = agent.predict(_task("hi"))
    assert pred.error is not None
    assert pred.output == ""


def test_openai_native_predict_handles_empty_content(monkeypatch):
    agent = OpenAiNativeAgent(model="m", api_key="k")
    monkeypatch.setattr(
        agent,
        "_post",
        lambda payload: {"choices": [{"message": {"content": ""}}]},
    )
    pred = agent.predict(_task("hi"))
    assert pred.error is None
    assert pred.output == ""


def test_openai_native_predict_latency_on_error(monkeypatch):
    import ageval.adapters.openai_native as mod

    monkeypatch.setattr(mod.time, "sleep", lambda *a, **k: None)
    agent = OpenAiNativeAgent(model="m", api_key="k")
    err = urllib.error.HTTPError(
        "http://x", 500, "server", http.client.HTTPMessage(), None
    )
    monkeypatch.setattr(agent, "_post", lambda payload: (_ for _ in ()).throw(err))
    pred = agent.predict(_task("hi"))
    assert pred.error is not None
    assert isinstance(pred.latency_ms, float)
    assert pred.latency_ms >= 0.0
