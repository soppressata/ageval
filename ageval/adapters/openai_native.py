from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request

from ageval.core import Prediction, Task, register_agent
from ageval.errors import ConfigError


@register_agent("openai_native")
class OpenAiNativeAgent:
    """Agent that calls an OpenAI-compatible chat completions API."""

    name: str = "openai_native"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigError(
                "OpenAI api_key must be provided or set via OPENAI_API_KEY env var"
            )
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")

    def _post(self, payload: dict) -> dict:
        """Make the HTTP POST to the chat completions endpoint. Isolated for monkeypatching."""
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def predict(self, task: Task) -> Prediction:
        start = time.perf_counter()
        try:
            messages: list[dict[str, str]] = []
            if self.system is not None:
                messages.append({"role": "system", "content": self.system})
            messages.append({"role": "user", "content": task.input})
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            response = self._with_retries(payload)
            output = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return Prediction(
                output=output,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                raw=response,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return Prediction(output="", error=str(e), latency_ms=latency_ms)

    def _with_retries(self, payload: dict) -> dict:
        """Retry on 429/5xx and transient URLError, up to 3 attempts."""
        backoff_delays = [0.5, 1.0, 2.0]
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return self._post(payload)
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    last_exc = e
                    time.sleep(backoff_delays[attempt] + random.uniform(0, 0.25))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_exc = e
                time.sleep(backoff_delays[attempt] + random.uniform(0, 0.25))
                continue
        raise last_exc  # type: ignore[misc]
