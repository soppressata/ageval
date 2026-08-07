from __future__ import annotations

import json
import time
import urllib.request

from ageval.core import Prediction, Task, register_agent


@register_agent("ollama")
class OllamaAgent:
    """Agent that calls a local Ollama chat completion API."""

    name: str = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _post(self, payload: dict) -> dict:
        """Make the HTTP POST to the Ollama chat endpoint. Isolated for monkeypatching."""
        url = f"{self.base_url}/api/chat"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def predict(self, task: Task) -> Prediction:
        start = time.perf_counter()
        try:
            payload = {
                "model": self.model,
                "stream": False,
                "messages": [{"role": "user", "content": task.input}],
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            response = self._post(payload)
            output = response["message"]["content"]
            tokens_in = response.get("prompt_eval_count", 0)
            tokens_out = response.get("eval_count", 0)
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
