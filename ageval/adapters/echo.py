from __future__ import annotations

import time

from ageval.core import Prediction, Task, register_agent


@register_agent("echo")
class EchoAgent:
    """Dependency-free agent that returns the task input, optionally transformed."""

    name: str = "echo"

    def __init__(self, prefix: str = "", uppercase: bool = False) -> None:
        self.prefix = prefix
        self.uppercase = uppercase

    def predict(self, task: Task) -> Prediction:
        start = time.perf_counter()
        try:
            text = self.prefix + task.input
            if self.uppercase:
                text = text.upper()
            tokens = len(text.split())
            latency_ms = (time.perf_counter() - start) * 1000.0
            return Prediction(
                output=text,
                latency_ms=latency_ms,
                tokens_in=tokens,
                tokens_out=tokens,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return Prediction(output="", error=str(e), latency_ms=latency_ms)
