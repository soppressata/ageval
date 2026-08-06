from __future__ import annotations

import subprocess
import time

from ageval.core import Prediction, Task, register_agent
from ageval.errors import ConfigError


@register_agent("subprocess")
class SubprocessAgent:
    """Agent that runs an external command, piping task.input to its stdin."""

    name: str = "subprocess"

    def __init__(self, command: list[str], timeout: float = 300.0) -> None:
        if not isinstance(command, list) or len(command) == 0:
            raise ConfigError("SubprocessAgent command must be a non-empty list[str]")
        self.command = command
        self.timeout = timeout

    def predict(self, task: Task) -> Prediction:
        start = time.perf_counter()
        raw: dict[str, object] = {}
        try:
            result = subprocess.run(
                self.command,
                input=task.input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            raw["returncode"] = result.returncode
            raw["stderr"] = result.stderr
            if result.returncode != 0:
                err_stderr = result.stderr.strip()[:500]
                latency_ms = (time.perf_counter() - start) * 1000.0
                return Prediction(
                    output="",
                    error=f"subprocess exited with code {result.returncode}: {err_stderr}",
                    latency_ms=latency_ms,
                    raw=raw,
                )
            output = result.stdout
            if output.endswith("\n"):
                output = output[:-1]
            latency_ms = (time.perf_counter() - start) * 1000.0
            return Prediction(output=output, latency_ms=latency_ms, raw=raw)
        except subprocess.TimeoutExpired:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return Prediction(
                output="",
                error=f"timeout after {self.timeout}s",
                latency_ms=latency_ms,
                raw=raw,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return Prediction(output="", error=str(e), latency_ms=latency_ms, raw=raw)
