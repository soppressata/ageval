from __future__ import annotations

from ageval.adapters import echo  # noqa: F401
from ageval.adapters import ollama  # noqa: F401
from ageval.adapters import openrouter  # noqa: F401
from ageval.adapters import openai_native  # noqa: F401
from ageval.adapters import subprocess_agent  # noqa: F401

from ageval.adapters.echo import EchoAgent
from ageval.adapters.ollama import OllamaAgent
from ageval.adapters.openrouter import OpenRouterAgent
from ageval.adapters.openai_native import OpenAiNativeAgent
from ageval.adapters.subprocess_agent import SubprocessAgent

__all__ = ["EchoAgent", "OllamaAgent", "OpenRouterAgent", "OpenAiNativeAgent", "SubprocessAgent"]
