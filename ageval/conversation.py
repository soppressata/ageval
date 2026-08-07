from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from ageval.core import Task


ALLOWED_ROLES: frozenset[str] = frozenset({"user", "assistant", "system", "tool"})


@dataclass
class Turn:
    """A single turn in a conversational agent exchange.

    Attributes:
        role: One of ``"user"``, ``"assistant"``, ``"system"``, ``"tool"``.
        content: The textual content of the turn.
        tool_call: Optional tool-call description (name + arguments).
        tool_result: Optional tool result payload.
        latency_ms: Time spent producing this turn, in milliseconds.
    """

    role: str
    content: str
    tool_call: dict[str, Any] | None = None
    tool_result: Any | None = None
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.role not in ALLOWED_ROLES:
            raise ValueError(
                f"invalid role {self.role!r}; allowed: {sorted(ALLOWED_ROLES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Turn:
        """Deserialize from a dict, validating the role."""
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            tool_call=d.get("tool_call"),
            tool_result=d.get("tool_result"),
            latency_ms=float(d.get("latency_ms", 0.0)),
        )


@dataclass
class ConversationResult:
    """The full result of a multi-turn conversation.

    Attributes:
        turns: Ordered list of conversation turns.
        final_output: The agent's final textual answer.
        total_latency_ms: Sum of all turn latencies plus overhead, in milliseconds.
        total_cost_usd: Total cost of the conversation in USD.
        raw: Free-form raw data from the underlying provider.
    """

    turns: list[Turn] = field(default_factory=list)
    final_output: str = ""
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "turns": [t.to_dict() for t in self.turns],
            "final_output": self.final_output,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConversationResult:
        """Deserialize from a dict, rebuilding each Turn."""
        return cls(
            turns=[Turn.from_dict(t) for t in d.get("turns", [])],
            final_output=d.get("final_output", ""),
            total_latency_ms=float(d.get("total_latency_ms", 0.0)),
            total_cost_usd=float(d.get("total_cost_usd", 0.0)),
            raw=dict(d.get("raw", {})),
        )


@runtime_checkable
class ConversationalAgent(Protocol):
    """Protocol for agents that support multi-turn conversation.

    Implementations must expose a ``converse`` method that accepts a task
    and optional conversation history, and returns a :class:`ConversationResult`.
    """

    def converse(
        self,
        task: Task,
        history: list[Turn] | None = None,
    ) -> ConversationResult:
        """Run a conversational exchange.

        Args:
            task: The task to process.
            history: Optional prior conversation turns to prepend.

        Returns:
            A :class:`ConversationResult` containing the turns and final output.
        """
        ...
