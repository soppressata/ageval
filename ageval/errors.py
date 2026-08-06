from __future__ import annotations


class AgevalError(Exception):
    """Base error for all ageval exceptions."""


class ConfigError(AgevalError):
    """Raised when configuration is invalid."""


class UnknownComponentError(AgevalError):
    """Raised when a requested agent or scorer is not registered."""

    def __init__(self, name: str, available: list[str], kind: str = "component") -> None:
        self.name = name
        self.available = sorted(available)
        super().__init__(
            f"Unknown {kind} '{name}'. Available: {self.available}"
        )


class TaskLoadError(AgevalError):
    """Raised when tasks cannot be loaded."""


class AgentError(AgevalError):
    """Raised when an agent fails."""


class ScoringError(AgevalError):
    """Raised when scoring fails."""
