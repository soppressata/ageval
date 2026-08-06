from __future__ import annotations

__version__ = "0.1.0"

from ageval.core import (
    Task,
    Prediction,
    Score,
    TaskResult,
    RunReport,
    Agent,
    Scorer,
    register_agent,
    register_scorer,
    get_agent,
    get_scorer,
    list_agents,
    list_scorers,
)

__all__ = [
    "Task",
    "Prediction",
    "Score",
    "TaskResult",
    "RunReport",
    "Agent",
    "Scorer",
    "register_agent",
    "register_scorer",
    "get_agent",
    "get_scorer",
    "list_agents",
    "list_scorers",
    "RunConfig",
    "load_tasks",
    "run_suite",
    "__version__",
]

# Import submodules last to trigger registration of agents and scorers.
import ageval.adapters  # noqa: F401
import ageval.scorers  # noqa: F401
from ageval.tasks import load_tasks  # noqa: F401
from ageval.runner import run_suite, RunConfig  # noqa: F401
