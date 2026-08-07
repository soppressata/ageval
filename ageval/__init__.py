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
    "RunStore",
    "RunMeta",
    "TaskDelta",
    "RunComparison",
    "paired_deltas",
    "compare_runs",
    "pass_rate_delta",
    "mean_score_delta",
    "variance_delta",
    "confidence_interval",
    "paired_significance_test",
    "effect_size",
    "detect_regression",
    "Dataset",
    "DatasetMetadata",
    "Turn",
    "ConversationResult",
    "ConversationalAgent",
    "ToolCall",
    "Trace",
    "Budget",
    "BudgetManager",
    "SubprocessSandbox",
    "__version__",
]

# Import submodules last to trigger registration of agents and scorers.
import ageval.adapters  # noqa: F401
import ageval.scorers  # noqa: F401
from ageval.tasks import load_tasks  # noqa: F401
from ageval.runner import run_suite, RunConfig  # noqa: F401

from ageval.store import RunStore, RunMeta  # noqa: F401
from ageval.stats import (  # noqa: F401
    TaskDelta,
    RunComparison,
    paired_deltas,
    compare_runs,
    pass_rate_delta,
    mean_score_delta,
    variance_delta,
    confidence_interval,
    paired_significance_test,
    effect_size,
    detect_regression,
)
from ageval.datasets import Dataset, DatasetMetadata  # noqa: F401
from ageval.conversation import (  # noqa: F401
    Turn,
    ConversationResult,
    ConversationalAgent,
)
from ageval.tracing import ToolCall, Trace  # noqa: F401
from ageval.budget import Budget, BudgetManager  # noqa: F401
from ageval.sandbox import SubprocessSandbox  # noqa: F401
