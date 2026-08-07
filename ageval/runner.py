from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ageval.budget import Budget, BudgetManager
from ageval.core import (
    Prediction,
    RunReport,
    Score,
    Task,
    TaskResult,
    get_scorer,
)
from ageval.errors import ConfigError
from ageval.sandbox import Sandbox


@dataclass
class RunConfig:
    concurrency: int = 4
    max_retries: int = 1
    retry_backoff: float = 0.5
    task_timeout: float = 300.0
    fail_fast: bool = False
    limit: int | None = None
    tags: list[str] = field(default_factory=list)
    cache_dir: str | None = None
    budget: Budget | None = None
    sandbox: Sandbox | None = None
    conversation: bool = False
    max_turns: int = 5


def _config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Convert RunConfig to a JSON-serializable dict.

    ``budget`` and ``sandbox`` are replaced with plain descriptors so the
    report config survives ``json.dumps`` without altering v1 fields.
    """
    d = dataclasses.asdict(config)
    if config.budget is not None:
        d["budget"] = {
            "max_cost_usd": config.budget.max_cost_usd,
            "max_tokens": config.budget.max_tokens,
            "max_latency_ms": config.budget.max_latency_ms,
        }
    if config.sandbox is not None:
        d["sandbox"] = getattr(config.sandbox, "__class__", type(config.sandbox)).__name__
    return d


def run_suite(
    tasks: list[Task],
    agent: Any,
    config: RunConfig | None = None,
    suite_name: str = "suite",
    progress: Callable[[TaskResult], None] | None = None,
) -> RunReport:
    """Run a suite of tasks through an agent with the given configuration."""
    if config is None:
        config = RunConfig()

    if not isinstance(config.concurrency, int) or isinstance(config.concurrency, bool) or config.concurrency <= 0:
        raise ConfigError("concurrency must be a positive integer")
    if not isinstance(config.max_retries, int) or isinstance(config.max_retries, bool) or config.max_retries < 0:
        raise ConfigError("max_retries must be a non-negative integer")
    if not isinstance(config.retry_backoff, (int, float)) or isinstance(config.retry_backoff, bool) or config.retry_backoff < 0:
        raise ConfigError("retry_backoff must be non-negative")
    if not isinstance(config.task_timeout, (int, float)) or isinstance(config.task_timeout, bool) or config.task_timeout <= 0:
        raise ConfigError("task_timeout must be positive")
    if config.limit is not None and (not isinstance(config.limit, int) or isinstance(config.limit, bool) or config.limit < 0):
        raise ConfigError("limit must be non-negative or None")
    if config.budget is not None and not isinstance(config.budget, Budget):
        raise ConfigError("budget must be a Budget instance or None")
    if config.sandbox is not None and not isinstance(config.sandbox, Sandbox):
        raise ConfigError("sandbox must be a Sandbox instance or None")
    if not isinstance(config.max_turns, int) or isinstance(config.max_turns, bool) or config.max_turns <= 0:
        raise ConfigError("max_turns must be a positive integer")

    agent_name = getattr(agent, "name", agent.__class__.__name__)

    # Filter by tags, then limit.
    filtered = list(tasks)
    if config.tags:
        wanted = set(config.tags)
        filtered = [t for t in filtered if wanted.intersection(t.tags)]
    if config.limit is not None:
        filtered = filtered[: config.limit]

    run_id = f"{suite_name}-{agent_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    started_at = datetime.now(timezone.utc).isoformat()

    results: list[TaskResult | None] = [None] * len(filtered)
    scorer_cache: dict[str, Any] = {}
    scorer_lock = threading.Lock()
    progress_lock = threading.Lock()

    if config.cache_dir:
        os.makedirs(config.cache_dir, exist_ok=True)

    budget_mgr = BudgetManager(config.budget) if config.budget is not None else None

    def _budget_exceeded_reason(status: Any) -> str:
        reasons: list[str] = []
        if status.total_cost_usd is not None:
            reasons.append(f"cost=${status.total_cost_usd:.4f}")
        if status.total_tokens is not None:
            reasons.append(f"tokens={status.total_tokens}")
        if status.total_latency_ms is not None:
            reasons.append(f"latency={status.total_latency_ms:.0f}ms")
        return "budget exceeded: " + ", ".join(reasons) if reasons else "budget exceeded"

    def _get_cached_prediction(task: Task) -> Prediction | None:
        if not config.cache_dir:
            return None
        key = hashlib.sha256(
            f"{agent_name}\0{task.id}\0{task.input}".encode()
        ).hexdigest()
        path = os.path.join(config.cache_dir, f"{key}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Prediction.from_dict(data)
        except Exception:
            return None

    def _write_cache(task: Task, prediction: Prediction) -> None:
        if not config.cache_dir:
            return
        key = hashlib.sha256(
            f"{agent_name}\0{task.id}\0{task.input}".encode()
        ).hexdigest()
        path = os.path.join(config.cache_dir, f"{key}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(prediction.to_dict(), f)
        except Exception:
            pass

    def _get_scorer(task: Task) -> Any:
        cache_key = (
            task.scorer,
            json.dumps(task.scorer_args, sort_keys=True, default=str),
        )
        with scorer_lock:
            if cache_key in scorer_cache:
                return scorer_cache[cache_key]
            try:
                scorer = get_scorer(task.scorer, **task.scorer_args)
            except Exception as e:
                scorer = _FailingScorer(str(e))
            scorer_cache[cache_key] = scorer
            return scorer

    def _score_result(task: Task, prediction: Prediction, attempts: int) -> TaskResult:
        scorer = _get_scorer(task)
        try:
            score = scorer.score(task, prediction)
        except Exception as e:
            score = Score(0.0, False, detail=f"scoring error: {e}")

        return TaskResult(task=task, prediction=prediction, score=score, attempts=attempts)

    if not filtered:
        finished_at = datetime.now(timezone.utc).isoformat()
        return RunReport(
            run_id=run_id,
            suite_name=suite_name,
            agent_name=agent_name,
            started_at=started_at,
            finished_at=finished_at,
            results=[],
            config=_config_to_dict(config),
        )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrency)
    pending: dict[Future, tuple[int, int, float]] = {}
    next_index = 0
    stopped = False

    def record(idx: int, task_result: TaskResult) -> None:
        results[idx] = task_result
        if progress is not None:
            with progress_lock:
                try:
                    progress(task_result)
                except Exception:
                    pass

    def record_cached(idx: int, prediction: Prediction) -> TaskResult:
        task_result = _score_result(filtered[idx], prediction, 1)
        record(idx, task_result)
        return task_result

    def _invoke_agent(agent: Any, task: Task) -> Prediction:
        """Invoke the agent, using conversation mode if configured and supported.

        When ``config.conversation`` is ``True`` and the agent exposes a
        callable ``converse`` method, delegate to ``agent.converse(task,
        history=[])`` and convert the returned ``ConversationResult`` into a
        ``Prediction``. Otherwise fall back to ``agent.predict(task)``.
        """
        if config.conversation and callable(getattr(agent, "converse", None)):
            conv = agent.converse(task, history=[])
            return Prediction(
                output=conv.final_output,
                latency_ms=conv.total_latency_ms,
                cost_usd=conv.total_cost_usd,
                raw=conv.to_dict(),
            )
        return agent.predict(task)

    def submit_attempt(idx: int, attempt: int) -> None:
        task = filtered[idx]
        if config.sandbox is not None:
            future = executor.submit(
                config.sandbox.run, agent, task, config.task_timeout
            )
        else:
            future = executor.submit(_invoke_agent, agent, task)
        pending[future] = (idx, attempt, time.monotonic())

    try:
        while next_index < len(filtered) and len(pending) < config.concurrency:
            cached = _get_cached_prediction(filtered[next_index])
            if cached is not None:
                task_result = record_cached(next_index, cached)
                next_index += 1
                if config.fail_fast and not task_result.ok:
                    stopped = True
                    break
                continue
            submit_attempt(next_index, 1)
            next_index += 1

        while pending and not stopped:
            now = time.monotonic()
            wait_timeout = min(
                max(0.0, started + config.task_timeout - now)
                for _, _, started in pending.values()
            )
            done, _ = concurrent.futures.wait(
                pending, timeout=wait_timeout, return_when=concurrent.futures.FIRST_COMPLETED
            )
            now = time.monotonic()
            expired = [
                f for f, (_, _, started) in pending.items()
                if f not in done and now - started >= config.task_timeout
            ]
            for future in done | set(expired):
                if future not in pending:
                    continue
                idx, attempt, _ = pending.pop(future)
                if future in done and future not in expired:
                    try:
                        prediction = future.result()
                    except Exception as e:
                        prediction = Prediction(error=str(e))
                else:
                    future.cancel()
                    prediction = Prediction(error=f"timeout after {config.task_timeout}s")
                if budget_mgr is not None:
                    status = budget_mgr.consume(prediction)
                    if status.stopped:
                        prediction = Prediction(error=_budget_exceeded_reason(status))
                if prediction.error is not None and attempt <= config.max_retries:
                    if config.retry_backoff:
                        time.sleep(config.retry_backoff * (2 ** (attempt - 1)))
                    submit_attempt(idx, attempt + 1)
                    continue
                _write_cache(filtered[idx], prediction)
                task_result = _score_result(filtered[idx], prediction, attempt)
                record(idx, task_result)
                if config.fail_fast and not task_result.ok:
                    stopped = True
                    break
                if budget_mgr is not None and budget_mgr.should_stop():
                    stopped = True
                    break
                if next_index < len(filtered):
                    cached = _get_cached_prediction(filtered[next_index])
                    if cached is not None:
                        task_result = record_cached(next_index, cached)
                        next_index += 1
                        if config.fail_fast and not task_result.ok:
                            stopped = True
                            break
                    else:
                        submit_attempt(next_index, 1)
                        next_index += 1
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    completed = [r for r in results if r is not None]
    finished_at = datetime.now(timezone.utc).isoformat()

    return RunReport(
        run_id=run_id,
        suite_name=suite_name,
        agent_name=agent_name,
        started_at=started_at,
        finished_at=finished_at,
        results=completed,
        config=_config_to_dict(config),
    )


class _FailingScorer:
    """A scorer that always returns a failed score, used when scorer lookup fails."""

    def __init__(self, detail: str) -> None:
        self.name = "_failing"
        self._detail = detail

    def score(self, task: Task, prediction: Prediction) -> Score:
        return Score(0.0, False, detail=self._detail)
