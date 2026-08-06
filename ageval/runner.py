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

from ageval.core import (
    Prediction,
    RunReport,
    Score,
    Task,
    TaskResult,
    get_scorer,
)


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
        if not config.cache_dir or prediction.error is not None:
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

    def _worker(idx: int, task: Task) -> tuple[int, TaskResult]:
        prediction = _get_cached_prediction(task)
        attempts = 1

        if prediction is None:
            for attempt_index in range(config.max_retries + 1):
                try:
                    prediction = agent.predict(task)
                except Exception as e:
                    prediction = Prediction(error=str(e))
                attempts = attempt_index + 1
                if prediction.error is None:
                    break
                if attempt_index < config.max_retries:
                    time.sleep(config.retry_backoff * (2 ** attempt_index))
            _write_cache(task, prediction)

        scorer = _get_scorer(task)
        try:
            score = scorer.score(task, prediction)
        except Exception as e:
            score = Score(0.0, False, detail=f"scoring error: {e}")

        return idx, TaskResult(
            task=task, prediction=prediction, score=score, attempts=attempts
        )

    if not filtered:
        finished_at = datetime.now(timezone.utc).isoformat()
        return RunReport(
            run_id=run_id,
            suite_name=suite_name,
            agent_name=agent_name,
            started_at=started_at,
            finished_at=finished_at,
            results=[],
            config=dataclasses.asdict(config),
        )

    future_to_idx: dict[Future, int] = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, config.concurrency)
    ) as executor:
        for idx, task in enumerate(filtered):
            future = executor.submit(_worker, idx, task)
            future_to_idx[future] = idx

        start = time.monotonic()
        finished: set[int] = set()
        stopped = False

        while not stopped and len(finished) < len(filtered):
            elapsed = time.monotonic() - start
            remaining = max(0.0, config.task_timeout - elapsed)
            if remaining <= 0.0:
                break

            unfinished = {f for f in future_to_idx if future_to_idx[f] not in finished}
            try:
                for future in concurrent.futures.as_completed(
                    unfinished, timeout=remaining
                ):
                    idx = future_to_idx[future]
                    try:
                        _, task_result = future.result()
                    except Exception as e:
                        task_result = TaskResult(
                            task=filtered[idx],
                            prediction=Prediction(error=str(e)),
                            score=Score(
                                0.0, False, detail=f"scoring error: {e}"
                            ),
                            attempts=1,
                        )
                    results[idx] = task_result
                    finished.add(idx)

                    if progress is not None:
                        with progress_lock:
                            try:
                                progress(task_result)
                            except Exception:
                                pass

                    if config.fail_fast and not task_result.ok:
                        stopped = True
                        break
            except concurrent.futures.TimeoutError:
                pass

        if stopped:
            for f in future_to_idx:
                f.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

    # Record timeouts for any tasks that didn't finish (unless fail_fast stopped us).
    if not stopped:
        for idx in range(len(filtered)):
            if results[idx] is None:
                results[idx] = TaskResult(
                    task=filtered[idx],
                    prediction=Prediction(error=f"timeout after {config.task_timeout}s"),
                    score=Score(0.0, False, "agent error"),
                    attempts=1,
                )

    completed = [r for r in results if r is not None]
    finished_at = datetime.now(timezone.utc).isoformat()

    return RunReport(
        run_id=run_id,
        suite_name=suite_name,
        agent_name=agent_name,
        started_at=started_at,
        finished_at=finished_at,
        results=completed,
        config=dataclasses.asdict(config),
    )


class _FailingScorer:
    """A scorer that always returns a failed score, used when scorer lookup fails."""

    def __init__(self, detail: str) -> None:
        self.name = "_failing"
        self._detail = detail

    def score(self, task: Task, prediction: Prediction) -> Score:
        return Score(0.0, False, detail=self._detail)
