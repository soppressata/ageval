# Changelog

All notable changes to `ageval` will be documented in this file.

## [Unreleased]

- Score clamping: `Score.__post_init__` enforces `value` in `[0.0, 1.0]` via `clamp01`.
- Task ID handling: `load_tasks` auto-assigns `"{stem}-{index}"` IDs to tasks missing them.
- Strict package initialization: `ageval/__init__.py` imports adapters/scorers last (after
  `__all__`) to trigger side-effect registration without circular imports.
- Runner behavior: per-task `task_timeout`, `max_retries` with exponential backoff,
  SHA-256-keyed prediction caching, and `fail-fast` early termination.
- Adapter robustness: all adapters wrap `predict` in try/except, returning
  `Prediction(error=str(e))` instead of raising; subprocess handles timeout/non-zero exit.
- Scorer fixes: `BaseScorer.score` short-circuits on `prediction.error`, wraps `_score` in
  try/except, and returns `Score(0.0, False, "scoring error: ...")` on internal failure.
- CLI validation: `validate` subcommand loads tasks, verifies scorers instantiate, warns on
  missing `expected` with non-regex scorers; exits 1 on failure.
- OpenRouter retry: `_with_retries` retries up to 3 times on HTTP 429/5xx and transient
  network errors, with exponential backoff (0.5s/1s/2s) plus jitter.
- Run history & diff: `store.RunStore` persists reports under
  `<store_dir>/runs/<run_id>/report.json` (`save_run`/`load_run`/`list_runs`/`latest`);
  the CLI gains a `history list/show/diff` command family and `stats.compare_runs`
  performs paired statistical diffs between stored runs.
- Statistics, datasets & sampling: `stats.py` adds paired comparison
  (`TaskDelta`/`RunComparison`) with a paired t-test (t-statistic, two-tailed and
  one-sided p-values), Cohen's d, student-t confidence intervals, and
  `detect_regression`; `datasets.py` adds `Dataset`/`DatasetMetadata` with SHA-256
  provenance and JSONL/CSV/JSON I/O; `sample`/`split`/`filter_by_tags` give
  reproducible task selection via `--sample`/`--seed`.
- Conversations & traces: `conversation.py` adds `Turn`/`ConversationResult` and the
  `ConversationalAgent` protocol (`converse(task, history)`); `tracing.py` adds
  `Trace`/`ToolCall` with `record_tool_call`, `validate_trace`, `tool_call_summary`,
  and `trace_matches_expected`.
- Budgets & sandboxing: `budget.py` adds `Budget`/`BudgetManager` (thread-safe
  cost/token/latency limits) and `sandbox.py` adds the `Sandbox` protocol with
  `SubprocessSandbox` for process-isolated prediction evaluation; both are wired into
  `RunConfig` and the CLI (`--budget-cost`/`--budget-tokens`/`--budget-latency-ms`).
- Adapters: `openai_native.py` registers `openai_native` (`OpenAiNativeAgent`) for
  OpenAI-compatible chat APIs with retry/backoff; `ollama.py` registers `ollama`
  (`OllamaAgent`) for the local Ollama `/api/chat` endpoint.
- CLI controls: `run` now persists every report to history via `--store-dir`, supports
  `--sample`/`--seed`, and accepts cost/token/latency budgets; a new `history
  list/show/diff` command family inspects stored runs.

## [0.1.0] - 2026-08-06

- Initial public release: core dataclasses, tasks, adapters (echo/openrouter/subprocess),
  scorers (exact/includes/regex/numeric/json_subset/llm_judge), runner, report, cli,
  suites, 189 tests green.
