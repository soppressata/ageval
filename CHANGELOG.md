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

## [0.1.0] - 2026-08-06

- Initial public release: core dataclasses, tasks, adapters (echo/openrouter/subprocess),
  scorers (exact/includes/regex/numeric/json_subset/llm_judge), runner, report, cli,
  suites, 189 tests green.
