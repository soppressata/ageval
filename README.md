# ageval

![Python >=3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tests](https://img.shields.io/badge/tests-189%20passed-brightgreen)
![CI](https://github.com/soppressata/ageval/actions/workflows/ci.yml/badge.svg)

**Pluggable agent evaluation harness** — run task suites, score, and report.

`ageval` is a stdlib-only Python library and CLI for evaluating LLM agents. Define
tasks as JSONL or JSON, pick (or register) an agent and a scorer, and get back
JSON / Markdown / HTML reports and a pass rate.

## Features

- **Pluggable registry** for agents and scorers — add your own with a decorator.
- **6 built-in scorers**: `exact`, `includes`, `regex`, `numeric`, `json_subset`, `llm_judge`.
- **3 built-in adapters**: `echo` (testing), `openrouter` (live LLM via `urllib`), `subprocess` (shell out).
- **ThreadPool runner** with caching, retries with backoff, per-task timeouts, `fail_fast`, progress, ordered results.
- **Reports** in JSON / Markdown / HTML + a `compare` table across runs.
- **CLI** with `run`, `list`, `compare`, `validate` subcommands.

## Installation

Requires Python 3.11+.

```bash
pip install -e .
```

For development:

```bash
pip install pytest
```

There are no runtime dependencies — `ageval` uses only the standard library.

## Quickstart

```bash
pip install -e .

python -m ageval validate suites/demo.jsonl
python -m ageval run suites/demo.jsonl --agent echo --tags smoke --out runs --format json,markdown,html
python -m ageval compare runs/*.json
```

Expected summary line:

```
summary: total=5 passed=5 pass_rate=1.000 mean_score=1.000 total_cost_usd=0.0
```

List registered components:

```bash
python -m ageval list agents
python -m ageval list scorers
```

## CLI Reference

```
python -m ageval <command>
```

| Command | Description |
|---|---|
| `run <suite_path>` | Run a suite through an agent, write reports. |
| `list agents` | Print registered agent names. |
| `list scorers` | Print registered scorer names. |
| `compare <report.json> […]` | Print a comparison table across reports. |
| `validate <suite_path>` | Load tasks and report count / problems. |

### `run` flags

| Flag | Default | Description |
|---|---|---|
| `--agent NAME` | `echo` | Agent name to use. |
| `--agent-arg k=v` | — | Agent kwarg, JSON-decoded (repeatable). |
| `--concurrency N` | `4` | Parallel workers. |
| `--limit N` |  | Only run the first N tasks (after tag filter). |
| `--tags a,b` |  | Only tasks having at least one of these tags. |
| `--max-retries N` | `1` | Extra attempts after the first (1 → up to 2 total). |
| `--timeout S` | `300.0` | Per-task timeout in seconds. |
| `--cache-dir D` |  | Cache predictions on disk (sha256 keyed). |
| `--fail-fast` | `False` | Stop after the first non-ok result. |
| `--out DIR` | `runs` | Output directory for reports. |
| `--format json,markdown,html` | `json,markdown` | Comma-separated output formats. |
| `--quiet` | `False` | Suppress per-task progress output. |

`--agent-arg` values are parsed as JSON first, falling back to the raw string:
`--agent-arg temperature=0.2` yields a float, `--agent-arg model=gpt-4` yields a string.

Exit code is `0` if `pass_rate == 1.0`, else `1`.

## Python API

```python
from ageval.tasks import load_tasks
from ageval.core import get_agent
from ageval.runner import run_suite, RunConfig
from ageval.report import to_json, to_markdown, to_html

tasks = load_tasks("suites/demo.jsonl")
agent = get_agent("echo")
config = RunConfig(concurrency=4, tags=["smoke"])
report = run_suite(tasks, agent, config=config, suite_name="demo")

print(report.summary())
# summary: total=5 passed=5 pass_rate=1.000 ...

to_json(report, path="runs/report.json")
to_markdown(report, path="runs/report.md")
to_html(report, path="runs/report.html")
```

## Writing Tasks

### JSONL (one task per line)

Blank lines and lines starting with `#` are skipped.

```jsonl
# a comment
{"id": "q1", "input": "What is 2+2?", "expected": 4, "scorer": "numeric", "tags": ["math"]}
{"input": "Say hello", "expected": "hello", "scorer": "exact"}
```

If a task object omits `id`, one is auto-generated as `f"{stem}-{index}"` (e.g. `demo-1`).

### JSON (single file)

Either a bare list or an object with a `"tasks"` key:

```json
{"tasks": [
  {"id": "q1", "input": "Capital of France?", "expected": "Paris", "scorer": "exact", "tags": ["smoke"]}
]}
```

### Directory

Pass a directory and `ageval` loads every `*.jsonl` and `*.json` inside, sorted by
filename, concatenated into one list.

Each task object supports these fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | auto | Unique within a suite. |
| `input` | `str` | required | Prompt given to the agent. |
| `expected` | any | `None` | Reference answer; scorer-defined shape. |
| `scorer` | `str` | `"exact"` | Registered scorer name. |
| `scorer_args` | `dict` | `{}` | Kwargs forwarded to the scorer. |
| `tags` | `list[str]` | `[]` | For slicing results. |
| `metadata` | `dict` | `{}` | Free-form. |

## Scorers

All scorers inherit from a `Scorer` protocol: `score(task, prediction) -> Score`.
`score()` **never raises** — on internal failure it returns `Score(0.0, False,
detail="scoring error: ...")`. If the agent errored (`prediction.error is not None`)
every scorer short-circuits with `Score(0.0, False, detail="agent error")`.

| Scorer | Args | `expected` shape |
|---|---|---|
| `exact` | `strip=True, case_sensitive=False` | `str` |
| `includes` | `case_sensitive=False` | `str` or `list[str]`; value = fraction present, passes at 1.0 |
| `regex` | `pattern=None, flags=""` | falls back to `task.expected`; `flags` is letters from `"ims"` |
| `numeric` | `tolerance=1e-6, relative=False` | number; extracts the last number from the output |
| `json_subset` |  | `dict`/`list`; passes when `expected` is a recursive subset of the parsed output; tolerates ```json fences |
| `llm_judge` | `agent="openrouter", agent_args=None, rubric=None, threshold=0.7` | any; builds a judge prompt, asks for `{"score": <0..1>, "reason": "..."}` |

Register your own:

```python
from ageval.core import register_scorer

@register_scorer("my_scorer")
class MyScorer:
    name = "my_scorer"
    def score(self, task, prediction): ...
```

## Adapters

Every adapter is a class registered with `@register_agent("<name>")`. `predict(task)
-> Prediction` **never raises** — catch every exception and return
`Prediction(error=str(e), output="")`. Always set `latency_ms` via `time.perf_counter()`.

| Adapter | Name | Notes |
|---|---|---|
| `EchoAgent` | `echo` | `__init__(prefix="", uppercase=False)`. Returns the input. Zero cost, zero network. |
| `OpenRouterAgent` | `openrouter` | `__init__(model, api_key=None, system=None, temperature=0.0, max_tokens=1024, timeout=120.0, base_url="https://openrouter.ai/api/v1")`. Reads `OPENROUTER_API_KEY` from env. Retries on 429/5xx with backoff + jitter. |
| `SubprocessAgent` | `subprocess` | `__init__(command: list[str], timeout=300.0)`. Pipes `task.input` on stdin, reads stdout. Non-zero exit or timeout → `error`. |

Register your own:

```python
from ageval.core import register_agent

@register_agent("my_agent")
class MyAgent:
    name = "my_agent"
    def predict(self, task): ...
```

## Runner details

`run_suite(tasks, agent, config=None, suite_name="suite", progress=None) -> RunConfig`.

### `RunConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `concurrency` | `int` | `4` | ThreadPool worker count. |
| `max_retries` | `int` | `1` | Extra attempts after the first. |
| `retry_backoff` | `float` | `0.5` | Seconds, doubled each retry. |
| `task_timeout` | `float` | `300.0` | Seconds per task attempt. |
| `fail_fast` | `bool` | `False` | Cancel remaining after first non-ok. |
| `limit` | `int | None` | `None` | Only run the first N tasks. |
| `tags` | `list[str]` | `[]` | Filter: tasks having ≥1 of these. |
| `cache_dir` | `str | None` | `None` | Cache predictions on disk. |

### Behavior

- Tasks are **filtered by tags first, then limited** (`limit` applies after filtering).
- `results` in the report are **ordered to match the filtered input order**, not completion order.
- Per task: `agent.predict` is called; if `Prediction.error` is set, retry up to
  `max_retries` extra times with exponential backoff. The final `attempts` count is recorded.
- `task_timeout` is enforced via `future.result(timeout=...)`; on timeout the result is
  `Prediction(error="timeout after Xs")`.
- Scorer instances are **cached per (name, args)** so a scorer is not rebuilt per task.
- `progress` is invoked once per completed `TaskResult`, guarded by a lock (thread-safe).
- `fail_fast=True`: after the first non-ok result, remaining futures are canceled and
  the partial report is returned.
- **Caching**: when `cache_dir` is set, the key is `sha256(agent.name + task.id + task.input)`.
  Predictions are stored as `<cache_dir>/<key>.json`; on a hit the agent call is skipped.

## Project Structure

```
ageval/
├── __init__.py        # re-exports, __version__ = "0.1.0", registers adapters/scorers
├── __main__.py        # calls cli.main()
├── core.py            # Task, Prediction, Score, TaskResult, RunReport, registries
├── errors.py          # AgevalError hierarchy
├── tasks.py           # load_tasks, save_tasks, suite_name
├── runner.py          # run_suite, RunConfig
├── report.py          # to_json, to_markdown, to_html, compare
├── cli.py             # argparse CLI
├── adapters/
│   ├── echo.py
│   ├── openrouter.py
│   └── subprocess_agent.py
└── scorers/
    ├── builtin.py     # exact, includes, regex, numeric, json_subset
    └── llm_judge.py   # llm_judge
suites/
├── demo.jsonl         # 10 tasks across all 6 scorers
└── math.jsonl         # 5 numeric tasks
tests/                 # 189 tests, pytest
```

## Development

Run the test suite:

```bash
python3 -m pytest -q
# 189 passed in 1.61s
```

Try the CLI:

```bash
python -m ageval --help
python -m ageval run suites/demo.jsonl --agent echo --tags smoke --out runs --quiet
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: fork, branch, run `pytest -q`,
and keep `SPEC.md` as law. `AGENTS.md` describes file ownership across the team —
please respect it. Stdlib only, no asyncio.

## License

[MIT](LICENSE) © 2026 soppressata.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
