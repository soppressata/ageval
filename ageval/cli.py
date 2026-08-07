from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from typing import Any

from ageval.core import TaskResult, get_agent, get_scorer, list_agents, list_scorers
from ageval.errors import TaskLoadError
from ageval.store import RunStore


def parse_kv(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` strings, decoding values as JSON when possible."""
    result: dict[str, Any] = {}
    if not pairs:
        return result
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"--agent-arg must be KEY=VALUE, got: {pair!r}"
            )
        key, _, value = pair.partition("=")
        try:
            parsed: Any = json.loads(value)
        except (ValueError, TypeError):
            parsed = value
        result[key] = parsed
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ageval")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a suite of tasks through an agent.")
    run_p.add_argument("suite_path", help="Path to a suite file or directory.")
    run_p.add_argument(
        "--agent", default="echo", help="Agent name to use (default: echo)."
    )
    run_p.add_argument(
        "--agent-arg",
        dest="agent_args",
        action="append",
        default=[],
        help="Agent kwargs as KEY=VALUE (may be repeated).",
    )
    run_p.add_argument(
        "--concurrency", type=int, default=4, help="Parallel workers (default: 4)."
    )
    run_p.add_argument(
        "--limit", type=int, default=None, help="Only run the first N tasks."
    )
    run_p.add_argument(
        "--tags",
        default=None,
        help="Comma-separated tags to filter tasks by.",
    )
    run_p.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Extra attempts per task after the first (default: 1).",
    )
    run_p.add_argument(
        "--timeout",
        dest="timeout",
        type=float,
        default=300.0,
        help="Per-task timeout in seconds (default: 300.0).",
    )
    run_p.add_argument("--cache-dir", dest="cache_dir", default=None, help="Cache directory for predictions.")
    run_p.add_argument(
        "--fail-fast", action="store_true", help="Stop after the first non-ok result."
    )
    run_p.add_argument(
        "--out", default="runs", help="Output directory for reports (default: runs)."
    )
    run_p.add_argument(
        "--format",
        default="json,markdown",
        help="Comma-separated formats: json, markdown, html.",
    )
    run_p.add_argument(
        "--quiet", action="store_true", help="Suppress per-task progress output."
    )
    run_p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Deterministically sample N tasks after tag filtering.",
    )
    run_p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic sampling (default: 0).",
    )
    run_p.add_argument(
        "--budget-cost",
        dest="budget_cost",
        type=float,
        default=None,
        help="Max cost budget in USD.",
    )
    run_p.add_argument(
        "--budget-tokens",
        dest="budget_tokens",
        type=int,
        default=None,
        help="Max tokens budget.",
    )
    run_p.add_argument(
        "--budget-latency-ms",
        dest="budget_latency_ms",
        type=float,
        default=None,
        help="Max latency budget in ms.",
    )
    run_p.add_argument(
        "--store-dir",
        default=os.path.expanduser("~/.ageval"),
        help="Directory for run history storage (default: ~/.ageval).",
    )

    list_p = sub.add_parser("list", help="List registered components.")
    list_p.add_argument(
        "what", choices=["agents", "scorers"], help="What to list."
    )

    cmp_p = sub.add_parser("compare", help="Compare multiple JSON reports.")
    cmp_p.add_argument(
        "reports", nargs="+", help="One or more report JSON files."
    )

    val_p = sub.add_parser("validate", help="Validate a suite file.")
    val_p.add_argument("suite_path", help="Path to a suite file or directory.")

    hist_p = sub.add_parser("history", help="View run history.")
    hist_sub = hist_p.add_subparsers(
        dest="history_command", required=True
    )

    hlist_p = hist_sub.add_parser("list", help="List stored runs.")
    hlist_p.add_argument(
        "--store-dir",
        default=os.path.expanduser("~/.ageval"),
        help="Directory for run history storage (default: ~/.ageval).",
    )
    hlist_p.add_argument(
        "--suite", default=None, help="Filter by suite name."
    )
    hlist_p.add_argument(
        "--agent", default=None, help="Filter by agent name."
    )
    hlist_p.add_argument(
        "--limit", type=int, default=None, help="Maximum runs to show."
    )
    hlist_p.add_argument(
        "--json", action="store_true", help="Output in JSON format."
    )

    hshow_p = hist_sub.add_parser("show", help="Show a specific run report.")
    hshow_p.add_argument("run_id", help="Run ID to display.")
    hshow_p.add_argument(
        "--store-dir",
        default=os.path.expanduser("~/.ageval"),
        help="Directory for run history storage (default: ~/.ageval).",
    )
    hshow_p.add_argument(
        "--format",
        default="json",
        choices=["json", "markdown", "html"],
        help="Output format (default: json).",
    )

    hdiff_p = hist_sub.add_parser(
        "diff", help="Statistically compare two stored runs."
    )
    hdiff_p.add_argument("base_id", help="Baseline run ID.")
    hdiff_p.add_argument("candidate_id", help="Candidate run ID.")
    hdiff_p.add_argument(
        "--store-dir",
        default=os.path.expanduser("~/.ageval"),
        help="Directory for run history storage (default: ~/.ageval).",
    )
    hdiff_p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level in (0,1) (default: 0.05).",
    )

    return parser


def _run(args: argparse.Namespace) -> int:
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    for fmt in formats:
        if fmt not in {"json", "markdown", "html"}:
            print(f"ageval: unknown format '{fmt}'", file=sys.stderr)
            return 1

    agent_args = parse_kv(args.agent_args)
    try:
        from ageval.budget import Budget
        from ageval.datasets import sample
        from ageval.report import to_html, to_json, to_markdown
        from ageval.runner import RunConfig, run_suite
        from ageval.tasks import load_tasks, suite_name
    except ImportError as e:
        print(f"ageval: component not available yet: {e}", file=sys.stderr)
        return 2

    tasks = load_tasks(args.suite_path)
    agent = get_agent(args.agent, **agent_args)
    tags = [t for t in args.tags.split(",") if t] if args.tags else []
    suite = suite_name(args.suite_path)
    budget = None
    if (
        args.budget_cost is not None
        or args.budget_tokens is not None
        or args.budget_latency_ms is not None
    ):
        budget = Budget(
            max_cost_usd=args.budget_cost,
            max_tokens=args.budget_tokens,
            max_latency_ms=args.budget_latency_ms,
        )
    config = RunConfig(
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        task_timeout=args.timeout,
        fail_fast=args.fail_fast,
        limit=args.limit,
        tags=tags,
        cache_dir=args.cache_dir,
        budget=budget,
    )
    selected_tasks = tasks
    if tags:
        wanted = set(tags)
        selected_tasks = [t for t in selected_tasks if wanted.intersection(t.tags)]
    if args.sample is not None:
        selected_tasks = sample(selected_tasks, args.sample, args.seed)
    if args.limit is not None:
        selected_tasks = selected_tasks[: args.limit]
    total = len(selected_tasks)
    state = {"done": 0}
    lock = threading.Lock()

    def progress(r: TaskResult) -> None:
        if args.quiet:
            return
        with lock:
            state["done"] += 1
            done = state["done"]
        width = len(str(total))
        if r.prediction.error is not None:
            status = "ERROR"
        elif r.score is not None and r.score.passed:
            status = "PASS"
        else:
            status = "FAIL"
        score = r.score.value if r.score is not None else 0.0
        print(
            f"[{done:>{width}}/{total}] {status} {r.task.id}  "
            f"score={score:.3f}  {r.prediction.latency_ms:.0f}ms"
        )

    report = run_suite(
        selected_tasks, agent, config=config, suite_name=suite, progress=progress
    )

    summary = report.summary()
    writers = {
        "json": (to_json, ".json"),
        "markdown": (to_markdown, ".md"),
        "html": (to_html, ".html"),
    }
    os.makedirs(args.out, exist_ok=True)
    for fmt in formats:
        fn, ext = writers[fmt]
        path = os.path.join(args.out, f"{report.run_id}{ext}")
        fn(report, path=path)
        print(path)

    print(
        f"summary: total={summary['total']} passed={summary['passed']} "
        f"pass_rate={summary['pass_rate']:.3f} "
        f"mean_score={summary['mean_score']:.3f} "
        f"total_cost_usd={summary['total_cost_usd']}"
    )
    RunStore(args.store_dir).save_run(report)
    return 0 if summary["pass_rate"] == 1.0 else 1


def _list(args: argparse.Namespace) -> int:
    if args.what == "agents":
        for name in list_agents():
            print(name)
    else:
        for name in list_scorers():
            print(name)
    return 0


def _compare(args: argparse.Namespace) -> int:
    try:
        from ageval.core import RunReport
        from ageval.report import compare
    except ImportError as e:
        print(f"ageval: component not available yet: {e}", file=sys.stderr)
        return 2
    reports = []
    for path in args.reports:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.pop("summary", None)
        reports.append(RunReport.from_dict(data))
    print(compare(reports))
    return 0


def _validate(args: argparse.Namespace) -> int:
    try:
        from ageval.tasks import load_tasks
    except ImportError as e:
        print(f"ageval: component not available yet: {e}", file=sys.stderr)
        return 2
    try:
        tasks = load_tasks(args.suite_path)
    except TaskLoadError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"ok: {len(tasks)} tasks in {args.suite_path}")
    counts: Counter = Counter(t.scorer for t in tasks)
    valid = True
    for scorer in sorted(counts):
        print(f"  {scorer}: {counts[scorer]}")
    for t in tasks:
        try:
            get_scorer(t.scorer, **t.scorer_args)
        except Exception as e:
            print(
                f"ageval: error: task {t.id} has invalid scorer "
                f"'{t.scorer}': {e}",
                file=sys.stderr,
            )
            valid = False
        if t.expected is None and t.scorer != "regex":
            print(
                f"ageval: warning: task {t.id} has no expected value "
                f"but uses scorer '{t.scorer}'",
                file=sys.stderr,
            )
    return 0 if valid else 1


def _history_list(args: argparse.Namespace) -> int:
    store = RunStore(args.store_dir)
    metas = store.list_runs(
        suite_name=args.suite,
        agent_name=args.agent,
    )
    if args.limit is not None:
        metas = metas[: args.limit]
    if args.json:
        payload = [
            {
                "run_id": m.run_id,
                "suite_name": m.suite_name,
                "agent_name": m.agent_name,
                "started_at": m.started_at,
                "finished_at": m.finished_at,
            }
            for m in metas
        ]
        print(json.dumps(payload, indent=2))
    else:
        for m in metas:
            print(
                f"{m.run_id}  {m.suite_name}  {m.agent_name}  "
                f"{m.started_at}  {m.finished_at}"
            )
    return 0


def _history_show(args: argparse.Namespace) -> int:
    store = RunStore(args.store_dir)
    try:
        report = store.load_run(args.run_id)
    except FileNotFoundError:
        print(f"ageval: run not found: {args.run_id}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ageval: could not load run: {e}", file=sys.stderr)
        return 1
    try:
        from ageval.report import to_html, to_json, to_markdown
    except ImportError as e:
        print(f"ageval: component not available yet: {e}", file=sys.stderr)
        return 2
    renderers = {
        "json": to_json,
        "markdown": to_markdown,
        "html": to_html,
    }
    renderer = renderers[args.format]
    output = renderer(report)
    print(output)
    return 0


def _history_diff(args: argparse.Namespace) -> int:
    alpha = args.alpha
    if not (0.0 < alpha < 1.0):
        print(f"ageval: alpha must be in (0,1), got {alpha}", file=sys.stderr)
        return 1
    store = RunStore(args.store_dir)
    try:
        baseline = store.load_run(args.base_id)
    except FileNotFoundError:
        print(f"ageval: run not found: {args.base_id}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ageval: could not load baseline run: {e}", file=sys.stderr)
        return 1
    try:
        candidate = store.load_run(args.candidate_id)
    except FileNotFoundError:
        print(f"ageval: run not found: {args.candidate_id}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ageval: could not load candidate run: {e}", file=sys.stderr)
        return 1
    try:
        from ageval.stats import compare_runs
    except ImportError as e:
        print(f"ageval: component not available yet: {e}", file=sys.stderr)
        return 2
    comp = compare_runs(
        baseline,
        candidate,
        confidence_level=1.0 - alpha,
        regression_threshold=alpha,
    )
    payload = {
        "baseline_run_id": comp.baseline_run_id,
        "candidate_run_id": comp.candidate_run_id,
        "paired": {
            "n_paired": comp.n_paired,
            "n_baseline_only": comp.n_baseline_only,
            "n_candidate_only": comp.n_candidate_only,
        },
        "pass_rate_delta": comp.pass_rate_delta,
        "mean_score_delta": comp.mean_score_delta,
        "confidence_interval": [comp.ci_lower, comp.ci_upper],
        "confidence_level": comp.confidence_level,
        "p_value": comp.p_value_two_tailed,
        "effect_size": comp.cohens_d,
        "regression_detected": comp.regression_detected,
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns an exit code; never calls ``sys.exit``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "list":
            return _list(args)
        if args.command == "compare":
            return _compare(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "history":
            if args.history_command == "list":
                return _history_list(args)
            if args.history_command == "show":
                return _history_show(args)
            if args.history_command == "diff":
                return _history_diff(args)
            parser.error(f"unknown history command: {args.history_command}")
        parser.error(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ageval: {e}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
