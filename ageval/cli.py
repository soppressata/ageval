from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from typing import Any

from ageval.core import TaskResult, get_agent, list_agents, list_scorers
from ageval.errors import TaskLoadError


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

    return parser


def _run(args: argparse.Namespace) -> int:
    agent_args = parse_kv(args.agent_args)
    try:
        from ageval.report import to_html, to_json, to_markdown
        from ageval.runner import RunConfig, run_suite
        from ageval.tasks import load_tasks, suite_name
    except ImportError as e:
        print(f"ageval: component not available yet: {e}", file=sys.stderr)
        return 2

    tasks = load_tasks(args.suite_path)
    agent = get_agent(args.agent, **agent_args)
    tags = [t for t in args.tags.split(",") if t] if args.tags else []
    config = RunConfig(
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        task_timeout=args.timeout,
        fail_fast=args.fail_fast,
        limit=args.limit,
        tags=tags,
        cache_dir=args.cache_dir,
    )
    suite = suite_name(args.suite_path)
    total = len(tasks)
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
        tasks, agent, config=config, suite_name=suite, progress=progress
    )

    summary = report.summary()
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    writers = {
        "json": (to_json, ".json"),
        "markdown": (to_markdown, ".md"),
        "html": (to_html, ".html"),
    }
    os.makedirs(args.out, exist_ok=True)
    for fmt in formats:
        if fmt not in writers:
            print(f"ageval: unknown format '{fmt}'", file=sys.stderr)
            return 1
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
    for scorer in sorted(counts):
        print(f"  {scorer}: {counts[scorer]}")
    for t in tasks:
        if t.expected is None and t.scorer != "regex":
            print(
                f"ageval: warning: task {t.id} has no expected value "
                f"but uses scorer '{t.scorer}'",
                file=sys.stderr,
            )
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
