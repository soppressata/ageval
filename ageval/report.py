from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ageval.core import RunReport, TaskResult


_SUMMARY_KEYS = (
    "total",
    "passed",
    "failed",
    "errored",
    "pass_rate",
    "mean_score",
    "total_cost_usd",
    "mean_latency_ms",
    "p95_latency_ms",
)


def _write(path: str | Path | None, content: str) -> None:
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _status(result: TaskResult) -> str:
    if result.prediction.error is not None:
        return "ERROR"
    if result.score is not None and result.score.passed:
        return "PASS"
    return "FAIL"


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _md_output(output: str, limit: int = 80) -> str:
    text = _escape_md_cell(output)
    if len(text) > limit:
        return text[:limit] + "\u2026"
    return text


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def to_json(
    report: RunReport,
    path: str | Path | None = None,
    indent: int = 2,
) -> str:
    """Render a RunReport as JSON, including a self-describing summary."""
    d = report.to_dict()
    d["summary"] = report.summary()
    result = json.dumps(d, indent=indent, ensure_ascii=False, default=str)
    _write(path, result)
    return result


def to_markdown(
    report: RunReport,
    path: str | Path | None = None,
    max_rows: int = 100,
) -> str:
    """Render a RunReport as markdown (summary, by-tag, and task tables)."""
    summary = report.summary()
    lines: list[str] = []
    lines.append(f"# {report.run_id}")
    lines.append("")

    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in _SUMMARY_KEYS:
        lines.append(f"| {key} | {_fmt(summary[key])} |")
    lines.append("")

    by_tag = summary["by_tag"]
    if by_tag:
        lines.append("## By tag")
        lines.append("")
        lines.append("| tag | total | passed | pass_rate |")
        lines.append("| --- | --- | --- | --- |")
        for tag in sorted(by_tag):
            t = by_tag[tag]
            lines.append(
                f"| {_escape_md_cell(tag)} | {t['total']} | {t['passed']} "
                f"| {_fmt(t['pass_rate'])} |"
            )
        lines.append("")

    lines.append("## Tasks")
    lines.append("")
    lines.append("| id | status | score | attempts | latency_ms | output |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    results = report.results
    show = results[:max_rows]
    for r in show:
        status = _status(r)
        score_str = _fmt(r.score.value) if r.score is not None else "-"
        output = _md_output(r.prediction.output or "")
        lines.append(
            f"| {_escape_md_cell(r.task.id)} | {status} | {score_str} | "
            f"{r.attempts} | {_fmt(r.prediction.latency_ms)} | {output} |"
        )
    if len(results) > max_rows:
        lines.append("")
        lines.append(f"\u2026 {len(results) - max_rows} more rows omitted")
    result = "\n".join(lines) + "\n"
    _write(path, result)
    return result


def to_html(report: RunReport, path: str | Path | None = None) -> str:
    """Render a RunReport as a single self-contained HTML document."""
    summary = report.summary()

    metric_divs = []
    for key in _SUMMARY_KEYS:
        metric_divs.append(
            f'      <div class="metric"><div class="k">{_esc(key)}</div>'
            f'<div class="v">{_fmt(summary[key])}</div></div>'
        )
    metrics_html = "\n".join(metric_divs)

    task_rows = []
    for r in report.results:
        status = _status(r)
        cls = {"PASS": "pass", "FAIL": "fail", "ERROR": "err"}.get(status, "")
        score = r.score
        score_str = _fmt(score.value) if score is not None else "-"
        detail = _esc(score.detail) if score is not None else ""
        output = _esc(r.prediction.output or "")
        error = _esc(r.prediction.error or "")
        task_rows.append(
            f'      <tr class="{cls}">\n'
            f'        <td>{_esc(r.task.id)}</td>\n'
            f'        <td><span class="badge {cls}">{_esc(status)}</span></td>\n'
            f'        <td>{score_str}</td>\n'
            f'        <td>{r.attempts}</td>\n'
            f'        <td>{_fmt(r.prediction.latency_ms)}</td>\n'
            f'        <td><div class="out">{output}</div></td>\n'
            f'        <td><div class="out">{error}</div></td>\n'
            f'        <td><div class="out">{detail}</div></td>\n'
            f'      </tr>'
        )
    tasks_body = "\n".join(task_rows)

    by_tag = summary["by_tag"]
    if by_tag:
        tag_rows = []
        for tag in sorted(by_tag):
            t = by_tag[tag]
            tag_rows.append(
                f'      <tr><td>{_esc(tag)}</td><td>{t["total"]}</td>'
                f'<td>{t["passed"]}</td><td>{_fmt(t["pass_rate"])}</td></tr>'
            )
        tag_table = (
            "  <h2>By tag</h2>\n"
            "  <table>\n"
            "    <thead><tr><th>tag</th><th>total</th><th>passed</th>"
            "<th>pass_rate</th></tr></thead>\n"
            "    <tbody>\n"
            + "\n".join(tag_rows)
            + "\n    </tbody>\n"
            "  </table>\n"
        )
    else:
        tag_table = ""

    css = (
        "    * { box-sizing: border-box; }\n"
        "    body { font-family: system-ui, -apple-system, BlinkMacSystemFont,\n"
        '      "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin:\n'
        "      1.5rem; color: #222; }\n"
        "    h1 { font-size: 1.3rem; margin-bottom: 0.2rem; }\n"
        "    .meta { color: #666; font-size: 0.85rem; }\n"
        "    h2 { font-size: 1.05rem; margin-top: 1.4rem; }\n"
        "    .card { display: flex; flex-wrap: wrap; gap: 0.4rem; margin:\n"
        "      0.8rem 0; }\n"
        "    .metric { background: #f6f7f8; border: 1px solid #e0e0e0;\n"
        "      border-radius: 6px; padding: 0.45rem 0.8rem; min-width: 110px; }\n"
        "    .metric .k { font-size: 0.68rem; text-transform: uppercase;\n"
        "      letter-spacing: 0.03em; color: #777; }\n"
        "    .metric .v { font-size: 0.95rem; font-weight: 600; }\n"
        "    table { border-collapse: collapse; width: 100%; margin-top:\n"
        "      0.4rem; }\n"
        "    th, td { border: 1px solid #ddd; padding: 0.4rem 0.5rem;\n"
        "      text-align: left; font-size: 0.82rem; }\n"
        "    th { position: sticky; top: 0; background: #f0f1f3; z-index: 1; }\n"
        "    tbody tr:nth-child(even) { background: #fafafa; }\n"
        "    tbody tr.pass { background: #e8f5e9; }\n"
        "    tbody tr.fail { background: #ffebee; }\n"
        "    tbody tr.err { background: #fff3e0; }\n"
        "    .badge { padding: 0.1rem 0.4rem; border-radius: 4px;\n"
        "      font-weight: 700; font-size: 0.75rem; }\n"
        "    .badge.pass { background: #a5d6a7; }\n"
        "    .badge.fail { background: #ef9a9a; }\n"
        "    .badge.err { background: #ffcc80; }\n"
        "    .out { max-height: 7rem; overflow: auto; white-space:\n"
        "      pre-wrap; word-break: break-word; }\n"
        "    .toolbar { margin: 0.4rem 0; }\n"
        "    .meta-line { margin: 0.3rem 0 0; color: #666; font-size:\n"
        "      0.82rem; }\n"
    )

    doc = (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>{_esc(report.run_id)}</title>\n"
        "  <style>\n"
        f"{css}"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>{_esc(report.run_id)}</h1>\n"
        f'  <div class="meta">{_esc(report.agent_name)} \u00b7 '
        f"{_esc(report.suite_name)}</div>\n"
        '  <div class="meta-line">'
        f"total cost: {_fmt(summary['total_cost_usd'])} usd \u00b7 "
        f"mean latency: {_fmt(summary['mean_latency_ms'])} ms \u00b7 "
        f"p95 latency: {_fmt(summary['p95_latency_ms'])} ms</div>\n"
        f'  <div class="card">\n{metrics_html}\n  </div>\n'
        '  <div class="toolbar"><label><input type="checkbox" '
        'id="filterFail"> show failures only</label></div>\n'
        "  <h2>Tasks</h2>\n"
        "  <table>\n"
        "    <thead>\n"
        "      <tr><th>id</th><th>status</th><th>score</th><th>attempts</th>"
        "<th>latency_ms</th><th>output</th><th>error</th><th>detail</th></tr>\n"
        "    </thead>\n"
        "    <tbody>\n"
        f"{tasks_body}\n"
        "    </tbody>\n"
        "  </table>\n"
        f"{tag_table}"
        "  <script>\n"
        "    document.getElementById('filterFail').addEventListener(\n"
        "      'change', function() {\n"
        "        var chk = this.checked;\n"
        "        document.querySelectorAll('tbody tr').forEach(function(r) {\n"
        "          var show = chk ? !r.classList.contains('pass') : true;\n"
        "          r.style.display = show ? '' : 'none';\n"
        "        });\n"
        "      });\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )
    _write(path, doc)
    return doc


def compare(reports: list[RunReport]) -> str:
    """Render a comparison table across multiple RunReports."""
    if not reports:
        return "_no reports_"
    summaries = [r.summary() for r in reports]
    max_rate = max(s["pass_rate"] for s in summaries)
    lines = [
        "| run_id | agent | suite | total | passed | pass_rate | "
        "mean_score | mean_latency_ms | total_cost_usd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    starred = False
    for r, s in zip(reports, summaries):
        star = ""
        if not starred and abs(s["pass_rate"] - max_rate) < 1e-12:
            star = " **\u2605**"
            starred = True
        run_id = _escape_md_cell(r.run_id) + star
        cells = [
            run_id,
            _escape_md_cell(r.agent_name),
            _escape_md_cell(r.suite_name),
            str(s["total"]),
            str(s["passed"]),
            _fmt(s["pass_rate"]),
            _fmt(s["mean_score"]),
            _fmt(s["mean_latency_ms"]),
            _fmt(s["total_cost_usd"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    result = "\n".join(lines) + "\n"
    return result
