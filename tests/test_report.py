from __future__ import annotations

import json
from pathlib import Path

import pytest

from ageval.core import Prediction, RunReport, Score, Task, TaskResult
from ageval.report import compare, to_html, to_json, to_markdown


def _task(task_id: str, scorer: str, tags: list[str]) -> Task:
    return Task(
        id=task_id,
        input=f"in:{task_id}",
        expected="x",
        scorer=scorer,
        tags=tags,
    )


def R(rid: str = "run-1", rows: list[TaskResult] | None = None, agent: str = "echo") -> RunReport:
    return RunReport(
        run_id=rid,
        suite_name="demo",
        agent_name=agent,
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:01:00+00:00",
        results=rows or [],
    )


@pytest.fixture
def good() -> TaskResult:
    return TaskResult(
        task=_task("good-1", "exact", ["smoke"]),
        prediction=Prediction(output="hello", latency_ms=10.0, cost_usd=0.0),
        score=Score(1.0, True, "ok"),
        attempts=1,
    )


@pytest.fixture
def bad() -> TaskResult:
    return TaskResult(
        task=_task("bad-1", "regex", ["math"]),
        prediction=Prediction(
            output="x | y\nline " + "z" * 200,
            latency_ms=20.0,
            cost_usd=0.0,
        ),
        score=Score(0.0, False, "nope"),
        attempts=1,
    )


@pytest.fixture
def err() -> TaskResult:
    return TaskResult(
        task=_task("err-1", "numeric", ["math"]),
        prediction=Prediction(
            output="",
            error="boom <script>",
            latency_ms=5.0,
            cost_usd=0.0,
        ),
        score=Score(0.0, False, "agent error"),
        attempts=2,
    )


@pytest.fixture
def rows(good: TaskResult, bad: TaskResult, err: TaskResult) -> list[TaskResult]:
    return [good, bad, err]


@pytest.fixture
def report(rows: list[TaskResult]) -> RunReport:
    return R(rows=rows)


def _none_row() -> TaskResult:
    return TaskResult(
        task=_task("none-1", "exact", []),
        prediction=Prediction(output="", latency_ms=1.0, cost_usd=0.0),
        score=None,
        attempts=1,
    )


# --- to_json (A) ---


def test_to_json_returns_parseable_with_run_id_and_summary_total(report: RunReport) -> None:
    result = to_json(report)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["run_id"] == "run-1"
    assert parsed["summary"]["total"] == 3


def test_to_json_round_trips_summary(report: RunReport) -> None:
    parsed = json.loads(to_json(report))
    summary = parsed.pop("summary")
    rebuilt = RunReport.from_dict(parsed)
    assert rebuilt.summary() == summary


def test_to_json_writes_file_creates_parent_dirs(
    report: RunReport, tmp_path: Path
) -> None:
    out = tmp_path / "deep" / "nested" / "report.json"
    rendered = to_json(report, path=out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == rendered


# --- to_markdown (B) ---


def test_to_markdown_basics(report: RunReport) -> None:
    md = to_markdown(report)
    assert md.splitlines()[0].startswith("#")
    assert "run-1" in md.splitlines()[0]
    assert "pass_rate" in md
    assert "By tag" in md
    assert "Tasks" in md
    assert "smoke" in md
    assert "PASS" in md
    assert "FAIL" in md
    assert "ERROR" in md


def test_to_markdown_escapes_pipe(report: RunReport) -> None:
    md = to_markdown(report)
    assert r"x \| y" in md


def test_to_markdown_truncates_long_output(report: RunReport) -> None:
    md = to_markdown(report)
    assert "\u2026" in md
    assert md.count("z") < 200


def test_to_markdown_max_rows_omits_rest(report: RunReport) -> None:
    md = to_markdown(report, max_rows=1)
    assert "more rows omitted" in md


def test_to_markdown_empty_and_score_none_handle_dash() -> None:
    md_empty = to_markdown(R(rows=[]))
    assert md_empty.splitlines()[0].startswith("#")
    assert "## Tasks" in md_empty
    md_none = to_markdown(R(rows=[_none_row()]))
    assert "| - |" in md_none


# --- to_html (C) ---


def test_to_html_structure(report: RunReport) -> None:
    doc = to_html(report)
    assert doc.lstrip().startswith("<!doctype html>")
    assert "<style>" in doc
    assert 'charset="utf-8"' in doc
    assert "http://" not in doc
    assert "https://" not in doc
    assert "cdn" not in doc.lower()


def test_to_html_escapes_script(report: RunReport) -> None:
    doc = to_html(report)
    assert "boom &lt;script&gt;" in doc
    assert "boom <script>" not in doc


def test_to_html_writes_file_creates_parent_dirs(
    report: RunReport, tmp_path: Path
) -> None:
    out = tmp_path / "deep" / "nested" / "report.html"
    rendered = to_html(report, path=out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == rendered


def test_to_html_empty_score_none_and_row_classes(report: RunReport) -> None:
    empty_doc = to_html(R(rows=[]))
    assert empty_doc.lstrip().startswith("<!doctype html>")
    assert "<style>" in empty_doc

    none_doc = to_html(R(rows=[_none_row()]))
    assert "<td>-</td>" in none_doc

    doc = to_html(report)
    assert '<tr class="pass">' in doc
    assert '<tr class="fail">' in doc
    assert '<tr class="err">' in doc


# --- compare (D) ---


def test_compare_empty_and_two_headers(
    report: RunReport, bad: TaskResult
) -> None:
    assert compare([]) == "_no reports_"
    md = compare([R(rid="r1", rows=report.results), R(rid="r2", rows=report.results)])
    assert md.splitlines()[0].startswith("|")
    assert "r1" in md.splitlines()[2]
    assert "r2" in md.splitlines()[3]


def test_compare_star_only_on_best(bad: TaskResult, good: TaskResult) -> None:
    best = R(rid="best", rows=[good])
    worst = R(rid="worst", rows=[bad])
    md = compare([worst, best])

    best_line = [line for line in md.splitlines() if line.startswith("| best")][0]
    worst_line = [line for line in md.splitlines() if line.startswith("| worst")][0]
    assert "\u2605" in best_line
    assert "\u2605" not in worst_line
    assert md.count("\u2605") == 1


def test_compare_star_first_tie_only(bad: TaskResult) -> None:
    tie1 = R(rid="tie1", rows=[bad])
    tie2 = R(rid="tie2", rows=[bad])
    md = compare([tie1, tie2])

    tie1_line = [line for line in md.splitlines() if line.startswith("| tie1")][0]
    tie2_line = [line for line in md.splitlines() if line.startswith("| tie2")][0]
    assert "\u2605" in tie1_line
    assert "\u2605" not in tie2_line
    assert md.count("\u2605") == 1
