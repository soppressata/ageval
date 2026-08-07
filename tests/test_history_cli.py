from __future__ import annotations

import json
from pathlib import Path

import pytest

from ageval.cli import main
from ageval.core import Task
from ageval.tasks import save_tasks

PASSING_TASKS = [
    Task(id="t1", input="hello world", expected="hello world", scorer="exact"),
    Task(id="t2", input="goodbye", expected="goodbye", scorer="exact"),
    Task(id="t3", input="foo bar", expected="foo bar", scorer="exact"),
]


def _run_suite(
    tmp_path: Path, suite_name: str, store: Path | None = None
) -> tuple[str, Path]:
    """Run a passing echo suite and return ``(run_id, store_dir)``."""
    suite = tmp_path / f"{suite_name}.jsonl"
    save_tasks(PASSING_TASKS, suite)
    if store is None:
        store = tmp_path / f"store_{suite_name}"
    out = tmp_path / f"out_{suite_name}"
    rc = main(
        [
            "run",
            str(suite),
            "--agent",
            "echo",
            "--out",
            str(out),
            "--store-dir",
            str(store),
            "--format",
            "json",
            "--quiet",
        ]
    )
    assert rc == 0
    reports = sorted(out.glob("*.json"))
    assert len(reports) == 1
    data = json.loads(reports[0].read_text())
    return data["run_id"], store


# ---------------------------------------------------------------------------
# A. run --store-dir saves a report
# ---------------------------------------------------------------------------


def test_run_store_dir_saves_report(tmp_path: Path) -> None:
    """``--store-dir`` persists the report to ``<store>/runs/<run_id>/report.json``."""
    run_id, store = _run_suite(tmp_path, "alpha")
    saved = store / "runs" / run_id / "report.json"
    assert saved.is_file()
    data = json.loads(saved.read_text())
    assert data["run_id"] == run_id
    assert data["agent_name"] == "echo"
    assert data["suite_name"] == "alpha"


# ---------------------------------------------------------------------------
# B. history list metadata and JSON mode
# ---------------------------------------------------------------------------


def test_history_list_json_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``history list --json`` emits run metadata with the expected fields."""
    run_id, store = _run_suite(tmp_path, "beta")
    capsys.readouterr()

    rc = main(["history", "list", "--store-dir", str(store), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert len(payload) >= 1
    entry = payload[0]
    assert set(entry.keys()) == {
        "run_id",
        "suite_name",
        "agent_name",
        "started_at",
        "finished_at",
    }
    assert entry["run_id"] == run_id
    assert entry["suite_name"] == "beta"
    assert entry["agent_name"] == "echo"
    assert entry["finished_at"] != ""


def test_history_list_text_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``history list`` (text) prints run metadata on each line."""
    run_id, store = _run_suite(tmp_path, "gamma")
    capsys.readouterr()

    rc = main(["history", "list", "--store-dir", str(store)])
    assert rc == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "echo" in out


def test_history_list_filters_and_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--suite``, ``--agent`` and ``--limit`` filter the JSON list."""
    store = tmp_path / "store_filter"
    base_id, _ = _run_suite(tmp_path, "base", store=store)
    _run_suite(tmp_path, "cand", store=store)
    capsys.readouterr()

    rc = main(
        ["history", "list", "--store-dir", str(store), "--suite", "base", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["run_id"] == base_id

    rc = main(
        ["history", "list", "--store-dir", str(store), "--agent", "echo", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2

    rc = main(
        ["history", "list", "--store-dir", str(store), "--limit", "1", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1


# ---------------------------------------------------------------------------
# C. history show formats
# ---------------------------------------------------------------------------


def test_history_show_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``history show --format json`` prints the full report as JSON."""
    run_id, store = _run_suite(tmp_path, "delta")
    capsys.readouterr()

    rc = main(
        ["history", "show", run_id, "--store-dir", str(store), "--format", "json"]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == run_id
    assert data["agent_name"] == "echo"
    assert "summary" in data
    assert data["summary"]["total"] == 3


def test_history_show_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``history show --format markdown`` renders a markdown report."""
    run_id, store = _run_suite(tmp_path, "epsilon")
    capsys.readouterr()

    rc = main(
        ["history", "show", run_id, "--store-dir", str(store), "--format", "markdown"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert f"# {run_id}" in out
    assert "| metric | value |" in out


def test_history_show_html(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``history show --format html`` renders a self-contained HTML document."""
    run_id, store = _run_suite(tmp_path, "zeta")
    capsys.readouterr()

    rc = main(
        ["history", "show", run_id, "--store-dir", str(store), "--format", "html"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "<html" in out
    assert run_id in out


# ---------------------------------------------------------------------------
# D. history diff output and invalid alpha
# ---------------------------------------------------------------------------


def test_history_diff_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``history diff`` emits a JSON comparison payload."""
    store = tmp_path / "store_diff"
    base_id, _ = _run_suite(tmp_path, "base", store=store)
    cand_id, _ = _run_suite(tmp_path, "cand", store=store)
    capsys.readouterr()

    rc = main(
        ["history", "diff", base_id, cand_id, "--store-dir", str(store)]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline_run_id"] == base_id
    assert payload["candidate_run_id"] == cand_id
    assert payload["paired"]["n_paired"] == 3
    assert payload["paired"]["n_baseline_only"] == 0
    assert payload["paired"]["n_candidate_only"] == 0
    assert "pass_rate_delta" in payload
    assert "mean_score_delta" in payload
    assert isinstance(payload["confidence_interval"], list)
    assert len(payload["confidence_interval"]) == 2
    assert "confidence_level" in payload
    assert "p_value" in payload
    assert "effect_size" in payload
    assert "regression_detected" in payload


@pytest.mark.parametrize("alpha", ["0", "1.0", "1.5", "-0.1"])
def test_history_diff_invalid_alpha(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], alpha: str
) -> None:
    """An out-of-range ``--alpha`` exits 1 with a stderr message."""
    store = tmp_path / "store_alpha"
    rc = main(
        [
            "history",
            "diff",
            "nope",
            "nope",
            "--store-dir",
            str(store),
            "--alpha",
            alpha,
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "alpha must be in (0,1)" in err


# ---------------------------------------------------------------------------
# E. missing run exit code
# ---------------------------------------------------------------------------


def test_history_show_missing_run_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``history show`` on a non-existent run exits 1 with a clear message."""
    store = tmp_path / "store_missing"
    store.mkdir()
    rc = main(
        ["history", "show", "does-not-exist", "--store-dir", str(store)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "run not found" in err


def test_history_diff_missing_run_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``history diff`` on a missing baseline exits 1 with a clear message."""
    store = tmp_path / "store_missing"
    store.mkdir()
    rc = main(
        ["history", "diff", "nope-a", "nope-b", "--store-dir", str(store)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "run not found" in err
