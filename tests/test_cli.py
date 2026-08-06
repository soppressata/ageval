from __future__ import annotations

import json
from pathlib import Path

import pytest

from ageval.cli import main, parse_kv
from ageval.core import Task
from ageval.tasks import save_tasks

PASSING_TASKS = [
    Task(id="t1", input="hello world", expected="hello world", scorer="exact"),
    Task(id="t2", input="goodbye", expected="goodbye", scorer="exact"),
    Task(id="t3", input="foo bar", expected="foo bar", scorer="exact"),
]

FAILING_TASKS = [
    Task(id="t1", input="hello", expected="hello", scorer="exact"),
    Task(id="t2", input="goodbye", expected="different", scorer="exact"),
]

TAGGED_TASKS = [
    Task(id="t1", input="hi", expected=">>hi", tags=["smoke"], scorer="exact"),
    Task(id="t2", input="yo", expected=">>yo", tags=["smoke"], scorer="exact"),
    Task(id="t3", input="nomatch", expected="nomatch", tags=["other"], scorer="exact"),
]


# ---------------------------------------------------------------------------
# A. parse_kv
# ---------------------------------------------------------------------------


def test_parse_kv_none_and_empty() -> None:
    """None and empty input both yield an empty dict."""
    assert parse_kv(None) == {}
    assert parse_kv([]) == {}


def test_parse_kv_float_value() -> None:
    """JSON-encoded floats decode to Python float."""
    result = parse_kv(["temperature=0.2"])
    assert result == {"temperature": 0.2}
    assert isinstance(result["temperature"], float)


def test_parse_kv_string_value() -> None:
    """Non-JSON values fall back to a raw string."""
    result = parse_kv(["model=openai/gpt-4"])
    assert result == {"model": "openai/gpt-4"}
    assert isinstance(result["model"], str)


def test_parse_kv_bool_value() -> None:
    """JSON booleans decode to Python bool."""
    result = parse_kv(["uppercase=true"])
    assert result["uppercase"] is True


def test_parse_kv_null_value() -> None:
    """JSON null decodes to Python None."""
    result = parse_kv(["system=null"])
    assert result["system"] is None


def test_parse_kv_splits_on_first_equals() -> None:
    """Only the first '=' is treated as the key/value separator."""
    result = parse_kv(["prefix=a=b"])
    assert result == {"prefix": "a=b"}


def test_parse_kv_missing_equals_raises() -> None:
    """A pair without '=' raises ValueError mentioning the offending value."""
    with pytest.raises(ValueError) as exc:
        parse_kv(["bogus"])
    assert "bogus" in str(exc.value)


# ---------------------------------------------------------------------------
# B. list agents / scorers
# ---------------------------------------------------------------------------


def test_list_agents(capsys: pytest.CaptureFixture[str]) -> None:
    """``list agents`` exits 0 and prints registered agent names."""
    rc = main(["list", "agents"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "echo" in out


def test_list_scorers(capsys: pytest.CaptureFixture[str]) -> None:
    """``list scorers`` exits 0 and prints registered scorer names."""
    rc = main(["list", "scorers"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "exact" in out


# ---------------------------------------------------------------------------
# C. argparse rejects invalid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [["run"], ["nope"], ["list", "planets"]],
)
def test_argparse_rejects_invalid(argv: list[str]) -> None:
    """Missing/invalid subcommands and choices raise SystemExit."""
    with pytest.raises(SystemExit):
        main(argv)


# ---------------------------------------------------------------------------
# D. validate
# ---------------------------------------------------------------------------


def test_validate_ok_reports_count_and_scorers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid suite prints the task count and per-scorer breakdown."""
    suite = tmp_path / "suite.jsonl"
    save_tasks(PASSING_TASKS[:1], suite)
    rc = main(["validate", str(suite)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ok: 1 tasks" in out
    assert "exact: 1" in out


def test_validate_missing_file_stderr_and_rc1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A nonexistent suite path reports to stderr and exits 1."""
    missing = tmp_path / "nope.jsonl"
    rc = main(["validate", str(missing)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "Path not found" in captured.err


def test_validate_duplicate_ids_stderr_and_rc1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Duplicate task ids fail validation with a stderr message."""
    suite = tmp_path / "dup.jsonl"
    save_tasks(
        [
            Task(id="dup", input="a", expected="a", scorer="exact"),
            Task(id="dup", input="b", expected="b", scorer="exact"),
        ],
        suite,
    )
    rc = main(["validate", str(suite)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "Duplicate" in captured.err


# ---------------------------------------------------------------------------
# E. run (offline echo) + compare
# ---------------------------------------------------------------------------


def test_run_writes_reports_and_compare(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run`` writes json/md/html with run_id filenames; ``compare`` lists run ids."""
    suite_a = tmp_path / "suite_a.jsonl"
    save_tasks(PASSING_TASKS, suite_a)
    out_dir = tmp_path / "out"

    rc = main(
        [
            "run",
            str(suite_a),
            "--agent",
            "echo",
            "--out",
            str(out_dir),
            "--format",
            "json,markdown,html",
            "--quiet",
        ]
    )
    capsys.readouterr()
    assert rc == 0

    jsons = sorted(out_dir.glob("*.json"))
    assert len(jsons) == 1
    report_a = json.loads(jsons[0].read_text())
    run_id_a = report_a["run_id"]
    assert (out_dir / f"{run_id_a}.json").exists()
    assert (out_dir / f"{run_id_a}.md").exists()
    assert (out_dir / f"{run_id_a}.html").exists()
    assert "summary" in report_a

    suite_b = tmp_path / "suite_b.jsonl"
    save_tasks(PASSING_TASKS, suite_b)
    out_b = tmp_path / "out_b"
    rc2 = main(
        [
            "run",
            str(suite_b),
            "--agent",
            "echo",
            "--out",
            str(out_b),
            "--format",
            "json",
            "--quiet",
        ]
    )
    capsys.readouterr()
    assert rc2 == 0
    jsons_b = sorted(out_b.glob("*.json"))
    run_id_b = json.loads(jsons_b[0].read_text())["run_id"]
    assert run_id_a != run_id_b

    rc3 = main(["compare", str(jsons[0]), str(jsons_b[0])])
    out = capsys.readouterr().out
    assert rc3 == 0
    assert run_id_a in out
    assert run_id_b in out


# ---------------------------------------------------------------------------
# F. run exit codes
# ---------------------------------------------------------------------------


def test_run_exit_code_all_pass_is_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fully-passing suite exits 0."""
    suite = tmp_path / "suite.jsonl"
    save_tasks(PASSING_TASKS, suite)
    out_dir = tmp_path / "out"
    rc = main(
        [
            "run",
            str(suite),
            "--agent",
            "echo",
            "--out",
            str(out_dir),
            "--format",
            "json",
            "--quiet",
        ]
    )
    capsys.readouterr()
    assert rc == 0


def test_run_exit_code_one_failing_is_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A suite with a failing task exits 1."""
    suite = tmp_path / "mixed.jsonl"
    save_tasks(FAILING_TASKS, suite)
    out_dir = tmp_path / "out"
    rc = main(
        [
            "run",
            str(suite),
            "--agent",
            "echo",
            "--out",
            str(out_dir),
            "--format",
            "json",
            "--quiet",
        ]
    )
    capsys.readouterr()
    assert rc == 1


# ---------------------------------------------------------------------------
# G. flags smoke test
# ---------------------------------------------------------------------------


def test_run_with_agent_arg_tags_limit_concurrency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--agent-arg, --tags, --limit and --concurrency all work together."""
    suite = tmp_path / "tagged.jsonl"
    save_tasks(TAGGED_TASKS, suite)
    out_dir = tmp_path / "out"
    rc = main(
        [
            "run",
            str(suite),
            "--agent",
            "echo",
            "--agent-arg",
            "prefix=>>",
            "--tags",
            "smoke",
            "--limit",
            "1",
            "--concurrency",
            "2",
            "--out",
            str(out_dir),
            "--quiet",
        ]
    )
    capsys.readouterr()
    assert rc == 0
    jsons = sorted(out_dir.glob("*.json"))
    assert len(jsons) == 1
    report = json.loads(jsons[0].read_text())
    assert report["summary"]["total"] == 1
    assert report["summary"]["passed"] == 1
