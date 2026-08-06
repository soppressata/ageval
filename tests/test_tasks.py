from __future__ import annotations

import json
from pathlib import Path

import pytest

from ageval.core import Task
from ageval.errors import TaskLoadError
from ageval.tasks import load_tasks, save_tasks, suite_name


# ---------------------------------------------------------------------------
# A. jsonl loading
# ---------------------------------------------------------------------------


def test_jsonl_valid_tasks(tmp_path: Path) -> None:
    p = tmp_path / "suite.jsonl"
    p.write_text(
        '{"id": "a", "input": "hi"}\n'
        '{"id": "b", "input": "bye"}\n'
    )
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["a", "b"]
    assert [t.input for t in tasks] == ["hi", "bye"]


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "suite.jsonl"
    p.write_text(
        '{"id": "a", "input": "hi"}\n'
        "\n"
        "   \n"
        '{"id": "b", "input": "bye"}\n'
    )
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["a", "b"]


def test_jsonl_skips_comment_lines(tmp_path: Path) -> None:
    p = tmp_path / "suite.jsonl"
    p.write_text(
        '{"id": "a", "input": "hi"}\n'
        "# a comment\n"
        "  # indented comment\n"
        '{"id": "b", "input": "bye"}\n'
    )
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["a", "b"]


def test_jsonl_autogen_id_stem_index(tmp_path: Path) -> None:
    p = tmp_path / "mysuite.jsonl"
    p.write_text(
        '{"id": "x", "input": "first"}\n'
        '{"input": "second"}\n'
        '{"input": "third"}\n'
    )
    tasks = load_tasks(p)
    assert tasks[0].id == "x"
    assert tasks[1].id == "mysuite-1"
    assert tasks[2].id == "mysuite-2"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    tasks = [
        Task(id="r1", input="hello", expected="hello", scorer="exact"),
        Task(id="r2", input="world", tags=["smoke"]),
    ]
    out = tmp_path / "out.jsonl"
    save_tasks(tasks, out)
    loaded = load_tasks(out)
    assert len(loaded) == 2
    assert loaded[0].id == "r1"
    assert loaded[0].expected == "hello"
    assert loaded[0].scorer == "exact"
    assert loaded[1].id == "r2"
    assert loaded[1].tags == ["smoke"]


# ---------------------------------------------------------------------------
# B. json loading (bare list and {"tasks":[...]})
# ---------------------------------------------------------------------------


def test_json_bare_list(tmp_path: Path) -> None:
    p = tmp_path / "suite.json"
    p.write_text(json.dumps([{"id": "a", "input": "hi"}, {"id": "b", "input": "bye"}]))
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["a", "b"]


def test_json_tasks_wrapper(tmp_path: Path) -> None:
    p = tmp_path / "suite.json"
    p.write_text(json.dumps({"tasks": [{"id": "a", "input": "hi"}]}))
    tasks = load_tasks(p)
    assert tasks[0].id == "a"


def test_json_invalid_structure(tmp_path: Path) -> None:
    p = tmp_path / "suite.json"
    p.write_text(json.dumps({"foo": "bar"}))
    with pytest.raises(TaskLoadError):
        load_tasks(p)


def test_json_autogen_id(tmp_path: Path) -> None:
    p = tmp_path / "abc.json"
    p.write_text(json.dumps([{"input": "one"}, {"input": "two"}]))
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["abc-0", "abc-1"]


# ---------------------------------------------------------------------------
# C. directory loading
# ---------------------------------------------------------------------------


def test_dir_sorted_concatenated(tmp_path: Path) -> None:
    (tmp_path / "b.jsonl").write_text('{"id": "b1", "input": "x"}\n')
    (tmp_path / "a.jsonl").write_text('{"id": "a1", "input": "y"}\n')
    (tmp_path / "c.json").write_text(json.dumps([{"id": "c1", "input": "z"}]))
    tasks = load_tasks(tmp_path)
    assert [t.id for t in tasks] == ["a1", "b1", "c1"]


def test_dir_empty_returns_empty(tmp_path: Path) -> None:
    assert load_tasks(tmp_path) == []


def test_dir_ignores_unknown_extension(tmp_path: Path) -> None:
    (tmp_path / "tasks.jsonl").write_text('{"id": "a", "input": "hi"}\n')
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "data.csv").write_text("x,y")
    tasks = load_tasks(tmp_path)
    assert [t.id for t in tasks] == ["a"]


def test_dir_duplicate_ids_across_files(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text('{"id": "dup", "input": "one"}\n')
    (tmp_path / "b.jsonl").write_text('{"id": "dup", "input": "two"}\n')
    with pytest.raises(TaskLoadError) as exc:
        load_tasks(tmp_path)
    assert "dup" in str(exc.value)


# ---------------------------------------------------------------------------
# D. errors
# ---------------------------------------------------------------------------


def test_missing_path_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(TaskLoadError) as exc:
        load_tasks(missing)
    assert str(missing) in str(exc.value)


def test_unknown_extension_error(tmp_path: Path) -> None:
    p = tmp_path / "tasks.txt"
    p.write_text("hello")
    with pytest.raises(TaskLoadError) as exc:
        load_tasks(p)
    assert str(p) in str(exc.value)


def test_malformed_jsonl_line_number(tmp_path: Path) -> None:
    p = tmp_path / "suite.jsonl"
    p.write_text(
        '{"id": "a", "input": "hi"}\n'
        "not json\n"
        '{"id": "b", "input": "bye"}\n'
    )
    with pytest.raises(TaskLoadError) as exc:
        load_tasks(p)
    assert "line 2" in str(exc.value)


def test_duplicate_ids_within_one_call(tmp_path: Path) -> None:
    p = tmp_path / "suite.jsonl"
    p.write_text(
        '{"id": "same", "input": "one"}\n'
        '{"id": "same", "input": "two"}\n'
    )
    with pytest.raises(TaskLoadError) as exc:
        load_tasks(p)
    assert "same" in str(exc.value)


# ---------------------------------------------------------------------------
# E. helpers and real suites
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/tmp/data/suite.jsonl", "suite"),
        ("/tmp/data/suite.json", "suite"),
        (Path("/tmp/data/mydir"), "mydir"),
    ],
)
def test_suite_name(path: str, expected: str) -> None:
    assert suite_name(path) == expected


def test_demo_suite_loads() -> None:
    tasks = load_tasks("suites/demo.jsonl")
    assert len(tasks) == 10
    scorers = {t.scorer for t in tasks}
    assert scorers == {"exact", "includes", "regex", "numeric", "json_subset"}


def test_math_suite_loads() -> None:
    tasks = load_tasks("suites/math.jsonl")
    assert len(tasks) == 5
    assert all(t.scorer == "numeric" for t in tasks)


# ---------------------------------------------------------------------------
# F. save_tasks behavior
# ---------------------------------------------------------------------------


def test_save_tasks_creates_parent_dirs(tmp_path: Path) -> None:
    tasks = [Task(id="t1", input="hello")]
    out = tmp_path / "nested" / "dir" / "tasks.jsonl"
    save_tasks(tasks, out)
    assert out.exists()


def test_save_tasks_compact_jsonl(tmp_path: Path) -> None:
    tasks = [Task(id="t1", input="hello", tags=["a"])]
    out = tmp_path / "tasks.jsonl"
    save_tasks(tasks, out)
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    # compact: no space after separators
    assert ": " not in lines[0]
    parsed = json.loads(lines[0])
    assert parsed["id"] == "t1"
    assert parsed["tags"] == ["a"]
