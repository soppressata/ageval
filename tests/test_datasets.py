from __future__ import annotations

import json
from pathlib import Path

import pytest

from ageval.core import Task
from ageval.datasets import (
    SCHEMA_VERSION,
    DEFAULT_COLUMNS,
    Dataset,
    DatasetMetadata,
    build_dataset,
    compute_sha256,
    filter_by_tags,
    load_csv,
    load_dataset,
    load_jsonl,
    make_metadata,
    sample,
    save_csv,
    save_dataset,
    save_jsonl,
    split,
)
from ageval.errors import TaskLoadError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_task() -> Task:
    return Task(
        id="t1",
        input="hello world",
        expected={"answer": 42},
        scorer="json_subset",
        scorer_args={"tolerance": 0.01},
        tags=["smoke", "unit"],
        metadata={"source": "test", "n": 1},
    )


@pytest.fixture
def tasks() -> list[Task]:
    return [Task(id=str(i), input=f"input-{i}") for i in range(10)]


# ---------------------------------------------------------------------------
# A. round trips
# ---------------------------------------------------------------------------


def test_jsonl_round_trip_preserves_all_fields(tmp_path: Path, full_task: Task) -> None:
    ds = build_dataset([full_task])
    p = tmp_path / "out.jsonl"
    save_jsonl(ds, p)
    loaded = load_jsonl(p)
    t = loaded.tasks[0]
    assert t.id == "t1"
    assert t.input == "hello world"
    assert t.expected == {"answer": 42}
    assert t.scorer == "json_subset"
    assert t.scorer_args == {"tolerance": 0.01}
    assert t.tags == ["smoke", "unit"]
    assert t.metadata == {"source": "test", "n": 1}


def test_csv_round_trip_preserves_all_fields(tmp_path: Path, full_task: Task) -> None:
    ds = build_dataset([full_task])
    p = tmp_path / "out.csv"
    save_csv(ds, p)
    loaded = load_csv(p)
    t = loaded.tasks[0]
    assert t.id == "t1"
    assert t.input == "hello world"
    assert t.expected == {"answer": 42}
    assert t.scorer == "json_subset"
    assert t.scorer_args == {"tolerance": 0.01}
    assert t.tags == ["smoke", "unit"]
    assert t.metadata == {"source": "test", "n": 1}


def test_jsonl_round_trip_multiple_tasks(tmp_path: Path, tasks: list[Task]) -> None:
    p = tmp_path / "multi.jsonl"
    save_jsonl(tasks, p)
    loaded = load_jsonl(p)
    assert len(loaded.tasks) == 10
    assert [t.id for t in loaded.tasks] == [str(i) for i in range(10)]


def test_csv_round_trip_multiple_tasks(tmp_path: Path, tasks: list[Task]) -> None:
    p = tmp_path / "multi.csv"
    save_csv(tasks, p)
    loaded = load_csv(p)
    assert len(loaded.tasks) == 10
    assert [t.id for t in loaded.tasks] == [str(i) for i in range(10)]


def test_jsonl_round_trip_subset_columns(tmp_path: Path, full_task: Task) -> None:
    p = tmp_path / "s.jsonl"
    save_jsonl([full_task], p, columns=["id", "input"])
    loaded = load_jsonl(p, columns=["id", "input"])
    t = loaded.tasks[0]
    assert t.id == "t1"
    assert t.input == "hello world"
    assert t.scorer == "exact"
    assert t.expected is None
    assert t.scorer_args == {}
    assert t.tags == []
    assert t.metadata == {}


def test_csv_round_trip_subset_columns(tmp_path: Path, full_task: Task) -> None:
    p = tmp_path / "s.csv"
    save_csv([full_task], p, columns=["id", "input"])
    loaded = load_csv(p, columns=["id", "input"])
    t = loaded.tasks[0]
    assert t.id == "t1"
    assert t.input == "hello world"
    assert t.scorer == "exact"
    assert t.expected is None
    assert t.scorer_args == {}
    assert t.tags == []
    assert t.metadata == {}


# ---------------------------------------------------------------------------
# B. quoted / tricky CSV fields
# ---------------------------------------------------------------------------


def test_csv_field_with_comma(tmp_path: Path) -> None:
    tasks = [Task(id="t1", input="hello, world")]
    p = tmp_path / "q.csv"
    save_csv(tasks, p)
    loaded = load_csv(p)
    assert loaded.tasks[0].input == "hello, world"


def test_csv_field_with_double_quote(tmp_path: Path) -> None:
    tasks = [Task(id="t1", input='say "hi"')]
    p = tmp_path / "q.csv"
    save_csv(tasks, p)
    loaded = load_csv(p)
    assert loaded.tasks[0].input == 'say "hi"'


def test_csv_field_with_newline(tmp_path: Path) -> None:
    tasks = [Task(id="t1", input="line1\nline2")]
    p = tmp_path / "q.csv"
    save_csv(tasks, p)
    loaded = load_csv(p)
    assert loaded.tasks[0].input == "line1\nline2"


def test_csv_non_scalar_with_comma_and_quote(tmp_path: Path) -> None:
    tasks = [
        Task(
            id="t1",
            input="prompt",
            expected={"msg": 'a, "b"'},
            scorer="json_subset",
            tags=["x,y"],
            metadata={"m": 'n"m'},
        )
    ]
    p = tmp_path / "q.csv"
    save_csv(tasks, p)
    loaded = load_csv(p)
    t = loaded.tasks[0]
    assert t.expected == {"msg": 'a, "b"'}
    assert t.tags == ["x,y"]
    assert t.metadata == {"m": 'n"m'}


def test_csv_numeric_expected_round_trip(tmp_path: Path) -> None:
    tasks = [Task(id="t1", input="calc", expected=4, scorer="numeric")]
    p = tmp_path / "n.csv"
    save_csv(tasks, p)
    loaded = load_csv(p)
    assert loaded.tasks[0].expected == 4


def test_jsonl_compact_encoding(tmp_path: Path) -> None:
    tasks = [Task(id="t1", input="hi", tags=["a"])]
    p = tmp_path / "compact.jsonl"
    save_jsonl(tasks, p)
    lines = p.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["tags"] == ["a"]


# ---------------------------------------------------------------------------
# C. metadata / hash
# ---------------------------------------------------------------------------


def test_compute_sha256_deterministic(tasks: list[Task]) -> None:
    assert compute_sha256(tasks) == compute_sha256(list(tasks))


def test_hash_changes_when_input_differs() -> None:
    t1 = Task(id="a", input="hi")
    t2 = Task(id="a", input="bye")
    assert compute_sha256([t1]) != compute_sha256([t2])


def test_hash_changes_when_scorer_args_differs() -> None:
    t1 = Task(id="a", input="hi", scorer_args={"max": 1})
    t2 = Task(id="a", input="hi", scorer_args={"max": 2})
    assert compute_sha256([t1]) != compute_sha256([t2])


def test_hash_changes_when_tags_differs() -> None:
    t1 = Task(id="a", input="hi", tags=["x"])
    t2 = Task(id="a", input="hi", tags=["y"])
    assert compute_sha256([t1]) != compute_sha256([t2])


def test_make_metadata_fields(tasks: list[Task]) -> None:
    m = make_metadata(tasks)
    assert m.schema_version == SCHEMA_VERSION
    assert m.count == len(tasks)
    assert m.columns == list(DEFAULT_COLUMNS)
    assert m.sha256 == compute_sha256(tasks)
    assert m.created_at  # non-empty


def test_make_metadata_explicit_created_at(tasks: list[Task]) -> None:
    ts = "2024-01-01T00:00:00+00:00"
    m = make_metadata(tasks, created_at=ts)
    assert m.created_at == ts


def test_make_metadata_computed_sha_matches_recomputed(tasks: list[Task]) -> None:
    ds = build_dataset(tasks)
    assert ds.metadata.sha256 == compute_sha256(tasks)


def test_metadata_hash_jsonl_consistent(tmp_path: Path, tasks: list[Task]) -> None:
    ds = build_dataset(tasks)
    p = tmp_path / "out.jsonl"
    save_jsonl(ds, p)
    loaded = load_jsonl(p)
    assert loaded.metadata.sha256 == compute_sha256(tasks)


def test_metadata_hash_csv_consistent(tmp_path: Path, tasks: list[Task]) -> None:
    ds = build_dataset(tasks)
    p = tmp_path / "out.csv"
    save_csv(ds, p)
    loaded = load_csv(p)
    assert loaded.metadata.sha256 == compute_sha256(tasks)


def test_dataset_metadata_dict_round_trip() -> None:
    m = DatasetMetadata(
        schema_version="v1",
        sha256="abc123",
        count=3,
        columns=["id", "input"],
        created_at="2024-01-01T00:00:00+00:00",
    )
    assert DatasetMetadata.from_dict(m.to_dict()) == m


# ---------------------------------------------------------------------------
# D. deterministic sampling
# ---------------------------------------------------------------------------


def test_sample_reproducible(tasks: list[Task]) -> None:
    a = sample(tasks, 5, seed=42)
    b = sample(tasks, 5, seed=42)
    assert [t.id for t in a] == [t.id for t in b]


def test_sample_different_seeds(tasks: list[Task]) -> None:
    a = sample(tasks, 5, seed=1)
    b = sample(tasks, 5, seed=2)
    assert [t.id for t in a] != [t.id for t in b]


def test_sample_returns_subset(tasks: list[Task]) -> None:
    out = sample(tasks, 3, seed=0)
    assert len(out) == 3
    assert {t.id for t in out} <= {t.id for t in tasks}


def test_sample_n_equals_len(tasks: list[Task]) -> None:
    out = sample(tasks, len(tasks), seed=0)
    assert {t.id for t in out} == {t.id for t in tasks}


def test_sample_n_zero(tasks: list[Task]) -> None:
    assert sample(tasks, 0, seed=3) == []


def test_sample_too_large_raises(tasks: list[Task]) -> None:
    with pytest.raises(ValueError):
        sample(tasks, len(tasks) + 1)


def test_sample_negative_raises(tasks: list[Task]) -> None:
    with pytest.raises(ValueError):
        sample(tasks, -1)


# ---------------------------------------------------------------------------
# E. tag filtering
# ---------------------------------------------------------------------------


def test_filter_by_tags_any() -> None:
    tasks = [
        Task(id="a", input="i", tags=["x"]),
        Task(id="b", input="i", tags=["y"]),
        Task(id="c", input="i", tags=["x", "y"]),
        Task(id="d", input="i", tags=[]),
    ]
    ids = [t.id for t in filter_by_tags(tasks, ["x"])]
    assert ids == ["a", "c"]


def test_filter_by_tags_all() -> None:
    tasks = [
        Task(id="a", input="i", tags=["x", "y"]),
        Task(id="b", input="i", tags=["x"]),
        Task(id="c", input="i", tags=["x", "y", "z"]),
    ]
    ids = [t.id for t in filter_by_tags(tasks, ["x", "y"], match="all")]
    assert ids == ["a", "c"]


def test_filter_by_tags_empty_returns_all(tasks: list[Task]) -> None:
    out = filter_by_tags(tasks, [])
    assert [t.id for t in out] == [t.id for t in tasks]


def test_filter_by_tags_no_match(tasks: list[Task]) -> None:
    assert filter_by_tags(tasks, ["nonexistent"], match="any") == []


def test_filter_by_tags_bad_match(tasks: list[Task]) -> None:
    with pytest.raises(ValueError):
        filter_by_tags(tasks, ["x"], match="none")


def test_filter_does_not_mutate(tasks: list[Task]) -> None:
    orig = list(tasks)
    filter_by_tags(tasks, ["x"], match="any")
    assert [t.id for t in tasks] == [t.id for t in orig]


# ---------------------------------------------------------------------------
# F. deterministic splits
# ---------------------------------------------------------------------------


def test_split_reproducible(tasks: list[Task]) -> None:
    a = split(tasks, [0.5, 0.5], seed=1)
    b = split(tasks, [0.5, 0.5], seed=1)
    assert [[t.id for t in p] for p in a] == [[t.id for t in p] for p in b]


def test_split_all_tasks_placed_once(tasks: list[Task]) -> None:
    parts = split(tasks, [0.5, 0.5], seed=0)
    ids = [t.id for p in parts for t in p]
    assert len(ids) == len(set(ids)) == len(tasks)


def test_split_three_ways(tasks: list[Task]) -> None:
    parts = split(tasks, [0.5, 0.3, 0.2], seed=7)
    assert len(parts) == 3
    assert sum(len(p) for p in parts) == len(tasks)


def test_split_single_partition(tasks: list[Task]) -> None:
    parts = split(tasks, [1.0], seed=0)
    assert len(parts) == 1
    assert len(parts[0]) == len(tasks)


def test_split_different_seeds(tasks: list[Task]) -> None:
    a = split(tasks, [0.5, 0.5], seed=1)
    b = split(tasks, [0.5, 0.5], seed=2)
    assert [[t.id for t in p] for p in a] != [[t.id for t in p] for p in b]


def test_split_bad_sum(tasks: list[Task]) -> None:
    with pytest.raises(ValueError):
        split(tasks, [0.5])


def test_split_negative_ratio(tasks: list[Task]) -> None:
    with pytest.raises(ValueError):
        split(tasks, [0.6, -0.5])


def test_split_empty_ratios(tasks: list[Task]) -> None:
    with pytest.raises(ValueError):
        split(tasks, [])


def test_split_does_not_mutate(tasks: list[Task]) -> None:
    orig = [t.id for t in tasks]
    split(tasks, [0.5, 0.5], seed=0)
    assert [t.id for t in tasks] == orig


# ---------------------------------------------------------------------------
# G. invalid inputs
# ---------------------------------------------------------------------------


def test_load_dataset_missing_path(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError) as exc:
        load_dataset(tmp_path / "nope.jsonl")
    assert "nope" in str(exc.value)


def test_load_jsonl_missing_path(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError):
        load_jsonl(tmp_path / "nope.jsonl")


def test_load_csv_missing_path(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError):
        load_csv(tmp_path / "nope.csv")


def test_load_dataset_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "d.txt"
    p.write_text("data")
    with pytest.raises(TaskLoadError) as exc:
        load_dataset(p)
    assert str(p) in str(exc.value)


def test_load_dataset_directory(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError):
        load_dataset(tmp_path)


def test_save_dataset_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError):
        save_dataset([Task(id="a", input="hi")], tmp_path / "out.txt")


def test_malformed_jsonl_raises(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text('{"id": "a", "input": "hi"}\nnot json\n')
    with pytest.raises(TaskLoadError) as exc:
        load_jsonl(p)
    assert "line 2" in str(exc.value)


def test_jsonl_non_object_entry_raises(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text('[1, 2, 3]\n')
    with pytest.raises(TaskLoadError):
        load_jsonl(p)


def test_csv_header_unknown_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "s.csv"
    p.write_text("id,input,notes\n")
    with pytest.raises(TaskLoadError):
        load_csv(p)


def test_csv_missing_requested_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "s.csv"
    p.write_text("id,input\n")
    with pytest.raises(TaskLoadError) as exc:
        load_csv(p, columns=["id", "input", "expected"])
    assert "expected" in str(exc.value)


def test_csv_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(TaskLoadError):
        load_csv(p)


def test_jsonl_duplicate_ids_raises(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text('{"id": "a", "input": "one"}\n{"id": "a", "input": "two"}\n')
    with pytest.raises(TaskLoadError) as exc:
        load_jsonl(p)
    assert "a" in str(exc.value)


def test_csv_duplicate_ids_raises(tmp_path: Path) -> None:
    p = tmp_path / "s.csv"
    p.write_text("id,input\na,one\na,two\n")
    with pytest.raises(TaskLoadError):
        load_csv(p)


def test_jsonl_autogen_id(tmp_path: Path) -> None:
    p = tmp_path / "mysuite.jsonl"
    p.write_text('{"input": "no id"}\n')
    loaded = load_jsonl(p)
    assert loaded.tasks[0].id == "mysuite-0"


def test_jsonl_defaults_scorer_and_expected(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text('{"id": "t1", "input": "hi"}\n')
    loaded = load_jsonl(p)
    t = loaded.tasks[0]
    assert t.scorer == "exact"
    assert t.scorer_args == {}
    assert t.tags == []
    assert t.metadata == {}
    assert t.expected is None


def test_build_dataset_unknown_column() -> None:
    with pytest.raises(ValueError):
        build_dataset([Task(id="a", input="hi")], columns=["id", "bogus"])


def test_save_dataset_creates_parent_dirs(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "dir" / "out.jsonl"
    save_dataset([Task(id="a", input="hi")], p)
    assert p.exists()


# ---------------------------------------------------------------------------
# H. JSON envelope (.json)
# ---------------------------------------------------------------------------


def test_json_envelope_round_trip(tmp_path: Path, full_task: Task) -> None:
    ds = build_dataset([full_task])
    p = tmp_path / "out.json"
    save_dataset(ds, p)
    loaded = load_dataset(p)
    assert loaded.metadata.sha256 == ds.metadata.sha256
    assert loaded.metadata.count == 1
    assert loaded.metadata.schema_version == SCHEMA_VERSION
    t = loaded.tasks[0]
    assert t.id == "t1"
    assert t.input == "hello world"
    assert t.expected == {"answer": 42}
    assert t.scorer == "json_subset"
    assert t.scorer_args == {"tolerance": 0.01}
    assert t.tags == ["smoke", "unit"]
    assert t.metadata == {"source": "test", "n": 1}


def test_json_envelope_from_list(tmp_path: Path) -> None:
    tasks = [Task(id="a", input="hi", tags=["x"])]
    p = tmp_path / "out.json"
    save_dataset(tasks, p)
    loaded = load_dataset(p)
    assert loaded.tasks[0].tags == ["x"]
    assert loaded.metadata.sha256 == compute_sha256(tasks)


def test_json_envelope_missing_tasks_key(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"foo": "bar"}))
    with pytest.raises(TaskLoadError):
        load_dataset(p)


def test_json_envelope_is_list_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(TaskLoadError):
        load_dataset(p)


def test_json_envelope_malformed(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    with pytest.raises(TaskLoadError):
        load_dataset(p)


def test_json_envelope_duplicate_ids_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": SCHEMA_VERSION,
                    "sha256": "x",
                    "count": 2,
                    "columns": [],
                    "created_at": "",
                },
                "tasks": [
                    {"id": "a", "input": "one"},
                    {"id": "a", "input": "two"},
                ],
            }
        )
    )
    with pytest.raises(TaskLoadError):
        load_dataset(p)


# ---------------------------------------------------------------------------
# I. Dataset methods
# ---------------------------------------------------------------------------


def test_dataset_dict_round_trip(full_task: Task) -> None:
    ds = build_dataset([full_task])
    d = ds.to_dict()
    ds2 = Dataset.from_dict(d)
    assert ds2.metadata.sha256 == ds.metadata.sha256
    assert ds2.tasks[0].scorer_args == {"tolerance": 0.01}


def test_dataset_dict_missing_keys_raises() -> None:
    with pytest.raises(ValueError):
        Dataset.from_dict({"tasks": []})


def test_dataset_save_load_method(tmp_path: Path, full_task: Task) -> None:
    ds = build_dataset([full_task])
    p = tmp_path / "out.jsonl"
    ds.save(p)
    loaded = Dataset.load(p)
    assert loaded.tasks[0].scorer_args == {"tolerance": 0.01}


def test_dataset_to_jsonl_method(tmp_path: Path, full_task: Task) -> None:
    ds = build_dataset([full_task])
    p = tmp_path / "out.jsonl"
    result = ds.to_jsonl(p)
    assert isinstance(result, Dataset)
    loaded = load_jsonl(p)
    assert loaded.tasks[0].scorer_args == {"tolerance": 0.01}


def test_dataset_to_csv_method(tmp_path: Path, full_task: Task) -> None:
    ds = build_dataset([full_task])
    p = tmp_path / "out.csv"
    result = ds.to_csv(p)
    assert isinstance(result, Dataset)
    loaded = load_csv(p)
    assert loaded.tasks[0].scorer_args == {"tolerance": 0.01}


def test_dataset_load_jsonl_csv_methods(tmp_path: Path, full_task: Task) -> None:
    ds = build_dataset([full_task])
    jp = tmp_path / "out.jsonl"
    cp = tmp_path / "out.csv"
    save_jsonl(ds, jp)
    save_csv(ds, cp)
    assert Dataset.load_jsonl(jp).tasks[0].scorer == "json_subset"
    assert Dataset.load_csv(cp).tasks[0].scorer == "json_subset"
