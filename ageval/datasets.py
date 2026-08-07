from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ageval.core import Task
from ageval.errors import TaskLoadError

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_COLUMNS",
    "DatasetMetadata",
    "Dataset",
    "compute_sha256",
    "make_metadata",
    "build_dataset",
    "load_dataset",
    "save_dataset",
    "load_jsonl",
    "save_jsonl",
    "load_csv",
    "save_csv",
    "sample",
    "filter_by_tags",
    "split",
]

SCHEMA_VERSION: str = "dataset_v1"

SCALAR_FIELDS: tuple[str, ...] = ("id", "input", "scorer")
NON_SCALAR_FIELDS: tuple[str, ...] = ("expected", "scorer_args", "tags", "metadata")
KNOWN_FIELDS: tuple[str, ...] = SCALAR_FIELDS + NON_SCALAR_FIELDS
DEFAULT_COLUMNS: tuple[str, ...] = ("id", "input", "expected", "scorer", "scorer_args", "tags", "metadata")

_NON_SCALAR: frozenset[str] = frozenset(NON_SCALAR_FIELDS)

_DEFAULTS: dict[str, Any] = {
    "expected": None,
    "scorer_args": {},
    "tags": [],
    "metadata": {},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _defaults_dict() -> dict[str, Any]:
    return {"id": "", "input": "", "expected": None, "scorer": "exact",
            "scorer_args": {}, "tags": [], "metadata": {}}


def _validate_columns(columns: Sequence[str]) -> list[str]:
    cols = list(columns)
    if not cols:
        raise ValueError("columns must be non-empty")
    if len(cols) != len(set(cols)):
        raise ValueError(f"duplicate columns: {cols}")
    for c in cols:
        if c not in KNOWN_FIELDS:
            raise ValueError(f"unknown column {c!r}; known columns: {list(KNOWN_FIELDS)}")
    return cols


def _field_value(task: Task, col: str) -> Any:
    return getattr(task, col)


def _value_to_cell(value: Any, field_name: str) -> str:
    if field_name in _NON_SCALAR:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if value is None:
        return ""
    return str(value)


def _cell_to_value(cell: str | None, field_name: str) -> Any:
    if cell is None:
        cell = ""
    if field_name in _NON_SCALAR:
        if cell == "":
            return _DEFAULTS[field_name]
        try:
            return json.loads(cell)
        except json.JSONDecodeError as e:
            raise TaskLoadError(f"malformed value for column '{field_name}': invalid JSON: {e}") from e
    return cell


def _row_to_dict_jsonl(obj: dict, columns: Sequence[str], stem: str, index: int) -> dict[str, Any]:
    d = _defaults_dict()
    for c in columns:
        if c in obj:
            d[c] = obj[c]
    if not d["id"]:
        d["id"] = f"{stem}-{index}"
    return d


def _row_to_dict_csv(row: dict[str, str | None], columns: Sequence[str], stem: str, index: int) -> dict[str, Any]:
    d = _defaults_dict()
    for c in columns:
        if c in row and row[c] is not None:
            d[c] = _cell_to_value(row[c], c)
    if not d["id"]:
        d["id"] = f"{stem}-{index}"
    return d


def _build_task(d: dict[str, Any]) -> Task:
    try:
        return Task.from_dict(d)
    except KeyError as e:
        raise TaskLoadError(f"task missing required field {e}") from e
    except (TypeError, ValueError) as e:
        raise TaskLoadError(f"invalid task entry: {e}") from e


def _check_duplicates(tasks: list[Task]) -> None:
    seen: set[str] = set()
    for t in tasks:
        if t.id in seen:
            raise TaskLoadError(f"duplicate dataset id: {t.id}")
        seen.add(t.id)


def compute_sha256(tasks: list[Task]) -> str:
    """Deterministic SHA-256 hex digest over the canonical JSON of task dicts."""
    payload = json.dumps(
        [t.to_dict() for t in tasks],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class DatasetMetadata:
    schema_version: str
    sha256: str
    count: int
    columns: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "count": self.count,
            "columns": list(self.columns),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetMetadata:
        return cls(
            schema_version=d["schema_version"],
            sha256=d["sha256"],
            count=int(d["count"]),
            columns=list(d.get("columns", [])),
            created_at=d.get("created_at", ""),
        )


@dataclass
class Dataset:
    tasks: list[Task]
    metadata: DatasetMetadata

    @classmethod
    def from_list(
        cls,
        tasks: Sequence[Task],
        *,
        columns: Sequence[str] = DEFAULT_COLUMNS,
        created_at: str | None = None,
    ) -> Dataset:
        return build_dataset(tasks, columns=columns, created_at=created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Dataset:
        if "metadata" not in d or "tasks" not in d:
            raise ValueError("dataset dict must contain 'metadata' and 'tasks' keys")
        meta = DatasetMetadata.from_dict(d["metadata"])
        tasks = [_build_task(t) for t in d["tasks"]]
        return cls(tasks=tasks, metadata=meta)

    def to_jsonl(self, path: str | Path) -> Dataset:
        return save_jsonl(self, path)

    def to_csv(self, path: str | Path, *, columns: Sequence[str] | None = None) -> Dataset:
        return save_csv(self, path, columns=columns)

    def save(self, path: str | Path, *, columns: Sequence[str] | None = None) -> None:
        save_dataset(self, path, columns=columns)

    @classmethod
    def load(cls, path: str | Path, *, columns: Sequence[str] = DEFAULT_COLUMNS) -> Dataset:
        return load_dataset(path, columns=columns)

    @classmethod
    def load_jsonl(cls, path: str | Path, *, columns: Sequence[str] = DEFAULT_COLUMNS) -> Dataset:
        return load_jsonl(path, columns=columns)

    @classmethod
    def load_csv(cls, path: str | Path, *, columns: Sequence[str] = DEFAULT_COLUMNS) -> Dataset:
        return load_csv(path, columns=columns)


def _normalize(tasks_or_dataset: list[Task] | Dataset) -> tuple[list[Task], list[str]]:
    if isinstance(tasks_or_dataset, Dataset):
        tasks = list(tasks_or_dataset.tasks)
        columns = list(tasks_or_dataset.metadata.columns)
    else:
        tasks = list(tasks_or_dataset)
        columns = list(DEFAULT_COLUMNS)
    return tasks, columns


def make_metadata(
    tasks: list[Task],
    columns: Sequence[str] = DEFAULT_COLUMNS,
    *,
    created_at: str | None = None,
) -> DatasetMetadata:
    """Build deterministic dataset metadata for a list of tasks."""
    cols = _validate_columns(columns)
    return DatasetMetadata(
        schema_version=SCHEMA_VERSION,
        sha256=compute_sha256(tasks),
        count=len(tasks),
        columns=cols,
        created_at=created_at or _now(),
    )


def build_dataset(
    tasks: Sequence[Task],
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    created_at: str | None = None,
) -> Dataset:
    """Construct a :class:`Dataset` with freshly computed metadata."""
    tasks_list = list(tasks)
    cols = _validate_columns(columns)
    return Dataset(
        tasks=tasks_list,
        metadata=make_metadata(tasks_list, cols, created_at=created_at),
    )


def save_jsonl(
    tasks_or_dataset: list[Task] | Dataset,
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    created_at: str | None = None,
) -> Dataset:
    """Write tasks as JSONL (one compact JSON object per line) and return a Dataset."""
    tasks, cols = _normalize(tasks_or_dataset)
    if columns is not None:
        cols = _validate_columns(columns)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for t in tasks:
            row = {c: _field_value(t, c) for c in cols}
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str))
            f.write("\n")
    return Dataset(tasks=tasks, metadata=make_metadata(tasks, cols, created_at=created_at))


def load_jsonl(
    path: str | Path,
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> Dataset:
    """Load a JSONL dataset file into a :class:`Dataset` with recomputed metadata."""
    p = Path(path)
    if not p.exists():
        raise TaskLoadError(f"Path not found: {p}")
    cols = _validate_columns(columns)
    stem = p.stem
    tasks: list[Task] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise TaskLoadError(f"Malformed JSON in {p} at line {lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise TaskLoadError(
                    f"Malformed entry in {p} at line {lineno}: expected JSON object"
                )
            d = _row_to_dict_jsonl(obj, cols, stem, len(tasks))
            tasks.append(_build_task(d))
    _check_duplicates(tasks)
    return Dataset(tasks=tasks, metadata=make_metadata(tasks, cols))


def save_csv(
    tasks_or_dataset: list[Task] | Dataset,
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    created_at: str | None = None,
) -> Dataset:
    """Write tasks as CSV with the given (or default) columns and return a Dataset."""
    tasks, cols = _normalize(tasks_or_dataset)
    if columns is not None:
        cols = _validate_columns(columns)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=cols, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for t in tasks:
            writer.writerow({c: _value_to_cell(_field_value(t, c), c) for c in cols})
    return Dataset(tasks=tasks, metadata=make_metadata(tasks, cols, created_at=created_at))


def load_csv(
    path: str | Path,
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> Dataset:
    """Load a CSV dataset file into a :class:`Dataset` with recomputed metadata."""
    p = Path(path)
    if not p.exists():
        raise TaskLoadError(f"Path not found: {p}")
    cols = _validate_columns(columns)
    stem = p.stem
    tasks: list[Task] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise TaskLoadError(f"CSV file {p} has no header row")
        header = [h for h in reader.fieldnames if h is not None]
        for h in header:
            if h not in KNOWN_FIELDS:
                raise TaskLoadError(f"unknown column {h!r} in CSV header of {p}")
        missing = [c for c in cols if c not in header]
        if missing:
            raise TaskLoadError(
                f"CSV header of {p} missing requested column(s): {missing}"
            )
        for index, row in enumerate(reader):
            if row is None:
                continue
            row = row  # type: ignore[assignment]
            d = _row_to_dict_csv({c: row.get(c) for c in cols}, cols, stem, len(tasks))
            tasks.append(_build_task(d))
    _check_duplicates(tasks)
    return Dataset(tasks=tasks, metadata=make_metadata(tasks, cols))


def _save_json_envelope(dataset: Dataset, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(dataset.to_dict(), f, sort_keys=True, indent=2, default=str)
        f.write("\n")


def _load_json_envelope(p: Path) -> Dataset:
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise TaskLoadError(f"Malformed JSON in {p}: {e}") from e
    if not isinstance(data, dict) or "tasks" not in data or "metadata" not in data:
        raise TaskLoadError(
            f"Invalid dataset envelope in {p}: expected an object with 'tasks' and 'metadata'"
        )
    tasks = [_build_task(t) for t in data["tasks"]]
    _check_duplicates(tasks)
    meta = DatasetMetadata.from_dict(data["metadata"])
    return Dataset(tasks=tasks, metadata=meta)


def load_dataset(
    path: str | Path,
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> Dataset:
    """Load a dataset from ``.jsonl``, ``.csv``, or ``.json`` (envelope) by extension."""
    p = Path(path)
    if not p.exists():
        raise TaskLoadError(f"Path not found: {p}")
    if p.is_dir():
        raise TaskLoadError(f"Expected a dataset file, got directory: {p}")
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path, columns=columns)
    if suffix == ".csv":
        return load_csv(path, columns=columns)
    if suffix == ".json":
        return _load_json_envelope(p)
    raise TaskLoadError(f"Unknown dataset extension for {p}")


def save_dataset(
    dataset: Dataset | list[Task],
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
) -> None:
    """Save a dataset to disk, dispatching by file extension."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        save_jsonl(dataset, path, columns=columns)
    elif suffix == ".csv":
        save_csv(dataset, path, columns=columns)
    elif suffix == ".json":
        ds = dataset if isinstance(dataset, Dataset) else build_dataset(dataset)
        _save_json_envelope(ds, p)
    else:
        raise TaskLoadError(f"Unknown dataset extension for {p}")


def sample(
    tasks: list[Task], n: int, seed: int = 0
) -> list[Task]:
    """Reproducibly sample ``n`` tasks without replacement using ``seed``."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n > len(tasks):
        raise ValueError(f"cannot sample {n} tasks without replacement from {len(tasks)}")
    rng = random.Random(seed)
    return [tasks[i] for i in rng.sample(range(len(tasks)), n)]


def filter_by_tags(
    tasks: list[Task], tags: Sequence[str], *, match: str = "any"
) -> list[Task]:
    """Filter tasks by tags.

    ``match="any"`` keeps tasks having at least one of ``tags``;
    ``match="all"`` keeps tasks having every tag in ``tags``.
    An empty ``tags`` list returns all tasks unchanged.
    """
    if match not in ("any", "all"):
        raise ValueError(f"match must be 'any' or 'all', got {match!r}")
    tags_set = set(tags)
    if not tags_set:
        return list(tasks)
    result: list[Task] = []
    for t in tasks:
        ttags = set(t.tags)
        if (match == "any" and bool(tags_set & ttags)) or (
            match == "all" and tags_set <= ttags
        ):
            result.append(t)
    return result


def split(
    tasks: list[Task], ratios: Sequence[float], *, seed: int = 0
) -> list[list[Task]]:
    """Split tasks into partitions with the given fractional ``ratios``.

    ``ratios`` must sum to 1.0 (within tolerance). Sampling is reproducible via
    ``seed``. The last partition absorbs any remainder so all tasks are placed
    exactly once.
    """
    ratios_list = list(ratios)
    if not ratios_list:
        raise ValueError("ratios must be non-empty")
    for r in ratios_list:
        if r < 0:
            raise ValueError(f"negative ratio {r}")
    total = sum(ratios_list)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"ratios must sum to 1.0, got {total}")
    n = len(tasks)
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    parts: list[list[Task]] = []
    start = 0
    for r in ratios_list[:-1]:
        size = int(round(r * n))
        parts.append([tasks[order[i]] for i in range(start, start + size)])
        start += size
    parts.append([tasks[order[i]] for i in range(start, n)])
    return parts
