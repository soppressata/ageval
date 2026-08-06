from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ageval.core import Task
from ageval.errors import TaskLoadError

_VALID_SUFFIXES = (".jsonl", ".json")


def _build_task(d: dict) -> Task:
    """Build a ``Task`` from a dict, translating key errors into ``TaskLoadError``."""
    try:
        return Task.from_dict(d)
    except KeyError as e:
        raise TaskLoadError(f"Task missing required key {e}") from e


def _parse_jsonl(path: Path) -> list[Task]:
    """Parse a ``.jsonl`` file into tasks.

    Blank lines and lines whose first non-whitespace character is ``#`` are
    skipped. A malformed JSON line raises ``TaskLoadError`` naming the file and
    the 1-based line number. Missing ``id`` is assigned ``f"{stem}-{index}"``
    where ``index`` is the 0-based position of the task within the file.
    """
    stem = path.stem
    tasks: list[Task] = []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    index = 0
    for lineno, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise TaskLoadError(
                f"Malformed JSON in {path} at line {lineno}: {e}"
            ) from e
        if not isinstance(obj, dict):
            raise TaskLoadError(
                f"Malformed JSON in {path} at line {lineno}: expected JSON object"
            )
        d = dict(obj)
        if not d.get("id"):
            d["id"] = f"{stem}-{index}"
        index += 1
        tasks.append(_build_task(d))
    return tasks


def _parse_json(path: Path) -> list[Task]:
    """Parse a ``.json`` file into tasks.

    Accepts a bare list or ``{"tasks": [...]}``. Anything else raises
    ``TaskLoadError`` with the path in the message.
    """
    stem = path.stem
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise TaskLoadError(f"Malformed JSON in {path}: {e}") from e

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
        items = data["tasks"]
    else:
        raise TaskLoadError(
            f"Invalid JSON structure in {path}: expected a list or {{\"tasks\": [...]}}"
        )

    tasks: list[Task] = []
    for index, obj in enumerate(items):
        if not isinstance(obj, dict):
            raise TaskLoadError(
                f"Invalid task entry in {path} at index {index}: expected JSON object"
            )
        d = dict(obj)
        if not d.get("id"):
            d["id"] = f"{stem}-{index}"
        tasks.append(_build_task(d))
    return tasks


def _parse_file(path: Path) -> list[Task]:
    """Dispatch parsing of a single task file by its suffix."""
    suffix = path.suffix
    if suffix == ".jsonl":
        return _parse_jsonl(path)
    if suffix == ".json":
        return _parse_json(path)
    raise TaskLoadError(f"Unknown file extension for {path}")


def _parse_dir(path: Path) -> list[Task]:
    """Load every top-level ``*.jsonl`` and ``*.json`` file in a directory.

    Files are sorted by filename and concatenated. An empty directory returns
    an empty list.
    """
    files = sorted(
        (p for p in path.iterdir() if p.is_file() and p.suffix in _VALID_SUFFIXES),
        key=lambda p: p.name,
    )
    tasks: list[Task] = []
    for p in files:
        tasks.extend(_parse_file(p))
    return tasks


def _check_duplicates(tasks: list[Task]) -> None:
    """Raise ``TaskLoadError`` if any id appears more than once."""
    seen: set[str] = set()
    for t in tasks:
        if t.id in seen:
            raise TaskLoadError(f"Duplicate task id: {t.id}")
        seen.add(t.id)


def load_tasks(path: str | Path) -> list[Task]:
    """Load tasks from a file or directory.

    - ``.jsonl``: one JSON object per line; blank lines and ``#`` comments skipped.
    - ``.json``: a bare list or ``{"tasks": [...]}``.
    - directory: top-level ``*.jsonl``/``*.json`` files, sorted by filename.
    - Missing path or unknown extension raises ``TaskLoadError`` containing the path.
    - Duplicate ids across the whole call raise ``TaskLoadError`` naming the id.
    """
    p = Path(path)
    if not p.exists():
        raise TaskLoadError(f"Path not found: {p}")
    if p.is_dir():
        tasks = _parse_dir(p)
    elif p.is_file():
        tasks = _parse_file(p)
    else:
        raise TaskLoadError(f"Invalid path: {p}")
    _check_duplicates(tasks)
    return tasks


def save_tasks(tasks: list[Task], path: str | Path) -> None:
    """Write tasks as JSONL (one compact JSON object per line, trailing newline).

    Parent directories are created as needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), separators=(",", ":")))
            f.write("\n")


def suite_name(path: str | Path) -> str:
    """Return the file stem for a file, or the directory name for a directory."""
    p = Path(path)
    if p.is_dir():
        return p.name
    return p.stem
