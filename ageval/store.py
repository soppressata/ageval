from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ageval.core import RunReport


@dataclass
class RunMeta:
    """Lightweight metadata about a stored run."""

    run_id: str
    suite_name: str
    agent_name: str
    started_at: str
    finished_at: str


class RunStore:
    """Manages run reports on disk under ``root/runs/<run_id>/report.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"

    def _run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def _report_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "report.json"

    def save_run(self, report: RunReport) -> str:
        """Persist a RunReport atomically. Returns the run_id."""
        run_id = report.run_id
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._report_path(run_id)
        data = json.dumps(report.to_dict(), indent=2, sort_keys=False)
        fd, tmp_path = tempfile.mkstemp(
            prefix=run_id + ".",
            suffix=".tmp",
            dir=str(run_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(report_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return run_id

    def load_run(self, run_id: str) -> RunReport:
        """Load a RunReport by run_id."""
        path = self._report_path(run_id)
        with open(path, "r", encoding="utf-8") as f:
            return RunReport.from_dict(json.load(f))

    def list_runs(
        self,
        suite_name: str | None = None,
        agent_name: str | None = None,
        since: str | None = None,
    ) -> list[RunMeta]:
        """List run metadata, newest first. Malformed dirs are skipped."""
        metas: list[RunMeta] = []
        if not self.runs_dir.is_dir():
            return metas
        for entry in self.runs_dir.iterdir():
            if not entry.is_dir():
                continue
            run_id = entry.name
            report_path = entry / "report.json"
            if not report_path.is_file():
                continue
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                suite = raw["suite_name"]
                agent = raw["agent_name"]
                started = raw["started_at"]
            except Exception:
                continue
            if suite_name is not None and suite != suite_name:
                continue
            if agent_name is not None and agent != agent_name:
                continue
            if since is not None and started < since:
                continue
            metas.append(
                RunMeta(
                    run_id=run_id,
                    suite_name=suite,
                    agent_name=agent,
                    started_at=started,
                    finished_at=raw.get("finished_at", ""),
                )
            )
        metas.sort(key=lambda m: m.started_at, reverse=True)
        return metas

    def latest(
        self, suite_name: str, agent_name: str
    ) -> RunReport | None:
        """Return the most recent RunReport matching suite/agent, or None."""
        metas = self.list_runs(suite_name=suite_name, agent_name=agent_name)
        if not metas:
            return None
        return self.load_run(metas[0].run_id)
