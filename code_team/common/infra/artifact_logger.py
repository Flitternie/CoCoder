from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json


class _PathEncoder(json.JSONEncoder):
    """JSON encoder that converts Path objects to strings."""
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class ArtifactLogger:
    """Manages persisting run artifacts to disk."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.logs_dir = run_dir / "logs"
        self.logs_workers_dir = self.logs_dir / "workers"
        self.artifacts_dir = run_dir / "artifacts"
        self.arch_dir = self.artifacts_dir / "architecture"
        self.rib_dir = self.artifacts_dir / "rib"
        self.code_dir = self.artifacts_dir / "code"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for path in [
            self.run_dir,
            self.logs_dir,
            self.logs_workers_dir,
            self.arch_dir,
            self.rib_dir,
            self.code_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def write_json(self, rel_path: str, data: dict | list) -> Path:
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, cls=_PathEncoder), encoding="utf-8")
        return path

    def write_text(self, rel_path: str, text: str) -> Path:
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def append_jsonl(self, rel_path: str, row: dict) -> Path:
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def lead_log(self, event: str, payload: dict) -> None:
        self.append_jsonl(
            "logs/lead.jsonl",
            {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "payload": payload,
            },
        )

    def worker_log(self, worker_id: str, event: str, payload: dict) -> None:
        self.append_jsonl(
            f"logs/workers/{worker_id}.jsonl",
            {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "payload": payload,
            },
        )
