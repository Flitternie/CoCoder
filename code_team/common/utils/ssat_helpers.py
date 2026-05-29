"""Shared helpers for tools that operate on SSAT architecture JSON."""
from __future__ import annotations

import json
from pathlib import Path


def load_architecture(arch_path: Path) -> list[dict]:
    """Read and parse an SSAT JSON file.

    Returns a list of module dicts (normalises a single dict to a list).
    Raises FileNotFoundError or ValueError on problems.
    """
    if not arch_path.exists():
        raise FileNotFoundError(f"SSAT file not found: {arch_path}")
    data = json.loads(arch_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return data


def flatten_files(architecture: list[dict]) -> list[dict]:
    """Extract a flat list of file items from SSAT architecture."""
    files = []
    for module in architecture:
        for f in module.get("files", []):
            files.append(f)
    return files


def find_file_item(architecture: list[dict], target_file: str) -> dict | None:
    """Find a specific file item in the architecture by path."""
    for module in architecture:
        for f in module.get("files", []):
            if f.get("path") == target_file:
                return f
    return None


def architecture_summary(architecture: list[dict], exclude: str = "") -> list[dict]:
    """Extract lightweight metadata (path + description) for each file."""
    summary = []
    for f_item in flatten_files(architecture):
        rel = f_item.get("path", "")
        if rel == exclude:
            continue
        summary.append({"path": rel, "description": f_item.get("description", "")})
    return summary
