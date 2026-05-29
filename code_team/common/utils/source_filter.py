"""Utility for identifying source-code files vs. data files."""
from __future__ import annotations

from pathlib import Path

SOURCE_EXTENSIONS = frozenset({
    ".py", ".sh", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".pl", ".r", ".swift", ".kt",
})

MAX_SOURCE_FILE_SIZE = 50 * 1024  # 50 KB – skip oversized data-like files


def is_source_file(path: str) -> bool:
    """Return True if *path* looks like a source code file (not data)."""
    return Path(path).suffix.lower() in SOURCE_EXTENSIONS
