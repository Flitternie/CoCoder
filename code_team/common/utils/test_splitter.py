"""Test splitter — map check_test files to source modules via AST import analysis.

Given a ``check_tests/`` directory and a source package name (e.g. ``"rsa"``),
this module analyses each test file's ``import`` / ``from … import`` statements
to determine which source files the test depends on.

Strategy
--------
1. ``import rsa.cli``            → direct mapping → ``rsa/cli.py``
2. ``from rsa.pkcs1 import X``   → direct mapping → ``rsa/pkcs1.py``
3. ``import rsa`` (package-only) → fallback       → all ``rsa/*.py`` files
"""
from __future__ import annotations

import ast
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_check_tests(
    check_tests_dir: Path,
    source_package: str,
    source_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Build a mapping ``{source_file: [test_file, …]}`` by analysing imports.

    Parameters
    ----------
    check_tests_dir:
        Absolute path to the ``check_tests/`` directory.
    source_package:
        Top-level Python package under test (e.g. ``"rsa"``, ``"tinydb"``).
    source_dir:
        Absolute path to the source package directory.  Used for enumerating
        ``.py`` files when a package-level import triggers the fallback.
        If *None*, defaults to ``check_tests_dir.parent / source_package``.

    Returns
    -------
    dict mapping *workspace-relative* source file paths (e.g. ``"rsa/cli.py"``)
    to lists of *workspace-relative* test file paths
    (e.g. ``"check_tests/test_cli.py"``).
    """
    if source_dir is None:
        source_dir = check_tests_dir.parent / source_package

    # Enumerate all .py files in the source package (for package-level fallback)
    all_source_files = _list_source_files(source_dir, source_package)

    # Analyse each test file
    mapping: dict[str, list[str]] = {}

    for test_path in iter_check_test_files(check_tests_dir):
        test_rel = str(test_path.relative_to(check_tests_dir.parent)).replace("\\", "/")
        imported_files = _extract_source_refs(
            test_path, source_package, all_source_files, source_dir,
        )

        for src_file in imported_files:
            mapping.setdefault(src_file, [])
            if test_rel not in mapping[src_file]:
                mapping[src_file].append(test_rel)

    return mapping


def get_tests_for_file(
    mapping: dict[str, list[str]],
    assigned_file: str,
) -> list[str]:
    """Return test files relevant to *assigned_file*.

    Tries an exact match first, then a package-level match
    (e.g. ``tinydb/__init__.py`` ↔ everything under ``tinydb/``).
    """
    # Exact match
    if assigned_file in mapping:
        return list(mapping[assigned_file])

    # Package-level match: if the assigned file is an __init__.py,
    # collect all tests that target any file in the same package.
    if assigned_file.endswith("/__init__.py"):
        pkg_prefix = assigned_file.rsplit("/", 1)[0] + "/"
        tests: list[str] = []
        for src, tfiles in mapping.items():
            if src.startswith(pkg_prefix):
                for t in tfiles:
                    if t not in tests:
                        tests.append(t)
        return tests

    return []


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def is_check_test_file(path: Path) -> bool:
    """Return True when *path* matches the project's check_test naming rules."""
    if path.suffix != ".py":
        return False

    name = path.name
    if name in {"__init__.py", "conftest.py"}:
        return False

    return (
        name == "test.py"
        or name == "unit_test.py"
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def iter_check_test_files(check_tests_dir: Path) -> list[Path]:
    """Enumerate real test modules inside *check_tests_dir*."""
    return [
        path
        for path in sorted(check_tests_dir.rglob("*.py"))
        if is_check_test_file(path)
    ]


def _list_source_files(source_dir: Path, source_package: str) -> list[str]:
    """Return workspace-relative paths for all ``.py`` files in the package."""
    result: list[str] = []
    if not source_dir.is_dir():
        return result
    for p in sorted(source_dir.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = f"{source_package}/{p.relative_to(source_dir)}"
        result.append(rel)
    return result


def _extract_source_refs(
    test_file: Path,
    source_package: str,
    all_source_files: list[str],
    source_dir: Path | None = None,
) -> set[str]:
    """Parse *test_file* with ``ast`` and extract referenced source files.

    Returns a set of workspace-relative source file paths.
    """
    try:
        source = test_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(test_file))
    except SyntaxError:
        return set()

    refs: set[str] = set()
    has_package_level_import = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _process_import_name(alias.name, source_package, refs, source_dir)
                if alias.name == source_package:
                    has_package_level_import = True

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                _process_import_name(node.module, source_package, refs, source_dir)
                if node.module == source_package:
                    has_package_level_import = True

    # Fallback: package-level import → map to all source files
    if has_package_level_import:
        refs.update(all_source_files)

    return refs


def _process_import_name(
    dotted_name: str,
    source_package: str,
    refs: set[str],
    source_dir: Path | None = None,
) -> bool:
    """Try to map a dotted import name to a source file.

    Returns *True* if a specific file was resolved,
    *False* if it's just the package root.

    Examples::

        "rsa.cli"       → adds "rsa/cli.py", returns True
        "rsa.key"       → adds "rsa/key.py", returns True
        "rsa"           → nothing added,     returns False
        "os.path"       → nothing (wrong package), returns False
    """
    parts = dotted_name.split(".")
    if not parts or parts[0] != source_package:
        return False

    if len(parts) == 1:
        # Just the package itself — can't resolve to a specific file
        return False

    # ``rsa.cli`` → ``rsa/cli.py``
    # ``rsa.sub.module`` → ``rsa/sub/module.py``
    file_path = "/".join(parts) + ".py"

    # Check if the path is actually a package (directory), not a module (file).
    # e.g. ``hone.utils`` → ``hone/utils/`` exists as a dir → use ``hone/utils/__init__.py``
    if source_dir is not None:
        # Resolve relative to source_dir's parent (since source_dir IS the package)
        sub_parts = parts[1:]  # strip the package name
        candidate_dir = source_dir / "/".join(sub_parts)
        if candidate_dir.is_dir():
            file_path = "/".join(parts) + "/__init__.py"

    refs.add(file_path)
    return True
