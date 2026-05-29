from __future__ import annotations

from pathlib import Path


def load_implementation_integrity_rules(workspace_dir: Path | str) -> str:
    """Read docs/implementation_integrity.md from the workspace and return it
    as a prompt section.  Returns empty string if the file does not exist."""
    p = Path(workspace_dir) / "docs" / "implementation_integrity.md"
    if not p.exists():
        return ""
    content = p.read_text(encoding="utf-8").strip()
    return (
        "\n## Project-Specific Implementation Integrity Rules\n\n"
        "The following rules are specific to this repository. "
        "You MUST obey them in addition to any general integrity rules.\n\n"
        f"{content}\n"
    )


APPLY_PATCH_FORMAT_REQUIREMENTS = """
- When using `apply_patch`, you MUST follow ALL rules below strictly:

  **General rules:**
  - The patch text MUST start with `*** Begin Patch` and end with `*** End Patch`.
  - Send raw patch text only; do NOT wrap it with keys like `patch:` or any extra prose.
  - `*** Begin Patch` / `*** End Patch` must be at column 1 (no leading indentation).
  - Use workspace-relative paths (never absolute paths) after `*** Add File:` / `*** Update File:` / `*** Delete File:`.
  - Do NOT use unified diff headers like `--- /dev/null` or `@@ -0,0 +1,N @@`.
  - The `@@` marker is for content-based context ONLY — NEVER use `@@ -N,M +N,M @@` line-number syntax anywhere.

  **Adding a new file (`*** Add File:`):**
  - Every content line MUST start with `+` (including blank lines, which should be just `+`).
  - Do NOT add `@@ ... @@` hunk headers for new files.
  - Example:
    ```
    *** Begin Patch
    *** Add File: path/to/new_file.py
    +\\"\\"\\"Module docstring.\\"\\"\\"
    +
    +import os
    +
    +
    +def hello():
    +    pass
    *** End Patch
    ```

  **Updating an existing file (`*** Update File:`):**
  - Use `@@ context_line` to locate the edit position. The text after `@@ ` is a **literal line from the file**, NOT line numbers.
  - CORRECT: `@@ def some_function():` (a real line from the file)
  - WRONG:   `@@ -1,12 +1,47 @@` (this is git diff syntax, NOT supported!)
  - Lines starting with ` ` (space) are unchanged context lines.
  - Lines starting with `-` are deleted lines.
  - Lines starting with `+` are inserted lines.
  - For multiple edits in ONE file, use multiple `@@ context_line` sections:
  - Example (single edit):
    ```
    *** Begin Patch
    *** Update File: path/to/existing.py
    @@ def some_function():
         existing_line
    -    old_line
    +    new_line
         another_existing_line
    *** End Patch
    ```
  - Example (multiple edits in one file):
    ```
    *** Begin Patch
    *** Update File: path/to/existing.py
    @@ def first_func():
    -    pass
    +    return 1
    @@ def second_func():
    -    pass
    +    return 2
    *** End Patch
    ```

  **Deleting a file (`*** Delete File:`):**
  - Simply specify the path; no content lines needed.
  - Example:
    ```
    *** Begin Patch
    *** Delete File: path/to/old_file.py
    *** End Patch
    ```

  **Common mistakes to avoid:**
  - WRONG: `@@ -0,0 +1,N @@` or `@@ -1,12 +1,47 @@` — this is git diff syntax, NOT supported anywhere.
  - WRONG: Forgetting `*** End Patch` at the end — every patch MUST end with `*** End Patch` on its own line.
  - WRONG: Forgetting `+` prefix on content lines in `*** Add File:`.
  - WRONG: Using absolute paths like `/home/user/project/file.py`.
  - WRONG: Sending an indented block like `patch:\\n    *** Begin Patch ...`.
""".strip()


FILE_EDITOR_USAGE_REQUIREMENTS = """
- When using `file_editor`, you MUST follow ALL rules below strictly:

  **General rules:**
  - Always use absolute paths for `file_editor`.
  - You may use `file_editor` to inspect related files for consistent imports/signatures.
  - For `command=view`, only read paths that are confirmed to exist.
  - If file existence is unknown, inspect the parent directory first (e.g., with `terminal ls`).

  **Creating missing files (`command=create`):**
  - When the workflow requires creating a missing target file, use `file_editor` (`command=create`).
  - Do NOT call `command=view` on a missing target file before creating it.
  - Ensure the parent directory exists before create.
  - `command=create` requires `file_text`; always provide it explicitly.
  - For intentionally empty files, set `file_text` to an empty string (`""`), not omitted.

  **String replacement (`command=str_replace`):**
  - `old_str` MUST uniquely match exactly one location in the current file.
  - If there are multiple matches, expand `old_str` with surrounding context (e.g., surrounding lines) until unique.
  - If there are zero matches, re-read the file and retry with the exact current content.
  - For append-style edits to the bottom of file, build `old_str` from the file tail (last 10-20 lines).
  - Ensure `old_str` contains distinctive text (e.g. code/JSON keys), not just whitespace or punctuation.

  **Common mistakes to avoid:**
  - WRONG: Calling `command=view` on a non-existent path (causes `Invalid path parameter`).
  - WRONG: Using an ambiguous `old_str` (causes "Multiple occurrences of old_str").
  - WRONG: Using `command=create` without `file_text` (causes "Parameter `file_text` is required").
  - WRONG: Using relative or guessed paths when absolute paths are required.
""".strip()


GREP_GLOB_USAGE_REQUIREMENTS = """
- When using `grep` or `glob`, you MUST follow ALL rules below strictly:

  **Path rules (critical):**
  - `path` MUST be an **absolute path to a directory**, not a file or relative path.
    - CORRECT: `path="/absolute/path/to/workspace"`
    - WRONG:  `path="bplustree"` (relative → error)
    - WRONG:  `path="/path/to/file.py"` (file → error)
  - Use the **workspace root** as base. The path is shown as `[Current working directory: ...]`.

  **`glob` pattern MUST start with `**` (critical — without `**` silently returns nothing):**
  - CORRECT: `**/*.py`, `**/hone/*.py`, `**/examples/**`
  - WRONG:  `hone/*.py`, `examples/*` — returns zero results even when files exist!

  **`grep` pattern is regex:**
  - Escape special characters: `(`, `)`, `[`, `.`, `*`, `+`, `?`, `|`, etc.
    - CORRECT: `pattern="Record\\\\("` for literal `Record(`
    - WRONG:  `pattern="Record("` → regex error

  **`grep` `include` MUST start with `**` (critical — without `**` silently returns nothing):**
  - CORRECT: `include="**/*.py"`, `include="**/check_tests/*.py"`, `include="**/geotext/data_file/*"`
  - WRONG:  `include="*.py"`, `include="check_tests/*.py"`, `include="geotext/data_file/*"` — may return zero results!
""".strip()


TERMINAL_USAGE_REQUIREMENTS = """
- When using `terminal`, you MUST follow ALL rules below strictly:

  **Timeout parameter (critical — unit is SECONDS, not milliseconds):**
  - The `timeout` parameter specifies the maximum time in **seconds**.
    - CORRECT: `timeout=30` for a 30-second timeout
    - CORRECT: `timeout=120` for a 2-minute timeout
    - WRONG:  `timeout=120000` — this is 120,000 seconds (33 hours), NOT 120 seconds!
  - Set `timeout` proportional to the expected command duration:
    - Simple scripts, `ls`, `echo`, compile checks: `timeout=10` to `timeout=30`
    - Test runs, installations: `timeout=120` to `timeout=300`
    - Never exceed `timeout=600` unless you have a specific reason.
  - If you omit `timeout`, the default soft-timeout behavior applies (pauses after 10 seconds of no output).

  **Common mistakes to avoid:**
  - WRONG: Setting `timeout=120000` thinking it is milliseconds — it is seconds!
  - WRONG: Using very large timeouts for simple diagnostic scripts.
""".strip()


INIT_PY_EXPORT_REQUIREMENTS = """
- **`__init__.py` re-export rule (CRITICAL)**: If the target file is a package
  `__init__.py`:
  1. **ONLY** write final package-level re-export imports. You may also write
     `__all__` or `__version__` when required. Do NOT implement classes, functions,
     algorithms, placeholders, or fake stubs in `__init__.py`.
  2. Use **explicit name imports** like `from .submodule import ClassName`.
     Do **NOT** use `from . import submodule` alone — that only exposes the module
     object, not the class names, and **WILL cause ImportError** in downstream code
     that does `from pkg import ClassName`.
     Both import styles MUST work after your `__init__.py` is written:
       - `from pkg import ClassName`  (package-level shortcut)
       - `from pkg.submodule import ClassName`  (fully qualified)
  3. **Determine exported symbols from the RIB** (CRITICAL): Read the sibling file
     entries in the RIB and use the **exact class/function names defined there**,
     including exact casing and spelling. The RIB is the authoritative source for
     symbol names. Do NOT guess, infer, or invent names — copy them verbatim from
     the RIB.
  4. Do NOT wrap imports in `try/except ImportError` — it silently hides missing
     exports and causes test failures.
""".strip()



IMPORT_SOURCE_REQUIREMENTS = """
- **Import source rule (CRITICAL)**: When importing a symbol (class, function,
  constant) from another module in the same project, you MUST import it from the
  file where it is **defined** in the RIB, not from a file that merely re-exports
  or uses it. This applies to both runtime imports and `TYPE_CHECKING` imports.
  For example, if the RIB defines `TreeConf` in `const.py`, write
  `from .const import TreeConf`, NOT `from .tree import TreeConf` — even if
  `tree.py` also imports `TreeConf`. Importing from the wrong module causes
  circular imports and violates the architectural dependency structure.
""".strip()

SHARED_TASK_LIST_USAGE_REQUIREMENTS = """
- When using `shared_task_list`, you MUST follow ALL rules below strictly:

  **CRITICAL: Do NOT confuse `shared_task_list` with `task_tracker`.**
  - `task_tracker` uses: `task_list=[{"title": ..., "notes": ..., "status": ...}]`  ← WRONG for shared_task_list
  - `shared_task_list` uses: `tasks=[{"id": ..., "deps": [...], ...}]`              ← CORRECT

  **`init` — Create the task list:**
  - Parameter: `tasks` (list of dicts, each MUST have `id` and `deps`)
  - `id`: a unique task identifier (e.g. a file path)
  - `deps`: list of other task `id`s this task depends on (use `[]` if none)
  - `owner` (optional): name of the agent responsible for this task
  - `description` (optional): short summary of the task
  - Any other fields are stored in `metadata`
  - Example:
    ```
    shared_task_list(command="init", tasks=[
        {"id": "pkg/const.py",  "deps": [],                               "owner": "group_hub"},
        {"id": "pkg/node.py",   "deps": ["pkg/const.py"],                 "owner": "group_entry"},
        {"id": "pkg/tree.py",   "deps": ["pkg/const.py", "pkg/node.py"],  "owner": "group_integration"},
    ])
    ```

  **`claim` — Claim a task before working on it:**
  - Parameter: `task_id` (the task's `id` string)
  - Returns `status="ready"` if newly claimed, `status="in_progress"` if already yours,
    `status="waiting"` with `blocked_by=[...]` if deps not met, `status="completed"` if done
  - Example: `shared_task_list(command="claim", task_id="pkg/node.py")`

  **`complete` — Mark a task done after finishing it:**
  - Parameter: `task_id`
  - Returns `newly_ready=[{task_id, owner, description}]` — other tasks just unblocked
  - Peer agents are automatically notified by the system — you do NOT need to
    call `send_to_agent` for newly_ready tasks.
  - Example: `shared_task_list(command="complete", task_id="pkg/node.py")`

  **`status` — Check overall progress:**
  - No extra parameter → full summary: `{total, completed, in_progress, ready, blocked, all_done}`
  - Example: `shared_task_list(command="status")`

  **Common mistakes to avoid:**
  - WRONG: `tasks=[{"title": "pkg/node.py", "notes": ..., "status": "todo"}]`  ← task_tracker schema!
  - WRONG: calling `claim` without `task_id`
  - WRONG: calling `complete` before `claim`
""".strip()
