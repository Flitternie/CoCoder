"""compile_check tool — Execution (direct command, no LLM).

Runs py_compile on all Python files in the workspace to check syntax.
"""
from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class CompileCheckAction(Action):
    model_config = {"extra": "ignore"}  # LLM may add extra fields like security_risk, summary


class CompileCheckObservation(Observation):
    passed: bool = False
    errors: list[str] = []
    file_count: int = 0

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.passed:
            return [TextContent(text=f"Compile check PASSED: {self.file_count} files OK")]
        err_text = "\n".join(f"  - {e}" for e in self.errors[:10])
        more = f"\n  ... and {len(self.errors) - 10} more" if len(self.errors) > 10 else ""
        return [TextContent(
            text=f"Compile check FAILED ({len(self.errors)} errors):\n{err_text}{more}"
        )]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class CompileCheckExecutor(ToolExecutor[CompileCheckAction, CompileCheckObservation]):
    """Run py_compile on all .py files."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    def __call__(
        self,
        action: CompileCheckAction,
        conversation=None,
    ) -> CompileCheckObservation:
        workspace = self.workspace_dir
        if not workspace.is_dir():
            return CompileCheckObservation(errors=[f"Directory not found: {workspace}"])

        # Find all .py files, excluding test/fixture dirs that may contain
        # non-Python files (e.g. Jinja templates) which would fail py_compile.
        _EXCLUDE_DIRS = {"__pycache__", "check_tests", "unit_tests", "tests"}
        py_files = sorted(
            p for p in workspace.rglob("*.py")
            if not (set(p.relative_to(workspace).parts) & _EXCLUDE_DIRS)
        )

        if not py_files:
            return CompileCheckObservation(passed=True, file_count=0)

        errors: list[str] = []
        for py_file in py_files:
            try:
                result = subprocess.run(
                    ["python", "-m", "py_compile", str(py_file)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    err = result.stderr.strip() or result.stdout.strip()
                    rel = str(py_file.relative_to(workspace))
                    errors.append(f"{rel}: {err}")
            except subprocess.TimeoutExpired:
                rel = str(py_file.relative_to(workspace))
                errors.append(f"{rel}: compile check timed out")
            except Exception as e:
                rel = str(py_file.relative_to(workspace))
                errors.append(f"{rel}: {e}")

        return CompileCheckObservation(
            passed=len(errors) == 0,
            errors=errors,
            file_count=len(py_files),
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Check syntax of all Python files in the workspace.

Runs py_compile on every .py file (excluding check_tests/) and reports
any syntax errors. This is a fast validation step — use it after generating
skeletons or code to quickly catch syntax issues before running full tests.

Returns passed/failed status and a list of errors.
"""


class CompileCheckTool(ToolDefinition[CompileCheckAction, CompileCheckObservation]):
    """Check Python syntax for all workspace files."""

    @classmethod
    def create(cls, conv_state, workspace_dir=None) -> Sequence[ToolDefinition]:
        if workspace_dir is None:
            raise ValueError("CompileCheckTool requires workspace_dir")
        executor = CompileCheckExecutor(Path(workspace_dir))
        return [cls(
            description=_DESCRIPTION,
            action_type=CompileCheckAction,
            observation_type=CompileCheckObservation,
            executor=executor,
        )]
