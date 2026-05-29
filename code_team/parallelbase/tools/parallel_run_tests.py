"""run_tests tool — parallel-pipeline variant.

Subclasses the codebase run_tests tool with parallel-specific observation
messages and agent-specific output file naming to avoid collisions.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from openhands.sdk import TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition

from codebase.tools.run_tests import (
    RunTestsAction,
    RunTestsObservation,
    RunTestsExecutor,
    _parse_pytest_counts,
)


# ---------------------------------------------------------------------------
# Action / Observation — distinct class names to avoid SDK duplicate-class errors
# ---------------------------------------------------------------------------

_LIMIT_REACHED_MSG = (
    "run_tests limit reached. "
    "You MUST dismiss all agents and call `finish` immediately. "
    "Do NOT dispatch more fixes or call run_tests again."
)


class ParallelRunTestsAction(RunTestsAction):
    """Same schema as RunTestsAction, distinct class for the SDK registry."""
    model_config = {"extra": "ignore"}  # LLM may add unexpected fields like 'timeout'


class ParallelRunTestsObservation(RunTestsObservation):
    """Parallel-pipeline variant with next-step guidance."""

    report_path: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"run_tests ERROR: {self.error}")]
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Tests {status}: {self.passed_count}/{self.total} passed",
        ]
        if self.passed and self.total > 0:
            lines.append(
                "\n✅ ALL TESTS PASSED. Call `finish` now."
            )
        elif self.limit_reached:
            lines.append(
                f"\nTest output: {self.output_path}"
                f"\n\n{_LIMIT_REACHED_MSG}"
            )
        else:
            lines.append(
                f"\nTest output: {self.output_path}"
                "\n\n⚠️ NEXT STEPS (you MUST call tools — do NOT output plain text):\n"
                "1. Call `file_editor` to read the test output above for failure details (test nodeids, crash messages, line numbers)\n"
                "2. Analyze failures and fix the code\n"
                "3. Call `compile_check`, then `run_tests` again"
            )
        text = "\n".join(lines)
        return [TextContent(text=text)]


# ---------------------------------------------------------------------------
# Executor — reuse the base executor, just map to our Observation type
# ---------------------------------------------------------------------------

class ParallelRunTestsExecutor(RunTestsExecutor):
    """Same execution logic, returns our Observation subclass.

    Overrides output file naming to include thread ID and timestamp,
    preventing concurrent agents from overwriting each other's test
    output files in the shared test_outputs/ directory.
    """

    def __call__(self, action, conversation=None):
        if self._call_count >= self.max_runs:
            return ParallelRunTestsObservation(error=_LIMIT_REACHED_MSG)

        workspace = self.workspace_dir
        if not workspace.is_dir():
            return ParallelRunTestsObservation(
                error=f"Workspace not found: {workspace}"
            )

        agent_name = getattr(conversation, '_agent_name', 'unknown') if conversation else 'unknown'
        run_id = self._call_count + 1

        # Prepare versioned output directories
        reports_dir = self.output_dir / "reports"
        raw_dir = self.output_dir / "raw_outputs"
        reports_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        report_file = reports_dir / f"check_test_report_{agent_name}_{run_id}.json"

        # Build command — rewrite or append --json-report-file to versioned path
        cmd = action.command
        if "--json-report-file" in cmd:
            cmd = re.sub(
                r"--json-report-file=\S+",
                f"--json-report-file={report_file}",
                cmd,
            )
        else:
            cmd += f" --json-report --json-report-file={report_file}"
        if "--continue-on-collection-errors" not in cmd:
            cmd += " --continue-on-collection-errors"
        cmd = f"PYTHONPATH=. PYTHONWARNINGS=ignore::DeprecationWarning {cmd}"

        # Execute
        import subprocess
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(workspace),
                capture_output=True, text=True, timeout=60,
            )
            output = result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            return ParallelRunTestsObservation(
                error="Test command timed out (60s)"
            )
        except Exception as e:
            return ParallelRunTestsObservation(
                error=f"Execution error: {e}"
            )

        # Count only after successful execution
        self._call_count += 1

        # Save raw output
        raw_file = raw_dir / f"test_output_{agent_name}_{run_id}.txt"
        raw_file.write_text(output, encoding="utf-8")

        # Parse results
        all_passed, passed_count, total = _parse_pytest_counts(output)

        return ParallelRunTestsObservation(
            passed=all_passed and total > 0,
            passed_count=passed_count,
            total=total,
            output_path=str(raw_file),
            report_path=str(report_file),
            limit_reached=self._call_count >= self.max_runs,
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Run a test command and get structured results.

Executes the given test command in the workspace directory and parses the output
for pass/fail counts. The raw test output is saved for failure analysis.

Parameters:
- command: Test command, e.g. 'python -m pytest check_tests -v -s'

Returns pass/fail status and counts. If tests fail, read the test output file
(path returned in the result) with file_editor for failure details.
"""


class ParallelRunTestsTool(ToolDefinition[ParallelRunTestsAction, ParallelRunTestsObservation]):
    """Run tests and parse results (no finish hint)."""

    @classmethod
    def create(cls, conv_state, workspace_dir=None, output_dir=None, max_runs: int = 10) -> Sequence[ToolDefinition]:
        if workspace_dir is None:
            raise ValueError("RunTestsTool requires workspace_dir")
        if output_dir is None:
            output_dir = Path("./test_outputs")
        executor = ParallelRunTestsExecutor(Path(workspace_dir), Path(output_dir), max_runs=max_runs)
        return [cls(
            description=_DESCRIPTION,
            action_type=ParallelRunTestsAction,
            observation_type=ParallelRunTestsObservation,
            executor=executor,
        )]
