"""run_tests tool — Execution (direct command, no LLM).

Runs a test command in the workspace and parses pytest output.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class RunTestsAction(Action):
    model_config = {"extra": "ignore"}  # LLM may add extra fields like timeout, security_risk
    command: str = Field(description="Test command to execute, e.g. 'python -m pytest check_tests -v'")
    setup_script: str = Field(default="", description="Optional setup script to run first, e.g. 'setup.sh'")


_LIMIT_REACHED_MSG = (
    "run_tests limit reached. "
    "You MUST call `finish` immediately to report your current status. "
    "Do NOT attempt more fixes or call run_tests again."
)


class RunTestsObservation(Observation):
    passed: bool = False
    passed_count: int = 0
    total: int = 0
    output_path: str = ""
    summary: str = ""
    error: str = ""
    limit_reached: bool = False

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"run_tests ERROR: {self.error}")]
        status = "PASSED" if self.passed else "FAILED"
        text = (
            f"Tests {status}: {self.passed_count}/{self.total} passed\n"
            f"Full output saved to: {self.output_path}\n"
            f"Read the output file with file_editor to see failure details."
        )
        if self.passed and self.total > 0:
            text += (
                "\n\n⚠️ ALL TESTS PASSED. Your task is COMPLETE. "
                "Call `finish` RIGHT NOW. Do NOT call any other tool. "
                "Do NOT regenerate architecture, skeleton, or code. Just call `finish`."
            )
        elif self.limit_reached:
            text += f"\n\n{_LIMIT_REACHED_MSG}"
        else:
            text += (
                "\n\n📌 REMINDER: When using `terminal`, the `timeout` parameter is in SECONDS. "
                "Use timeout=10~30 for simple commands, timeout=120~300 for test runs. "
                "NEVER use timeout>600. Setting timeout=120000 means 33 HOURS and will hang the run."
            )
        return [TextContent(text=text)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pytest_counts(output: str) -> tuple[bool, int, int]:
    """Parse pytest output for pass/fail counts.

    Returns (all_passed, passed_count, total_count).
    Skipped tests are counted in total and treated as failures.
    """
    # Extract each count independently — pytest can emit them in any order:
    # "1 failed, 8 passed, 2 skipped in 0.08s" etc.
    passed = 0
    failed = 0
    errors = 0
    skipped = 0

    m = re.search(r"(\d+)\s+passed", output)
    if m:
        passed = int(m.group(1))

    m = re.search(r"(\d+)\s+failed", output)
    if m:
        failed = int(m.group(1))

    m = re.search(r"(\d+)\s+error", output)
    if m:
        errors = int(m.group(1))

    m = re.search(r"(\d+)\s+skipped", output)
    if m:
        skipped = int(m.group(1))

    total = passed + failed + errors + skipped
    if total > 0:
        return (failed == 0 and errors == 0 and skipped == 0), passed, total

    # Fallback: count PASSED/FAILED marker lines
    passed = len(re.findall(r"\bPASSED\b", output))
    failed = len(re.findall(r"\bFAILED\b", output))
    total = passed + failed
    if total > 0:
        return failed == 0, passed, total

    return False, 0, 0


def _extract_summary(output: str, max_lines: int = 30) -> str:
    """Extract the last N lines as a summary."""
    lines = output.strip().splitlines()
    if len(lines) <= max_lines:
        return output.strip()
    return "\n".join(["...(truncated)..."] + lines[-max_lines:])


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class RunTestsExecutor(ToolExecutor[RunTestsAction, RunTestsObservation]):
    """Execute test command and parse results."""

    def __init__(self, workspace_dir: Path, output_dir: Path, max_runs: int = 10):
        self.workspace_dir = workspace_dir
        self.output_dir = output_dir
        self.max_runs = max_runs
        self._call_count = 0

    def __call__(
        self,
        action: RunTestsAction,
        conversation=None,
    ) -> RunTestsObservation:
        if self._call_count >= self.max_runs:
            return RunTestsObservation(error=_LIMIT_REACHED_MSG)

        workspace = self.workspace_dir
        if not workspace.is_dir():
            return RunTestsObservation(error=f"Workspace not found: {workspace}")

        # Build command
        cmd = action.command
        if "--continue-on-collection-errors" not in cmd:
            cmd += " --continue-on-collection-errors"
        cmd = f"PYTHONPATH=. {cmd}"

        if action.setup_script:
            script_path = workspace / action.setup_script
            if script_path.exists():
                cmd = f"sh {action.setup_script} && {cmd}"

        # Execute
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            return RunTestsObservation(error="Test command timed out (60s)")
        except Exception as e:
            return RunTestsObservation(error=f"Execution error: {e}")

        # Save full output
        self._call_count += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / f"test_output_{self._call_count}.txt"
        output_file.write_text(output, encoding="utf-8")

        # Parse results
        all_passed, passed_count, total = _parse_pytest_counts(output)

        return RunTestsObservation(
            passed=all_passed and total > 0,
            passed_count=passed_count,
            total=total,
            output_path=str(output_file),
            limit_reached=self._call_count >= self.max_runs,
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Run a test command and get structured results.

Executes the given test command in the workspace directory and parses the output
for pass/fail counts. The full test output is saved to a file for later review.

Parameters:
- command: Test command, e.g. 'python -m pytest check_tests -v -s'
- setup_script: Optional setup script to run first (e.g. 'setup.sh')

Returns pass/fail status, counts, and a summary. The full output is saved
to a file whose path is returned in output_path. If you need to see the
full error details, read that file with file_editor.
"""


class RunTestsTool(ToolDefinition[RunTestsAction, RunTestsObservation]):
    """Run tests and parse results."""

    @classmethod
    def create(cls, conv_state, workspace_dir=None, output_dir=None, max_runs: int = 10) -> Sequence[ToolDefinition]:
        if workspace_dir is None:
            raise ValueError("RunTestsTool requires workspace_dir")
        if output_dir is None:
            output_dir = Path("./test_outputs")
        executor = RunTestsExecutor(Path(workspace_dir), Path(output_dir), max_runs=max_runs)
        return [cls(
            description=_DESCRIPTION,
            action_type=RunTestsAction,
            observation_type=RunTestsObservation,
            executor=executor,
        )]
