"""read_ssat tool — deterministic SSAT lookup (no LLM, no sub-agent).

Given a target file path, reads architecture/ssat.json from disk and returns:
  1. The full SSAT entry for the target file (classes, functions, parameters).
  2. A lightweight summary of all other files (path + description only).

This guarantees group agents always receive the exact architecture spec
without relying on the Leader to copy the SSAT into the initial message.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor

from common.utils.ssat_helpers import load_architecture, find_file_item, architecture_summary


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class ReadSSATAction(Action):
    target_file: str = Field(
        description="Source file to look up (workspace-relative, e.g. 'cookiecutter/prompt.py')",
    )
    architecture_path: str = Field(
        default="architecture/ssat.json",
        description="Path to SSAT JSON file (relative to workspace)",
    )


class ReadSSATObservation(Observation):
    target_file: str = ""
    status: str = "unknown"
    file_spec: str = ""        # full SSAT JSON for target file
    other_files: str = ""      # lightweight summary of sibling files
    error: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"read_ssat FAILED for {self.target_file}: {self.error}")]
        return [TextContent(
            text=(
                f"## SSAT spec for `{self.target_file}`\n\n"
                f"Implement exactly what is listed below. Follow the SSAT Compliance Rules in your system prompt.\n\n"
                f"```json\n{self.file_spec}\n```\n\n"
                f"## Other files in the project (for import/context reference)\n\n"
                f"```json\n{self.other_files}\n```"
            ),
        )]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ReadSSATExecutor(ToolExecutor[ReadSSATAction, ReadSSATObservation]):
    """Pure file read — no LLM, no sub-agent."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    def __call__(
        self,
        action: ReadSSATAction,
        conversation=None,
    ) -> ReadSSATObservation:
        arch_path = self.workspace_dir / action.architecture_path
        try:
            arch_data = load_architecture(arch_path)
        except (FileNotFoundError, Exception) as e:
            return ReadSSATObservation(
                target_file=action.target_file,
                error=str(e),
            )

        file_item = find_file_item(arch_data, action.target_file)
        if file_item is None:
            return ReadSSATObservation(
                target_file=action.target_file,
                error=f"File '{action.target_file}' not found in SSAT",
            )

        summary = architecture_summary(arch_data, exclude=action.target_file)

        return ReadSSATObservation(
            target_file=action.target_file,
            status="ok",
            file_spec=json.dumps(file_item, ensure_ascii=False, indent=2),
            other_files=json.dumps(summary, ensure_ascii=False, indent=2),
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Look up the SSAT architecture spec for a single source file.

Returns the exact interface specification (function names, class names,
parameters, descriptions) that you MUST implement, plus a summary of
other project files for import/context reference.

Call this tool BEFORE writing code for each file.

Parameters:
- target_file: Which source file to look up (e.g. 'cookiecutter/prompt.py')
- architecture_path: Path to SSAT JSON (default: 'architecture/ssat.json')
"""


class ReadSSATTool(ToolDefinition[ReadSSATAction, ReadSSATObservation]):
    """Look up SSAT spec for one file — deterministic, no LLM."""

    @classmethod
    def create(cls, conv_state, workspace_dir: Path | None = None) -> Sequence[ToolDefinition]:
        if workspace_dir is None:
            raise ValueError("ReadSSATTool requires workspace_dir")
        executor = ReadSSATExecutor(workspace_dir)
        return [cls(
            description=_DESCRIPTION,
            action_type=ReadSSATAction,
            observation_type=ReadSSATObservation,
            executor=executor,
        )]
