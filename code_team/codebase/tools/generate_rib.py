"""generate_rib tool — Generative (sub-agent).

Generates skeleton code for a single file based on RIB architecture.
The executor reads the RIB JSON, extracts the target file's description,
and runs a sub-agent that uses tools to create the skeleton.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor

from common.utils.rib_helpers import load_architecture, find_file_item, architecture_summary


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class GenerateRIBAction(Action):
    architecture_path: str = Field(description="Path to RIB JSON file (relative to workspace)")
    target_file: str = Field(description="Target source file to generate skeleton for, e.g. 'src/utils/io.py'")


class GenerateRIBObservation(Observation):
    path: str = ""  # Absolute path on disk
    status: str = "unknown"
    code_preview: str = ""
    error: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"generate_rib FAILED for {self.path}: {self.error}")]
        preview = self.code_preview[:500] if self.code_preview else "(empty)"
        return [TextContent(
            text=f"Skeleton generated: {self.path}\n```\n{preview}\n```"
        )]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class GenerateRIBExecutor(ToolExecutor[GenerateRIBAction, GenerateRIBObservation]):
    """Run a sub-agent to generate skeleton code for one file."""

    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(
        self,
        action: GenerateRIBAction,
        conversation=None,
    ) -> GenerateRIBObservation:
        from common.prompts import rib_prompt

        workspace = Path(self.runtime.workspace_dir)
        target_abs = workspace / action.target_file

        # Defensive: skip if the file already exists in workspace (pre-existing
        # data/resource file copied by workspace_builder).
        if target_abs.exists():
            return GenerateRIBObservation(
                path=str(target_abs.resolve()),
                status="skipped",
                code_preview="# File already exists in workspace, skipped to avoid overwriting",
            )

        # Load architecture
        try:
            arch_data = load_architecture(workspace / action.architecture_path)
        except (FileNotFoundError, Exception) as e:
            return GenerateRIBObservation(
                path=str(target_abs.resolve()), error=str(e),
            )

        # Find target file in architecture
        file_item = find_file_item(arch_data, action.target_file)
        if file_item is None:
            return GenerateRIBObservation(
                path=str(target_abs.resolve()),
                error=f"File '{action.target_file}' not found in RIB",
            )

        from common.prompt_constants import load_implementation_integrity_rules
        prompt = rib_prompt(
            file_item=file_item,
            architecture_summary=architecture_summary(arch_data, exclude=action.target_file),
            integrity_rules=load_implementation_integrity_rules(workspace),
        )

        try:
            self.runtime.run_agent("rib", prompt)
        except Exception as e:
            return GenerateRIBObservation(
                path=str(target_abs.resolve()),
                status="failed",
                error=f"Sub-agent crashed: {e}",
            )

        # Read generated file
        if target_abs.exists():
            code = target_abs.read_text(encoding="utf-8", errors="replace")
            preview = "\n".join(code.splitlines()[:50])
            return GenerateRIBObservation(
                path=str(target_abs.resolve()),
                status="success",
                code_preview=preview,
            )
        return GenerateRIBObservation(
            path=str(target_abs.resolve()),
            status="failed",
            error="Sub-agent did not produce the skeleton file",
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Generate skeleton code for a single source file based on the RIB architecture.

The skeleton contains class/function signatures, docstrings, and structural
elements (imports, constants) but no implementation (bodies are `pass`).

Parameters:
- architecture_path: Relative path to the RIB JSON file (e.g. 'architecture/rib.json')
- target_file: Which source file to generate (e.g. 'src/utils/io.py')

Call this tool once per source file. Use judge_rib after generating all
skeletons to evaluate quality. If the skeleton needs improvement, edit it
directly with file_editor based on judge feedback.
"""


class GenerateRIBTool(ToolDefinition[GenerateRIBAction, GenerateRIBObservation]):
    """Generate skeleton code for one file from RIB architecture."""

    @classmethod
    def create(cls, conv_state, runtime=None) -> Sequence[ToolDefinition]:
        if runtime is None:
            raise ValueError("GenerateRIBTool requires a runtime instance")
        executor = GenerateRIBExecutor(runtime)
        return [cls(
            description=_DESCRIPTION,
            action_type=GenerateRIBAction,
            observation_type=GenerateRIBObservation,
            executor=executor,
        )]
