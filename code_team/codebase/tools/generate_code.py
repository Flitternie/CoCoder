"""generate_code tool — Generative (sub-agent).

Implements full code for a single file based on its skeleton and context.
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

class GenerateCodeAction(Action):
    architecture_path: str = Field(description="Path to RIB JSON file (relative to workspace)")
    target_file: str = Field(description="Source file to implement (workspace-relative)")


class GenerateCodeObservation(Observation):
    path: str = ""  # Absolute path on disk
    status: str = "unknown"
    code_preview: str = ""
    error: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"generate_code FAILED for {self.path}: {self.error}")]
        preview = self.code_preview[:500] if self.code_preview else "(empty)"
        return [TextContent(
            text=f"Code generated: {self.path}\n```\n{preview}\n```"
        )]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class GenerateCodeExecutor(ToolExecutor[GenerateCodeAction, GenerateCodeObservation]):
    """Run a sub-agent to implement full code for one file."""

    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(
        self,
        action: GenerateCodeAction,
        conversation=None,
    ) -> GenerateCodeObservation:
        from common.prompts import code_generation_prompt

        workspace = Path(self.runtime.workspace_dir)
        target_abs = workspace / action.target_file

        # Load architecture
        try:
            arch_data = load_architecture(workspace / action.architecture_path)
        except (FileNotFoundError, Exception) as e:
            return GenerateCodeObservation(
                path=str(target_abs.resolve()), error=str(e),
            )

        # Find target file item
        file_item = find_file_item(arch_data, action.target_file)
        if file_item is None:
            return GenerateCodeObservation(
                path=str(target_abs.resolve()),
                error=f"File '{action.target_file}' not found in RIB",
            )

        # Attach current skeleton/code to file_item
        if target_abs.exists():
            file_item = dict(file_item)
            file_item["code"] = target_abs.read_text(encoding="utf-8", errors="replace")

        from common.prompt_constants import load_implementation_integrity_rules
        prompt = code_generation_prompt(
            file_item=file_item,
            architecture_summary=architecture_summary(arch_data, exclude=action.target_file),
            integrity_rules=load_implementation_integrity_rules(workspace),
        )

        try:
            self.runtime.run_agent("code", prompt)
        except Exception as e:
            return GenerateCodeObservation(
                path=str(target_abs.resolve()),
                status="failed",
                error=f"Sub-agent crashed: {e}",
            )

        # Read implemented file
        if target_abs.exists():
            code = target_abs.read_text(encoding="utf-8", errors="replace")
            preview = "\n".join(code.splitlines()[:50])
            return GenerateCodeObservation(
                path=str(target_abs.resolve()),
                status="success",
                code_preview=preview,
            )
        return GenerateCodeObservation(
            path=str(target_abs.resolve()),
            status="failed",
            error="Sub-agent did not produce the code file",
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Implement full code for a single source file based on its skeleton.

The tool reads the file's skeleton code and RIB description, and runs a
sub-agent that generates a complete implementation using workspace tools.

Parameters:
- architecture_path: Relative path to the RIB JSON file (e.g. 'architecture/rib.json')
- target_file: Which source file to implement (e.g. 'src/utils/io.py')

Call this tool once per source file after skeletons are generated.
After generating all code, use run_tests to validate. If tests fail,
analyze the errors yourself and fix code with apply_patch, then re-run tests.
"""


class GenerateCodeTool(ToolDefinition[GenerateCodeAction, GenerateCodeObservation]):
    """Implement full code for one file from skeleton."""

    @classmethod
    def create(cls, conv_state, runtime=None) -> Sequence[ToolDefinition]:
        if runtime is None:
            raise ValueError("GenerateCodeTool requires a runtime instance")
        executor = GenerateCodeExecutor(runtime)
        return [cls(
            description=_DESCRIPTION,
            action_type=GenerateCodeAction,
            observation_type=GenerateCodeObservation,
            executor=executor,
        )]
