"""generate_architecture tool — Generative (sub-agent).

Creates a RIB (Repository Interface Blueprint) JSON from requirement docs.
The executor reads docs from workspace paths, builds the architecture prompt,
and runs a sub-agent to generate the RIB file.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class GenerateArchitectureAction(Action):
    prd_path: str = Field(description="Path to PRD document (relative to workspace)")
    uml_class_path: str = Field(description="Path to UML class diagram")
    uml_sequence_path: str = Field(description="Path to UML sequence diagram")
    arch_design_path: str = Field(description="Path to architecture design document")
    output_path: str = Field(description="Path where RIB JSON will be written")


class GenerateArchitectureObservation(Observation):
    path: str = ""  # Absolute path on disk
    status: str = "unknown"
    module_count: int = 0
    file_count: int = 0
    error: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"generate_architecture FAILED: {self.error}")]
        return [TextContent(
            text=(
                f"Architecture generated successfully.\n"
                f"  Output: {self.path}\n"
                f"  Modules: {self.module_count}\n"
                f"  Files: {self.file_count}"
            )
        )]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class GenerateArchitectureExecutor(ToolExecutor[GenerateArchitectureAction, GenerateArchitectureObservation]):
    """Run a sub-agent to generate RIB JSON from requirement docs."""

    def __init__(self, runtime):
        """
        Args:
            runtime: SerialRuntime instance for sub-agent execution.
        """
        self.runtime = runtime
        self.workspace_dir = Path(runtime.workspace_dir)

    def __call__(
        self,
        action: GenerateArchitectureAction,
        conversation=None,
    ) -> GenerateArchitectureObservation:
        from common.prompts import architecture_prompt

        workspace = Path(self.runtime.workspace_dir)
        try:
            def _read(p: str) -> str:
                return (workspace / p).read_text(encoding="utf-8") if p else ""

            prd = _read(action.prd_path)
            uml_class = _read(action.uml_class_path)
            uml_seq = _read(action.uml_sequence_path)
            arch_design = _read(action.arch_design_path)
        except FileNotFoundError as e:
            return GenerateArchitectureObservation(error=f"File not found: {e}")

        output_abs = workspace / action.output_path
        output_abs.parent.mkdir(parents=True, exist_ok=True)

        # Scan workspace for pre-existing files so the LLM excludes them
        # from the RIB (they are data/resource files, not source to generate).
        pre_existing = sorted(
            str(p.relative_to(workspace))
            for p in workspace.rglob("*")
            if p.is_file()
        )

        from common.prompt_constants import load_implementation_integrity_rules
        prompt = architecture_prompt(
            requirement=prd,
            uml_class=uml_class,
            uml_sequence=uml_seq,
            arch_design=arch_design,
            output_json_path=str(output_abs),
            pre_existing_files=pre_existing,
            integrity_rules=load_implementation_integrity_rules(workspace),
        )

        try:
            self.runtime.run_agent("architecture", prompt, timeout_sec=600)
        except Exception as e:
            return GenerateArchitectureObservation(
                path=str(output_abs),
                status="failed",
                error=f"Sub-agent crashed: {e}",
            )

        # Parse generated RIB
        if not output_abs.exists():
            return GenerateArchitectureObservation(
                path=str(output_abs),
                status="failed",
                error="Sub-agent did not produce RIB file",
            )
        try:
            data = json.loads(output_abs.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = [data]
            module_count = len(data)
            file_count = sum(
                len(m.get("files", []))
                for m in data
            )
        except (json.JSONDecodeError, TypeError) as e:
            return GenerateArchitectureObservation(
                path=str(output_abs),
                status="failed",
                error=f"RIB JSON parse error: {e}",
            )

        return GenerateArchitectureObservation(
            path=str(output_abs),
            status="success",
            module_count=module_count,
            file_count=file_count,
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Generate a Repository Interface Blueprint (RIB) from project requirement documents.

This tool reads the PRD, UML class diagram, UML sequence diagram, and architecture
design document, then generates a structured RIB JSON file describing the project's
module/file/class/function hierarchy.

Use this tool once at the beginning of a project to establish the architecture.
After generation, use judge_architecture to evaluate the result.

If the architecture needs improvement, edit the RIB JSON directly using file_editor
based on judge feedback, then re-evaluate with judge_architecture.
"""


class GenerateArchitectureTool(ToolDefinition[GenerateArchitectureAction, GenerateArchitectureObservation]):
    """Generate project architecture (RIB) from requirement documents."""

    @classmethod
    def create(cls, conv_state, runtime=None) -> Sequence[ToolDefinition]:
        if runtime is None:
            raise ValueError("GenerateArchitectureTool requires a runtime instance")
        executor = GenerateArchitectureExecutor(runtime)
        return [cls(
            description=_DESCRIPTION,
            action_type=GenerateArchitectureAction,
            observation_type=GenerateArchitectureObservation,
            executor=executor,
        )]
