"""generate_architecture_dep tool — Enhanced version with dependencies.

Migrated from depgraphbase/tools/generate_architecture.py into parallelbase.

Uses architecture_prompt_dep which has the `dependencies` field integrated
directly into the schema definition and output example, instead of appending
a separate instruction block at the end.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from openhands.sdk.tool import ToolDefinition, ToolExecutor


# Reuse Action/Observation from codebase version
from codebase.tools.generate_architecture import (
    GenerateArchitectureAction,
    GenerateArchitectureObservation,
)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class GenerateArchitectureDepExecutor(ToolExecutor[GenerateArchitectureAction, GenerateArchitectureObservation]):
    """Run a sub-agent to generate RIB JSON with dependencies from requirement docs."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.workspace_dir = Path(runtime.workspace_dir)

    def __call__(
        self,
        action: GenerateArchitectureAction,
        conversation=None,
    ) -> GenerateArchitectureObservation:
        from common.prompts import architecture_prompt_dep

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

        pre_existing = sorted(
            str(p.relative_to(workspace))
            for p in workspace.rglob("*")
            if p.is_file()
        )

        from common.prompt_constants import load_implementation_integrity_rules
        prompt = architecture_prompt_dep(
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
            file_count = sum(len(m.get("files", [])) for m in data)
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
Generate a Repository Interface Blueprint (RIB) with file-level dependencies
from project requirement documents.

This enhanced version includes a `dependencies` field in each file entry,
listing internal files that the file depends on at runtime. This is used by
plan_dependency_waves to compute execution wave ordering.

Use this tool once at the beginning. After generation, use judge_architecture
to evaluate the result.
"""


class GenerateArchitectureDepTool(ToolDefinition[GenerateArchitectureAction, GenerateArchitectureObservation]):
    """Generate project architecture (RIB) with dependencies."""

    @classmethod
    def create(cls, conv_state, runtime=None) -> Sequence[ToolDefinition]:
        if runtime is None:
            raise ValueError("GenerateArchitectureDepTool requires a runtime instance")
        executor = GenerateArchitectureDepExecutor(runtime)
        return [cls(
            description=_DESCRIPTION,
            action_type=GenerateArchitectureAction,
            observation_type=GenerateArchitectureObservation,
            executor=executor,
        )]
