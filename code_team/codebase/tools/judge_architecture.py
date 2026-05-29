"""judge_architecture tool — Judge (single LLM call).

Evaluates a RIB architecture against project requirements.
Returns a structured score and per-dimension feedback.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from openhands.sdk import LLM, Action, Observation, TextContent, ImageContent, Message
from openhands.sdk.tool import ToolDefinition, ToolExecutor

from common.utils.parsers import parse_judge_score


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class JudgeArchitectureAction(Action):
    prd_path: str = Field(description="Path to PRD document")
    uml_class_path: str = Field(description="Path to UML class diagram")
    uml_sequence_path: str = Field(description="Path to UML sequence diagram")
    arch_design_path: str = Field(description="Path to architecture design document")
    architecture_path: str = Field(description="Path to RIB JSON to evaluate")


class JudgeArchitectureObservation(Observation):
    score: int = 0
    feedback: dict[str, str] = {}
    raw_text: str = ""
    error: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"judge_architecture FAILED: {self.error}")]
        fb_lines = "\n".join(f"  - {k}: {v}" for k, v in self.feedback.items())
        text = f"Architecture Score: {self.score}/10\nFeedback:\n{fb_lines}"
        if self.score < 8:
            text += (
                "\n\n⚠️ SCORE BELOW THRESHOLD (< 8). "
                "You MUST edit the RIB JSON based on the feedback above using file_editor, "
                "then call judge_architecture again. "
                "Do NOT proceed to generate_rib until score ≥ 8 or you have used 3 judge attempts."
            )
        return [TextContent(text=text)]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class JudgeArchitectureExecutor(ToolExecutor[JudgeArchitectureAction, JudgeArchitectureObservation]):
    """Single LLM.completion() call to evaluate architecture."""

    def __init__(self, llm: LLM, workspace_dir: Path):
        self.llm = llm
        self.workspace_dir = workspace_dir

    def __call__(
        self,
        action: JudgeArchitectureAction,
        conversation=None,
    ) -> JudgeArchitectureObservation:
        from common.prompts import architecture_judge_prompt

        def _read(p: str) -> str:
            return (self.workspace_dir / p).read_text(encoding="utf-8") if p else ""

        try:
            prd = _read(action.prd_path)
            uml_class = _read(action.uml_class_path)
            uml_seq = _read(action.uml_sequence_path)
            arch_design = _read(action.arch_design_path)
            arch_json = _read(action.architecture_path)
        except FileNotFoundError as e:
            return JudgeArchitectureObservation(error=f"File not found: {e}")

        # Collect pre-existing workspace files so the judge won't penalise
        # the RIB for omitting them (they are assets, not generated code).
        pre_existing = sorted(
            str(p.relative_to(self.workspace_dir))
            for p in self.workspace_dir.rglob("*")
            if p.is_file()
        )

        prompt = architecture_judge_prompt(
            requirement=prd,
            uml_class=uml_class,
            uml_sequence=uml_seq,
            arch_design=arch_design,
            architecture_json=arch_json,
            pre_existing_files=pre_existing,
        )

        response = self.llm.completion(
            messages=[Message(role="user", content=[TextContent(text=prompt)])],
        )
        # LLMResponse.message is a Message; extract text from content blocks
        text = "".join(
            c.text for c in response.message.content if isinstance(c, TextContent)
        )

        judge = parse_judge_score(text)
        return JudgeArchitectureObservation(
            score=judge.final_score,
            feedback=judge.feedback if hasattr(judge, "feedback") and judge.feedback else {"raw": text[:500]},
            raw_text=text,
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Evaluate a RIB architecture against project requirements.

The tool reads the requirement documents and the RIB JSON, then scores the
architecture on multiple dimensions (requirement coverage, module partitioning,
interface consistency, dependency management).

Returns a score (1-10) and per-dimension feedback.

Use this after generate_architecture. If score < 8, edit the RIB JSON
with file_editor based on the feedback, then call this tool again.
"""


class JudgeArchitectureTool(ToolDefinition[JudgeArchitectureAction, JudgeArchitectureObservation]):
    """Evaluate RIB architecture quality."""

    @classmethod
    def create(cls, conv_state, llm=None, workspace_dir=None) -> Sequence[ToolDefinition]:
        if llm is None or workspace_dir is None:
            raise ValueError("JudgeArchitectureTool requires llm and workspace_dir")
        executor = JudgeArchitectureExecutor(llm, workspace_dir)
        return [cls(
            description=_DESCRIPTION,
            action_type=JudgeArchitectureAction,
            observation_type=JudgeArchitectureObservation,
            executor=executor,
        )]
