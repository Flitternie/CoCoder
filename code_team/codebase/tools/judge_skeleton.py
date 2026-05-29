"""judge_skeleton tool — Judge (single LLM call).

Evaluates skeleton code against the SSAT architecture.
Returns a structured score and feedback.
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

class JudgeSkeletonAction(Action):
    architecture_path: str = Field(description="Path to SSAT JSON (relative to workspace)")


class JudgeSkeletonObservation(Observation):
    score: int = 0
    feedback: dict[str, str] = {}
    raw_text: str = ""
    error: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"judge_skeleton FAILED: {self.error}")]
        fb_lines = "\n".join(f"  - {k}: {v}" for k, v in self.feedback.items())
        text = f"Skeleton Score: {self.score}/10\nFeedback:\n{fb_lines}"
        if self.score < 8:
            text += (
                "\n\n⚠️ SCORE BELOW THRESHOLD (< 8). "
                "You MUST fix the skeleton files based on the feedback above using file_editor, "
                "then call judge_skeleton again. "
                "Do NOT proceed to generate_code until score ≥ 8 or you have used 3 judge attempts."
            )
        return [TextContent(text=text)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from common.utils.source_filter import is_source_file as _is_source_file, MAX_SOURCE_FILE_SIZE
from common.utils.ssat_helpers import load_architecture, flatten_files


def _collect_skeleton_snapshot(workspace: Path, architecture: list[dict]) -> list[dict]:
    """Read all skeleton files from workspace based on architecture.

    Only source-code files are included; data files (e.g. .txt, .csv)
    and oversized source files (> MAX_SOURCE_FILE_SIZE) are skipped to
    avoid blowing up the LLM prompt.
    """
    snapshot = []
    for f_item in flatten_files(architecture):
        rel = f_item.get("path", "")
        if not _is_source_file(rel):
            continue
        fpath = workspace / rel
        if fpath.exists():
            size = fpath.stat().st_size
            if size > MAX_SOURCE_FILE_SIZE:
                snapshot.append({"path": rel, "code": f"# [SKIPPED: file too large ({size:,} bytes > {MAX_SOURCE_FILE_SIZE:,} limit)]"})
                continue
            code = fpath.read_text(encoding="utf-8", errors="replace")
            snapshot.append({"path": rel, "code": code})
    return snapshot


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class JudgeSkeletonExecutor(ToolExecutor[JudgeSkeletonAction, JudgeSkeletonObservation]):
    """Single LLM.completion() call to evaluate skeleton code."""

    def __init__(self, llm: LLM, root_workspace: Path):
        self.llm = llm
        self.root_workspace = root_workspace

    def __call__(
        self,
        action: JudgeSkeletonAction,
        conversation=None,
    ) -> JudgeSkeletonObservation:
        from common.prompts import skeleton_judge_prompt

        workspace = self.root_workspace

        # Load architecture
        try:
            arch_data = load_architecture(workspace / action.architecture_path)
        except (FileNotFoundError, Exception) as e:
            return JudgeSkeletonObservation(error=str(e))

        # Collect skeleton snapshot (source files only)
        skeleton = _collect_skeleton_snapshot(workspace, arch_data)
        if not skeleton:
            return JudgeSkeletonObservation(error="No skeleton files found in workspace")

        # Filter architecture to match — only source files that are not
        # oversized, so the judge doesn't penalise for "missing" or
        # placeholder skeletons of data-heavy files.
        source_arch = []
        for module in arch_data:
            filtered_files = []
            for f in module.get("files", []):
                rel = f.get("path", "")
                if not _is_source_file(rel):
                    continue
                fpath = workspace / rel
                if fpath.exists() and fpath.stat().st_size > MAX_SOURCE_FILE_SIZE:
                    continue
                filtered_files.append(f)
            if filtered_files:
                source_arch.append({**module, "files": filtered_files})

        # Also strip placeholder entries from the skeleton snapshot so the
        # judge only sees real skeleton code.
        skeleton = [s for s in skeleton if not s["code"].startswith("# [SKIPPED:")]

        prompt = skeleton_judge_prompt(
            architecture=source_arch,
            skeleton=skeleton,
        )

        response = self.llm.completion(
            messages=[Message(role="user", content=[TextContent(text=prompt)])],
        )
        # LLMResponse.message is a Message; extract text from content blocks
        text = "".join(
            c.text for c in response.message.content if isinstance(c, TextContent)
        )

        judge = parse_judge_score(text)
        return JudgeSkeletonObservation(
            score=judge.final_score,
            feedback=judge.feedback if hasattr(judge, "feedback") and judge.feedback else {"raw": text[:500]},
            raw_text=text,
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Evaluate skeleton code quality against the SSAT architecture.

The tool reads the SSAT JSON and all generated skeleton files from the workspace,
then scores on structural alignment and compilation readiness.

Parameters:
- architecture_path: Relative path to SSAT JSON (e.g. 'architecture/ssat.json')

Returns a score (1-10) and feedback.

Use this after generating all skeleton files. If score < 8, fix skeletons
with file_editor based on feedback, then call this tool again.
"""


class JudgeSkeletonTool(ToolDefinition[JudgeSkeletonAction, JudgeSkeletonObservation]):
    """Evaluate skeleton code quality."""

    @classmethod
    def create(cls, conv_state, llm=None, workspace_dir=None) -> Sequence[ToolDefinition]:
        if llm is None or workspace_dir is None:
            raise ValueError("JudgeSkeletonTool requires llm and workspace_dir")
        executor = JudgeSkeletonExecutor(llm, workspace_dir)
        return [cls(
            description=_DESCRIPTION,
            action_type=JudgeSkeletonAction,
            observation_type=JudgeSkeletonObservation,
            executor=executor,
        )]
