"""judge_file_rib tool — per-file skeleton quality evaluator.

Evaluates a single file's skeleton code against its RIB architecture entry.
Used by Subagents to self-evaluate their skeleton before proceeding to code.

Based on JudgeRIBTool but operates on a single file, not the full project.
"""
from __future__ import annotations

import json


from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

from pydantic import Field

from openhands.sdk import LLM, Action, Observation, TextContent, ImageContent, Message
from openhands.sdk.tool import ToolDefinition, ToolExecutor

from common.utils.parsers import parse_judge_score
from common.utils.rib_helpers import load_architecture, find_file_item
from common.utils.source_filter import MAX_SOURCE_FILE_SIZE

JUDGE_FILE_SKELETON_MAX_OUTPUT_TOKENS = 12000
JUDGE_FILE_SKELETON_TIMEOUT_SEC = 180


def file_rib_judge_prompt(file_architecture: dict, skeleton_code: str, file_path: str) -> str:
    """Prompt for evaluating a single file's skeleton against its RIB entry."""
    return f"""You are an expert software architecture reviewer.
You will be given two inputs:

1. **File Architecture Specification (ARCH)** - the RIB entry for a single file, including its intended classes, functions, parameters, and descriptions.
2. **Generated Skeleton Code (SKEL)** - the Python skeleton code produced for this file, including imports, class definitions, and function signatures (with `pass` as placeholders).

Your task is to evaluate the quality of the skeleton for file `{file_path}` based on the following criteria:

1. **Interface Matching** - Do the classes and functions (including names, parameters, and default values) align with the architecture definition? Are all expected interfaces present? Are there inconsistencies or omissions?
2. **Import Correctness** - Are the import statements reasonable given the file's role and dependencies described in the architecture?
3. **Syntax Validity** - Is the skeleton syntactically valid Python code?

For each criterion, provide a short justification of your evaluation.
Then, give an **overall score** for the skeleton between **1 (poor) and 10 (excellent)**.

Format your output as follows:

```
Interface Matching: <justification>
Import Correctness: <justification>
Syntax Validity: <justification>

Final Score: <a single number between 1 and 10>
```

------

### **Inputs:**

#### **ARCH** (for `{file_path}`):

```json
{json.dumps(file_architecture, ensure_ascii=False, indent=2)}
```

#### **SKEL** (for `{file_path}`):

```python
{skeleton_code}
```

""".strip()


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class JudgeFileRIBAction(Action):
    architecture_path: str = Field(description="Path to RIB JSON (relative to workspace)")
    target_file: str = Field(description="Source file to evaluate (workspace-relative), e.g. 'rsa/key.py'")


class JudgeFileRIBObservation(Observation):
    score: int = 0
    feedback: dict[str, str] = {}
    raw_text: str = ""
    error: str = ""
    target_file: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"judge_file_rib FAILED: {self.error}")]
        fb_lines = "\n".join(f"  - {k}: {v}" for k, v in self.feedback.items())
        text = f"Skeleton Score for {self.target_file}: {self.score}/10\nFeedback:\n{fb_lines}"
        if self.score < 8:
            text += (
                "\n\n⚠️ SCORE BELOW THRESHOLD (< 8). "
                "Fix your skeleton based on the feedback above, "
                "then call judge_file_rib again. "
                "Do NOT proceed to full code implementation until score ≥ 8 "
                "or you have used 3 judge attempts."
            )
        return [TextContent(text=text)]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class JudgeFileRIBExecutor(ToolExecutor[JudgeFileRIBAction, JudgeFileRIBObservation]):
    """Single LLM.completion() call to evaluate one file's skeleton code."""

    def __init__(self, llm: LLM, root_workspace: Path):
        self.llm = llm
        self.root_workspace = root_workspace

    def _completion_with_timeout(self, prompt: str):
        judge_llm = self.llm.model_copy(
            update={
                "max_output_tokens": JUDGE_FILE_SKELETON_MAX_OUTPUT_TOKENS,
                "timeout": JUDGE_FILE_SKELETON_TIMEOUT_SEC,
            }
        )

        def _call():
            return judge_llm.completion(
                messages=[Message(role="user", content=[TextContent(text=prompt)])],
            )

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge_file_rib")
        future = executor.submit(_call)
        try:
            return future.result(timeout=JUDGE_FILE_SKELETON_TIMEOUT_SEC)
        except FutureTimeoutError as e:
            future.cancel()
            raise TimeoutError(
                f"LLM judge call timed out after {JUDGE_FILE_SKELETON_TIMEOUT_SEC} seconds"
            ) from e
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def __call__(
        self,
        action: JudgeFileRIBAction,
        conversation=None,
    ) -> JudgeFileRIBObservation:


        workspace = self.root_workspace

        # Load architecture
        try:
            arch_data = load_architecture(workspace / action.architecture_path)
        except (FileNotFoundError, Exception) as e:
            return JudgeFileRIBObservation(
                error=str(e), target_file=action.target_file,
            )

        # Find the target file's RIB entry
        file_item = find_file_item(arch_data, action.target_file)
        if not file_item:
            return JudgeFileRIBObservation(
                error=f"File '{action.target_file}' not found in RIB architecture",
                target_file=action.target_file,
            )

        # Read the skeleton code from disk
        target_abs = workspace / action.target_file
        if not target_abs.exists():
            return JudgeFileRIBObservation(
                error=f"Skeleton file not found: {action.target_file}",
                target_file=action.target_file,
            )

        size = target_abs.stat().st_size
        if size > MAX_SOURCE_FILE_SIZE:
            return JudgeFileRIBObservation(
                error=f"File too large ({size:,} bytes > {MAX_SOURCE_FILE_SIZE:,} limit)",
                target_file=action.target_file,
            )

        skeleton_code = target_abs.read_text(encoding="utf-8", errors="replace")

        # Build prompt and call judge LLM
        prompt = file_rib_judge_prompt(
            file_architecture=file_item,
            skeleton_code=skeleton_code,
            file_path=action.target_file,
        )

        try:
            response = self._completion_with_timeout(prompt)
        except Exception as e:
            return JudgeFileRIBObservation(
                error=f"LLM judge call failed: {e}",
                target_file=action.target_file,
            )
        text = "".join(
            c.text for c in response.message.content if isinstance(c, TextContent)
        )

        judge = parse_judge_score(text)
        return JudgeFileRIBObservation(
            score=judge.final_score,
            feedback=judge.feedback if hasattr(judge, "feedback") and judge.feedback else {"raw": text[:500]},
            raw_text=text,
            target_file=action.target_file,
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Evaluate a single file's skeleton code against its RIB architecture entry.

Call this after generating your skeleton to check quality. The tool reads
your file's RIB entry and skeleton code, then scores on interface matching,
import correctness, and syntax validity.

Parameters:
- architecture_path: Relative path to RIB JSON (e.g. 'architecture/rib.json')
- target_file: Your assigned source file (workspace-relative, e.g. 'rsa/key.py')

Returns a score (1-10) and feedback.

If score < 8, fix your skeleton based on feedback, then call this tool again.
Do NOT proceed to full code implementation until score ≥ 8 or 3 attempts used.
"""


class JudgeFileRIBTool(ToolDefinition[JudgeFileRIBAction, JudgeFileRIBObservation]):
    """Evaluate a single file's skeleton code quality."""

    @classmethod
    def create(cls, conv_state, llm=None, workspace_dir=None) -> Sequence[ToolDefinition]:
        if llm is None or workspace_dir is None:
            raise ValueError("JudgeFileRIBTool requires llm and workspace_dir")
        executor = JudgeFileRIBExecutor(llm, workspace_dir)
        return [cls(
            description=_DESCRIPTION,
            action_type=JudgeFileRIBAction,
            observation_type=JudgeFileRIBObservation,
            executor=executor,
        )]
