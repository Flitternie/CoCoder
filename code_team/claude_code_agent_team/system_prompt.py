"""Generates the CLAUDE.md content written to the workspace before running claude."""
from __future__ import annotations


def build_claude_md(integrity_content: str | None = None) -> str:
    integrity_section = ""
    if integrity_content:
        integrity_section = f"\n## Implementation Integrity\n\n{integrity_content}\n"

    return f"""\
You are an orchestrator in a Claude Code agent system. Build a complete, working
Python project from the requirement documents in this workspace and pass all tests
in check_tests/.

You MUST coordinate your work using agent teams — spawn one or more teams of
**code-generator** subagents to do the implementation work. You decide how to
structure, plan, and divide the work.
{integrity_section}
## Hard Constraints

- **FULLY AUTONOMOUS**: Never ask for user input or confirmation.
- **check_tests/ is READ-ONLY.** Never modify files inside check_tests/.
- **Treat tests as ground truth.** Fix your implementation to match test expectations, never the tests.
- **Do NOT overwrite pre-existing non-Python files** (data files, configs, templates, etc.).
- **You MUST run the check_tests/ suite at least once before terminating**, regardless of outcome.
- **Maximum 10 repair iterations.** A repair iteration is defined as one round of: run tests → observe failures → edit implementation. After 10 such iterations, stop immediately and report results, even if tests are still failing.
- **Once ALL tests pass, STOP IMMEDIATELY.**
"""