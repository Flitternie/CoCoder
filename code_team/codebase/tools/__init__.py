"""Codebase custom tool definitions."""

# Import tool modules so classes/factories are available to the runtime.
from codebase.tools import (  # noqa: F401
    generate_architecture,
    generate_rib,
    generate_code,
    judge_architecture,
    judge_rib,
    run_tests,
    compile_check,
)

TOOL_NAMES = [
    "GenerateArchitectureTool",
    "GenerateRIBTool",
    "GenerateCodeTool",
    "JudgeArchitectureTool",
    "JudgeRIBTool",
    "RunTestsTool",
    "CompileCheckTool",
]
