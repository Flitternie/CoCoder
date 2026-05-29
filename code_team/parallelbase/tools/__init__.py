"""Parallel-agent tools: communication + skeleton evaluation support."""

from parallelbase.tools import (  # noqa: F401
    send_to_agent,
    agent_manager,
    judge_file_rib,
)

TOOL_NAMES = [
    "SendToAgentTool",
    "AgentManagerTool",
    "JudgeFileRIBTool",
]

