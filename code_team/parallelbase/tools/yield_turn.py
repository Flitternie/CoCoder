"""yield_turn tool — agent explicitly signals it is waiting for messages.

When an agent calls this tool, its run() loop ends cleanly after the tool
call (not after a text message).  The Orchestrator recognises YieldTurnTool
as a legitimate pause signal and will NOT nudge the agent to continue.
"""
from __future__ import annotations

from collections.abc import Sequence

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class YieldTurnAction(Action):
    """No parameters — just signals 'I am done for now'."""
    pass


class YieldTurnObservation(Observation):
    """Confirmation that the agent's turn has been yielded."""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(
            text="OK, yielding. You will be woken when messages arrive."
        )]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class YieldTurnExecutor(ToolExecutor[YieldTurnAction, YieldTurnObservation]):
    """Executor that pauses the agent's run() loop.

    Sets execution_status to PAUSED so the SDK's run() loop breaks
    immediately after this tool call — no further agent.step() iterations.
    The next run() call (triggered by a new message delivery) will
    transition PAUSED → RUNNING automatically (SDK line 500-505).
    """

    def __call__(
        self,
        action: YieldTurnAction,
        conversation=None,
    ) -> YieldTurnObservation:
        if conversation is not None:
            from openhands.sdk.conversation.state import ConversationExecutionStatus
            conversation.state.execution_status = ConversationExecutionStatus.PAUSED
        return YieldTurnObservation()


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Signal that you are done for now and waiting for messages from other agents.
Call this instead of outputting a text message when you want to pause and wait.
After calling this tool, your run() will end cleanly.
"""


class YieldTurnTool(ToolDefinition[YieldTurnAction, YieldTurnObservation]):
    """Agent signals it is waiting for messages."""

    @classmethod
    def create(cls, conv_state) -> Sequence[ToolDefinition]:
        return [cls(
            description=_DESCRIPTION,
            action_type=YieldTurnAction,
            observation_type=YieldTurnObservation,
            executor=YieldTurnExecutor(),
        )]
