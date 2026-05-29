"""send_to_agent tool — available to all agents.

Sends a message to another agent (direct) or broadcasts to all agents of a role.
The executor puts the message into the shared MessageBus; the Orchestrator's
routing loop delivers it asynchronously.
"""
from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor

from parallelbase.message_bus import AgentMessage, MessageBus


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class SendToAgentAction(Action):
    mode: str = Field(description="'direct' to send to one agent, 'broadcast' to send to all agents of a role")
    target: str = Field(description="direct: agent name (e.g. 'leader'); broadcast: role prefix (e.g. 'code')")
    message: str = Field(description="Message content to send")


class SendToAgentObservation(Observation):
    success: bool = False
    detail: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.detail)]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class SendToAgentExecutor(ToolExecutor[SendToAgentAction, SendToAgentObservation]):
    """Put a message into the MessageBus for async delivery by Orchestrator.

    The sender name is resolved dynamically from ``conversation._agent_name``
    at call time, so a single executor instance can be shared across agents.
    """

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus

    def __call__(
        self,
        action: SendToAgentAction,
        conversation=None,
    ) -> SendToAgentObservation:
        self_name = getattr(conversation, "_agent_name", "unknown")

        if action.mode == "direct":
            to = action.target
        elif action.mode == "broadcast":
            to = f"broadcast:{action.target}"
        else:
            return SendToAgentObservation(
                success=False,
                detail=f"Unknown mode: {action.mode}. Use 'direct' or 'broadcast'.",
            )

        self.bus.send(AgentMessage(
            from_agent=self_name,
            to_agent=to,
            content=action.message,
            timestamp=time.time(),
        ))
        return SendToAgentObservation(
            success=True,
            detail=f"Message sent (to={to}).",
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Send a message to another agent.

- mode="direct", target="leader" → send to the leader agent
- mode="direct", target="code_1" → send to a specific code agent
- mode="broadcast", target="code" → send to ALL agents whose name starts with "code"

Messages are delivered asynchronously. After sending, your current run() ends \
and the recipient will be woken up by the Orchestrator.
"""


class SendToAgentTool(ToolDefinition[SendToAgentAction, SendToAgentObservation]):
    """Send a message to another agent (direct or broadcast)."""

    @classmethod
    def create(cls, conv_state, message_bus: MessageBus | None = None) -> Sequence[ToolDefinition]:
        if message_bus is None:
            raise ValueError("SendToAgentTool requires a message_bus instance")
        executor = SendToAgentExecutor(message_bus)
        return [cls(
            description=_DESCRIPTION,
            action_type=SendToAgentAction,
            observation_type=SendToAgentObservation,
            executor=executor,
        )]
