"""spawn_teammate tool — Leader-only.

Synchronously creates a new agent (Agent + LocalConversation) and registers
it in the Orchestrator.  If ``initial_message`` is provided, the message is
automatically placed into the MessageBus for async delivery, saving the
Leader an extra ``send_to_agent`` call.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor

from parallelbase.message_bus import AgentMessage, MessageBus

if TYPE_CHECKING:
    from parallelbase.orchestrator import ParallelOrchestrator


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class SpawnTeammateAction(Action):
    name: str = Field(description="Unique name for the new agent (e.g. 'architecture', 'code_1')")
    role: str = Field(description="Agent role: 'architecture' or 'code'")
    files: list[str] = Field(
        default_factory=list,
        description="Files assigned to this agent (for code role, e.g. ['src/utils.py', 'src/main.py'])",
    )
    initial_message: str = Field(
        default="",
        description="Optional task message to send immediately after creation (saves a separate send_to_agent call)",
    )


class SpawnTeammateObservation(Observation):
    success: bool = False
    agent_name: str = ""
    detail: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.detail)]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class SpawnTeammateExecutor(ToolExecutor[SpawnTeammateAction, SpawnTeammateObservation]):
    """Create a new agent via the Orchestrator (synchronous).

    If *initial_message* is non-empty, the executor also enqueues it into the
    MessageBus so the Orchestrator will push it to the new agent automatically.

    The sender name is resolved dynamically from ``conversation._agent_name``
    at call time, so a single executor instance can be shared across agents.
    """

    def __init__(self, orchestrator: ParallelOrchestrator, message_bus: MessageBus):
        self.orchestrator = orchestrator
        self.bus = message_bus

    def __call__(
        self,
        action: SpawnTeammateAction,
        conversation=None,
    ) -> SpawnTeammateObservation:
        self_name = getattr(conversation, "_agent_name", "unknown")

        try:
            self.orchestrator.spawn_agent(
                name=action.name,
                role=action.role,
                files=action.files,
            )
        except Exception as e:
            return SpawnTeammateObservation(
                success=False,
                agent_name=action.name,
                detail=f"Failed to create agent '{action.name}': {e}",
            )

        # Optionally enqueue the initial task message
        detail = f"Agent '{action.name}' (role={action.role}) created successfully."
        if action.initial_message:
            self.bus.send(AgentMessage(
                from_agent=self_name,
                to_agent=action.name,
                content=action.initial_message,
                timestamp=time.time(),
            ))
            detail += f" Initial message queued."

        return SpawnTeammateObservation(
            success=True,
            agent_name=action.name,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Create a new teammate agent. Only the Leader can use this tool.

Parameters:
- name: Unique name (e.g. 'architecture', 'code_1', 'code_2')
- role: 'architecture' (generates RIB) or 'code' (generates skeleton/code)
- files: List of source files assigned to this agent (code role only)
- initial_message: (optional) Task message to send right after creation — \
saves a separate send_to_agent call. Leave empty to create without sending.

Example:
  spawn_teammate(name="code_1", role="code", files=["src/a.py"], \
initial_message="Generate skeletons for your assigned files")
"""


class SpawnTeammateTool(ToolDefinition[SpawnTeammateAction, SpawnTeammateObservation]):
    """Create a new teammate agent (Leader-only)."""

    @classmethod
    def create(
        cls,
        conv_state,
        orchestrator: ParallelOrchestrator | None = None,
        message_bus: MessageBus | None = None,
    ) -> Sequence[ToolDefinition]:
        if orchestrator is None:
            raise ValueError("SpawnTeammateTool requires an orchestrator instance")
        if message_bus is None:
            raise ValueError("SpawnTeammateTool requires a message_bus instance")
        executor = SpawnTeammateExecutor(orchestrator, message_bus)
        return [cls(
            description=_DESCRIPTION,
            action_type=SpawnTeammateAction,
            observation_type=SpawnTeammateObservation,
            executor=executor,
        )]
