"""agent_manager tool — unified agent lifecycle management.

Consolidates spawn_teammate + dismiss_teammate into a single tool with
an additional ``query`` operation that lets any agent discover the current
set of agents, their roles, status, and assigned files.
"""
from __future__ import annotations

import json
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

class AgentManagerAction(Action):
    operation: str = Field(description="Operation: 'spawn', 'dismiss', or 'query'")
    # spawn parameters
    name: str = Field(default="", description="Agent name (spawn/dismiss)")
    role: str = Field(default="", description="Agent role, e.g. 'code' (spawn)")
    files: list[str] = Field(default_factory=list, description="Files assigned to agent (spawn)")
    initial_message: str = Field(default="", description="Optional initial task message (spawn)")


class AgentManagerObservation(Observation):
    success: bool = False
    detail: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.detail)]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class AgentManagerExecutor(ToolExecutor[AgentManagerAction, AgentManagerObservation]):
    """Dispatch to spawn/dismiss/query based on operation field."""

    def __init__(self, orchestrator: ParallelOrchestrator, message_bus: MessageBus):
        self.orchestrator = orchestrator
        self.bus = message_bus

    def __call__(
        self,
        action: AgentManagerAction,
        conversation=None,
    ) -> AgentManagerObservation:
        op = action.operation.lower().strip()
        if op == "spawn":
            return self._spawn(action, conversation)
        elif op == "dismiss":
            return self._dismiss(action)
        elif op == "query":
            return self._query()
        else:
            return AgentManagerObservation(
                success=False,
                detail=f"Unknown operation: '{action.operation}'. Use 'spawn', 'dismiss', or 'query'.",
            )

    def _spawn(self, action: AgentManagerAction, conversation) -> AgentManagerObservation:
        if not action.name:
            return AgentManagerObservation(success=False, detail="spawn requires 'name'")
        if not action.role:
            return AgentManagerObservation(success=False, detail="spawn requires 'role'")

        try:
            self.orchestrator.spawn_agent(
                name=action.name,
                role=action.role,
                files=action.files,
            )
        except Exception as e:
            return AgentManagerObservation(
                success=False,
                detail=f"Failed to spawn '{action.name}': {e}",
            )

        detail = f"Agent '{action.name}' (role={action.role}) created."
        if action.initial_message:
            self_name = getattr(conversation, "_agent_name", "unknown")
            self.bus.send(AgentMessage(
                from_agent=self_name,
                to_agent=action.name,
                content=action.initial_message,
                timestamp=time.time(),
            ))
            detail += " Initial message queued."

        return AgentManagerObservation(success=True, detail=detail)

    def _dismiss(self, action: AgentManagerAction) -> AgentManagerObservation:
        if not action.name:
            return AgentManagerObservation(success=False, detail="dismiss requires 'name'")
        try:
            self.orchestrator.dismiss_agent(action.name)
            return AgentManagerObservation(
                success=True,
                detail=f"Agent '{action.name}' dismissed.",
            )
        except Exception as e:
            return AgentManagerObservation(
                success=False,
                detail=f"Failed to dismiss '{action.name}': {e}",
            )

    def _query(self) -> AgentManagerObservation:
        agents_list = []
        for name, handle in self.orchestrator.agents.items():
            agents_list.append({
                "name": name,
                "role": handle.role,
                "status": "running",
                "files": handle.files,
            })
        detail = json.dumps({"agents": agents_list}, ensure_ascii=False)
        return AgentManagerObservation(success=True, detail=detail)


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Manage agents in the parallel pipeline.

Operations:
- operation="spawn": Create a new agent.
    Required: name, role. Optional: files, initial_message.
    Example: agent_manager(operation="spawn", name="code_1", role="code", \
files=["src/a.py"], initial_message="Generate skeleton for your file")

- operation="dismiss": Remove an agent.
    Required: name.
    Example: agent_manager(operation="dismiss", name="code_1")

- operation="query": List all agents with their name, role, status, and files.
    No parameters needed.
    Example: agent_manager(operation="query")
"""


class AgentManagerTool(ToolDefinition[AgentManagerAction, AgentManagerObservation]):
    """Unified agent management: spawn, dismiss, query."""

    @classmethod
    def create(
        cls,
        conv_state,
        orchestrator: ParallelOrchestrator | None = None,
        message_bus: MessageBus | None = None,
    ) -> Sequence[ToolDefinition]:
        if orchestrator is None:
            raise ValueError("AgentManagerTool requires an orchestrator instance")
        if message_bus is None:
            raise ValueError("AgentManagerTool requires a message_bus instance")
        executor = AgentManagerExecutor(orchestrator, message_bus)
        return [cls(
            description=_DESCRIPTION,
            action_type=AgentManagerAction,
            observation_type=AgentManagerObservation,
            executor=executor,
        )]
