"""dismiss_teammate tool — Leader-only.

Synchronously closes an agent's conversation and removes it from the Orchestrator.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor

if TYPE_CHECKING:
    from parallelbase.orchestrator import ParallelOrchestrator


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class DismissTeammateAction(Action):
    name: str = Field(description="Name of the agent to dismiss (e.g. 'code_1')")


class DismissTeammateObservation(Observation):
    success: bool = False
    detail: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.detail)]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class DismissTeammateExecutor(ToolExecutor[DismissTeammateAction, DismissTeammateObservation]):
    """Remove an agent via the Orchestrator (synchronous)."""

    def __init__(self, orchestrator: ParallelOrchestrator):
        self.orchestrator = orchestrator

    def __call__(
        self,
        action: DismissTeammateAction,
        conversation=None,
    ) -> DismissTeammateObservation:
        try:
            self.orchestrator.dismiss_agent(action.name)
            return DismissTeammateObservation(
                success=True,
                detail=f"Agent '{action.name}' dismissed successfully.",
            )
        except Exception as e:
            return DismissTeammateObservation(
                success=False,
                detail=f"Failed to dismiss agent '{action.name}': {e}",
            )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Dismiss a teammate agent. Only the Leader can use this tool.

This closes the agent's conversation and removes it from the system.
Use this after all work is done to clean up resources.

Pass the agent's name (e.g. 'architecture', 'code_1').
"""


class DismissTeammateTool(ToolDefinition[DismissTeammateAction, DismissTeammateObservation]):
    """Dismiss a teammate agent (Leader-only)."""

    @classmethod
    def create(cls, conv_state, orchestrator: ParallelOrchestrator | None = None) -> Sequence[ToolDefinition]:
        if orchestrator is None:
            raise ValueError("DismissTeammateTool requires an orchestrator instance")
        executor = DismissTeammateExecutor(orchestrator)
        return [cls(
            description=_DESCRIPTION,
            action_type=DismissTeammateAction,
            observation_type=DismissTeammateObservation,
            executor=executor,
        )]
