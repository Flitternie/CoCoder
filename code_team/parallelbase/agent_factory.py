"""Agent factory — builds OpenHands Agent objects for each parallel role.

Creates Agent instances with the correct tool set, system prompt, and condenser
for Main Agent / Subagent roles.  Reuses tool executors from codebase/tools/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Sequence

from openhands.sdk import Agent, LLM, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context import AgentContext
from openhands.sdk.tool import register_tool

# Standard tool imports (must happen before Agent creation)
import openhands.tools.terminal  # noqa: F401
import openhands.tools.file_editor  # noqa: F401
import openhands.tools.apply_patch  # noqa: F401
import openhands.tools.glob  # noqa: F401
import openhands.tools.grep  # noqa: F401
import openhands.tools.task_tracker  # noqa: F401

# Custom tool imports (registers definitions)
import codebase.tools  # noqa: F401

from common.agent_runtime import AgentRuntime, _set_tool_text_content_limit
from common.config import TERMINAL_NO_CHANGE_TIMEOUT_SECONDS
from codebase.config import SUB_AGENT_TOOLS
from codebase.tools.judge_architecture import JudgeArchitectureTool
from codebase.tools.compile_check import CompileCheckTool
from openhands.tools.delegate import DelegateTool as _DelegateTool
from parallelbase.message_bus import MessageBus
from parallelbase.prompts import (
    main_agent_system_prompt,
    subagent_system_prompt,
)
# from parallelbase.tools.generate_architecture_refine import RefineArchitectureTool
from codebase.tools.generate_architecture import GenerateArchitectureTool
from parallelbase.tools.judge_file_rib import JudgeFileRIBTool
from parallelbase.tools.parallel_run_tests import ParallelRunTestsTool
from parallelbase.tools.send_to_agent import SendToAgentTool
from parallelbase.tools.agent_manager import AgentManagerTool
from parallelbase.tools.yield_turn import YieldTurnTool

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool sets per role
# ---------------------------------------------------------------------------

_STANDARD_TOOLS = [
    Tool(name="terminal", params={"no_change_timeout_seconds": TERMINAL_NO_CHANGE_TIMEOUT_SECONDS}),
    Tool(name="file_editor"),
    Tool(name="apply_patch"),
    Tool(name="glob"),
    Tool(name="grep"),
]

# Main Agent: file_editor, terminal, glob, grep, task_tracker + custom tools.
_MAIN_AGENT_STANDARD_TOOLS = [
    Tool(name="file_editor"),
    Tool(name="terminal", params={"no_change_timeout_seconds": TERMINAL_NO_CHANGE_TIMEOUT_SECONDS}),
    Tool(name="glob"),
    Tool(name="grep"),
    Tool(name="task_tracker"),
]

_MAIN_AGENT_CUSTOM_TOOLS = [
    # Architecture tools
    Tool(name="GenerateArchitectureTool"),
    Tool(name="JudgeArchitectureTool"),
    # Validation tools
    Tool(name="RunTestsTool"),
    Tool(name="CompileCheckTool"),
    # Communication / management tools
    Tool(name="SendToAgentTool"),
    Tool(name="AgentManagerTool"),
    Tool(name="YieldTurnTool"),
]

_SUBAGENT_CUSTOM_TOOLS = [
    Tool(name="JudgeFileRIBTool"),     # per-file skeleton quality evaluator
    Tool(name="CompileCheckTool"),
    Tool(name="SendToAgentTool"),
    Tool(name="YieldTurnTool"),
]


class AgentFactory:
    """Create OpenHands Agent instances for each parallel role.

    Holds shared resources (sub-agent runtime, judge LLM, workspace) and
    registers custom tool factories **once** at init time so that
    ``Tool(name=...)`` references resolve correctly when any Agent is built.

    All tool factories are agent-agnostic: per-agent identity (e.g. sender
    name for ``SendToAgentTool``) is resolved dynamically at tool-call time
    via ``conversation._agent_name``, which is set by the Orchestrator.
    """

    def __init__(
        self,
        *,
        workspace_dir: Path,
        run_dir: Path,
        runtime_cfg,  # RuntimeConfig — stored for creating per-agent runtimes
        main_llm: LLM,
        judge_llm: LLM,
        message_bus: MessageBus,
        orchestrator,  # ParallelOrchestrator (forward-ref to avoid circular import)
        testfix_iters: int = 10,
    ):
        self.workspace_dir = workspace_dir
        self.run_dir = run_dir
        self.runtime_cfg = runtime_cfg
        self.main_llm = main_llm
        self.judge_llm = judge_llm
        self.message_bus = message_bus
        self.orchestrator = orchestrator
        self.testfix_iters = testfix_iters

        # Raise OpenHands tool text content limit (matches serial pipeline)
        _set_tool_text_content_limit(120000)

        # Track all LLMs and runtimes for metrics aggregation / cleanup
        self._llms: list[LLM] = [main_llm, judge_llm]
        self._runtimes: list[AgentRuntime] = []

        # --- Shared runtime (used by all generate tools) ---
        self._shared_runtime = AgentRuntime(
            workspace_dir,
            run_dir / "shared_runtime",
            runtime_cfg,
            agent_tool_configs=SUB_AGENT_TOOLS,
        )
        self._runtimes.append(self._shared_runtime)

        # --- Register ALL tool factories once ---
        self._register_all_tool_factories()

    # ------------------------------------------------------------------
    # One-time tool factory registration
    # ------------------------------------------------------------------

    def _register_all_tool_factories(self) -> None:
        """Register all custom tool factories once at init time.

        Factories capture shared resources (runtime, judge_llm, workspace, bus,
        orchestrator) via closures.  Per-agent identity is resolved dynamically
        at call time via ``conversation._agent_name``.
        """
        runtime = self._shared_runtime
        judge_llm = self.judge_llm
        workspace_dir = self.workspace_dir
        test_output_dir = self.workspace_dir / "test_outputs"
        bus = self.message_bus
        orchestrator = self.orchestrator

        # --- Register DelegateTool with increased max_children ---
        def _delegate_factory(conv_state):
            return list(_DelegateTool.create(conv_state, max_children=20))

        register_tool("DelegateTool", _delegate_factory)

        # --- Generative tools (shared runtime) ---

        def gen_arch_factory(conv_state):
            return list(GenerateArchitectureTool.create(conv_state, runtime=runtime))

        # --- Judge tools ---
        def judge_arch_factory(conv_state):
            return list(JudgeArchitectureTool.create(conv_state, llm=judge_llm, workspace_dir=workspace_dir))

        def judge_file_rib_factory(conv_state):
            return list(JudgeFileRIBTool.create(conv_state, llm=judge_llm, workspace_dir=workspace_dir))

        # --- Execution tools ---
        def run_tests_factory(conv_state):
            return list(ParallelRunTestsTool.create(conv_state, workspace_dir=workspace_dir, output_dir=test_output_dir, max_runs=self.testfix_iters))

        def compile_check_factory(conv_state):
            return list(CompileCheckTool.create(conv_state, workspace_dir=workspace_dir))

        # --- Communication tools (agent-agnostic, name resolved at call time) ---
        def send_to_agent_factory(conv_state):
            return list(SendToAgentTool.create(conv_state, message_bus=bus))

        def agent_manager_factory(conv_state):
            return list(AgentManagerTool.create(conv_state, orchestrator=orchestrator, message_bus=bus))

        # Register all factories (one-time, stable registry)
        register_tool("GenerateArchitectureTool", gen_arch_factory)
        register_tool("JudgeArchitectureTool", judge_arch_factory)
        register_tool("JudgeFileRIBTool", judge_file_rib_factory)
        register_tool("RunTestsTool", run_tests_factory)
        register_tool("CompileCheckTool", compile_check_factory)


        # --- Communication tools (agent-agnostic, name resolved at call time) ---
        register_tool("SendToAgentTool", send_to_agent_factory)
        register_tool("AgentManagerTool", agent_manager_factory)

        # --- Yield turn tool (all agents) ---
        def yield_turn_factory(conv_state):
            return list(YieldTurnTool.create(conv_state))

        register_tool("YieldTurnTool", yield_turn_factory)

    # ------------------------------------------------------------------
    # Agent creation
    # ------------------------------------------------------------------

    def create_agent(self, name: str, role: str) -> Agent:
        """Build an Agent for the given role.

        Args:
            name: Unique agent name (e.g. 'leader', 'code_key')
            role: One of 'leader', 'code'

        Returns:
            Configured Agent instance ready for conversation creation.
        """
        from common.prompt_constants import load_implementation_integrity_rules
        integrity_rules = load_implementation_integrity_rules(self.workspace_dir)
        if role == "leader":
            tools = _MAIN_AGENT_STANDARD_TOOLS + _MAIN_AGENT_CUSTOM_TOOLS
            system_prompt = main_agent_system_prompt(testfix_iters=self.testfix_iters, integrity_rules=integrity_rules)
        elif role == "code":
            tools = _STANDARD_TOOLS + _SUBAGENT_CUSTOM_TOOLS
            system_prompt = subagent_system_prompt(integrity_rules=integrity_rules)
        else:
            raise ValueError(f"Unknown role: {role}")

        # Per-agent LLM: same config, separate completion logs
        # Completions go to runs/<name>/logs/completions/ so the
        # progress tracker can attribute them by directory name.
        agent_comp_dir = self.run_dir / name / "logs" / "completions"
        agent_comp_dir.mkdir(parents=True, exist_ok=True)
        agent_llm = self.main_llm.model_copy(
            update={"usage_id": name, "log_completions_folder": str(agent_comp_dir)}
        )
        self._llms.append(agent_llm)

        # Build a condenser to keep context bounded
        condenser_comp_dir = self.run_dir / name / "logs" / "completions" / "condenser"
        condenser_comp_dir.mkdir(parents=True, exist_ok=True)
        condenser_llm = self.main_llm.model_copy(
            update={"usage_id": f"condenser_{name}", "stream": False,
                    "log_completions_folder": str(condenser_comp_dir)}
        )
        self._llms.append(condenser_llm)

        condenser = LLMSummarizingCondenser(
            llm=condenser_llm,
            max_size=300,
            keep_first=2,
        )

        ctx = AgentContext(system_message_suffix=system_prompt)
        agent = Agent(
            llm=agent_llm,
            tools=tools,
            condenser=condenser,
            agent_context=ctx,
            system_prompt_kwargs={"cli_mode": True},
        )

        log.info(f"[AgentFactory] Created agent '{name}' (role={role}) with {len(tools)} tools")
        return agent

    def get_all_llms(self) -> list[LLM]:
        """Return LLM instances for metrics aggregation (deduplicated by metrics identity).

        model_copy() produces shallow copies that share the same metrics object,
        so we must deduplicate to avoid counting the same cost/tokens N times.
        """
        all_llms = list(self._llms)
        for rt in self._runtimes:
            all_llms.extend(rt._llms)
        # Deduplicate: keep only one LLM per unique metrics object
        seen_metrics: set[int] = set()
        unique: list[LLM] = []
        for llm in all_llms:
            mid = id(llm.metrics)
            if mid not in seen_metrics:
                seen_metrics.add(mid)
                unique.append(llm)
        return unique

    def close(self) -> None:
        """Close all per-agent runtimes."""
        for rt in self._runtimes:
            try:
                rt.close()
            except Exception:
                pass
        self._runtimes.clear()
