"""Agent factory for cohesionbase — builds agents for leader / group roles.

Differences from depgraphbase:
  - Adds partition_into_groups tool (Role Detection + Louvain, Leader only)
  - Adds shared_task_list tool (Leader + Group agents)
  - Removes plan_dependency_waves (replaced by shared_task_list)
  - Removes judge_rib (no global skeleton phase)
  - Role "group" replaces "code" (multi-file agent)
  - Uses cohesionbase prompts
"""
from __future__ import annotations

import logging
from pathlib import Path

from openhands.sdk import Agent, LLM, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context import AgentContext
from openhands.sdk.tool import register_tool

# Standard tool imports
import openhands.tools.terminal  # noqa: F401
import openhands.tools.file_editor  # noqa: F401
import openhands.tools.apply_patch  # noqa: F401
import openhands.tools.glob  # noqa: F401
import openhands.tools.grep  # noqa: F401
import openhands.tools.task_tracker  # noqa: F401

import codebase.tools  # noqa: F401

from common.agent_runtime import AgentRuntime, _set_tool_text_content_limit
from common.config import TERMINAL_NO_CHANGE_TIMEOUT_SECONDS
from codebase.config import SUB_AGENT_TOOLS
from codebase.tools.judge_architecture import JudgeArchitectureTool
from codebase.tools.compile_check import CompileCheckTool
from openhands.tools.delegate import DelegateTool as _DelegateTool
from parallelbase.message_bus import MessageBus
from parallelbase.tools.parallel_run_tests import ParallelRunTestsTool
from parallelbase.tools.send_to_agent import SendToAgentTool
from parallelbase.tools.agent_manager import AgentManagerTool
from parallelbase.tools.yield_turn import YieldTurnTool

# cohesionbase-specific tools
from parallelbase.tools.generate_architecture_dep import GenerateArchitectureDepTool
from codebase.tools.generate_architecture import GenerateArchitectureTool
from cohesionbase.tools.partition_into_groups import PartitionIntoGroupsTool
from cohesionbase.tools.shared_task_list import SharedTaskListTool
from cohesionbase.tools.read_rib import ReadRIBTool
from cohesionbase.prompts import leader_system_prompt, group_agent_system_prompt

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

_LEADER_STANDARD_TOOLS = [
    Tool(name="file_editor"),
    Tool(name="terminal", params={"no_change_timeout_seconds": TERMINAL_NO_CHANGE_TIMEOUT_SECONDS}),
    Tool(name="glob"),
    Tool(name="grep"),
    Tool(name="task_tracker"),
]

_LEADER_CUSTOM_TOOLS = [
    Tool(name="GenerateArchitectureTool"),
    Tool(name="JudgeArchitectureTool"),
    Tool(name="PartitionIntoGroupsTool"),        # Cosine Similarity + InfoMap + Role Detection (computes weights internally)
    Tool(name="SharedTaskListTool"),
    Tool(name="RunTestsTool"),
    Tool(name="CompileCheckTool"),
    Tool(name="SendToAgentTool"),
    Tool(name="AgentManagerTool"),
    Tool(name="YieldTurnTool"),
]

_GROUP_CUSTOM_TOOLS = [
    Tool(name="ReadRIBTool"),                   # RIB lookup — call before writing each file
    Tool(name="SharedTaskListTool"),             # next/complete
    Tool(name="CompileCheckTool"),
    Tool(name="SendToAgentTool"),
    Tool(name="YieldTurnTool"),
]


class AgentFactory:
    """Create Agent instances for cohesionbase roles (leader / group)."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        run_dir: Path,
        runtime_cfg,
        main_llm: LLM,
        judge_llm: LLM,
        message_bus: MessageBus,
        orchestrator,
        testfix_iters: int = 10,
        rib_dep_tool: bool = False,
    ):
        self.workspace_dir = workspace_dir
        self.run_dir = run_dir
        self.runtime_cfg = runtime_cfg
        self.main_llm = main_llm
        self.judge_llm = judge_llm
        self.message_bus = message_bus
        self.orchestrator = orchestrator
        self.testfix_iters = testfix_iters
        self.rib_dep_tool = rib_dep_tool

        _set_tool_text_content_limit(120000)

        self._llms: list[LLM] = [main_llm, judge_llm]
        self._runtimes: list[AgentRuntime] = []

        self._shared_runtime = AgentRuntime(
            workspace_dir,
            run_dir / "shared_runtime",
            runtime_cfg,
            agent_tool_configs=SUB_AGENT_TOOLS,
        )
        self._runtimes.append(self._shared_runtime)

        self._register_all_tool_factories()

    def _register_all_tool_factories(self) -> None:
        runtime = self._shared_runtime
        judge_llm = self.judge_llm
        workspace_dir = self.workspace_dir
        test_output_dir = self.workspace_dir / "test_outputs"
        bus = self.message_bus
        orchestrator = self.orchestrator

        def _delegate_factory(conv_state):
            return list(_DelegateTool.create(conv_state, max_children=20))
        register_tool("DelegateTool", _delegate_factory)

        # --- Architecture ---
        rib_dep_tool = self.rib_dep_tool
        dep_runtime = runtime
        if rib_dep_tool:
            from common.config import get_rib_dep_runtime_cfg
            override_cfg = get_rib_dep_runtime_cfg()
            if override_cfg is not None:
                dep_runtime = AgentRuntime(
                    self.workspace_dir,
                    self.run_dir / "rib_dep_runtime",
                    override_cfg,
                    agent_tool_configs=SUB_AGENT_TOOLS,
                )
                self._runtimes.append(dep_runtime)

        def gen_arch_factory(conv_state):
            if rib_dep_tool:
                return list(GenerateArchitectureDepTool.create(conv_state, runtime=dep_runtime))
            else:
                return list(GenerateArchitectureTool.create(conv_state, runtime=runtime))
        register_tool("GenerateArchitectureTool", gen_arch_factory)

        # --- Judge architecture ---
        def judge_arch_factory(conv_state):
            return list(JudgeArchitectureTool.create(conv_state, llm=judge_llm, workspace_dir=workspace_dir))
        register_tool("JudgeArchitectureTool", judge_arch_factory)

        # --- Partition into groups (composite: partition + init task list + spawn agents) ---
        task_list_path = workspace_dir / "architecture" / "task_list.json"
        def partition_factory(conv_state):
            return list(PartitionIntoGroupsTool.create(
                conv_state,
                workspace_dir=workspace_dir,
                orchestrator=orchestrator,
                message_bus=bus,
                task_list_path=task_list_path,
            ))
        register_tool("PartitionIntoGroupsTool", partition_factory)

        # --- Shared task list (dependency-aware scheduling) ---
        def shared_task_list_factory(conv_state):
            return list(SharedTaskListTool.create(
                conv_state,
                task_list_path=workspace_dir / "architecture" / "task_list.json",
                orchestrator=orchestrator,
            ))
        register_tool("SharedTaskListTool", shared_task_list_factory)

        # --- Execution tools ---
        def run_tests_factory(conv_state):
            return list(ParallelRunTestsTool.create(conv_state, workspace_dir=workspace_dir, output_dir=test_output_dir, max_runs=self.testfix_iters))
        register_tool("RunTestsTool", run_tests_factory)

        def compile_check_factory(conv_state):
            return list(CompileCheckTool.create(conv_state, workspace_dir=workspace_dir))
        register_tool("CompileCheckTool", compile_check_factory)

        # --- Communication tools ---
        def send_to_agent_factory(conv_state):
            return list(SendToAgentTool.create(conv_state, message_bus=bus))
        register_tool("SendToAgentTool", send_to_agent_factory)

        def agent_manager_factory(conv_state):
            return list(AgentManagerTool.create(conv_state, orchestrator=orchestrator, message_bus=bus))
        register_tool("AgentManagerTool", agent_manager_factory)

        def yield_turn_factory(conv_state):
            return list(YieldTurnTool.create(conv_state))
        register_tool("YieldTurnTool", yield_turn_factory)

        # --- Read RIB (deterministic lookup, no LLM) ---
        def read_rib_factory(conv_state):
            return list(ReadRIBTool.create(conv_state, workspace_dir=workspace_dir))
        register_tool("ReadRIBTool", read_rib_factory)

    def create_agent(self, name: str, role: str) -> Agent:
        from common.prompt_constants import load_implementation_integrity_rules
        integrity_rules = load_implementation_integrity_rules(self.workspace_dir)
        if role == "leader":
            tools = _LEADER_STANDARD_TOOLS + _LEADER_CUSTOM_TOOLS
            system_prompt = leader_system_prompt(testfix_iters=self.testfix_iters, integrity_rules=integrity_rules)
        elif role == "group":
            tools = _STANDARD_TOOLS + _GROUP_CUSTOM_TOOLS
            system_prompt = group_agent_system_prompt(integrity_rules=integrity_rules)
        else:
            raise ValueError(f"Unknown role: {role}")

        agent_comp_dir = self.run_dir / name / "logs" / "completions"
        agent_comp_dir.mkdir(parents=True, exist_ok=True)
        agent_llm = self.main_llm.model_copy(
            update={"usage_id": name, "log_completions_folder": str(agent_comp_dir)}
        )
        self._llms.append(agent_llm)

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
        all_llms = list(self._llms)
        for rt in self._runtimes:
            all_llms.extend(rt._llms)
        seen_metrics: set[int] = set()
        unique: list[LLM] = []
        for llm in all_llms:
            mid = id(llm.metrics)
            if mid not in seen_metrics:
                seen_metrics.add(mid)
                unique.append(llm)
        return unique

    def close(self) -> None:
        for rt in self._runtimes:
            try:
                rt.close()
            except Exception:
                pass
        self._runtimes.clear()
