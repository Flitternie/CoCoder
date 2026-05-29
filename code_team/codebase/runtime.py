"""Autonomous runtime: single main Agent with 7 custom tools + standard tools.

Internally uses SerialRuntime for generative tool sub-agents and a separate
LLM instance for judge tool evaluations.
"""
from __future__ import annotations

import os
from pathlib import Path

from openhands.sdk import Agent, LLM, LocalConversation, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context import AgentContext
from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.sdk.tool import register_tool

# Register standard tools (terminal timeout hard-cap applied via common/sdk_patches.py)
import openhands.tools.terminal  # noqa: F401
import openhands.tools.file_editor  # noqa: F401
import openhands.tools.apply_patch  # noqa: F401
import openhands.tools.glob  # noqa: F401
import openhands.tools.grep  # noqa: F401

# Load custom tool modules
import codebase.tools  # noqa: F401

from codebase.config import MAX_ITERATION_PER_RUN, SUB_AGENT_TOOLS, is_bedrock_model, get_aws_region
from common.config import TERMINAL_NO_CHANGE_TIMEOUT_SECONDS, get_portkey_kwargs, get_sampling_params
from codebase.system_prompt import build_system_prompt
from common.agent_runtime import AgentRuntime


from common.agent_runtime import _set_tool_text_content_limit


class CodebaseRuntime:
    """Runtime for the codebase agent with 7 custom tools."""

    def __init__(self, workspace_dir: Path, run_dir: Path, runtime_cfg, *, testfix_iters: int = 10, rib_dep_tool: bool = False, no_visualizer: bool = False):
        from common.config import RuntimeConfig

        if isinstance(runtime_cfg, str):
            runtime_cfg = RuntimeConfig(model_name=runtime_cfg)

        self.workspace_dir = workspace_dir.resolve()
        self.run_dir = run_dir
        self.testfix_iters = testfix_iters
        self.model_name = runtime_cfg.model_name
        self.api_key = runtime_cfg.api_key
        self.base_url = runtime_cfg.base_url
        self.num_retries = runtime_cfg.num_retries
        self.llm_call_timeout_sec = runtime_cfg.llm_call_timeout_sec
        self.is_bedrock = is_bedrock_model(self.model_name)
        self.aws_region = get_aws_region() if self.is_bedrock else ""
        self.tool_text_content_limit = 120000
        _set_tool_text_content_limit(self.tool_text_content_limit)
        self.max_output_tokens = runtime_cfg.max_output_tokens
        self.rib_dep_tool = rib_dep_tool
        self.no_visualizer = no_visualizer

        self.completions_dir = self.run_dir / "logs" / "completions"
        self.completions_dir.mkdir(parents=True, exist_ok=True)

        # Internal AgentRuntime for generative tool sub-agents
        self.sub_agent_runtime = AgentRuntime(workspace_dir, run_dir, runtime_cfg, agent_tool_configs=SUB_AGENT_TOOLS)

        # Separate LLM for judge tools — uses stream=False because judges
        # call llm.completion() directly (not through a Conversation).
        self.judge_llm = self._build_llm(usage_id="judge", timeout_sec=self.llm_call_timeout_sec, stream=False)

        # Main agent LLM
        self.main_llm = self._build_llm(usage_id="main", timeout_sec=self.llm_call_timeout_sec)

        # Collect all LLMs for metrics
        self._llms: list[LLM] = [self.main_llm, self.judge_llm]

        # Register custom tool factories with runtime-specific dependencies
        self._register_custom_tool_factories()

        # Build agent and conversation
        self.agent = self._build_main_agent()
        self.conversation: LocalConversation | None = None

    def _build_llm(
        self,
        usage_id: str = "agent",
        completions_dir: Path | None = None,
        timeout_sec: int | None = None,
        stream: bool | None = None,
    ) -> LLM:
        if stream is None:
            stream = os.environ.get("LLM_STREAM", "true").strip().lower() != "false"

        litellm_extra_body: dict[str, object] = {}
        if isinstance(self.model_name, str) and "deepseek" in self.model_name.lower():
            litellm_extra_body["max_tokens"] = self.max_output_tokens

        log_dir = completions_dir or self.completions_dir

        llm_kwargs: dict[str, object] = dict(
            usage_id=usage_id,
            model=self.model_name,
            num_retries=self.num_retries,
            stream=stream,
            **get_sampling_params(self.model_name),
            max_output_tokens=self.max_output_tokens,
            litellm_extra_body=litellm_extra_body,
            log_completions=True,
            log_completions_folder=str(log_dir),
        )
        if timeout_sec is not None:
            llm_kwargs["timeout"] = timeout_sec

        if self.is_bedrock:
            llm_kwargs["aws_region_name"] = self.aws_region
        else:
            llm_kwargs["api_key"] = self.api_key

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        llm_kwargs.update(get_portkey_kwargs(self.model_name))

        return LLM(**llm_kwargs)

    def _register_custom_tool_factories(self) -> None:
        """Override registered factories to inject runtime dependencies."""
        runtime = self.sub_agent_runtime
        judge_llm = self.judge_llm
        workspace_dir = self.workspace_dir
        test_output_dir = self.workspace_dir / "test_outputs"

        # Generative tools need SerialRuntime
        rib_dep_tool = self.rib_dep_tool
        dep_runtime = runtime
        if rib_dep_tool:
            from common.config import get_rib_dep_runtime_cfg
            override_cfg = get_rib_dep_runtime_cfg()
            if override_cfg is not None:
                from codebase.config import SUB_AGENT_TOOLS as _SUB_AGENT_TOOLS
                dep_runtime = AgentRuntime(
                    self.workspace_dir,
                    self.run_dir / "rib_dep_runtime",
                    override_cfg,
                    agent_tool_configs=_SUB_AGENT_TOOLS,
                )
                self._dep_runtime = dep_runtime

        def gen_arch_factory(conv_state):
            if rib_dep_tool:
                from parallelbase.tools.generate_architecture_dep import GenerateArchitectureDepTool
                return list(GenerateArchitectureDepTool.create(conv_state, runtime=dep_runtime))
            else:
                from codebase.tools.generate_architecture import GenerateArchitectureTool
                return list(GenerateArchitectureTool.create(conv_state, runtime=runtime))

        def gen_skel_factory(conv_state):
            from codebase.tools.generate_rib import GenerateRIBTool
            return list(GenerateRIBTool.create(conv_state, runtime=runtime))

        def gen_code_factory(conv_state):
            from codebase.tools.generate_code import GenerateCodeTool
            return list(GenerateCodeTool.create(conv_state, runtime=runtime))

        # Judge tools need LLM + workspace
        def judge_arch_factory(conv_state):
            from codebase.tools.judge_architecture import JudgeArchitectureTool
            return list(JudgeArchitectureTool.create(conv_state, llm=judge_llm, workspace_dir=workspace_dir))

        def judge_skel_factory(conv_state):
            from codebase.tools.judge_rib import JudgeRIBTool
            return list(JudgeRIBTool.create(conv_state, llm=judge_llm, workspace_dir=workspace_dir))

        # Execution tools
        def run_tests_factory(conv_state):
            from codebase.tools.run_tests import RunTestsTool
            return list(RunTestsTool.create(conv_state, workspace_dir=workspace_dir, output_dir=test_output_dir, max_runs=self.testfix_iters))

        def compile_check_factory(conv_state):
            from codebase.tools.compile_check import CompileCheckTool
            return list(CompileCheckTool.create(conv_state, workspace_dir=workspace_dir))

        # Re-register with injected dependencies
        register_tool("GenerateArchitectureTool", gen_arch_factory)
        register_tool("GenerateRIBTool", gen_skel_factory)
        register_tool("GenerateCodeTool", gen_code_factory)
        register_tool("JudgeArchitectureTool", judge_arch_factory)
        register_tool("JudgeRIBTool", judge_skel_factory)
        register_tool("RunTestsTool", run_tests_factory)
        register_tool("CompileCheckTool", compile_check_factory)

    def _build_main_agent(self) -> Agent:
        tools = [
            # Standard tools
            Tool(name="terminal", params={"no_change_timeout_seconds": TERMINAL_NO_CHANGE_TIMEOUT_SECONDS}),
            Tool(name="file_editor"),
            Tool(name="apply_patch"),
            Tool(name="glob"),
            Tool(name="grep"),
            # Custom tools
            Tool(name="GenerateArchitectureTool"),
            Tool(name="GenerateRIBTool"),
            Tool(name="GenerateCodeTool"),
            Tool(name="JudgeArchitectureTool"),
            Tool(name="JudgeRIBTool"),
            Tool(name="RunTestsTool"),
            Tool(name="CompileCheckTool"),
        ]

        condenser_llm = self.main_llm.model_copy(update={"usage_id": "condenser", "stream": False})
        assert not condenser_llm.stream, "Condenser LLM must not use streaming (model_copy failed to set stream=False)"
        condenser = LLMSummarizingCondenser(
            llm=condenser_llm,
            max_size=300,
            keep_first=2,
        )

        # Inject workflow instructions as system prompt suffix
        from common.prompt_constants import load_implementation_integrity_rules
        integrity_rules = load_implementation_integrity_rules(self.workspace_dir)
        ctx = AgentContext(system_message_suffix=build_system_prompt(testfix_iters=self.testfix_iters, integrity_rules=integrity_rules))
        return Agent(
            llm=self.main_llm,
            tools=tools,
            condenser=condenser,
            agent_context=ctx,
            system_prompt_kwargs={"cli_mode": True},
        )

    @staticmethod
    def _noop_token_callback(chunk) -> None:
        """No-op callback required by OpenHands SDK when stream=True."""
        pass

    def create_conversation(self) -> LocalConversation:
        """Create the main agent's conversation for long-running sessions."""
        persistence_dir = self.run_dir / "logs" / "conversations" / "main_agent"
        persistence_dir.mkdir(parents=True, exist_ok=True)

        conv_kwargs = dict(
            agent=self.agent,
            workspace=str(self.workspace_dir.resolve()),
            persistence_dir=str(persistence_dir),
            max_iteration_per_run=MAX_ITERATION_PER_RUN,
            delete_on_close=False,
            token_callbacks=[self._noop_token_callback],
        )
        if self.no_visualizer:
            conv_kwargs["visualizer"] = None
        self.conversation = LocalConversation(**conv_kwargs)
        return self.conversation

    def run(self, prompt: str) -> str:
        """Send prompt and run the codebase agent."""
        if self.conversation is None:
            self.create_conversation()
        self.conversation.send_message(prompt, sender="user")
        self.conversation.run()
        return get_agent_final_response(self.conversation.state.events)

    def get_llm_metrics(self) -> dict:
        """Aggregate cost and token usage from all LLM instances."""
        # Include sub-agent runtime's LLMs too
        all_llms = self._llms + self.sub_agent_runtime._llms
        if hasattr(self, "_dep_runtime"):
            all_llms = all_llms + self._dep_runtime._llms
        total_cost = 0.0
        total_prompt = 0
        total_completion = 0
        for llm in all_llms:
            total_cost += llm.metrics.accumulated_cost
            if llm.metrics.accumulated_token_usage:
                total_prompt += llm.metrics.accumulated_token_usage.prompt_tokens
                total_completion += llm.metrics.accumulated_token_usage.completion_tokens
        return {
            "total_cost": total_cost,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
        }

    def close(self) -> None:
        """Cleanup resources."""
        if self.conversation:
            try:
                self.conversation.close()
            except Exception:
                pass
        self.sub_agent_runtime.close()
        if hasattr(self, "_dep_runtime"):
            self._dep_runtime.close()
