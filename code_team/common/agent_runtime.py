"""Generic sub-agent executor.

Spins up an independent LocalConversation per run_agent() call.
Each caller passes its own agent_tool_configs so the runtime is
not tied to any package-specific tool configuration.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from openhands.sdk import Agent, LLM, LocalConversation, Tool
from openhands.sdk.context import AgentContext
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.conversation.response_utils import get_agent_final_response

# Register standard tools
import openhands.tools.terminal  # noqa: F401
import openhands.tools.file_editor  # noqa: F401
import openhands.tools.apply_patch  # noqa: F401
import openhands.tools.task_tracker  # noqa: F401
import openhands.tools.glob  # noqa: F401
import openhands.tools.grep  # noqa: F401

from common.config import (
    RuntimeConfig,
    TERMINAL_NO_CHANGE_TIMEOUT_SECONDS,
    get_aws_region,
    get_portkey_kwargs,
    get_sampling_params,
    is_bedrock_model,
)


# Re-export from canonical location for backward compatibility
from common.sdk_patches import _set_tool_text_content_limit  # noqa: F401


class AgentRuntime:
    """Generic sub-agent executor.

    Each run_agent() call creates a brand-new LocalConversation — no
    history accumulates across calls, so token usage stays bounded.

    Args:
        workspace_dir: Project workspace (agents' working directory).
        run_dir: Run output directory (logs, completions, etc.).
        runtime_cfg: RuntimeConfig or model name string.
        agent_tool_configs: Maps agent_name → list of allowed tool names.
            If None, agents run with no tools (prompt-only).
    """

    def __init__(
        self,
        workspace_dir: Path,
        run_dir: Path,
        runtime_cfg: RuntimeConfig | str,
        agent_tool_configs: dict[str, list[str]] | None = None,
    ):
        if isinstance(runtime_cfg, str):
            runtime_cfg = RuntimeConfig(model_name=runtime_cfg)

        self.workspace_dir = workspace_dir.resolve()
        self.run_dir = run_dir
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
        self.max_iteration_per_run = 120

        self.completions_dir = self.run_dir / "logs" / "completions"
        self.completions_dir.mkdir(parents=True, exist_ok=True)

        # agent_name → allowed tool names (None means any/no tools)
        self.agent_tool_configs: dict[str, list[str]] = agent_tool_configs or {}

        self._llms: list[LLM] = []
        self._agent_call_count: dict[str, int] = {}
        self._call_count_lock = threading.Lock()

    def _build_llm(
        self,
        usage_id: str = "agent",
        completions_dir: Path | None = None,
        timeout_sec: int | None = None,
    ) -> LLM:
        litellm_extra_body: dict[str, object] = {}
        if isinstance(self.model_name, str) and "deepseek" in self.model_name.lower():
            litellm_extra_body["max_tokens"] = self.max_output_tokens

        log_dir = completions_dir or self.completions_dir

        llm_kwargs: dict[str, object] = dict(
            usage_id=usage_id,
            model=self.model_name,
            num_retries=self.num_retries,
            stream=os.environ.get("LLM_STREAM", "true").strip().lower() != "false",
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

        llm = LLM(**llm_kwargs)
        self._llms.append(llm)
        return llm

    def _build_agent(
        self,
        allowed_tools: list[str] | None,
        completions_dir: Path | None = None,
        timeout_sec: int | None = None,
    ) -> Agent:
        llm = self._build_llm(completions_dir=completions_dir, timeout_sec=timeout_sec)
        tools = [
            Tool(name=t, params={"no_change_timeout_seconds": TERMINAL_NO_CHANGE_TIMEOUT_SECONDS}) if t == "terminal" else Tool(name=t)
            for t in allowed_tools
        ] if allowed_tools else []
        condenser_llm = llm.model_copy(update={"usage_id": "condenser", "stream": False})
        assert not condenser_llm.stream, "Condenser LLM must not use streaming (model_copy failed to set stream=False)"
        condenser = LLMSummarizingCondenser(
            llm=condenser_llm,
            max_size=300,
            keep_first=2,
        )
        return Agent(
            llm=llm,
            tools=tools,
            condenser=condenser,
            agent_context=AgentContext(),
            system_prompt_kwargs={"cli_mode": True},
        )

    def run_agent(
        self,
        agent_name: str,
        prompt: str,
        allowed_tools: list[str] | None = None,
        timeout_sec: int | None = None,
    ) -> str:
        """Run a prompt in a fresh isolated conversation.

        Args:
            agent_name: Logical name used for logging/persistence paths.
            prompt: User message to send to the agent.
            allowed_tools: Tool list override. If None, falls back to
                agent_tool_configs[agent_name] (or no tools if not configured).
        """
        tools = allowed_tools if allowed_tools is not None else self.agent_tool_configs.get(agent_name)

        with self._call_count_lock:
            call_count = self._agent_call_count.get(agent_name, 0) + 1
            self._agent_call_count[agent_name] = call_count

        # Sub-agents log completions to their own directory to avoid
        # mixing with main agent completions in progress tracking.
        sub_completions_dir = self.completions_dir / f"{agent_name}_{call_count}"
        sub_completions_dir.mkdir(parents=True, exist_ok=True)

        agent = self._build_agent(
            tools,
            completions_dir=sub_completions_dir,
            timeout_sec=timeout_sec if timeout_sec is not None else self.llm_call_timeout_sec,
        )
        persistence_dir = (
            self.run_dir / "logs" / "conversations" / agent_name / f"call_{call_count}"
        )
        persistence_dir.mkdir(parents=True, exist_ok=True)

        def _noop_token_callback(chunk) -> None:
            pass

        conv = LocalConversation(
            agent=agent,
            workspace=str(self.workspace_dir),
            persistence_dir=str(persistence_dir),
            max_iteration_per_run=self.max_iteration_per_run,
            delete_on_close=False,
            visualizer=None,
            token_callbacks=[_noop_token_callback],
        )

        try:
            conv.send_message(prompt, sender="user")
            conv.run()
            return get_agent_final_response(conv.state.events)
        finally:
            try:
                conv.close()
            except Exception:
                pass

    def get_llm_metrics(self) -> dict:
        """Aggregate cost and token usage from all LLM instances."""
        total_cost = 0.0
        total_prompt = 0
        total_completion = 0
        for llm in self._llms:
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
        """No-op — each conversation is closed inside run_agent."""
        pass
