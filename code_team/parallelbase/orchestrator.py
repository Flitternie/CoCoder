"""Parallel Orchestrator — starts Leader, routes messages, manages agent lifecycle.

The Orchestrator is a *pure message router*: it does not maintain any phase state
or completion counters.  All business logic lives in the Leader agent's LLM context.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from openhands.sdk import LLM, LocalConversation
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.tool.builtins.finish import FinishTool

from common.config import RuntimeConfig, is_bedrock_model, get_aws_region, get_portkey_kwargs, get_sampling_params
from parallelbase.tools.yield_turn import YieldTurnTool

from parallelbase.message_bus import MessageBus, ThreadSafeDict, FutureTracker
from parallelbase.agent_factory import AgentFactory

log = logging.getLogger(__name__)

# Maximum iterations per agent conversation run
MAX_ITERATION_PER_RUN = 500

# Deadlock detection: if no messages flow and no futures are in-flight
# for this many seconds after leader's initial run, wake the leader.
DEADLOCK_TIMEOUT_SEC = 15
MAX_DEADLOCK_WAKEUPS = 3

# Task list poller: check for ready tasks and wake agents
POLL_INTERVAL_IDLE = 0.5    # seconds between polls when nothing was sent
POLL_INTERVAL_ACTIVE = 5.0  # seconds to wait after sending a wake message


@dataclass
class AgentHandle:
    """A living agent: its conversation + a lock to serialize run() calls."""
    conversation: LocalConversation
    role: str = ""
    files: list[str] = field(default_factory=list)
    run_lock: threading.Lock = field(default_factory=threading.Lock)


class ParallelOrchestrator:
    """Start Leader, route messages between agents, manage spawning/dismissal.

    Threading model:
      - route_messages() runs in a background thread, polling the MessageBus
      - Each message delivery (_push_to_agent) is submitted to a ThreadPoolExecutor
      - send_message() is called WITHOUT the lock (SDK's internal FIFO lock
        handles thread safety), so messages can be injected even while run()
        is active on another thread
      - Per-agent run_lock serializes run() calls to prevent concurrent LLM
        execution on the same agent
    """

    def __init__(
        self,
        *,
        workspace_dir: Path,
        run_dir: Path,
        runtime_cfg: RuntimeConfig,
        tracker=None,
        testfix_iters: int = 10,
        no_visualizer: bool = False,
    ):
        self.workspace_dir = workspace_dir.resolve()
        self.run_dir = run_dir
        self.runtime_cfg = runtime_cfg
        self.tracker = tracker  # ParallelProgressTracker (optional)
        self.testfix_iters = testfix_iters
        self.no_visualizer = no_visualizer

        # --- Shared resources ---
        self.bus = MessageBus()
        self.agents: ThreadSafeDict[str, AgentHandle] = ThreadSafeDict()
        self.running = False
        self._executor = ThreadPoolExecutor(max_workers=runtime_cfg.max_workers, thread_name_prefix="push")
        self._futures = FutureTracker()  # track in-flight deliveries

        # --- Build LLMs ---
        self.main_llm = self._build_llm("main")
        self.judge_llm = self._build_llm("judge", stream=False)

        # --- Agent factory (creates per-agent AgentRuntime instances) ---
        self.factory = AgentFactory(
            workspace_dir=self.workspace_dir,
            run_dir=self.run_dir,
            runtime_cfg=runtime_cfg,
            main_llm=self.main_llm,
            judge_llm=self.judge_llm,
            message_bus=self.bus,
            orchestrator=self,
            testfix_iters=self.testfix_iters,
        )

        # Background thread handles
        self._router_thread: threading.Thread | None = None
        self._deadlock_thread: threading.Thread | None = None
        # Event to signal when Leader's initial run finishes
        self._leader_done = threading.Event()
        # Event to signal when Leader calls finish (the ONLY valid termination signal)
        self._finished = threading.Event()
        # Deadlock detection counter
        self._deadlock_wakeups = 0

    # ------------------------------------------------------------------
    # LLM builder (mirrors CodebaseRuntime._build_llm)
    # ------------------------------------------------------------------

    def _build_llm(self, usage_id: str, stream: bool | None = None) -> LLM:
        if stream is None:
            stream = os.environ.get("LLM_STREAM", "true").strip().lower() != "false"

        cfg = self.runtime_cfg
        model = cfg.model_name
        is_bedrock = is_bedrock_model(model)

        litellm_extra_body: dict[str, object] = {}
        if isinstance(model, str) and "deepseek" in model.lower():
            litellm_extra_body["max_tokens"] = cfg.max_output_tokens

        completions_dir = self.run_dir / "logs" / "completions"
        completions_dir.mkdir(parents=True, exist_ok=True)

        llm_kwargs: dict[str, object] = dict(
            usage_id=usage_id,
            model=model,
            num_retries=cfg.num_retries,
            stream=stream,
            **get_sampling_params(model),
            max_output_tokens=cfg.max_output_tokens,
            litellm_extra_body=litellm_extra_body,
            log_completions=True,
            log_completions_folder=str(completions_dir),
        )
        if cfg.llm_call_timeout_sec:
            llm_kwargs["timeout"] = cfg.llm_call_timeout_sec
        if is_bedrock:
            llm_kwargs["aws_region_name"] = get_aws_region()
        else:
            llm_kwargs["api_key"] = cfg.api_key
        if cfg.base_url:
            llm_kwargs["base_url"] = cfg.base_url

        llm_kwargs.update(get_portkey_kwargs(model))

        return LLM(**llm_kwargs)

    # ------------------------------------------------------------------
    # Conversation creation
    # ------------------------------------------------------------------

    def _create_conversation(self, name: str, agent) -> LocalConversation:
        """Create a LocalConversation for the given agent."""
        persistence_dir = self.run_dir / "logs" / "conversations" / name
        persistence_dir.mkdir(parents=True, exist_ok=True)

        def _noop_cb(chunk):
            pass

        conv_kwargs = dict(
            agent=agent,
            workspace=str(self.workspace_dir),
            persistence_dir=str(persistence_dir),
            max_iteration_per_run=MAX_ITERATION_PER_RUN,
            delete_on_close=False,
            token_callbacks=[_noop_cb],
        )
        if self.no_visualizer:
            conv_kwargs["visualizer"] = None
        return LocalConversation(**conv_kwargs)

    # ------------------------------------------------------------------
    # Agent lifecycle (called by spawn/dismiss tools)
    # ------------------------------------------------------------------

    def spawn_agent(self, name: str, role: str, files: list[str] | None = None) -> None:
        """Create a new agent and register it. Called by AgentManagerExecutor."""
        if name in self.agents:
            raise ValueError(f"Agent '{name}' already exists")

        agent = self.factory.create_agent(name, role)
        conv = self._create_conversation(name, agent)
        # Set the agent name on the conversation so that tool executors
        # (SendToAgentTool, AgentManagerTool) can resolve it dynamically
        # at call time via ``conversation._agent_name``.
        conv._agent_name = name
        #
        self.agents[name] = AgentHandle(conversation=conv, role=role, files=files or [])
        log.info(f"[Orchestrator] Spawned agent '{name}' (role={role}, files={files})")
        if self.tracker:
            self.tracker.agent_spawned(name, role)

    def dismiss_agent(self, name: str) -> None:
        """Close an agent's conversation and remove it. Called by AgentManagerExecutor."""
        handle = self.agents.pop(name, None)
        if handle is None:
            log.warning(f"[Orchestrator] Cannot dismiss unknown agent: {name}")
            return
        try:
            handle.conversation.close()
        except Exception as e:
            log.warning(f"[Orchestrator] Error closing conversation for '{name}': {e}")
        log.info(f"[Orchestrator] Dismissed agent '{name}'")
        if self.tracker:
            self.tracker.agent_dismissed(name)

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    def route_messages(self) -> None:
        """Main routing loop: poll MessageBus → deliver to target agent(s).

        Runs in a background thread.  Pure routing, no business logic.
        """
        log.info("[Orchestrator] Message router started")
        while self.running:
            msg = self.bus.poll(timeout=1.0)
            if msg is None:
                # Exit only when leader has called finish and all deliveries are done
                if self._finished.is_set() and not self._futures.has_active():
                    log.info("[Orchestrator] Leader finished, no in-flight tasks, stopping router")
                    self.running = False
                    break
                continue

            log.info(f"[Orchestrator] Routing message: {msg.from_agent} → {msg.to_agent}")
            if self.tracker:
                self.tracker.message_sent(msg.from_agent, msg.to_agent, msg.content)

            # Broadcast: expand to individual targets
            if msg.to_agent.startswith("broadcast:"):
                role_prefix = msg.to_agent.split(":", 1)[1]
                targets = self.agents.keys_matching(role_prefix)
                if not targets:
                    log.warning(f"[Orchestrator] No agents match broadcast prefix '{role_prefix}'")
                for t in targets:
                    self._push_to_agent(t, msg.from_agent, msg.content)
                continue

            # Direct message
            self._push_to_agent(msg.to_agent, msg.from_agent, msg.content)

        log.info("[Orchestrator] Message router stopped")

    def _submit_agent_run(self, name: str, fn) -> bool:
        """Submit agent work unless shutdown has already started."""
        if not self.running or self._finished.is_set():
            log.info(f"[Orchestrator] Skipping wake for '{name}' during shutdown")
            return False
        try:
            future = self._executor.submit(fn)
            self._futures.add(future)
            return True
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" in str(e):
                log.info(f"[Orchestrator] Ignoring late wake for '{name}' after executor shutdown")
                return False
            raise

    def _push_to_agent(self, target_name: str, from_name: str, content: str) -> None:
        """Submit message delivery to the thread pool.

        send_message() is called OUTSIDE the lock so that messages can be
        injected even while the agent's run() loop is active on another
        thread.  The SDK's internal FIFO lock on ConversationState ensures
        thread-safe event insertion.  run_lock only serializes run() calls.
        """
        handle = self.agents.get(target_name)
        if handle is None:
            log.warning(f"[Orchestrator] Unknown target agent: {target_name}")
            return

        formatted = f"[Message from {from_name}]: {content}"

        def _run():
            try:
                # Inject message without holding run_lock — always succeeds
                # via SDK FIFO lock, even if run() is active on another thread
                handle.conversation.send_message(formatted)
                # Run with auto-continue (nudges on text-only exit)
                self._run_agent(handle, target_name)
                # Detect if leader called finish
                if target_name == "leader":
                    self._detect_leader_finish(handle)
            except Exception as e:
                log.error(f"[Orchestrator] Error running agent '{target_name}': {e}")
                for retry in range(2):
                    log.info(f"[Orchestrator] Retrying agent '{target_name}' ({retry+1}/2)")
                    try:
                        self._run_agent(handle, target_name)
                        break
                    except Exception as e2:
                        log.error(f"[Orchestrator] Retry {retry+1} failed for '{target_name}': {e2}")
                else:
                    log.error("[Orchestrator] All retries exhausted, terminating process")
                    import os, sys
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(1)

        self._submit_agent_run(target_name, _run)

    def _wake_leader_on_deadlock(self) -> None:
        """Send a diagnostic message to leader when deadlock is detected.

        All agents are idle and no messages are in transit — the system
        cannot make progress on its own.  Wake the leader with a generic
        diagnostic so it can inspect state and decide how to continue.

        Submits to the thread pool so the caller (deadlock monitor) is not
        blocked while the leader runs.
        """
        handle = self.agents.get("leader")
        if handle is None:
            return

        agent_names = list(self.agents.keys())
        msg = (
            "[SYSTEM DEADLOCK DETECTED]: You called yield_turn but no agents "
            "are running and no messages are in transit. The system cannot "
            "make progress. "
            f"Registered agents: {agent_names}. "
            "Action required: inspect your pending work, ensure every agent "
            "you are waiting for has actually been given a task, and continue."
        )

        def _run():
            try:
                handle.conversation.send_message(msg)
                self._run_agent(handle, "leader")
                self._detect_leader_finish(handle)
            except Exception as e:
                log.error(f"[Orchestrator] Failed to wake leader on deadlock: {e}")

        self._submit_agent_run("leader", _run)

    # ------------------------------------------------------------------
    # Deadlock detection (independent monitor thread)
    # ------------------------------------------------------------------

    def _has_progressable_tasks(self) -> bool:
        """Check task_list.json for tasks that can still make progress.

        Returns True if any task is 'ready' or 'in_progress' — meaning
        work is happening or can be picked up by the poller.  Tasks stuck
        in 'pending' (e.g. circular deps) don't count.
        """
        import json as _json
        task_list_path = self.workspace_dir / "architecture" / "task_list.json"
        try:
            if task_list_path.exists():
                data = _json.loads(task_list_path.read_text(encoding="utf-8"))
                for t in data.get("tasks", {}).values():
                    if t.get("status") in ("ready", "in_progress"):
                        return True
        except Exception:
            pass
        return False

    def _deadlock_monitor(self) -> None:
        """Periodically check if the system is stuck and wake the leader.

        Runs in its own daemon thread, independent of message routing.
        Condition: bus empty + no in-flight futures + no tasks in_progress +
        leader initial run done + not finished.  If this holds for
        DEADLOCK_TIMEOUT_SEC consecutive seconds, the system is stuck.
        """
        log.info("[Orchestrator] Deadlock monitor started")
        idle_since: float | None = None

        # Wait until leader's initial run completes before monitoring
        self._leader_done.wait()

        while self.running and not self._finished.is_set():
            time.sleep(1.0)

            if self._futures.has_active() or not self.bus.empty:
                # Work is happening — reset timer
                idle_since = None
                continue

            if self._has_progressable_tasks():
                idle_since = None
                continue

            # System is idle
            if idle_since is None:
                idle_since = time.monotonic()
                continue

            elapsed = time.monotonic() - idle_since
            if elapsed < DEADLOCK_TIMEOUT_SEC:
                continue

            # --- Deadlock confirmed ---
            if self._deadlock_wakeups < MAX_DEADLOCK_WAKEUPS:
                self._deadlock_wakeups += 1
                log.warning(
                    f"[Orchestrator] Potential deadlock detected "
                    f"(idle {elapsed:.0f}s, wakeup "
                    f"{self._deadlock_wakeups}/{MAX_DEADLOCK_WAKEUPS})"
                )
                self._wake_leader_on_deadlock()
                idle_since = None  # reset, give leader time to react
            else:
                log.error(
                    "[Orchestrator] Max deadlock wakeups reached, "
                    "forcing termination"
                )
                self._finished.set()
                self.running = False

        log.info("[Orchestrator] Deadlock monitor stopped")

    # ------------------------------------------------------------------
    # Task list poller — auto-wake agents when tasks become ready
    # ------------------------------------------------------------------

    def _task_list_poller(self) -> None:
        """Poll task_list.json for ready+unclaimed tasks and wake the owner agent.

        Runs in a daemon thread. If no task list file exists, sleeps and retries.
        After sending wake messages, backs off to POLL_INTERVAL_ACTIVE to avoid
        spamming agents. Otherwise polls at POLL_INTERVAL_IDLE.
        """
        import json as _json

        task_list_path = self.workspace_dir / "architecture" / "task_list.json"
        log.info("[Orchestrator] Task list poller started")

        self._leader_done.wait()

        while self.running and not self._finished.is_set():
            sent_any = False
            try:
                if task_list_path.exists():
                    raw = task_list_path.read_text(encoding="utf-8")
                    data = _json.loads(raw)
                    tasks = data.get("tasks", {})

                    for tid, task in tasks.items():
                        status = task.get("status")

                        # --- Wake agents for ready + unclaimed tasks ---
                        if status == "ready" and not task.get("claimed_by"):
                            owner = task.get("owner")
                            if owner:
                                if owner in self.agents:
                                    self._push_to_agent(
                                        owner, "system",
                                        f"Task '{tid}' is ready. "
                                        f"Claim it: shared_task_list(command=\"claim\", task_id=\"{tid}\")"
                                    )
                                    sent_any = True
                            else:
                                for agent_name in list(self.agents.keys()):
                                    if agent_name == "leader":
                                        continue
                                    self._push_to_agent(
                                        agent_name, "system",
                                        f"Task '{tid}' is ready (unassigned). "
                                        f"Claim it: shared_task_list(command=\"claim\", task_id=\"{tid}\")"
                                    )
                                if len(self.agents) > 1:
                                    sent_any = True
                            continue

                        # --- Remind paused agents about their in_progress tasks ---
                        if status == "in_progress":
                            claimed_by = task.get("claimed_by")
                            if not claimed_by:
                                continue
                            handle = self.agents.get(claimed_by)
                            if handle is None:
                                continue
                            if handle.conversation.state.execution_status == ConversationExecutionStatus.PAUSED:
                                self._push_to_agent(
                                    claimed_by, "system",
                                    f"You still have task '{tid}' in_progress. "
                                    f"Please complete it: shared_task_list(command=\"complete\", task_id=\"{tid}\") "
                                    f"or continue working on it."
                                )
                                sent_any = True
            except Exception as e:
                log.debug(f"[Orchestrator] Task list poller error: {e}")

            sleep_time = POLL_INTERVAL_ACTIVE if sent_any else POLL_INTERVAL_IDLE
            time.sleep(sleep_time)

        log.info("[Orchestrator] Task list poller stopped")

    def _push_to_leader_error(self, error_content: str) -> None:
        """Send an error notification to the leader (best-effort)."""
        handle = self.agents.get("leader")
        if handle is None:
            return
        try:
            handle.conversation.send_message(f"[SYSTEM ERROR]: {error_content}")
            self._run_agent(handle, "leader")
        except Exception as e:
            log.error(f"[Orchestrator] Failed to notify leader of error: {e}")

    def _detect_leader_finish(self, handle: AgentHandle) -> None:
        """Check if leader's last action was finish. If so, signal termination."""
        try:
            for event in reversed(handle.conversation.state.events):
                if (
                    isinstance(event, ActionEvent)
                    and event.source == "agent"
                    and event.tool_name == FinishTool.name
                ):
                    log.info("[Orchestrator] Leader called finish, signaling termination")
                    self._finished.set()
                    return
                # Only inspect the last agent event
                if isinstance(event, ActionEvent) and event.source == "agent":
                    return
        except Exception as e:
            log.warning(f"[Orchestrator] Could not inspect leader events: {e}")

    def _has_unprocessed_messages(self, handle: AgentHandle) -> bool:
        """Check if there are user messages the agent hasn't responded to yet.

        Walks backward through events.  Returns True as soon as a user
        MessageEvent is found that is more recent than the last agent
        action or agent message (meaning the LLM hasn't seen it yet).
        """
        for event in reversed(handle.conversation.state.events):
            if isinstance(event, MessageEvent) and event.source == "user":
                return True  # user message found before any agent response
            if isinstance(event, ActionEvent) and event.source == "agent":
                return False  # agent already acted after last user msg
            if isinstance(event, MessageEvent) and event.source == "agent":
                return False  # agent already responded
            # ObservationEvents, system events, etc.: keep scanning
        return False

    def _agent_should_continue(self, handle: AgentHandle) -> bool:
        """Check if agent exited run() with text output and should be nudged.

        Returns True if the agent's last events show a text message (not a
        tool call).  Exempt: YieldTurnTool and FinishTool are legitimate
        pause/stop signals — do NOT nudge after those.
        """
        try:
            from openhands.sdk.event import ObservationEvent
            saw_agent_text = False
            for event in reversed(handle.conversation.state.events):
                # Skip agent text messages — mark that we saw one
                if isinstance(event, MessageEvent) and event.source == "agent":
                    saw_agent_text = True
                    continue
                # Skip observation events (tool results) — no source attr
                if isinstance(event, ObservationEvent):
                    continue
                # Found the last agent tool call before the text
                if isinstance(event, ActionEvent) and event.source == "agent":
                    if not saw_agent_text:
                        return False  # run() ended after tool call, not text
                    return event.tool_name not in (YieldTurnTool.name, FinishTool.name)
                # Hit a non-agent event (user message, system event, etc.)
                # If we saw agent text but no preceding tool call → nudge
                return saw_agent_text
        except Exception as e:
            log.warning(f"[Orchestrator] Could not inspect events: {e}")
        return False

    def _run_agent(self, handle: AgentHandle, agent_name: str) -> None:
        """Run an agent with auto-continue: nudge up to 3 times on text-only exit.

        If the agent's run() exits because the LLM produced text-only output
        (no tool call), inject a system message and re-invoke run().  This
        prevents deadlocks caused by the LLM failing to call the next tool.

        Guards against redundant run() calls: when multiple _push_to_agent
        threads inject messages concurrently, one thread's run() may consume
        all pending messages.  Threads that subsequently acquire run_lock
        check _has_unprocessed_messages and skip run() if their message was
        already processed, preventing wasted LLM calls and stuck_detector
        false positives.
        """
        with handle.run_lock:
            # Skip if another thread's run() already processed our message
            if not self._has_unprocessed_messages(handle):
                log.debug(
                    f"[Orchestrator] No unprocessed messages for '{agent_name}', "
                    f"skipping redundant run()"
                )
                return
            # Reset STUCK status if stuck_detector triggered (e.g. after
            # empty LLM responses).  Without this, run() is a no-op because
            # STUCK is not in the SDK's [IDLE, PAUSED, ERROR] → RUNNING list.
            if handle.conversation.state.execution_status == ConversationExecutionStatus.STUCK:
                log.info(f"[Orchestrator] Resetting STUCK status for '{agent_name}'")
                handle.conversation.state.execution_status = ConversationExecutionStatus.IDLE
            handle.conversation.run()
        for nudge_i in range(3):
            if not self._agent_should_continue(handle):
                break
            log.info(
                f"[Orchestrator] Agent '{agent_name}' "
                f"exited with text, nudge {nudge_i + 1}/3"
            )
            handle.conversation.send_message(
                "[System] Your turn ended with a text message instead of "
                "a tool call. Resume your workflow and call the next "
                "required tool now."
            )
            with handle.run_lock:
                if handle.conversation.state.execution_status == ConversationExecutionStatus.STUCK:
                    log.info(f"[Orchestrator] Resetting STUCK status for '{agent_name}' before nudge")
                    handle.conversation.state.execution_status = ConversationExecutionStatus.IDLE
                handle.conversation.run()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def start(self, prompt: str) -> None:
        """Create the Leader, send initial prompt, start routing, and block until done.

        Args:
            prompt: The initial user prompt for the Leader agent.
        """
        # 1. Create Leader agent
        self.spawn_agent("leader", "leader")
        leader_handle = self.agents["leader"]

        # 2. Start router and deadlock monitor in background
        self.running = True
        self._router_thread = threading.Thread(
            target=self.route_messages, daemon=True, name="msg-router"
        )
        self._router_thread.start()
        self._deadlock_thread = threading.Thread(
            target=self._deadlock_monitor, daemon=True, name="deadlock-monitor"
        )
        self._deadlock_thread.start()
        self._poller_thread = threading.Thread(
            target=self._task_list_poller, daemon=True, name="task-list-poller"
        )
        self._poller_thread.start()

        # 3. Send initial prompt and run Leader
        log.info("[Orchestrator] Sending initial prompt to Leader")
        try:
            leader_handle.conversation.send_message(prompt, sender="user")
            self._run_agent(leader_handle, "leader")
        except Exception as e:
            log.error(f"[Orchestrator] Leader initial run failed: {e}")
            self.running = False
            raise

        log.info("[Orchestrator] Leader initial run() completed")
        self._leader_done.set()
        # Check if leader finished in the initial run (unlikely but safe)
        leader_handle_check = self.agents.get("leader")
        if leader_handle_check:
            self._detect_leader_finish(leader_handle_check)

        # 4. Wait for router to finish (all messages delivered, all agents dismissed)
        if self._router_thread:
            self._router_thread.join(timeout=3600)  # 1 hour max
        if self._deadlock_thread:
            self._deadlock_thread.join(timeout=5)
        if self._poller_thread:
            self._poller_thread.join(timeout=5)

        # 5. Shut down the thread pool (waits for in-flight deliveries)
        self._executor.shutdown(wait=True)

        log.info("[Orchestrator] Orchestration complete")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop routing and close all conversations."""
        self.running = False
        self._finished.set()
        if self._deadlock_thread and self._deadlock_thread.is_alive():
            self._deadlock_thread.join(timeout=5)
        if hasattr(self, '_poller_thread') and self._poller_thread and self._poller_thread.is_alive():
            self._poller_thread.join(timeout=5)
        if self._router_thread and self._router_thread.is_alive():
            self._router_thread.join(timeout=10)
        self._executor.shutdown(wait=False)

        for name, handle in list(self.agents.items()):
            try:
                handle.conversation.close()
            except Exception:
                pass
        self.agents.clear()
        self.factory.close()
        log.info("[Orchestrator] All resources closed")

    def get_llm_metrics(self) -> dict:
        """Aggregate cost and token usage from all LLM instances."""
        all_llms = self.factory.get_all_llms()
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
