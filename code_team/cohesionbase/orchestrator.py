"""Cohesion orchestrator — thin wrapper over parallelbase.orchestrator.

Only overrides the AgentFactory import to use cohesionbase's version.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.config import RuntimeConfig
from parallelbase.orchestrator import ParallelOrchestrator, AgentHandle
from parallelbase.message_bus import MessageBus, ThreadSafeDict, FutureTracker
from cohesionbase.agent_factory import AgentFactory


class CohesionOrchestrator(ParallelOrchestrator):
    """ParallelOrchestrator with cohesionbase's AgentFactory."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        run_dir: Path,
        runtime_cfg: RuntimeConfig,
        tracker=None,
        testfix_iters: int = 10,
        rib_dep_tool: bool = False,
        no_visualizer: bool = False,
    ):
        self.workspace_dir = workspace_dir.resolve()
        self.run_dir = run_dir
        self.runtime_cfg = runtime_cfg
        self.tracker = tracker
        self.testfix_iters = testfix_iters
        self.no_visualizer = no_visualizer

        self.bus = MessageBus()
        self.agents: ThreadSafeDict[str, AgentHandle] = ThreadSafeDict()
        self.running = False
        self._executor = ThreadPoolExecutor(max_workers=runtime_cfg.max_workers, thread_name_prefix="push")
        self._futures = FutureTracker()

        self.main_llm = self._build_llm("main")
        self.judge_llm = self._build_llm("judge", stream=False)

        self.factory = AgentFactory(
            workspace_dir=self.workspace_dir,
            run_dir=self.run_dir,
            runtime_cfg=runtime_cfg,
            main_llm=self.main_llm,
            judge_llm=self.judge_llm,
            message_bus=self.bus,
            orchestrator=self,
            testfix_iters=self.testfix_iters,
            rib_dep_tool=rib_dep_tool,
        )

        self._router_thread: threading.Thread | None = None
        self._deadlock_thread: threading.Thread | None = None
        self._leader_done = threading.Event()
        self._finished = threading.Event()
        self._deadlock_wakeups = 0
