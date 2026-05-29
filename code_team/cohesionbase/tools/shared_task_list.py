"""shared_task_list tool — dependency-aware task scheduling.

A shared state machine that tracks tasks with dependencies.
When a task's dependencies are all completed, it becomes ready.

The tool is a pure scheduler — it only manages: id + deps + status.
It does not know what a task represents (file, module, feature, etc.).
Any registered agent can claim any ready task.
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TYPE_CHECKING

from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor

if TYPE_CHECKING:
    from parallelbase.orchestrator import ParallelOrchestrator


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class SharedTaskListAction(Action):
    command: Literal["init", "claim", "complete", "status"] = Field(
        description="Operation: init (create task list), claim (claim a task), "
                    "complete (mark task done), status (get progress)",
    )
    tasks: list[dict] | None = Field(
        default=None,
        description="For 'init': list of task definitions. "
                    "Each must have 'id' and 'deps'. Optional: 'owner', 'description'. "
                    "Any other fields are stored in 'metadata'.",
    )
    task_id: str | None = Field(
        default=None,
        description="Task identifier. Required for 'claim' and 'complete'. "
                    "Optional for 'status' (query single task).",
    )


class SharedTaskListObservation(Observation):
    result: dict = Field(default_factory=dict)
    error: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"shared_task_list ERROR: {self.error}")]
        return [TextContent(text=json.dumps(self.result, ensure_ascii=False, indent=2))]


# ---------------------------------------------------------------------------
# Task state management
# ---------------------------------------------------------------------------

_STATUS_PENDING = "pending"
_STATUS_READY = "ready"
_STATUS_IN_PROGRESS = "in_progress"
_STATUS_COMPLETED = "completed"

_RESERVED_KEYS = {"id", "description", "owner", "claimed_by", "status",
                  "completed_at", "deps", "blocks", "metadata"}


class _TaskListLock:
    """Context manager for atomic load-modify-save on the task list file."""

    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(".lock")

    def __enter__(self) -> dict:
        _acquire_lock(self.lock_path)
        if not self.path.exists():
            return {"tasks": {}, "created_at": None}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def __exit__(self, exc_type, exc_val, exc_tb):
        _release_lock(self.lock_path)
        return False

    def save(self, data: dict) -> None:
        """Save data while lock is held."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _acquire_lock(lock_path: Path, timeout: float = 5.0, retry_interval: float = 0.05) -> None:
    """Simple file-based lock with retry."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = lock_path.open("x")  # exclusive create — fails if exists
            fd.close()
            return
        except FileExistsError:
            time.sleep(retry_interval)
    # Timeout: force acquire (stale lock)
    try:
        lock_path.unlink(missing_ok=True)
        fd = lock_path.open("x")
        fd.close()
    except Exception:
        raise TimeoutError(f"Cannot acquire lock: {lock_path}")


def _release_lock(lock_path: Path) -> None:
    """Release file lock."""
    lock_path.unlink(missing_ok=True)


def _update_readiness_for(tasks: dict[str, dict], completed_task_id: str) -> list[str]:
    """Check tasks blocked by completed_task_id; transition to ready if all deps done.

    Returns list of task ids that became ready.
    """
    newly_ready = []
    blocked_tasks = tasks[completed_task_id].get("blocks", [])
    for tid in blocked_tasks:
        task = tasks.get(tid)
        if task and task["status"] == _STATUS_PENDING:
            if all(tasks.get(d, {}).get("status") == _STATUS_COMPLETED
                   for d in task.get("deps", [])):
                task["status"] = _STATUS_READY
                newly_ready.append(tid)
    return newly_ready


def _update_readiness(tasks: dict[str, dict]) -> None:
    """Update status of all pending tasks whose deps are all completed."""
    for task in tasks.values():
        if task["status"] == _STATUS_PENDING:
            deps = task.get("deps", [])
            if all(tasks.get(d, {}).get("status") == _STATUS_COMPLETED for d in deps):
                task["status"] = _STATUS_READY


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _cmd_init(task_list_path: Path, task_defs: list[dict]) -> dict:
    """Initialize the shared task list.

    Refuses to overwrite an existing task list — only the first successful init
    takes effect.  Callers that need to re-initialize must delete the file first.
    """
    if task_list_path.exists():
        try:
            existing = json.loads(task_list_path.read_text(encoding="utf-8"))
            if existing.get("tasks"):
                return {
                    "error": "Task list already initialized. "
                             "Use 'status' to check progress. "
                             "Re-initialization is not allowed."
                }
        except (json.JSONDecodeError, OSError):
            pass  # corrupt / empty file — safe to overwrite

    tasks = {}
    for i, td in enumerate(task_defs):
        tid = td.get("id")
        if not tid:
            return {"error": f"Task at index {i} missing 'id' field. Got keys: {list(td.keys())}"}
        deps = td.get("deps", [])
        status = _STATUS_READY if not deps else _STATUS_PENDING

        # Collect metadata: anything not in reserved keys
        metadata = {k: v for k, v in td.items() if k not in _RESERVED_KEYS}

        tasks[tid] = {
            "id": tid,
            "description": td.get("description", ""),
            "owner": td.get("owner"),
            "claimed_by": None,
            "status": status,
            "completed_at": None,
            "deps": deps,
            "blocks": [],  # computed below
            "metadata": metadata,
        }

    # Validate that all dep ids exist in the task list.
    all_ids = set(tasks.keys())
    bad_deps: dict[str, list[str]] = {}
    for tid, task in tasks.items():
        unknown = [d for d in task["deps"] if d not in all_ids]
        if unknown:
            bad_deps[tid] = unknown
    if bad_deps:
        lines = [f"  {tid}: {udeps}" for tid, udeps in bad_deps.items()]
        return {"error": "Unknown dep ids (not in task list):\n" + "\n".join(lines)}

    # Build reverse dependency map: blocks
    for tid, task in tasks.items():
        for dep_id in task["deps"]:
            tasks[dep_id]["blocks"].append(tid)

    data = {
        "tasks": tasks,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    lock = _TaskListLock(task_list_path)
    lock.__enter__()
    try:
        lock.save(data)
    finally:
        lock.__exit__(None, None, None)

    ready_count = sum(1 for t in tasks.values() if t["status"] == _STATUS_READY)
    return {
        "initialized": True,
        "total_tasks": len(tasks),
        "immediately_ready": ready_count,
    }


def _cmd_claim(task_list_path: Path, task_id: str, agent_name: str) -> dict:
    """Claim a task. Atomic load-modify-save.

    - Already claimed by same agent → return it (idempotent resume).
    - Already claimed by different agent → error.
    - Deps not met → return blocked_by list.
    - Ready → transition to in_progress, record claimed_by.
    """
    lock = _TaskListLock(task_list_path)
    data = lock.__enter__()
    try:
        tasks = data.get("tasks", {})
        _update_readiness(tasks)

        if task_id not in tasks:
            return {"error": f"Unknown task: {task_id}"}

        task = tasks[task_id]
        status = task["status"]

        if status == _STATUS_COMPLETED:
            lock.save(data)
            return {"task_id": task_id, "status": "completed"}

        if status == _STATUS_IN_PROGRESS:
            lock.save(data)
            if task["claimed_by"] == agent_name:
                return {"task_id": task_id, "status": "in_progress"}
            return {"error": f"'{task_id}' already claimed by '{task['claimed_by']}'"}

        if status == _STATUS_READY:
            task["status"] = _STATUS_IN_PROGRESS
            task["claimed_by"] = agent_name
            lock.save(data)
            return {"task_id": task_id, "status": "ready"}

        # Pending — deps not met
        lock.save(data)
        blocked_by = [
            d for d in task["deps"]
            if tasks.get(d, {}).get("status") != _STATUS_COMPLETED
        ]
        return {"task_id": task_id, "status": "waiting", "blocked_by": blocked_by}
    finally:
        lock.__exit__(None, None, None)


def _cmd_complete(task_list_path: Path, task_id: str, agent_name: str) -> dict:
    """Mark a task as completed and return newly ready tasks."""
    lock = _TaskListLock(task_list_path)
    data = lock.__enter__()
    try:
        tasks = data.get("tasks", {})

        if task_id not in tasks:
            return {"error": f"Unknown task: {task_id}"}

        task = tasks[task_id]

        if task["status"] == _STATUS_COMPLETED:
            return {"error": f"Already completed: {task_id}"}
        if task["status"] != _STATUS_IN_PROGRESS:
            return {"error": f"Cannot complete '{task_id}': status is '{task['status']}', "
                            f"expected 'in_progress'. Call 'claim' first."}
        if task["claimed_by"] != agent_name:
            return {"error": f"Cannot complete '{task_id}': claimed by '{task['claimed_by']}', "
                            f"not '{agent_name}'"}

        task["status"] = _STATUS_COMPLETED
        task["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Use blocks to find newly unblocked tasks (targeted, not full scan)
        newly_ready_ids = _update_readiness_for(tasks, task_id)
        newly_ready = [
            {"task_id": tid, "owner": tasks[tid].get("owner"),
             "description": tasks[tid].get("description", "")}
            for tid in newly_ready_ids
        ]

        lock.save(data)

        all_done = all(t["status"] == _STATUS_COMPLETED for t in tasks.values())

        return {
            "completed": task_id,
            "newly_ready": newly_ready,
            "all_done": all_done,
        }
    finally:
        lock.__exit__(None, None, None)


def _cmd_status(task_list_path: Path, task_id: str | None = None) -> dict:
    """Get progress. If task_id given, return single task detail; else full summary."""
    lock = _TaskListLock(task_list_path)
    data = lock.__enter__()
    try:
        tasks = data.get("tasks", {})

        # Recalculate readiness in memory (don't persist — read-only)
        _update_readiness(tasks)

        # Single-task query
        if task_id:
            if task_id not in tasks:
                return {"error": f"Unknown task: {task_id}"}
            task = tasks[task_id]
            result = {
                "task_id": task_id,
                "status": task["status"],
                "deps": task["deps"],
                "blocks": task.get("blocks", []),
                "owner": task.get("owner"),
                "description": task.get("description", ""),
                "metadata": task.get("metadata", {}),
            }
            if task["claimed_by"]:
                result["claimed_by"] = task["claimed_by"]
            if task["status"] == _STATUS_PENDING:
                blocked = [d for d in task["deps"]
                           if tasks.get(d, {}).get("status") != _STATUS_COMPLETED]
                if blocked:
                    result["blocked_by"] = blocked
            return result

        # Full summary
        counts = {_STATUS_PENDING: 0, _STATUS_READY: 0, _STATUS_IN_PROGRESS: 0, _STATUS_COMPLETED: 0}
        for task in tasks.values():
            counts[task["status"]] = counts.get(task["status"], 0) + 1

        task_details = []
        for tid, task in tasks.items():
            entry = {"task_id": tid, "status": task["status"]}
            if task.get("owner"):
                entry["owner"] = task["owner"]
            if task["claimed_by"]:
                entry["claimed_by"] = task["claimed_by"]
            if task["status"] == _STATUS_PENDING:
                blocked = [d for d in task["deps"]
                           if tasks.get(d, {}).get("status") != _STATUS_COMPLETED]
                if blocked:
                    entry["blocked_by"] = blocked
            task_details.append(entry)

        return {
            "total": len(tasks),
            "completed": counts[_STATUS_COMPLETED],
            "in_progress": counts[_STATUS_IN_PROGRESS],
            "ready": counts[_STATUS_READY],
            "blocked": counts[_STATUS_PENDING],
            "all_done": counts[_STATUS_COMPLETED] == len(tasks) and len(tasks) > 0,
            "tasks": task_details,
        }
    finally:
        lock.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class SharedTaskListExecutor(ToolExecutor[SharedTaskListAction, SharedTaskListObservation]):
    """Shared task list with dependency-based auto-unlocking."""

    def __init__(self, task_list_path: Path, orchestrator: ParallelOrchestrator):
        self.task_list_path = task_list_path
        self.orchestrator = orchestrator

    def _resolve_agent(self, conversation) -> str | None:
        """Get caller agent name and verify it exists in the registry."""
        name = getattr(conversation, "_agent_name", None)
        if name and name in self.orchestrator.agents:
            return name
        return None

    def __call__(
        self,
        action: SharedTaskListAction,
        conversation=None,
    ) -> SharedTaskListObservation:
        try:
            if action.command == "init":
                if not action.tasks:
                    return SharedTaskListObservation(error="'init' requires 'tasks' parameter")
                result = _cmd_init(self.task_list_path, action.tasks)

            elif action.command == "claim":
                if not action.task_id:
                    return SharedTaskListObservation(error="'claim' requires 'task_id' parameter")
                agent_name = self._resolve_agent(conversation)
                if not agent_name:
                    return SharedTaskListObservation(error="Cannot resolve caller agent. Are you a registered agent?")
                result = _cmd_claim(self.task_list_path, action.task_id, agent_name)

            elif action.command == "complete":
                if not action.task_id:
                    return SharedTaskListObservation(error="'complete' requires 'task_id' parameter")
                agent_name = self._resolve_agent(conversation)
                if not agent_name:
                    return SharedTaskListObservation(error="Cannot resolve caller agent. Are you a registered agent?")
                result = _cmd_complete(self.task_list_path, action.task_id, agent_name)

            elif action.command == "status":
                result = _cmd_status(self.task_list_path, action.task_id)

            else:
                return SharedTaskListObservation(error=f"Unknown command: {action.command}")

            return SharedTaskListObservation(result=result)

        except Exception as e:
            return SharedTaskListObservation(error=f"shared_task_list error: {e}")


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Shared task list — dependency-aware task scheduling.

Tracks tasks with dependencies. When a task's dependencies are all completed,
it becomes ready to claim. A task can represent anything (file, module, etc.).
Peer agents are auto-notified by the system when tasks become ready.

Commands:
- init: Create the task list.
  Parameters: tasks (list of {id, deps, owner?, description?, ...extras})
  'id' and 'deps' are required. 'owner' (optional) assigns a responsible agent.
  'description' (optional) is a short summary. Any other fields go into 'metadata'.
- claim: Claim a task to work on.
  Parameters: task_id
  Returns: {task_id, status:"ready"} if newly claimed,
           {task_id, status:"in_progress"} if you already claimed it (resume!),
           {task_id, status:"waiting", blocked_by:[...]} if deps not met,
           {task_id, status:"completed"} if already done.
  Your identity is resolved automatically.
- complete: Mark a task as completed. Auto-unlocks dependent tasks.
  Parameters: task_id
  Returns: {completed, newly_ready:[{task_id, owner, description}], all_done}
  Peer agents are automatically notified when their tasks become ready —
  you do NOT need to call send_to_agent for newly_ready tasks.
- status: Get progress. Pass task_id for single task, or omit for full summary.
  Parameters: task_id (optional)
  Returns: single → {task_id, status, deps, blocks, owner, description, metadata, claimed_by?, blocked_by?}
           full → {total, completed, in_progress, ready, blocked, all_done, tasks:[...]}
"""


class SharedTaskListTool(ToolDefinition[SharedTaskListAction, SharedTaskListObservation]):
    """Shared task list with dependency-aware scheduling."""

    @classmethod
    def create(cls, conv_state, task_list_path=None, orchestrator=None) -> Sequence[ToolDefinition]:
        if task_list_path is None:
            raise ValueError("SharedTaskListTool requires task_list_path")
        if orchestrator is None:
            raise ValueError("SharedTaskListTool requires orchestrator")
        executor = SharedTaskListExecutor(Path(task_list_path), orchestrator)
        return [cls(
            description=_DESCRIPTION,
            action_type=SharedTaskListAction,
            observation_type=SharedTaskListObservation,
            executor=executor,
        )]
