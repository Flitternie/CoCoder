"""Thread-safe message bus and data structures for inter-agent communication.

Agents put messages via send(); Orchestrator consumes via poll().
ThreadSafeDict wraps a plain dict with a lock for concurrent access.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True)
class AgentMessage:
    """A single message between agents."""
    from_agent: str
    to_agent: str        # "agent_name" for direct, "broadcast:role_prefix" for broadcast
    content: str
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """Thread-safe message queue, consumed by Orchestrator's routing loop."""

    def __init__(self) -> None:
        self._queue: queue.Queue[AgentMessage] = queue.Queue()

    def send(self, msg: AgentMessage) -> None:
        """Put a message into the bus (called by send_to_agent executor)."""
        self._queue.put(msg)

    def poll(self, timeout: float = 1.0) -> AgentMessage | None:
        """Blocking poll with timeout. Returns None if no message available."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def empty(self) -> bool:
        return self._queue.empty()


class ThreadSafeDict(dict):
    """A dict subclass that wraps every operation with a reentrant lock.

    Supports all standard dict operations, plus helpers commonly needed
    by the Orchestrator (e.g. ``keys_matching``).

    Usage::

        agents = ThreadSafeDict()
        agents["leader"] = handle          # __setitem__ — locked
        handle = agents.get("leader")      # get — locked
        handle = agents.pop("leader", None) # pop — locked
        names = agents.keys_matching("code")# custom — locked
    """

    def __init__(self, *args, **kwargs):
        self._lock = threading.RLock()
        super().__init__(*args, **kwargs)

    # --- standard dict overrides ---

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __delitem__(self, key):
        with self._lock:
            super().__delitem__(key)

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)

    def __len__(self):
        with self._lock:
            return super().__len__()

    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)

    def pop(self, key, *args):
        with self._lock:
            return super().pop(key, *args)

    def items(self):
        with self._lock:
            return list(super().items())

    def keys(self):
        with self._lock:
            return list(super().keys())

    def values(self):
        with self._lock:
            return list(super().values())

    def clear(self):
        with self._lock:
            super().clear()

    # --- custom helpers ---

    def keys_matching(self, prefix: str) -> list[str]:
        """Return keys that start with *prefix* (snapshot)."""
        with self._lock:
            return [k for k in super().keys() if k.startswith(prefix)]

    def keys_not_equal(self, exclude: str) -> list[str]:
        """Return keys != *exclude* (snapshot)."""
        with self._lock:
            return [k for k in super().keys() if k != exclude]

    def set_if_absent(self, key, value) -> bool:
        """Atomically set key=value only if key is not present. Returns True if set."""
        with self._lock:
            if key in self:
                return False
            self[key] = value
            return True


class FutureTracker:
    """Thread-safe tracker for in-flight Future objects.

    All access is serialized through a lock.  ``has_active()`` atomically
    prunes completed futures and reports whether work is still in progress.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._futures: list[Future] = []

    def add(self, future: Future) -> None:
        """Register a newly-submitted future."""
        with self._lock:
            self._futures.append(future)

    def has_active(self) -> bool:
        """Prune done futures and return True if any are still running."""
        with self._lock:
            self._futures = [f for f in self._futures if not f.done()]
            return bool(self._futures)

