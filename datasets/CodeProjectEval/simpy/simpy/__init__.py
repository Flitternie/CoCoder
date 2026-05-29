"""
A small package initializer that exposes a compact public API for this
lightweight SimPy-compatible package.
"""
from __future__ import annotations

from typing import Any, Iterable, Tuple

# Re-export commonly used public API from submodules so callers can do
# `from simpy import Environment, Event, Process, Timeout, ...`.
from simpy.core import Environment
from simpy.events import Event, Timeout, Process, AllOf, AnyOf
from simpy.exceptions import SimPyException, Interrupt
from simpy.rt import RealtimeEnvironment
from simpy.util import start_delayed

__all__ = [
    'Environment',
    'Event',
    'Timeout',
    'Process',
    'AllOf',
    'AnyOf',
    'SimPyException',
    'Interrupt',
    'RealtimeEnvironment',
    'start_delayed',
]

__version__ = '0.0'


def _compile_toc(entries: Iterable[Tuple[str, Iterable[Any]]], section_marker: str = '=') -> str:
    """Return a simple Sphinx autosummary string for given (section, objects).

    One-line helper: build autosummary directives for documentation generation.
    """
    parts: list[str] = []
    for section, objects in entries:
        parts.append(section)
        parts.append(section_marker * len(section))
        parts.append('.. autosummary::')
        parts.append('')
        for obj in objects:
            # Allow either objects given as strings or actual objects with
            # __module__/__name__ attributes.
            if isinstance(obj, str):
                parts.append(f'    {obj}')
            else:
                parts.append(f'    ~{obj.__module__}.{obj.__name__}')
        parts.append('')
    return "\n".join(parts)
