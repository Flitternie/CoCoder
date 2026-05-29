"""Execution environment for events that synchronizes passing of time
with the real-time (aka *wall-clock time*).

This implementation keeps the mapping between simulation time and wall-clock
using a reference wall-clock time recorded at construction (or when sync()
is called) and the environment's simulation time at that moment.
"""
from time import time, sleep

from simpy.core import EmptySchedule, Environment, Infinity, SimTime


class RealtimeEnvironment(Environment):
    """Execution environment for an event-based simulation which is
    synchronized with the real-time (wall-clock time).

    A time step in simulation units is scaled by *factor* to compute the
    corresponding amount of wall-clock seconds that must elapse. For
    example, with factor=0.5 a simulation advance of 3 time units requires
    1.5 seconds of real time.

    If *strict* is True, step() will raise RuntimeError when the environment
    is already behind the desired wall-clock time for the next event.
    """

    def __init__(
        self,
        initial_time: SimTime = 0,
        factor: float = 1.0,
        strict: bool = True,
    ):
        super().__init__(initial_time)

        # Record the simulation time and the corresponding wall-clock time.
        # Mapping: wall_time = _real_ref + (sim_time - _sim_ref) * factor
        self._sim_ref = initial_time
        self._real_ref = time()
        self._factor = float(factor)
        self._strict = bool(strict)

    @property
    def factor(self) -> float:
        """Scaling factor between simulation time units and seconds of wall time."""
        return self._factor

    @property
    def strict(self) -> bool:
        """Whether to raise when the simulation is behind real time."""
        return self._strict

    def sync(self) -> None:
        """Reset the wall-clock reference so that the current wall-clock time
        corresponds to the current simulation time.

        This is useful to avoid declaring the simulation "too slow" if a lot
        of real time passed between creating the RealtimeEnvironment and
        starting it.
        """
        self._real_ref = time()
        self._sim_ref = self.now

    def step(self) -> None:
        """Process the next scheduled event, but only after the corresponding
        real time has passed.

        The desired wall-clock time for the next event is computed from the
        mapping recorded at construction (or last sync()). If *strict* is
        True and the current wall-clock time is already past the desired
        time, a RuntimeError is raised.
        """
        evt_time = self.peek()

        if evt_time is Infinity:
            raise EmptySchedule

        # Compute the wall-clock time by mapping the simulation time using the
        # stored reference points and the scaling factor.
        desired_real = self._real_ref + (evt_time - self._sim_ref) * self.factor

        now = time()

        if self.strict and now > desired_real:
            # The environment is already behind real time.
            delta = now - desired_real
            raise RuntimeError(f'Simulation too slow for real time ({delta:.3f}s).')

        # Sleep until the desired wall-clock time is reached. Use a simple
        # loop to handle potential spurious wakeups; keep it minimal.
        while True:
            now = time()
            remaining = desired_real - now
            if remaining <= 0:
                break
            sleep(remaining)

        # Process the event as usual.
        super().step()
