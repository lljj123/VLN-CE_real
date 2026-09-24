"""ROS-independent timing helpers for responsive periodic control loops."""

import math
import threading
import time


class WakeableRate:
    """Wait at a fixed period, but return early when new control work arrives.

    ``rospy.Rate`` is appropriate for a purely periodic publisher, but it can
    add up to one full period before a newly received action is handled.  This
    helper retains periodic execution while using a generation counter so a
    wake-up that happens just before ``wait`` cannot be lost.
    """

    def __init__(self, frequency_hz):
        if (
            isinstance(frequency_hz, bool)
            or not isinstance(frequency_hz, (int, float))
            or not math.isfinite(float(frequency_hz))
            or frequency_hz <= 0.0
        ):
            raise ValueError("frequency_hz must be a positive finite number")
        self.period_seconds = 1.0 / float(frequency_hz)
        self._condition = threading.Condition()
        self._generation = 0

    def snapshot(self):
        """Return the current wake generation."""

        with self._condition:
            return self._generation

    def wake(self):
        """Wake a waiter and record the event even if it is not waiting yet."""

        with self._condition:
            self._generation += 1
            self._condition.notify_all()

    def wait(self, observed_generation):
        """Wait for the next period or a generation newer than the observed one."""

        deadline = time.monotonic() + self.period_seconds
        with self._condition:
            while self._generation == observed_generation:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            return self._generation
