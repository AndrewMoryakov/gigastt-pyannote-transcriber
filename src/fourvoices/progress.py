"""Plain-text progress for the long stages.

Recognition and diarization of an hour of audio each take minutes on a CPU,
and both used to run without a word. Progress is printed as ordinary lines, a
few per minute at most: no carriage returns or progress bars, so it reads the
same in a console, in a redirected log and in a CI transcript.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any, TextIO

DEFAULT_INTERVAL = 30.0
INDENT = "      "  # lines up under "[3/5] "


def format_duration(seconds: float) -> str:
    """``7:05`` below an hour, ``1:02:03`` above it."""

    total = max(0, round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class Reporter:
    """Prints ``label: text`` lines, throttled to one per ``interval`` seconds."""

    def __init__(
        self,
        label: str,
        *,
        interval: float = DEFAULT_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
        stream: TextIO | None = None,
    ) -> None:
        self.label = label
        self.interval = interval
        self._clock = clock
        self._stream = stream
        self.started = clock()
        self._last_printed: float | None = None

    def elapsed(self) -> float:
        return self._clock() - self.started

    def update(self, text: str, *, force: bool = False) -> bool:
        """Print ``text`` if forced or if the interval has passed; say whether it did."""

        now = self._clock()
        due = self._last_printed is None or now - self._last_printed >= self.interval
        if not (force or due):
            return False
        self._last_printed = now
        print(f"{INDENT}{self.label}: {text}", file=self._stream or sys.stdout, flush=True)
        return True


class DiarizationProgress:
    """A pyannote pipeline hook that reports each step and its share done.

    pyannote calls ``hook(step_name, artifact, file=..., total=..., completed=...)``
    repeatedly during a step and once more, with no counts, when the step ends.
    """

    def __init__(self, reporter: Reporter) -> None:
        self.reporter = reporter
        self._step: str | None = None

    def __call__(
        self,
        step_name: str,
        step_artifact: Any = None,
        file: Any = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        new_step = step_name != self._step
        self._step = step_name
        elapsed = format_duration(self.reporter.elapsed())
        if total and completed is not None:
            # pyannote advances by whole batches, so the last call can overshoot.
            percent = min(100, completed * 100 // total)
            self.reporter.update(f"{step_name} {percent}% ({elapsed})", force=new_step)
        else:
            self.reporter.update(f"{step_name} done ({elapsed})", force=True)
