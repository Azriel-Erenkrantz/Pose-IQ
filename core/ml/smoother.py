"""
Temporal smoothing for per-frame phase predictions.

Raw frame-level predictions flicker: a single mis-classified frame mid-descent
shouldn't flip the phase. Two layers fix that:

  1. Majority vote over a sliding window of recent raw predictions —
     erases one-frame noise.
  2. Legal-cycle constraint — exercise phases follow a fixed cycle
     (standing → descending → ascending → standing …). A winner that is not
     the current phase or its legal successor is rejected.

A rejected winner that persists (`resync_after` consecutive frames) forces a
resync — this recovers from missed phases (e.g. a short hold the model never
fired on) or a video that starts mid-cycle, instead of deadlocking.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Optional

from core.exercise.exercise_model import Exercise

WINDOW = 5          # raw predictions in the majority vote
RESYNC_AFTER = 8    # consecutive rejected frames before trusting the winner


class PhaseSmoother:
    def __init__(self, exercise: Exercise, window: int = WINDOW,
                 resync_after: int = RESYNC_AFTER):
        self._cycle = [p.name for p in sorted(exercise.phases, key=lambda p: p.order)]
        self._buf: deque = deque(maxlen=window)
        self._disagree = 0
        self._resync_after = resync_after
        self.current: Optional[str] = None

    def update(self, raw_phase: Optional[str]) -> Optional[str]:
        """Feed one raw prediction; returns the smoothed phase."""
        if raw_phase is None:
            return self.current
        self._buf.append(raw_phase)

        winner, _ = Counter(self._buf).most_common(1)[0]

        if self.current is None or winner == self.current:
            self.current = winner
            self._disagree = 0
            return self.current

        if winner == self._legal_next():
            self.current = winner
            self._disagree = 0
            return self.current

        # Illegal jump — hold the current phase, but count the disagreement
        # so a persistent winner eventually wins (resync).
        self._disagree += 1
        if self._disagree >= self._resync_after:
            self.current = winner
            self._disagree = 0
        return self.current

    def _legal_next(self) -> Optional[str]:
        if self.current not in self._cycle:
            return None
        i = self._cycle.index(self.current)
        return self._cycle[(i + 1) % len(self._cycle)]

    def reset(self):
        self._buf.clear()
        self._disagree = 0
        self.current = None
