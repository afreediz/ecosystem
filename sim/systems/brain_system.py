"""Global, batched brain invocation (§3, §7.1 of v1.md).

The ONLY caller of ``brain.decide``. It takes the observation matrix built by
perception, runs the (single) brain over the whole population at once, and returns the
action matrix aligned to the same alive-index ordering. Pointless for rules, built now
so a neural brain is a literal drop-in.
"""
from __future__ import annotations

import numpy as np

from sim.brain import Brain


class BrainSystem:
    def __init__(self, brain: Brain):
        self.brain = brain

    def decide(self, obs: np.ndarray) -> np.ndarray:
        return self.brain.decide(obs)
