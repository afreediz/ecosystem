"""Brain interface + hardcoded RuleBrain (§7.1, §13 of v1.md).

The contract is the spine of the whole project: ``decide(obs) -> act`` where obs is
(N, OBS_DIM) and act is (N, ACT_DIM). The brain sees ONLY the observation matrix --
exactly what a future PyTorch brain will get. Swapping RuleBrain for a TorchBrain
changes nothing else.

Adjacency and reproduction *eligibility* are only proxied here from the observation;
the consumption / reproduction systems enforce the authoritative conditions, so the
brain never needs hidden state. Exploration momentum lives in the movement system
(turn-rate-limited steering), so the stateless brain can emit a fresh random heading
each tick and still produce a smooth directed wander.
"""
from __future__ import annotations

import numpy as np

OBS_DIM = 29
ACT_DIM = 5

# action indices
A_DX, A_DY, A_EAT, A_DRINK, A_REPRO = range(ACT_DIM)

# how close (as a fraction of sensory_range) a target must read before the brain raises
# the eat/drink/reproduce gate. The relevant system re-checks true world adjacency.
_ADJ_NORM = 0.25
# need urgency below which an animal won't actively pursue food/water
_NEED_URGENCY = 0.4
# flee only when a predator is within this fraction of the sensory range (close), so prey
# tolerate distant predators and keep foraging/breeding
_FLEE_TRIGGER = 0.45


class Brain:
    def decide(self, obs: np.ndarray) -> np.ndarray:
        """obs: (N, OBS_DIM) float32 -> actions: (N, ACT_DIM) float32."""
        raise NotImplementedError


def _norm(dx, dy):
    mag = np.sqrt(dx * dx + dy * dy)
    safe = mag > 1e-6
    ox = np.where(safe, dx / np.where(safe, mag, 1.0), 0.0)
    oy = np.where(safe, dy / np.where(safe, mag, 1.0), 0.0)
    return ox.astype(np.float32), oy.astype(np.float32)


class RuleBrain(Brain):
    """Vectorized priority arbitration (throwaway logic; exercises the contract)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def decide(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        act = np.zeros((n, ACT_DIM), dtype=np.float32)
        if n == 0:
            return act

        hunger, thirst, energy = obs[:, 0], obs[:, 1], obs[:, 2]
        food_dx, food_dy, food_d, food_p = obs[:, 6], obs[:, 7], obs[:, 8], obs[:, 9]
        thr_dx, thr_dy, thr_d, thr_p = obs[:, 10], obs[:, 11], obs[:, 12], obs[:, 13]
        mate_dx, mate_dy, mate_d, mate_p = obs[:, 14], obs[:, 15], obs[:, 16], obs[:, 17]
        wat_dx, wat_dy, wat_d, wat_p = obs[:, 18], obs[:, 19], obs[:, 20], obs[:, 21]

        # --- priority 4: explore (default) -- fresh random heading; movement smooths it
        ang = self.rng.uniform(0.0, 2 * np.pi, size=n).astype(np.float32)
        head_x = np.cos(ang)
        head_y = np.sin(ang)

        # --- priority 3: reproduce (rough eligibility; reproduction system enforces) ---
        repro_fit = (energy > 0.5) & (hunger < 0.55) & (thirst < 0.55)
        repro_go = repro_fit & (mate_p > 0.5)
        mx, my = _norm(mate_dx, mate_dy)
        head_x = np.where(repro_go, mx, head_x)
        head_y = np.where(repro_go, my, head_y)
        act[:, A_REPRO] = np.where(repro_go & (mate_d < _ADJ_NORM), 1.0, 0.0)

        # --- priority 2: needs ---
        # food drive responds to BOTH hunger and energy deficit, so an animal seeks food
        # before its energy reserve runs out (hunger alone rises too slowly to prevent
        # starvation).
        food_need = np.maximum(hunger, 1.0 - energy)
        want_water = thirst >= food_need
        urgent = np.maximum(food_need, thirst) > _NEED_URGENCY
        wx, wy = _norm(wat_dx, wat_dy)
        fx, fy = _norm(food_dx, food_dy)
        need_p = np.where(want_water, wat_p, food_p) > 0.5
        need_x = np.where(want_water, wx, fx)
        need_y = np.where(want_water, wy, fy)
        # only an *urgent* need overrides the reproduce/explore heading
        do_need = urgent & need_p
        head_x = np.where(do_need, need_x, head_x)
        head_y = np.where(do_need, need_y, head_y)
        # OPPORTUNISTIC eat/drink: top up whenever a resource is adjacent and we are not
        # already full -- this keeps thirst/hunger low without forcing "need" mode, so the
        # animal can still spend most of its time free to reproduce/explore.
        drink_go = (thirst > 0.05) & (wat_p > 0.5) & (wat_d < _ADJ_NORM)
        eat_go = ((hunger > 0.05) | (energy < 0.9)) & (food_p > 0.5) & (food_d < _ADJ_NORM)
        act[:, A_DRINK] = np.where(drink_go, 1.0, 0.0)
        act[:, A_EAT] = np.where(eat_go, 1.0, 0.0)
        # an urgent need suppresses reproduction
        act[:, A_REPRO] = np.where(urgent, 0.0, act[:, A_REPRO])

        # --- priority 1: flee threat (overrides all) ---
        # Only flee when the predator is genuinely CLOSE (within _FLEE_TRIGGER of the
        # sensory range), not for any predator anywhere in sight. Constant fleeing from
        # distant foxes would stop prey eating/breeding entirely (a runaway "landscape of
        # fear" that crashes the prey and then starves the predator).
        flee = (thr_p > 0.5) & (thr_d < _FLEE_TRIGGER)
        flx, fly = _norm(-thr_dx, -thr_dy)
        head_x = np.where(flee, flx, head_x)
        head_y = np.where(flee, fly, head_y)
        act[:, A_EAT] = np.where(flee, 0.0, act[:, A_EAT])
        act[:, A_DRINK] = np.where(flee, 0.0, act[:, A_DRINK])
        act[:, A_REPRO] = np.where(flee, 0.0, act[:, A_REPRO])

        act[:, A_DX] = head_x
        act[:, A_DY] = head_y
        return act
