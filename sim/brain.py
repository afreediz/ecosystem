"""Brain interface + hardcoded RuleBrain (§7.1, §13 of v1.md).

The contract is the spine of the whole project: ``decide(obs_by_species, idx) -> act``
where ``obs_by_species`` maps each species to its ``Observation`` (per-species egocentric
perception grids + a scalar vector, see ``sim/perception.py``) and ``act`` is the
(N, ACT_DIM) action matrix aligned to the GLOBAL alive ordering ``idx``. The brain sees
ONLY the observations -- exactly what a future PyTorch/CNN brain will get. The CNN would
consume each species' ``obs.grids`` (N, C, K, K) as image channels.

Because a *rule* brain can't run a convolution, it first **decodes** the relevant grid
channels back into the simple targets it reasons over -- "nearest threat", "best grass
cell", "nearest mate/water/prey" (``nearest_in_channel`` / ``best_in_channel``). A neural
brain skips the decoding and learns straight off the channels.

Each species is decoded separately (its grids carry only the channels it uses), but the
explore-heading RNG is drawn ONCE over the global ordering so the action stream is identical
regardless of how perception is partitioned (keeps runs deterministic + comparable).

Adjacency and reproduction *eligibility* are only proxied here from the observation; the
consumption / reproduction systems enforce the authoritative conditions, so the brain
never needs hidden state. Exploration momentum lives in the movement system, so the
stateless brain can emit a fresh random heading each tick and still wander smoothly.
"""
from __future__ import annotations

import numpy as np

from config import SHEEP, FOX
from sim.perception import (
    SH_WATER, SH_FOOD, SH_THREAT, SH_MATE,
    FX_WATER, FX_FOOD, FX_MATE,
    S_HUNGER, S_THIRST, S_ENERGY, S_SENSORY,
)

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

# default vegetation threshold a grass cell must clear to be worth grazing (config override
# is passed into RuleBrain; mirrors cfg.sim.food_eat_threshold)
_DEFAULT_FOOD_THR = 0.15


# --------------------------------------------------------------------- grid decoding
_STENCIL_CACHE: dict[int, tuple] = {}


def _stencil(K: int):
    """Cached (ox, oy, dcell) flat offset/distance stencils for a KxK egocentric window."""
    s = _STENCIL_CACHE.get(K)
    if s is None:
        R = (K - 1) // 2
        offs = np.arange(-R, R + 1)
        oy, ox = np.meshgrid(offs, offs, indexing="ij")
        ox = ox.ravel().astype(np.float32)
        oy = oy.ravel().astype(np.float32)
        dcell = np.sqrt(ox * ox + oy * oy)
        s = (ox, oy, dcell)
        _STENCIL_CACHE[K] = s
    return s


def nearest_in_channel(chan: np.ndarray):
    """Reduce a (N,K,K) channel to its NEAREST present cell, relative to the window centre.

    Returns ``(present, dx_cells, dy_cells, dist_cells)`` -- arrays of shape (N,). ``present``
    is 1.0 where any cell of the channel is non-zero, else 0.0 (with the offsets/dist zeroed).
    """
    n = chan.shape[0]
    K = chan.shape[-1]
    ox, oy, dcell = _stencil(K)
    flat = chan.reshape(n, -1)
    d = np.where(flat > 0.0, dcell[None], np.inf)
    j = np.argmin(d, axis=1)
    ar = np.arange(n)
    dist = d[ar, j]
    present = np.isfinite(dist)
    return (present.astype(np.float32),
            np.where(present, ox[j], 0.0),
            np.where(present, oy[j], 0.0),
            np.where(present, dist, 0.0))


def best_in_channel(chan: np.ndarray, thr: float):
    """Reduce a scalar-valued (N,K,K) channel (e.g. vegetation) to its BEST cell.

    Score = value - 0.02 * distance, over cells whose value exceeds ``thr`` -- i.e. the
    richest patch in sight, with a mild pull toward closer cells (faithful to the old
    "best grass cell within sensory_range" forage rule, v1.md §18). Returns
    ``(present, dx_cells, dy_cells, dist_cells)``.
    """
    n = chan.shape[0]
    K = chan.shape[-1]
    ox, oy, dcell = _stencil(K)
    flat = chan.reshape(n, -1)
    score = np.where(flat > thr, flat - 0.02 * dcell[None], -np.inf)
    j = np.argmax(score, axis=1)
    ar = np.arange(n)
    present = np.isfinite(score[ar, j])
    return (present.astype(np.float32),
            np.where(present, ox[j], 0.0),
            np.where(present, oy[j], 0.0),
            np.where(present, dcell[j], 0.0))


class Brain:
    def decide(self, obs_by_species, idx) -> np.ndarray:
        """obs_by_species: {species_id: Observation}, idx: global alive slot ids.
        Returns actions: (len(idx), ACT_DIM) float32 aligned to ``idx``."""
        raise NotImplementedError


def _norm(dx, dy):
    mag = np.sqrt(dx * dx + dy * dy)
    safe = mag > 1e-6
    ox = np.where(safe, dx / np.where(safe, mag, 1.0), 0.0)
    oy = np.where(safe, dy / np.where(safe, mag, 1.0), 0.0)
    return ox.astype(np.float32), oy.astype(np.float32)


class RuleBrain(Brain):
    """Vectorized priority arbitration over decoded perception grids (throwaway logic)."""

    def __init__(self, rng: np.random.Generator, food_threshold: float = _DEFAULT_FOOD_THR):
        self.rng = rng
        self.food_thr = float(food_threshold)

    def decide(self, obs_by_species, idx) -> np.ndarray:
        n_global = idx.shape[0]
        act = np.zeros((n_global, ACT_DIM), dtype=np.float32)
        if n_global == 0:
            return act
        # Draw explore headings ONCE over the global ordering, then hand each species the
        # slice for its rows. This keeps the random stream identical to a single batched
        # draw, so partitioning perception by species does not change the trajectory.
        ang = self.rng.uniform(0.0, 2 * np.pi, size=n_global).astype(np.float32)
        for sid in (SHEEP, FOX):
            obs = obs_by_species.get(sid)
            if obs is None or obs.grids.shape[0] == 0:
                continue
            pos = np.searchsorted(idx, obs.idx)        # rows of this species in global act
            act[pos] = self._decide_species(obs, ang[pos])
        return act

    def _decide_species(self, obs, ang) -> np.ndarray:
        grids = obs.grids                       # (n, C, K, K) for this species' layout
        sc = obs.scalars                        # (n, SCALAR_DIM)
        n = grids.shape[0]
        act = np.zeros((n, ACT_DIM), dtype=np.float32)

        hunger, thirst, energy = sc[:, S_HUNGER], sc[:, S_THIRST], sc[:, S_ENERGY]
        sens = np.maximum(sc[:, S_SENSORY], 1e-6)

        # --- decode the channels this species carries into nearest/best targets ---
        if obs.species == SHEEP:
            # herbivore: food is the best grass cell; threat is the nearest fox
            food_p, food_dx, food_dy, food_dc = best_in_channel(grids[:, SH_FOOD], self.food_thr)
            wat_p, wat_dx, wat_dy, wat_dc = nearest_in_channel(grids[:, SH_WATER])
            mate_p, mate_dx, mate_dy, mate_dc = nearest_in_channel(grids[:, SH_MATE])
            thr_p, thr_dx, thr_dy, thr_dc = nearest_in_channel(grids[:, SH_THREAT])
        else:  # FOX
            # carnivore: food is the nearest prey; no threat channel (no predators)
            food_p, food_dx, food_dy, food_dc = nearest_in_channel(grids[:, FX_FOOD])
            wat_p, wat_dx, wat_dy, wat_dc = nearest_in_channel(grids[:, FX_WATER])
            mate_p, mate_dx, mate_dy, mate_dc = nearest_in_channel(grids[:, FX_MATE])
            z = np.zeros(n, dtype=np.float32)
            thr_p, thr_dx, thr_dy, thr_dc = z, z, z, z

        food_d = np.clip(food_dc / sens, 0.0, 1.0)
        thr_d = np.clip(thr_dc / sens, 0.0, 1.0)
        mate_d = np.clip(mate_dc / sens, 0.0, 1.0)
        wat_d = np.clip(wat_dc / sens, 0.0, 1.0)

        # --- priority 4: explore (default) -- random heading (drawn globally); movement smooths it
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
