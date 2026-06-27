"""Circadian rest: animals sleep at night (diurnal behavior).

As dusk falls each animal heads for the nearest safe spot (forest cover) and beds down;
near dawn it wakes. The onset/wake times are population means shifted per-individual by
the heritable ``chronotype`` gene, so the herd does NOT drop unconscious all at once --
some turn in early, some stay up late, and the timing can drift over generations.

Sleep is real sim state (it gates movement, consumption, reproduction and lowers
metabolism), so it lives on the entity store and is managed here as a system -- the same
way metabolism owns aging. The brain still governs all *waking* behavior; this layer only
arbitrates rest vs. wake and steers sleepers toward shelter. A predator right next to a
sheep overrides sleep (you wake to flee), which keeps the predator-prey dynamics intact.

Mutates ``act`` (headings + eat/drink/repro gates) and ``ent.asleep`` in place.
"""
from __future__ import annotations

import numpy as np

from sim import genome as gn
from sim.brain import A_DX, A_DY, A_EAT, A_DRINK, A_REPRO, nearest_in_channel
from sim.perception import CH_THREAT, S_SENSORY

# A threat within this fraction of sensory range keeps a sheep awake to flee (mirrors the
# brain's _FLEE_TRIGGER so sleep never suppresses an escape from a close fox).
_WAKE_THREAT = 0.45


def apply(cfg, world, ent, idx, act, obs, env) -> int:
    """Run one tick of rest arbitration. Returns the number of sleeping animals."""
    n = idx.shape[0]
    if n == 0:
        return 0
    c = cfg.env
    t = float(env.time_of_day)

    # per-individual night window: onset shifted by the chronotype gene; the night length
    # (mean wake - mean onset, wrapped) is shared, so each animal's wake shifts in step.
    chrono = gn.gene(ent.genome[idx], "chronotype")
    onset = c.sleep_onset + chrono
    rest_dur = (c.sleep_wake - c.sleep_onset) % 1.0

    # how far into its own night each animal is (wraps across midnight)
    since_onset = (t - onset) % 1.0
    in_rest = since_onset < rest_dur
    early = since_onset < c.sleep_shelter_window     # grace window to reach shelter

    px = ent.pos_x[idx]
    py = ent.pos_y[idx]
    in_cover = world.in_cover(px, py)

    # a close predator overrides sleep -- the sheep stays awake and the brain's flee
    # heading (already in ``act``) is preserved. Decode the threat grid the same way the
    # brain does (nearest predator, as a fraction of the animal's own sensory range).
    thr_p, _, _, thr_dc = nearest_in_channel(obs.grids[:, CH_THREAT])
    thr_frac = thr_dc / np.maximum(obs.scalars[:, S_SENSORY], 1e-6)
    threat_close = (thr_p > 0.5) & (thr_frac < _WAKE_THREAT)

    # seeking shelter: night, still within the grace window, not yet safe, no near threat
    seeking = in_rest & early & ~in_cover & ~threat_close
    # asleep: night, and either already in cover or past the grace window (collapse where
    # you stand). Never asleep with a predator right next to you.
    asleep = in_rest & ~seeking & ~threat_close

    # --- steer seekers toward their nearest cover cell ---
    if seeking.any():
        cx, cy = world.world_to_cell(px, py)
        has_cover = np.isfinite(world.cover_dist[cy, cx])
        dx = world.cover_nearest_x[cy, cx] - px
        dy = world.cover_nearest_y[cy, cx] - py
        mag = np.sqrt(dx * dx + dy * dy)
        safe = mag > 1e-6
        hx = np.where(safe, dx / np.where(safe, mag, 1.0), act[:, A_DX])
        hy = np.where(safe, dy / np.where(safe, mag, 1.0), act[:, A_DY])
        go = seeking & has_cover
        act[:, A_DX] = np.where(go, hx, act[:, A_DX])
        act[:, A_DY] = np.where(go, hy, act[:, A_DY])
        # rushing to bed -- don't stop to forage, drink or mate on the way
        act[seeking, A_EAT] = 0.0
        act[seeking, A_DRINK] = 0.0
        act[seeking, A_REPRO] = 0.0

    # --- sleepers take no action; movement reads ent.asleep to hold their position ---
    if asleep.any():
        act[asleep, A_EAT] = 0.0
        act[asleep, A_DRINK] = 0.0
        act[asleep, A_REPRO] = 0.0

    ent.asleep[idx] = asleep
    return int(asleep.sum())
