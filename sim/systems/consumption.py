"""Consumption: sheep grazing, fox predation, drinking (§14 of v1.md).

Acts on the gates produced by the brain (eat / drink), but enforces the *authoritative*
world conditions (true adjacency to vegetation / prey / freshwater). Predation kills prey
slots immediately and returns the killed slots so the caller can drop them from the rest
of the tick.
"""
from __future__ import annotations

import numpy as np

from config import SHEEP, FOX
from sim import genome as gn
from sim.brain import A_EAT, A_DRINK


def apply(cfg, world, ent, idx, act, veg, species_grids, rng):
    """Returns (killed_prey_slots, n_drink, n_graze, n_predation)."""
    if idx.shape[0] == 0:
        return np.empty(0, dtype=np.intp), 0, 0, 0

    eat_r = cfg.sim.eat_radius
    px = ent.pos_x[idx]
    py = ent.pos_y[idx]
    spec = ent.species[idx]
    eat_gate = act[:, A_EAT] > 0.5
    drink_gate = act[:, A_DRINK] > 0.5

    # --- DRINK: gate + adjacent freshwater -> thirst 0 ---
    n_drink = 0
    if drink_gate.any():
        cx, cy = world.world_to_cell(px, py)
        on_fw = world.freshwater[cy, cx]
        near_fw = world.fw_dist[cy, cx] <= eat_r      # standing on/adjacent freshwater
        can_drink = drink_gate & (on_fw | near_fw)
        slots = idx[can_drink]
        ent.thirst[slots] = 0.0
        n_drink = int(slots.shape[0])

    # --- SHEEP GRAZE: gate + vegetation in current cell above threshold ---
    n_graze = 0
    is_sheep = spec == SHEEP
    graze = eat_gate & is_sheep
    if graze.any():
        rows = np.nonzero(graze)[0]
        for k in rows:
            slot = idx[k]
            cx = int(min(max(px[k], 0), world.w - 1))
            cy = int(min(max(py[k], 0), world.h - 1))
            available = veg[cy, cx]
            if available < cfg.sim.food_eat_threshold:
                continue
            take = available * cfg.sim.veg_graze_amount
            size = gn.gene(ent.genome[slot:slot + 1], "size")[0]
            spec_cfg = cfg.species[SHEEP]
            gain = spec_cfg.eat_value * take * (0.7 + 0.3 * size)
            ent.energy[slot] = min(1.0, ent.energy[slot] + gain)
            ent.hunger[slot] = max(0.0, ent.hunger[slot] - take * 1.5)
            veg[cy, cx] = available - take
            world.nutrients[cy, cx] = max(0.0, world.nutrients[cy, cx] - take * 0.15)
            n_graze += 1

    # --- FOX PREDATION: gate + adjacent sheep -> kill prey, gain energy ---
    killed = []
    n_pred = 0
    is_fox = spec == FOX
    hunt = eat_gate & is_fox
    sheep_grid = species_grids.get(SHEEP)
    # Type III functional response: when prey are scarce they are harder to find / warier,
    # so hunt success drops with prey abundance. This low-density prey refuge is the
    # classic mechanism that prevents predators from driving prey (and then themselves) to
    # extinction, turning runaway crashes into a bounded limit cycle.
    n_sheep = int(ent.count_species(SHEEP))
    scarcity = (n_sheep ** 2) / (n_sheep ** 2 + cfg.species[FOX].hunt_halfsat ** 2)
    if hunt.any() and sheep_grid is not None:
        rows = np.nonzero(hunt)[0]
        for k in rows:
            slot = idx[k]
            cand, cpx, cpy = sheep_grid.query_radius(float(px[k]), float(py[k]), eat_r)
            # only living sheep not already killed this tick, and not hidden in cover
            cand = cand[ent.alive[cand]]
            if cand.shape[0] == 0:
                continue
            cand = cand[~world.in_cover(ent.pos_x[cand], ent.pos_y[cand])]
            if cand.shape[0] == 0:
                continue
            d2 = (ent.pos_x[cand] - px[k]) ** 2 + (ent.pos_y[cand] - py[k]) ** 2
            j = int(np.argmin(d2))
            prey = int(cand[j])
            aggression = gn.gene(ent.genome[slot:slot + 1], "aggression")[0]
            # most chases fail: real predators have low hunt success, giving prey a real
            # chance to flee and survive an encounter.
            kill_prob = aggression * cfg.species[FOX].hunt_success * scarcity
            if rng.random() > kill_prob:
                continue                       # failed attack -- prey escapes
            prey_size = gn.gene(ent.genome[prey:prey + 1], "size")[0]
            gain = cfg.species[FOX].predation_gain * (0.4 + 0.5 * prey_size)
            ent.energy[slot] = min(1.0, ent.energy[slot] + gain)
            ent.hunger[slot] = max(0.0, ent.hunger[slot] - 0.6)
            killed.append(prey)
            n_pred += 1

    killed_arr = np.array(sorted(set(killed)), dtype=np.intp) if killed else np.empty(0, dtype=np.intp)
    if killed_arr.shape[0] > 0:
        ent.kill(killed_arr)
    return killed_arr, n_drink, n_graze, n_pred
