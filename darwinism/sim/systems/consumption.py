"""Consumption: grazing (herbivores), predation (carnivores), drinking (§14 of v1.md).

Acts on the gates produced by the brain (eat / drink), but enforces the *authoritative*
world conditions (true adjacency to a food field / prey / freshwater). Predation kills prey
slots immediately and returns the killed slots so the caller can drop them from the rest of
the tick.

Data-driven: which species graze and which hunt (and their food/hunt parameters) come from
each species' declarative ``diet`` (``FieldFood`` / ``PreyFood``), not hardcoded per species.
For the default sheep+fox config this is byte-identical to the old hardcoded logic.
"""
from __future__ import annotations

import numpy as np

from darwinism.config import FieldFood, PreyFood
from darwinism.sim import genome as gn
from darwinism.sim.brain import A_DRINK, A_EAT


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

    # diet lookups (first source of each kind per species). Field grazers reduce food to a
    # world field; prey hunters reduce it to entities.
    field_food = {}     # sid -> FieldFood
    prey_food = {}      # sid -> PreyFood
    for sid, s in cfg.species.items():
        for src in s.diet:
            if isinstance(src, FieldFood) and sid not in field_food:
                field_food[sid] = src
            elif isinstance(src, PreyFood) and sid not in prey_food:
                prey_food[sid] = src

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

    # --- GRAZE: field-eaters with gate + vegetation in current cell above threshold ---
    n_graze = 0
    if field_food:
        graze = eat_gate & np.isin(spec, np.fromiter(field_food, dtype=spec.dtype))
        if graze.any():
            for k in np.nonzero(graze)[0]:            # ascending row order; no RNG here
                slot = idx[k]
                cx = int(min(max(px[k], 0), world.w - 1))
                cy = int(min(max(py[k], 0), world.h - 1))
                available = veg[cy, cx]
                if available < cfg.sim.food_eat_threshold:
                    continue
                take = available * cfg.sim.veg_graze_amount
                size = gn.gene(ent.genome[slot:slot + 1], "size")[0]
                eat_value = field_food[int(spec[k])].eat_value
                gain = eat_value * take * (0.7 + 0.3 * size)
                ent.energy[slot] = min(1.0, ent.energy[slot] + gain)
                ent.hunger[slot] = max(0.0, ent.hunger[slot] - take * 1.5)
                veg[cy, cx] = available - take
                world.nutrients[cy, cx] = max(0.0, world.nutrients[cy, cx] - take * 0.15)
                n_graze += 1

    # --- PREDATION: prey-hunters with gate + adjacent EXPOSED prey -> kill, gain energy ---
    killed = []
    n_pred = 0
    if prey_food:
        hunt = eat_gate & np.isin(spec, np.fromiter(prey_food, dtype=spec.dtype))
        # Type III functional response per predator species: when its prey are scarce they are
        # harder to find / warier, so hunt success drops with prey abundance. This low-density
        # prey refuge is the classic mechanism that turns runaway crashes into a bounded limit
        # cycle (do not over-weaken it -- see v1.md §18).
        scarcity = {}
        for psid, pf in prey_food.items():
            n_prey = sum(int(ent.count_species(t)) for t in pf.prey)
            scarcity[psid] = (n_prey ** 2) / (n_prey ** 2 + pf.hunt_halfsat ** 2)
        has_aggression = "aggression" in gn.GENE_INDEX
        if hunt.any():
            # iterate rows in ascending GLOBAL index (as before) so the per-attempt rng.random()
            # stream is consumed in the exact same order for the single-predator default.
            for k in np.nonzero(hunt)[0]:
                slot = idx[k]
                psid = int(spec[k])
                pf = prey_food[psid]
                # gather candidate prey across this predator's prey species (default: one grid)
                cand_list = []
                for tsid in pf.prey:
                    g = species_grids.get(tsid)
                    if g is None:
                        continue
                    c, _cpx, _cpy = g.query_radius(float(px[k]), float(py[k]), eat_r)
                    if c.shape[0]:
                        cand_list.append(c)
                if not cand_list:
                    continue
                cand = cand_list[0] if len(cand_list) == 1 else np.concatenate(cand_list)
                # only living prey not already killed this tick, and not hidden in cover
                cand = cand[ent.alive[cand]]
                if cand.shape[0] == 0:
                    continue
                cand = cand[~world.in_cover(ent.pos_x[cand], ent.pos_y[cand])]
                if cand.shape[0] == 0:
                    continue
                d2 = (ent.pos_x[cand] - px[k]) ** 2 + (ent.pos_y[cand] - py[k]) ** 2
                prey = int(cand[int(np.argmin(d2))])
                aggression = (gn.gene(ent.genome[slot:slot + 1], "aggression")[0]
                              if has_aggression else 1.0)
                # most chases fail: low hunt success gives prey a real chance to flee
                kill_prob = aggression * pf.hunt_success * scarcity[psid]
                if rng.random() > kill_prob:
                    continue                       # failed attack -- prey escapes
                prey_size = gn.gene(ent.genome[prey:prey + 1], "size")[0]
                gain = pf.predation_gain * (0.4 + 0.5 * prey_size)
                ent.energy[slot] = min(1.0, ent.energy[slot] + gain)
                ent.hunger[slot] = max(0.0, ent.hunger[slot] - 0.6)
                killed.append(prey)
                n_pred += 1

    killed_arr = np.array(sorted(set(killed)), dtype=np.intp) if killed else np.empty(0, dtype=np.intp)
    if killed_arr.shape[0] > 0:
        ent.kill(killed_arr)
    return killed_arr, n_drink, n_graze, n_pred
