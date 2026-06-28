"""Sexual reproduction: mate-finding, crossover, offspring spawn (§11, §14 of v1.md).

Eligibility (authoritative): adult, energy > repro_threshold gene, cooldown elapsed,
hunger/thirst below limits, the reproduce gate raised, and an eligible opposite-sex
same-species adult within ``repro_radius``. Pairs are formed deterministically by
ascending slot index. Each child genome = uniform crossover(parentA, parentB) + mutation.
Both parents pay ``repro_cost`` energy and enter cooldown. Respects population caps.
"""
from __future__ import annotations

import numpy as np

from config import SHEEP, FOX
from sim import genome as gn
from sim.brain import A_REPRO


def apply(cfg, world, ent, idx, act, species_grids, rng):
    if idx.shape[0] == 0:
        return 0

    repro_gate = act[:, A_REPRO] > 0.5
    total_births = 0

    for species_id in (SHEEP, FOX):
        spec = cfg.species[species_id]
        # current population (for cap)
        pop = ent.count_species(species_id)
        if pop >= spec.population_cap:
            continue

        # eligibility mask over the tick's idx rows
        rows = np.nonzero((ent.species[idx] == species_id) & repro_gate)[0]
        if rows.shape[0] == 0:
            continue
        slots = idx[rows]
        genome = ent.genome[slots]
        repro_threshold = gn.gene(genome, "repro_threshold")
        eligible = (
            (ent.age[slots] >= spec.maturity_age)
            & (ent.energy[slots] >= repro_threshold)
            & (ent.repro_cooldown[slots] <= 0.0)
            & (ent.hunger[slots] <= spec.repro_max_hunger)
            & (ent.thirst[slots] <= spec.repro_max_thirst)
        )
        elig_slots = slots[eligible]
        if elig_slots.shape[0] < 2:
            continue

        grid = species_grids.get(species_id)
        if grid is None:
            continue

        elig_set = set(elig_slots.tolist())
        used = set()
        pairs = []
        # deterministic: pair greedily by ascending slot index
        for a in sorted(elig_slots.tolist()):
            if a in used:
                continue
            ax, ay = float(ent.pos_x[a]), float(ent.pos_y[a])
            cand, cpx, cpy = grid.query_radius(ax, ay, cfg.sim.repro_radius)
            if cand.shape[0] == 0:
                continue
            # eligible, opposite sex, not self, not used
            best = None
            best_d2 = np.inf
            for c, cx, cy in zip(cand.tolist(), cpx.tolist(), cpy.tolist()):
                if c == a or c in used or c not in elig_set:
                    continue
                if ent.sex[c] == ent.sex[a]:
                    continue
                d2 = (cx - ax) ** 2 + (cy - ay) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = c
            if best is not None:
                used.add(a)
                used.add(best)
                pairs.append((a, best))

        if not pairs:
            continue

        # build offspring genomes via crossover+mutation
        room = spec.population_cap - pop
        n_pairs = len(pairs)
        # respect both pop cap and pool capacity
        parents_a = np.array([p[0] for p in pairs], dtype=np.intp)
        parents_b = np.array([p[1] for p in pairs], dtype=np.intp)

        litter = spec.litter_size
        ga_rows = []
        gb_rows = []
        mom = []
        for li in range(litter):
            ga_rows.append(parents_a)
            gb_rows.append(parents_b)
            mom.append(parents_a)
        ga = ent.genome[np.concatenate(ga_rows)]
        gb = ent.genome[np.concatenate(gb_rows)]
        n_children = ga.shape[0]
        n_children = min(n_children, max(0, room))
        if n_children <= 0:
            continue
        ga = ga[:n_children]
        gb = gb[:n_children]
        mom_slots = np.concatenate(mom)[:n_children]

        children = gn.crossover(ga, gb, spec, rng)
        # spawn near the mother with a small offset
        offs = rng.uniform(-1.5, 1.5, size=(n_children, 2)).astype(np.float32)
        pos = np.stack([ent.pos_x[mom_slots] + offs[:, 0],
                        ent.pos_y[mom_slots] + offs[:, 1]], axis=1)
        pos[:, 0] = np.clip(pos[:, 0], 0, world.w - 1e-3)
        pos[:, 1] = np.clip(pos[:, 1], 0, world.h - 1e-3)
        new_slots = ent.spawn(spec, children, pos, rng, energy=0.6)
        born = new_slots.shape[0]
        total_births += born

        # parents pay cost + cooldown (only those that actually produced a child)
        produced = min(n_pairs, born // max(1, litter) + (1 if born % max(1, litter) else 0))
        pay_a = parents_a[:produced] if produced > 0 else parents_a[:0]
        pay_b = parents_b[:produced] if produced > 0 else parents_b[:0]
        for pset in (pay_a, pay_b):
            ent.energy[pset] = np.maximum(0.0, ent.energy[pset] - spec.repro_cost)
            ent.repro_cooldown[pset] = spec.repro_cooldown

    return total_births
