"""Metabolism: energy/hunger/thirst/health/aging and death (§14 of v1.md).

Runs per dt over the given alive indices. Vectorized across all species (per-species
rates pulled from config). Returns a death-cause tally for logging.
"""
from __future__ import annotations

import numpy as np

from config import SHEEP, FOX
from sim import genome as gn


def apply(cfg, world, ent, idx, temp_field, env, rng):
    causes = {"starve": 0, "thirst": 0, "age": 0, "health": 0}
    if idx.shape[0] == 0:
        return causes
    dt = cfg.sim.dt

    genome = ent.genome[idx]
    size = gn.gene(genome, "size")
    max_speed = gn.gene(genome, "max_speed")
    metab = gn.gene(genome, "metabolism_rate")
    max_age = gn.gene(genome, "max_age")
    spec = ent.species[idx]

    # per-species scalar rates broadcast to each agent
    base_burn = np.where(spec == FOX, cfg.species[FOX].base_burn,
                         cfg.species[SHEEP].base_burn).astype(np.float32)
    move_cost = np.where(spec == FOX, cfg.species[FOX].move_cost,
                         cfg.species[SHEEP].move_cost).astype(np.float32)
    hunger_rate = np.where(spec == FOX, cfg.species[FOX].hunger_rate,
                           cfg.species[SHEEP].hunger_rate).astype(np.float32)
    thirst_rate = np.where(spec == FOX, cfg.species[FOX].thirst_rate,
                           cfg.species[SHEEP].thirst_rate).astype(np.float32)

    # energy burn scales with metabolism gene, plus a movement cost proxy (speed*size)
    burn = (base_burn + move_cost * max_speed * size) * metab * dt
    ent.energy[idx] = ent.energy[idx] - burn

    # hunger / thirst accumulate (thirst scaled by local heat)
    ent.hunger[idx] = np.clip(ent.hunger[idx] + hunger_rate * metab * dt, 0.0, 1.5)
    px, py = ent.pos_x[idx], ent.pos_y[idx]
    cx, cy = world.world_to_cell(px, py)
    local_temp = temp_field[cy, cx]
    heat_factor = 0.6 + 1.2 * local_temp            # hotter cells -> thirstier
    if env.weather == 2:  # HEAT
        heat_factor *= cfg.env.heat_thirst_factor
    ent.thirst[idx] = np.clip(ent.thirst[idx] + thirst_rate * heat_factor * dt, 0.0, 1.5)

    # high hunger/thirst drain energy and health
    starving = ent.hunger[idx] > 0.85
    parched = ent.thirst[idx] > 0.85
    ent.energy[idx] = ent.energy[idx] - np.where(starving, 0.01 * dt, 0.0)
    ent.health[idx] = ent.health[idx] - np.where(starving, 0.012 * dt, 0.0)
    ent.health[idx] = ent.health[idx] - np.where(parched, 0.018 * dt, 0.0)
    # recover health slowly when well-fed and watered
    healthy = (ent.hunger[idx] < 0.5) & (ent.thirst[idx] < 0.5) & (ent.energy[idx] > 0.3)
    ent.health[idx] = np.clip(ent.health[idx] + np.where(healthy, 0.006 * dt, 0.0), 0.0, 1.0)

    # aging
    ent.age[idx] = ent.age[idx] + dt
    ent.repro_cooldown[idx] = np.maximum(0.0, ent.repro_cooldown[idx] - dt)

    # --- deaths ---
    energy = ent.energy[idx]
    thirst = ent.thirst[idx]
    health = ent.health[idx]
    age = ent.age[idx]

    dead_energy = energy <= 0.0
    dead_thirst = thirst >= 1.5
    dead_health = health <= 0.0
    # rising death probability as age approaches/exceeds max_age
    age_frac = age / np.maximum(max_age, 1e-6)
    age_prob = np.clip((age_frac - 0.9) * 2.0, 0.0, 1.0) * dt * 0.5
    dead_age = rng.random(idx.shape[0]) < age_prob
    dead_age |= age >= max_age

    dead = dead_energy | dead_thirst | dead_health | dead_age
    if dead.any():
        # attribute a single cause (priority: thirst > starve > health > age)
        causes["thirst"] = int((dead & dead_thirst).sum())
        rem = dead & ~dead_thirst
        causes["starve"] = int((rem & dead_energy).sum())
        rem = rem & ~dead_energy
        causes["health"] = int((rem & dead_health).sum())
        rem = rem & ~dead_health
        causes["age"] = int((rem & dead_age).sum())
        ent.kill(idx[dead])
    return causes
