"""Metabolism: energy/hunger/thirst/health/aging and death (§14 of v1.md).

Runs per dt over the given alive indices. Vectorized across all species (per-species
rates pulled from config). Returns a death-cause tally for logging.
"""
from __future__ import annotations

import numpy as np

from darwinism.sim import genome as gn
from darwinism.sim.brain import A_SPEED


def _rate_lut(cfg, attr):
    """Per-species scalar rate as a lookup array indexed by species id, so ``lut[spec]``
    broadcasts the right rate to each agent -- generalises the old 2-way np.where(spec==FOX,..)
    to any number of species (numerically identical for contiguous ids)."""
    table = np.zeros(max(cfg.species) + 1, dtype=np.float32)
    for sid, s in cfg.species.items():
        table[sid] = getattr(s, attr)
    return table


def apply(cfg, world, ent, idx, act, temp_field, env, rng):
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

    # per-species scalar rates broadcast to each agent (via id-indexed lookup tables)
    base_burn = _rate_lut(cfg, "base_burn")[spec]
    move_cost = _rate_lut(cfg, "move_cost")[spec]
    hunger_rate = _rate_lut(cfg, "hunger_rate")[spec]
    thirst_rate = _rate_lut(cfg, "thirst_rate")[spec]

    # sleepers rest: only a reduced basal burn (no locomotion cost) and needs accrue slower.
    # locomotion cost scales with the throttle the brain actually used this tick (0 = stood
    # still, 1 = full speed), so an animal that eased off to feed pays less than a sprinter.
    asleep = ent.asleep[idx]
    throttle = np.clip(act[:, A_SPEED], 0.0, 1.0)
    awake_burn = base_burn + move_cost * throttle * max_speed * size
    rest_burn = base_burn * cfg.env.sleep_burn_factor
    burn = np.where(asleep, rest_burn, awake_burn) * metab * dt
    ent.energy[idx] = ent.energy[idx] - burn

    # hunger / thirst accumulate (thirst scaled by local heat; slower while asleep)
    need_factor = np.where(asleep, cfg.env.sleep_need_factor, 1.0).astype(np.float32)
    ent.hunger[idx] = np.clip(
        ent.hunger[idx] + hunger_rate * metab * need_factor * dt, 0.0, 1.5)
    px, py = ent.pos_x[idx], ent.pos_y[idx]
    cx, cy = world.world_to_cell(px, py)
    local_temp = temp_field[cy, cx]
    heat_factor = 0.6 + 1.2 * local_temp            # hotter cells -> thirstier
    if env.weather == 2:  # HEAT
        heat_factor *= cfg.env.heat_thirst_factor
    ent.thirst[idx] = np.clip(
        ent.thirst[idx] + thirst_rate * heat_factor * need_factor * dt, 0.0, 1.5)

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
    # cosmetic mating glow fades out (viewer-only; does not influence any decision)
    ent.mating_glow[idx] = np.maximum(0.0, ent.mating_glow[idx] - dt)

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
