"""Vegetation field growth (§10, §14 of v1.md).

Vegetation is a per-cell field (NOT thousands of plant entities). Each land cell grows
logistically toward a carrying capacity set by local nutrients x moisture x plant
suitability x seasonal growth. Growth slowly consumes nutrients; nutrients regenerate
toward 1.0. No growth on water or unsuitable cells.
"""
from __future__ import annotations

import numpy as np


def initial_field(world, rng: np.random.Generator) -> np.ndarray:
    """Seed the vegetation field at a fraction of each cell's carrying capacity."""
    cap = _carrying_capacity(world, growth_mult=1.0)
    veg = cap * rng.uniform(0.2, 0.6, size=cap.shape).astype(np.float32)
    veg[world.water_any] = 0.0
    return veg.astype(np.float32)


def _carrying_capacity(world, growth_mult: float) -> np.ndarray:
    cap = world.plant_suitability * world.nutrients * (0.4 + 0.6 * world.moisture)
    cap = cap * growth_mult
    cap[world.water_any] = 0.0
    return np.clip(cap, 0.0, 1.0).astype(np.float32)


def grow(cfg, world, env, veg: np.ndarray, dt: float) -> None:
    """Logistic growth toward carrying capacity; consumes & regenerates nutrients."""
    growth_mult = env.growth_multiplier()
    cap = _carrying_capacity(world, growth_mult)

    # logistic growth: dV = rate * V * (1 - V/cap); seed a tiny base so empty cells recover
    rate = cfg.sim.veg_regrow_rate * dt
    safe_cap = np.maximum(cap, 1e-4)
    seed = 0.01 * cap                       # lets fully-grazed suitable cells restart
    effective = np.maximum(veg, seed)
    growth = rate * effective * (1.0 - effective / safe_cap)
    growth = np.where(cap > 1e-4, growth, 0.0)
    veg += growth
    np.clip(veg, 0.0, cap, out=veg)
    veg[world.water_any] = 0.0

    # nutrient dynamics: growth consumes a little; regen toward 1.0 on land
    world.nutrients -= growth * 0.05
    land = ~world.water_any
    world.nutrients[land] += cfg.env.nutrient_regen_rate * dt
    np.clip(world.nutrients, 0.0, 1.0, out=world.nutrients)
