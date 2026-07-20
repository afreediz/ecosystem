"""Systems registry: a uniform ``System.apply(ctx)`` interface + an ordered default pipeline.

The simulation tick was historically a fixed hand-written sequence of function calls inside
``Simulation.step``. It is now an ordered list of ``System`` objects that each read/write a
shared per-tick ``StepContext``. This lets a developer insert, replace, or reorder tick
systems (a disease system, a migration system, ...) without editing the core -- e.g.::

    sim = Simulation(cfg, systems=[*default_pipeline(cfg), MyDiseaseSystem()])

The ``default_pipeline`` reproduces the exact original order, so the run stays byte-identical:

    Environment -> Grid -> Perception -> Brain -> Sleep -> Movement -> Consumption
    -> Metabolism -> Reproduction -> Vegetation -> Stats

Determinism warning: several systems draw from the run RNG (movement, consumption, metabolism,
reproduction) at fixed points; REORDERING those relative to each other changes the RNG stream
and therefore the run. The thin System classes wrap the existing free-function ``apply`` in
each system module, which remain callable directly (used by the notebooks/tests).
"""
from __future__ import annotations

import numpy as np

from darwinism.sim.systems import consumption, metabolism, movement, reproduction, sleep, vegetation


class StepContext:
    """Mutable shared state threaded through every system for one tick.

    Persistent handles (``cfg``/``world``/``env``/``ent``/``veg``/``species_grids``/
    ``perception``/``rng``) plus the per-tick working set (``obs``/``idx``/``act``/
    ``temp_field``) and result tallies the Stats system reads. ``compact_dead`` narrows the
    working ``idx``/``act`` to survivors -- called by the systems that kill (consumption,
    metabolism) at the exact points deaths were dropped in the original step.
    """

    __slots__ = ("sim", "cfg", "world", "env", "ent", "rng", "dt", "tick", "veg",
                 "species_grids", "perception", "veg_growth_paused", "temp_field",
                 "obs", "idx", "act", "killed", "n_drink", "n_graze", "n_pred",
                 "n_asleep", "causes", "births", "stats")

    def __init__(self, sim, dt):
        self.sim = sim
        self.cfg = sim.cfg
        self.world = sim.world
        self.env = sim.env
        self.ent = sim.entities
        self.rng = sim.rng
        self.dt = dt
        self.tick = sim.tick
        self.veg = sim.veg
        self.species_grids = sim._species_grids
        self.perception = sim.perception
        self.veg_growth_paused = sim.veg_growth_paused
        # per-tick working set (filled by systems)
        self.temp_field = None
        self.obs = None
        self.idx = None
        self.act = None
        # result tallies (for Stats / logger)
        self.killed = None
        self.n_drink = 0
        self.n_graze = 0
        self.n_pred = 0
        self.n_asleep = 0
        self.causes = {"starve": 0, "thirst": 0, "age": 0, "health": 0}
        self.births = 0
        self.stats = {}

    def compact_dead(self) -> None:
        """Drop entities that died this tick from the working set (idx/act)."""
        alive_mask = self.ent.alive[self.idx]
        self.idx = self.idx[alive_mask]
        self.act = self.act[alive_mask]


class System:
    """Base class for a tick system. Subclass and implement ``apply(ctx)``; insert the
    instance into ``Simulation.systems`` (or a list passed to ``Simulation(systems=...)``)."""

    def apply(self, ctx: StepContext) -> None:
        raise NotImplementedError


class EnvironmentSystem(System):
    def apply(self, ctx):
        ctx.env.update(ctx.dt)
        ctx.temp_field = ctx.env.temperature_field(ctx.world.static_temp)


class GridSystem(System):
    """Rebuild the per-species spatial hashes for this tick's positions."""
    def apply(self, ctx):
        ent = ctx.ent
        for sid, g in ctx.species_grids.items():
            sidx = np.nonzero(ent.species_mask(sid))[0]
            g.rebuild(sidx, ent.pos_x, ent.pos_y)


class PerceptionSystem(System):
    def apply(self, ctx):
        p = ctx.perception
        p._species_grids = ctx.species_grids
        p.veg = ctx.veg
        p.temp_field = ctx.temp_field
        ctx.obs, ctx.idx = p.build()


class BrainSystem(System):
    """Batched decision: the (possibly composite/learned) brain maps observations -> actions."""
    def apply(self, ctx):
        ctx.act = ctx.sim.brain_system.decide(ctx.obs, ctx.idx)


class SleepSystem(System):
    def apply(self, ctx):
        ctx.n_asleep = sleep.apply(ctx.cfg, ctx.world, ctx.ent, ctx.idx, ctx.act, ctx.obs, ctx.env)


class MovementSystem(System):
    def apply(self, ctx):
        movement.apply(ctx.cfg, ctx.world, ctx.ent, ctx.idx, ctx.act, ctx.rng)


class ConsumptionSystem(System):
    def apply(self, ctx):
        ctx.killed, ctx.n_drink, ctx.n_graze, ctx.n_pred = consumption.apply(
            ctx.cfg, ctx.world, ctx.ent, ctx.idx, ctx.act, ctx.veg, ctx.species_grids, ctx.rng)
        if ctx.killed.shape[0] > 0:      # drop dead prey before later systems read the set
            ctx.compact_dead()


class MetabolismSystem(System):
    def apply(self, ctx):
        ctx.causes = metabolism.apply(ctx.cfg, ctx.world, ctx.ent, ctx.idx, ctx.act,
                                      ctx.temp_field, ctx.env, ctx.rng)
        ctx.compact_dead()               # drop starved/aged/etc before reproduction


class ReproductionSystem(System):
    def apply(self, ctx):
        ctx.births = reproduction.apply(ctx.cfg, ctx.world, ctx.ent, ctx.idx, ctx.act,
                                        ctx.species_grids, ctx.rng)


class VegetationSystem(System):
    def apply(self, ctx):
        if not ctx.veg_growth_paused:    # paused: grazed cells stay depleted (live-viewer toggle)
            vegetation.grow(ctx.cfg, ctx.world, ctx.env, ctx.veg, ctx.dt)


class StatsSystem(System):
    """Assemble the per-tick stats dict consumed by the logger / HUD."""
    def apply(self, ctx):
        ent = ctx.ent
        deaths_total = sum(ctx.causes.values()) + ctx.n_pred
        stats = {"tick": ctx.tick}
        # per-species live counts, keyed n_<name> (n_sheep/n_fox for the default config)
        for sid in sorted(ctx.cfg.species):
            stats[f"n_{ctx.cfg.species[sid].name}"] = ent.count_species(sid)
        stats.update({
            "veg_biomass": float(ctx.veg.sum()),
            "births": ctx.births,
            "deaths": deaths_total,
            "death_starve": ctx.causes["starve"],
            "death_thirst": ctx.causes["thirst"],
            "death_age": ctx.causes["age"],
            "death_health": ctx.causes["health"],
            "death_predation": ctx.n_pred,
            "n_drink": ctx.n_drink,
            "n_graze": ctx.n_graze,
            "n_asleep": ctx.n_asleep,
        })
        ctx.stats = stats


def default_pipeline(cfg) -> list:
    """The ordered default tick pipeline (reproduces the original ``Simulation.step`` order)."""
    return [
        EnvironmentSystem(),
        GridSystem(),
        PerceptionSystem(),
        BrainSystem(),
        SleepSystem(),
        MovementSystem(),
        ConsumptionSystem(),
        MetabolismSystem(),
        ReproductionSystem(),
        VegetationSystem(),
        StatsSystem(),
    ]
