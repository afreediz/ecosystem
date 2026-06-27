"""Simulation: owns world + entities + systems; exposes ``step(dt)`` (§7.4 of v1.md).

Headless and deterministic. A single seeded RNG (from config) is threaded into every
system that needs randomness. This module imports NOTHING from render/.

Fixed tick order:
  1 environment.update      5 sleep.apply           9  metabolism.apply
  2 grid.rebuild            6 movement.apply        10 reproduction.apply
  3 perception.build        7 consumption.apply     11 vegetation.grow
  4 brain_system.decide                             12 logger.record (caller)
                                                       (sleep gates act before movement)
"""
from __future__ import annotations

import numpy as np

from config import Config, SHEEP, FOX
from sim.world import World
from sim.environment import Environment
from sim.entities import Entities
from sim.grid import SpatialGrid
from sim.perception import Perception
from sim.brain import RuleBrain
from sim.systems.brain_system import BrainSystem
from sim.systems import movement, consumption, metabolism, reproduction, vegetation, sleep
from sim import genome as gn


class Simulation:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.rng = self.cfg.make_rng()

        # world is generated once from the master seed
        self.world = World(self.cfg.world, self.rng)
        self.env = Environment(self.cfg.env, self.rng)
        self.entities = Entities(self.cfg)

        # vegetation per-cell field (owned here; mutated by the vegetation system)
        self.veg = vegetation.initial_field(self.world, self.rng)
        # when True, vegetation stops regrowing (grazing still depletes it) -- a live
        # viewer toggle; defaults off so headless runs are unaffected.
        self.veg_growth_paused = False

        # spatial grids: one global (all animals) + one per species, rebuilt each tick
        cell = self.cfg.sim.grid_cell_size
        self.grid = SpatialGrid(self.world.w, self.world.h, cell)
        self._species_grids = {
            SHEEP: SpatialGrid(self.world.w, self.world.h, cell),
            FOX: SpatialGrid(self.world.w, self.world.h, cell),
        }

        self.perception = Perception(self.cfg, self.world, self.entities, self.grid, self.env)
        self.brain = RuleBrain(self.rng, self.cfg.sim.food_eat_threshold)
        self.brain_system = BrainSystem(self.brain)

        self.tick = 0
        # per-tick stats populated by step() for the logger / HUD
        self.stats = {}

        self._seed_population()

    # ------------------------------------------------------------------ setup
    def _seed_population(self):
        # Spawn each species in a handful of tight herds/packs (not scattered uniformly):
        # group starts bootstrap mate-finding and form persistent breeding demes, which is
        # what lets the predator avoid the lone-disperser extinction trap.
        cluster_cfg = {SHEEP: (8, 6.0), FOX: (5, 4.0)}
        for species_id in (SHEEP, FOX):
            spec = self.cfg.species[species_id]
            n = spec.init_count
            n_clusters, spread = cluster_cfg[species_id]
            genomes = gn.random_genomes(spec, n, self.rng)
            pos = self.world.clustered_land_positions(
                n, self.rng, n_clusters=n_clusters, spread=spread, near_freshwater=True)
            # Seed founders as ADULTS (age from maturity up to ~half their lifespan) so the
            # population can breed from tick 0. Spawning everyone at age 0 leaves a long
            # juvenile window in which the founders die off before they can reproduce.
            ages = self.rng.uniform(spec.maturity_age, spec.maturity_age * 3.0,
                                    size=n).astype(np.float32)
            self.entities.spawn(spec, genomes, pos, self.rng, energy=0.8, age=ages)

    def spawn_at(self, species_id: int, x: float, y: float, n: int = 1) -> np.ndarray:
        """Spawn ``n`` adult animals of ``species_id`` near world coords ``(x, y)``.

        Used by the live viewer for manual spawning. Clamps into the world and nudges to
        the nearest passable land cell if the target is water/blocked; returns [] if none
        is found nearby. NOTE: this draws from the master RNG, so manual spawning (like any
        live interaction) breaks run reproducibility -- it is never used by the headless path.
        """
        spec = self.cfg.species[species_id]
        x = float(np.clip(x, 0.0, self.world.w - 1e-3))
        y = float(np.clip(y, 0.0, self.world.h - 1e-3))
        if not bool(self.world.is_passable(x, y)):
            spot = self._nearest_passable(x, y)
            if spot is None:
                return np.empty(0, dtype=np.intp)
            x, y = spot
        genomes = gn.random_genomes(spec, n, self.rng)
        jitter = self.rng.uniform(-0.3, 0.3, size=(n, 2)).astype(np.float32) if n > 1 else 0.0
        pos = (np.array([x, y], dtype=np.float32) + jitter).reshape(n, 2)
        ages = self.rng.uniform(spec.maturity_age, spec.maturity_age * 3.0,
                                size=n).astype(np.float32)
        return self.entities.spawn(spec, genomes, pos, self.rng, energy=0.8, age=ages)

    def _nearest_passable(self, x: float, y: float):
        """Nearest passable land cell center to (x, y), searching outward up to 6 cells."""
        cx, cy = int(x), int(y)
        for r in range(0, 7):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.world.w and 0 <= ny < self.world.h:
                        if self.world.passable[ny, nx]:
                            return (nx + 0.5, ny + 0.5)
        return None

    def _rebuild_grids(self):
        ent = self.entities
        alive = ent.alive_indices()
        self.grid.rebuild(alive, ent.pos_x, ent.pos_y)
        for sid, g in self._species_grids.items():
            sidx = np.nonzero(ent.species_mask(sid))[0]
            g.rebuild(sidx, ent.pos_x, ent.pos_y)

    # ------------------------------------------------------------------ tick
    def step(self, dt: float | None = None):
        dt = self.cfg.sim.dt if dt is None else dt
        ent = self.entities
        world = self.world

        # 1. environment
        self.env.update(dt)
        temp_field = self.env.temperature_field(world.static_temp)

        # 2. spatial hashes
        self._rebuild_grids()

        # wire per-tick context into perception
        self.perception._species_grids = self._species_grids
        self.perception.veg = self.veg

        # 3. perception -> obs matrix (LOCAL, radius-gated)
        obs, idx = self.perception.build(temp_field)

        # 4. batched brain decision
        act = self.brain_system.decide(obs)

        # 5. circadian rest: night-time sleepers head for cover, then bed down. Gates the
        #    action matrix (and sets ent.asleep) BEFORE movement/consumption read it.
        n_asleep = sleep.apply(self.cfg, world, ent, idx, act, obs, self.env)

        # 6. movement
        movement.apply(self.cfg, world, ent, idx, act, self.rng)

        # 7. consumption (grids may now be slightly stale re: positions, fine for adjacency
        #    checks which we recompute exactly). Predation kills prey slots immediately.
        killed, n_drink, n_graze, n_pred = consumption.apply(
            self.cfg, world, ent, idx, act, self.veg, self._species_grids, self.rng)

        # drop dead prey from the tick's working set so later systems skip them
        if killed.shape[0] > 0:
            alive_mask = ent.alive[idx]
            idx = idx[alive_mask]
            act = act[alive_mask]

        # 9. metabolism (energy/hunger/thirst/health/aging/death; sleepers burn less)
        causes = metabolism.apply(self.cfg, world, ent, idx, temp_field, self.env, self.rng)

        # drop dead from the working set before reproduction
        alive_mask = ent.alive[idx]
        idx = idx[alive_mask]
        act = act[alive_mask]

        # 10. reproduction
        births = reproduction.apply(self.cfg, world, ent, idx, act,
                                    self._species_grids, self.rng)

        # 11. vegetation growth (skipped when paused: grazed cells stay depleted)
        if not self.veg_growth_paused:
            vegetation.grow(self.cfg, world, self.env, self.veg, dt)

        # 12. stats for logger / HUD
        self.tick += 1
        deaths_total = sum(causes.values()) + n_pred
        self.stats = {
            "tick": self.tick,
            "n_sheep": ent.count_species(SHEEP),
            "n_fox": ent.count_species(FOX),
            "veg_biomass": float(self.veg.sum()),
            "births": births,
            "deaths": deaths_total,
            "death_starve": causes["starve"],
            "death_thirst": causes["thirst"],
            "death_age": causes["age"],
            "death_health": causes["health"],
            "death_predation": n_pred,
            "n_drink": n_drink,
            "n_graze": n_graze,
            "n_asleep": n_asleep,
        }
        return self.stats

    # ------------------------------------------------------------------ analysis helpers
    def trait_means(self, species_id: int) -> dict:
        """Mean of each heritable gene over the living members of a species."""
        ent = self.entities
        mask = ent.species_mask(species_id)
        out = {}
        if mask.sum() == 0:
            for name in gn.GENE_NAMES:
                out[name] = float("nan")
            return out
        g = ent.genome[mask]
        for name in gn.GENE_NAMES:
            out[name] = float(gn.gene(g, name).mean())
        return out

    @property
    def populations(self) -> dict:
        return {"sheep": self.entities.count_species(SHEEP),
                "fox": self.entities.count_species(FOX)}
