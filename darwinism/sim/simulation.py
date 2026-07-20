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

from darwinism.config import Config
from darwinism.sim import genome as gn
from darwinism.sim.brain import CompositeBrain, RuleBrain
from darwinism.sim.entities import Entities
from darwinism.sim.environment import Environment
from darwinism.sim.grid import SpatialGrid
from darwinism.sim.perception import Perception
from darwinism.sim.systems import vegetation  # initial_field used at construction
from darwinism.sim.systems.brain_system import BrainSystem
from darwinism.sim.systems.pipeline import StepContext, default_pipeline
from darwinism.sim.world import World


class Simulation:
    def __init__(self, cfg: Config | None = None, brain=None, systems=None):
        self.cfg = cfg or Config()
        # build the gene layout from the registered species BEFORE the entity store is sized
        # (its genome array is (max_entities, N_GENES)). For the default sheep+fox set this
        # reproduces the historical 9-gene layout, so the run stays byte-identical.
        gn.build_registry(self.cfg.species)
        # run RNG: drives all stochastic dynamics; resolves a random seed if none was set
        self.rng = self.cfg.make_rng()

        # world is generated once from the WORLD seed only (independent of the run RNG), so
        # the same world seed reproduces the same map regardless of the run/determinism seed
        self.world = World(self.cfg.world)
        self.env = Environment(self.cfg.env, self.rng)
        self.entities = Entities(self.cfg)

        # vegetation per-cell field (owned here; mutated by the vegetation system)
        self.veg = vegetation.initial_field(self.world, self.rng)
        # when True, vegetation stops regrowing (grazing still depletes it) -- a live
        # viewer toggle; defaults off so headless runs are unaffected.
        self.veg_growth_paused = False

        # spatial grids: one per registered species, rebuilt each tick (queries are species-scoped)
        cell = self.cfg.sim.grid_cell_size
        self._species_grids = {
            sid: SpatialGrid(self.world.w, self.world.h, cell)
            for sid in sorted(self.cfg.species)
        }

        self.perception = Perception(self.cfg, self.world, self.entities, self.env)
        # brain is pluggable: default is the hardcoded RuleBrain, but any object honouring the
        # Brain contract (e.g. sim.policy_brain.PolicyBrain) can be injected. A brain that
        # keeps per-agent memory (e.g. an LSTM) is given a handle on the entity store via
        # bind(), so it can reset an agent's memory when its slot is recycled.
        self.brain = self._resolve_brain(brain)
        if hasattr(self.brain, "bind"):
            self.brain.bind(self.entities)
        self.brain_system = BrainSystem(self.brain)

        # ordered tick pipeline (a list of System objects sharing a StepContext). Defaults to
        # the canonical order; pass ``systems=`` (or edit this list) to insert/replace/reorder
        # tick systems. Reordering RNG-drawing systems changes the run -- see systems.pipeline.
        self.systems = systems if systems is not None else default_pipeline(self.cfg)

        self.tick = 0
        # per-tick stats populated by step() for the logger / HUD
        self.stats = {}
        # latest per-species perception {species_id: Observation}, exposed for observers
        # (live viewer entity inspector); set by step(). Each Observation carries its own idx.
        self.last_obs = None

        self._seed_population()

    # ------------------------------------------------------------------ brain wiring
    def _resolve_brain(self, brain):
        """Turn the ``brain`` argument into a concrete Brain.

        - ``None``        -> the default hardcoded RuleBrain (on the run RNG).
        - a ``Brain``     -> used as-is (a single brain drives every species, back-compat).
        - a ``dict``      -> a per-species spec ``{species_id: Brain | None}``: species mapped to
          a Brain use it; species mapped to ``None`` fall back to a shared RuleBrain built here
          on the run RNG (so its explore headings come from the single run Generator, per the
          determinism contract). Wrapped in a ``CompositeBrain`` that routes per species.
        """
        if brain is None:
            return RuleBrain(self.rng)
        if isinstance(brain, dict):
            rule = None
            resolved = {}
            for sid in sorted(self.cfg.species):
                b = brain.get(sid)
                if b is None:
                    if rule is None:
                        rule = RuleBrain(self.rng)
                    b = rule
                resolved[sid] = b
            return CompositeBrain(resolved)
        return brain

    # ------------------------------------------------------------------ setup
    def _seed_population(self):
        # Spawn each species in a handful of tight herds/packs (not scattered uniformly):
        # group starts bootstrap mate-finding and form persistent breeding demes, which is
        # what lets the predator avoid the lone-disperser extinction trap.
        for species_id in sorted(self.cfg.species):
            spec = self.cfg.species[species_id]
            n = spec.init_count
            n_clusters, spread = spec.cluster
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
        for sid, g in self._species_grids.items():
            sidx = np.nonzero(ent.species_mask(sid))[0]
            g.rebuild(sidx, ent.pos_x, ent.pos_y)

    # ------------------------------------------------------------------ tick
    def step(self, dt: float | None = None):
        """Advance one tick by running the ordered system pipeline over a shared context.

        The fixed sequence (environment -> grid -> perception -> brain -> sleep -> movement ->
        consumption -> metabolism -> reproduction -> vegetation -> stats) now lives in
        ``self.systems``; each ``System`` reads/writes the ``StepContext``. Behaviour is
        identical to the original hand-written sequence.
        """
        dt = self.cfg.sim.dt if dt is None else dt
        self.tick += 1
        ctx = StepContext(self, dt)          # captures tick, veg, grids, perception, paused flag
        for system in self.systems:
            system.apply(ctx)
        # expose this tick's per-species observations for an observer (the live viewer's entity
        # inspector). Each Observation carries its own ``idx`` captured at perception time, so a
        # row still maps to its slot even though deaths were filtered from the working set later.
        self.last_obs = ctx.obs
        self.stats = ctx.stats
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
        """Live count per species, keyed by species name (e.g. {"sheep": .., "fox": ..})."""
        return {self.cfg.species[sid].name: self.entities.count_species(sid)
                for sid in sorted(self.cfg.species)}
