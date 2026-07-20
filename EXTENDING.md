# Extending darwinism

`darwinism` is a framework, not just an app: you `import darwinism`, compose a `Config`, and
build around four extension points — **species**, **brains**, **tick-systems**, and
**traits** — without editing the core. This guide covers each, plus the determinism rules you
must respect.

```python
import darwinism as dw

cfg = dw.make_config(world_seed=12345, seed=7)
sim = dw.Simulation(cfg)
for _ in range(9000):
    stats = sim.step()
print(sim.populations)          # {'sheep': ..., 'fox': ...}
```

Runnable versions of everything below live in [`examples/`](examples/).

## The determinism contract (read this first)

The whole simulation is **deterministic**: *same `world_seed` + same `Config` + same run
`seed` ⇒ byte-identical run*. Two seeds are independent — `world.seed` (terrain/hydrology)
and `Config.seed` (all stochastic dynamics, via one `numpy.random.Generator`). There is no
global `np.random`; every system draws from the one run RNG threaded through it.

What this asks of your extensions:

- **Iterate species in `sorted(cfg.species)` order** (ascending id) anywhere order is
  observable. New species should take ids after the existing ones.
- **Don't reorder the RNG-drawing systems** (movement, consumption, metabolism,
  reproduction) relative to each other — that changes the RNG stream and hence the run.
- A **new species is a new `Config`**, so it legitimately produces a *different* (still
  reproducible) run. Adding one never breaks the built-in sheep+fox run.
- A custom `Brain` that draws randomness should use **its own** `Generator` (not the sim's),
  or act deterministically, so it doesn't perturb the systems' shared stream.

## 1. Add a new species

A species is declared, not coded. Build a `SpeciesConfig`, give it a `diet`, gene ranges, and
metabolic/reproductive parameters, then add it to `cfg.species`. Perception channels, the
genome layout, the consumption/reproduction systems, and the stats/logger all adapt from the
declaration.

```python
RABBIT = 2
rabbit = dw.SpeciesConfig(
    name="rabbit", species_id=RABBIT, init_count=90,
    diet=[dw.FieldFood(field="vegetation", eat_value=0.7)],   # herbivore: grazes a world field
    cluster=(5, 5.0),                                         # founder herds (n_clusters, spread)
    gene_ranges={
        "max_speed": dw.GeneRange(0.8, 2.2),
        "sensory_range": dw.GeneRange(6.0, 18.0),
        "metabolism_rate": dw.GeneRange(0.7, 1.3),
        "size": dw.GeneRange(0.4, 0.9),
        "max_age": dw.GeneRange(1000.0, 2000.0),
        "repro_threshold": dw.GeneRange(0.45, 0.75),
        "chronotype": dw.GeneRange(-0.06, 0.06),
    },
    maturity_age=80.0, repro_cost=0.2, repro_cooldown=70.0, litter_size=3,
    hunger_rate=0.0045, thirst_rate=0.0022, base_burn=0.0022, move_cost=0.005,
    population_cap=800,
)
cfg = dw.make_config(world_seed=12345, seed=7)
cfg.species[RABBIT] = rabbit
sim = dw.Simulation(cfg)
```

**Diet** is the crux — it makes food/threat data-driven:

- `FieldFood(field="vegetation", eat_value=...)` — a **herbivore** that grazes a per-cell
  world field. Perception reduces food to the *best* cell; the rule brain heads there.
  (`"vegetation"` is the only grazeable field today.)
- `PreyFood(prey=[...], predation_gain=, hunt_success=, hunt_halfsat=)` — a **carnivore** that
  hunts the listed species. Perception reduces food to the *nearest* exposed prey; predation
  uses a Type III kill probability.

**Relationships are derived, never double-declared.** Because fox declares
`PreyFood(prey=[SHEEP])`, the framework computes `cfg.predators_of()[SHEEP] == [FOX]`, which is
what gives sheep a `threat` perception channel and a predator-wake reflex. To make foxes also
hunt your rabbit: `cfg.species[dw.FOX].diet[0].prey.append(RABBIT)`. A species with **no**
predators (like the apex fox) simply has no threat channel and never flees.

Requirements/notes: a species must declare `sensory_range` and `max_age` genes (used by
perception/metabolism); the `aggression` gene modulates a predator's kill probability (absent
⇒ treated as 1.0). See [`examples/custom_species.py`](examples/custom_species.py).

## 2. Custom heritable traits

The genome layout is **built at runtime** from the union of every species' `gene_ranges`
(`darwinism.sim.genome.build_registry`, run once when the `Simulation` is constructed). The
nine built-in genes keep a fixed canonical order (so the default run is unchanged); any
**novel** gene you declare is appended after them.

```python
gene_ranges={..., "burrow_depth": dw.GeneRange(0.0, 1.0)}   # novel trait
```

A species that doesn't declare a gene pins it to a neutral value (an unused column). Read a
gene in your own system with `genome.gene(entities.genome[slots], "burrow_depth")`. Traits are
inherited by uniform crossover + mutation, so a novel trait drifts and evolves like any other.

> One gene registry per process: build one `Simulation` at a time, or reuse the same species
> set, if you run several in one process.

## 3. Add / replace a tick-system

The tick is an ordered list of `System` objects sharing a per-tick `StepContext`. Subclass
`System`, implement `apply(ctx)`, and insert it into the pipeline:

```python
class DroughtSystem(dw.System):
    def apply(self, ctx):
        if ctx.tick % 400 == 0:
            ctx.veg *= 0.6           # knock the vegetation field down

cfg = dw.make_config(world_seed=12345, seed=7)
pipeline = dw.default_pipeline(cfg)
pipeline.insert(-1, DroughtSystem())          # before the final StatsSystem
sim = dw.Simulation(cfg, systems=pipeline)
```

`StepContext` exposes everything a system needs: `cfg`, `world`, `env`, `ent` (the
Structure-of-Arrays entity store), `veg` (the live per-cell field), `species_grids`, `rng`,
`dt`, `tick`, the per-tick `obs`/`idx`/`act`, and result tallies. `ctx.compact_dead()` drops
entities that died this tick from the working set. The default pipeline order is
`Environment → Grid → Perception → Brain → Sleep → Movement → Consumption → Metabolism →
Reproduction → Vegetation → Stats`; keep the RNG-drawing systems in their relative order.
See [`examples/custom_system.py`](examples/custom_system.py).

## 4. Custom brain

Every decision goes through `Brain.decide(obs_by_species, idx) -> act`. `obs_by_species` maps
each species id to its `Observation`; `act` is a `(len(idx), ACT_DIM)` float32 matrix aligned
to the global alive ordering `idx`. Action columns: `A_DX, A_DY` (unit heading),
`A_EAT, A_DRINK, A_REPRO` (gates in [0,1]), `A_SPEED` (throttle in [0,1]).

```python
class MyBrain(dw.Brain):
    def decide(self, obs_by_species, idx):
        act = np.zeros((len(idx), dw.ACT_DIM), dtype=np.float32)
        for obs in obs_by_species.values():
            # obs.grids: (N, C, K, K) egocentric channels; obs.channels: {role: index}
            # obs.scalars: (N, SCALAR_DIM) internal state + env
            rows = np.searchsorted(idx, obs.idx)   # this species' rows in the global order
            act[rows, dw.A_EAT] = 1.0              # systems enforce true adjacency
        return act

sim = dw.Simulation(cfg, brain={dw.SHEEP: MyBrain(), dw.FOX: None})  # None -> shared RuleBrain
```

The grid is **self-describing**: read a channel by role via `obs.channels["food"]` rather than
a hardcoded index, so the same brain works for any species' layout. Helpers
`dw.nearest_in_channel` / `dw.best_in_channel` reduce a channel to a target direction for
rule-style brains. Per-species routing is handled by a `CompositeBrain`; passing a single
`Brain` instead of a dict drives every species with it. A learned, memoryless PyTorch policy
(`dw.PolicyBrain`, needs the `[torch]` extra) is a drop-in behind the same contract. See
[`examples/custom_brain.py`](examples/custom_brain.py).

## Out of scope (for now)

Terrain and biomes are fixed (a noise-generated world with 7 biomes, hydrology, weather, and
seasons); world generation is not yet a pluggable extension point.
