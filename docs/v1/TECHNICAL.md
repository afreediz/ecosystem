# Ecosystem + Evolution Simulation — v1 Technical Reference

> **What this document is.** The code-level companion to [OVERVIEW.md](OVERVIEW.md). It
> covers how the code is organized, the patterns that recur everywhere, the public API of
> every module (classes, key functions, signatures), and *why* the code is written the way
> it is. Read OVERVIEW.md first for the conceptual model. The build spec is
> [../../v1.md](../../v1.md); live calibration notes are in [../../CLAUDE.md](../../CLAUDE.md).

---

## 1. Repository layout

```
ecosystem/
├── config.py              # ALL tunables (dataclasses) + world seed + run seed + RNG factory
├── run_experiment.py      # entry: headless, fast-forward, writes CSV  (the measurement path)
├── run_live.py            # entry: Arcade window, watch live  (observer)
├── v1.md                  # the authoritative build spec
├── CLAUDE.md              # working guidance + calibration notes
├── docs/v1/               # this documentation set
├── sim/                   # PURE NUMBERS — never imports render/
│   ├── simulation.py      # Simulation: owns world+entities+systems; .step(dt)
│   ├── world.py           # noise fields, hydrology call, biomes, derived fields
│   ├── hydrology.py       # ocean / river / lake / beach generation (BFS)
│   ├── environment.py     # clock, season, weather, temperature offsets
│   ├── entities.py        # Structure-of-Arrays entity store + free list
│   ├── genome.py          # gene layout, mutation, crossover
│   ├── perception.py      # builds per-species egocentric grids + scalars (local only)
│   ├── brain.py           # Brain interface + RuleBrain (hardcoded) + grid decoders
│   ├── grid.py            # uniform spatial hash for radius queries
│   └── systems/
│       ├── brain_system.py    # global batched obs→act wrapper
│       ├── sleep.py           # circadian rest arbitration
│       ├── movement.py        # actions→positions, steering, terrain, water block
│       ├── consumption.py     # graze / predation / drink
│       ├── metabolism.py      # energy/hunger/thirst/health/aging/death
│       ├── reproduction.py    # mate-finding, crossover, spawn
│       └── vegetation.py      # logistic plant-field growth
├── render/
│   └── viewer.py          # Arcade observer window (the ONLY file that imports arcade)
└── analysis/
    ├── logger.py          # CSV writer (per-tick stats + per-species trait means)
    └── plots.py           # pandas + matplotlib report
```

**The one hard structural rule:** `sim/` must never import `render/`. Verify with:

```bash
grep -rn --include=*.py "^\s*\(import render\|from render\)" sim/   # must be empty
```

This keeps the core headless and fast, and guarantees the live viewer and headless runner
exercise *identical* simulation code.

---

## 2. Recurring code patterns

Understanding these five patterns makes every module readable.

### 2.1 Structure-of-Arrays (SoA)

There is **no `Animal` class**. All entity state lives in parallel NumPy arrays on
`Entities`, each of length `max_entities` (4000), indexed by *slot*:

```python
ent.pos_x[i], ent.energy[i], ent.genome[i], ent.alive[i], ...
```

A boolean `alive` mask marks live slots; a free list recycles dead ones. Operating on "all
sheep" is `ent.genome[ent.species_mask(SHEEP)]` — a vectorized array op, not a Python loop
over objects. **Why:** the batched brain wants observations as a matrix anyway; vectorized
NumPy is orders of magnitude faster than per-object Python; and it's zero-refactor when
neural brains arrive (they consume matrices natively).

### 2.2 Systems are stateless functions, not classes

Every system in `sim/systems/` is a module exposing a single `apply(...)` (or `grow`)
function. They own no state — they receive `(cfg, world, ent, idx, act, rng, ...)`, mutate
the entity arrays / world fields in place, and return small summary values for the logger.
**Why:** the simulation's state lives in exactly two places (the `World` and the `Entities`
store); systems are pure transformations over it. This makes the tick order explicit and
auditable in `Simulation.step`, and makes any system trivially testable in isolation.

### 2.3 The `idx` working set

`Simulation.step` computes `idx = ent.alive_indices()` once (via perception) and threads it
through every system. Systems index arrays as `ent.energy[idx]`, operate, and write back.
After a system that can kill (predation, metabolism), the dead are filtered out of `idx`
(and the aligned `act` matrix) so later systems never touch a corpse:

```python
alive_mask = ent.alive[idx]
idx = idx[alive_mask]
act = act[alive_mask]
```

**Why:** one canonical "who is alive and acting this tick" array keeps every system aligned
to the same rows of `obs` / `act`, and the explicit re-filter after deaths is the only place
mortality interacts with ordering.

### 2.4 Single threaded RNG, no globals — and two independent seeds

There are **two seeds**, deliberately separate:
- **World seed** (`cfg.world.seed`) drives world generation *only* — terrain noise **and**
  hydrology. `World` takes no RNG; it derives its own generator from the world seed, so the
  map never depends on the run dynamics.
- **Run seed** (`cfg.seed`) drives all stochastic *dynamics*. One `numpy.random.Generator` is
  created in `Config.make_rng()` and passed into every system that needs randomness. **No
  module ever calls `np.random` directly.** A run seed of `None` makes `make_rng` draw a fresh
  random seed and **record it back** onto `cfg.seed` (so each run differs but is still
  reproducible after the fact — the entry points print it).

**Why:** same world seed + same config + same run seed ⇒ identical run; same world seed + a
different run seed ⇒ a different run on the *same* world. The *only* deliberate exception is
the live viewer's manual-spawn feature (it draws from the run RNG, breaking reproducibility —
which is precisely why the headless path never uses it).

### 2.5 Brain proposes, systems enforce

The brain emits *gates* (eat / drink / reproduce in `[0,1]`) from the observation alone. It
proxies adjacency/eligibility from normalized distances in `obs`. The **systems** then
re-check the authoritative world conditions (true Euclidean adjacency to vegetation/prey/
water, true reproduction eligibility). **Why:** this lets the brain stay stateless and
world-blind (it sees only `obs`), which is exactly the constraint a future neural brain must
satisfy — so the rule brain can be swapped out with zero changes elsewhere.

---

## 3. `config.py` — the single source of tunables

Everything tunable is a dataclass field; there are no magic numbers scattered in code.

| Dataclass | Holds |
|---|---|
| `WorldConfig` | size, **world seed**, noise octaves/scales, sea level, river sources, biome thresholds, lapse rate, moisture-boost radius |
| `EnvConfig` | day/year length, weather change rate, diurnal/seasonal amplitudes, nutrient regen, heat thirst factor, **and all sleep parameters** |
| `SpeciesConfig` | per-species: init count, gene ranges, maturity, repro cost/cooldown/litter, hunger/thirst/burn/move rates, population cap, mutation rate/strength, need thresholds, eat value, predation gain, hunt success, hunt halfsat |
| `SimConfig` | dt, grid cell size, max entities, log interval, vegetation rates, eat/repro radii, food threshold, mating-glow duration |
| `Config` | composes the four above + the **run seed** (`seed: int \| None`); `make_rng()` returns the seeded Generator |
| `GeneRange` | `(lo, hi)` clamp bounds for one gene |

Key helpers:
- `default_species() -> {SHEEP: SpeciesConfig, FOX: SpeciesConfig}` — builds both species with
  their gene-range dicts. This is where the calibrated numbers live.
- `make_config(world_seed=None, seed=None, **world_overrides) -> Config` — convenience builder
  used by the entry points. `world_seed` sets `cfg.world.seed` (terrain/rivers); `seed` sets
  the run seed (`None` ⇒ random per run). They are independent (§2.4).
- `Config.make_rng()` — returns the seeded run `Generator`. If `cfg.seed is None` it first
  draws a random seed and **writes it back** onto `cfg.seed`, so the resolved value is
  recoverable after the run.

Species ids are module constants: `PLANT = -1`, `SHEEP = 0`, `FOX = 1`. They double as
indices, so `cfg.species[SHEEP]` works directly.

**Why dataclasses:** one obvious place to retune, trivial to `replace()` for experiments, and
no import-time side effects beyond constructing plain objects.

---

## 4. `sim/world.py` — the static world

`World(cfg: WorldConfig)` builds everything once in `__init__` — **it takes no RNG**. All
arrays are `[y, x]`. Everything that needs randomness is derived *solely* from the world seed
(`cfg.seed`), so the map is independent of the run dynamics (§2.4).

Construction sequence:
1. Two independent `OpenSimplex` generators seeded off the world seed (`seed`, `seed+9973`)
   so elevation and moisture are uncorrelated, plus a `world_rng =
   np.random.default_rng(cfg.seed)` for hydrology's random river sources — **never** the
   shared run RNG (which would otherwise make the same world's map drift between runs).
2. `_fractal_noise(gen, w, h, scale, octaves)` — module function: sums octaves (amp ×0.5,
   freq ×2 each), normalizes to [0,1]. Uses `gen.noise2array` (the vectorized OpenSimplex
   grid call) for speed rather than scalar `noise2`.
3. `hydrology.generate(elevation, cfg, world_rng)` → ocean/river/lake/beach/freshwater/
   water_any + moisture boost.
4. Moisture re-clamped after the freshwater boost.
5. Static temperature: latitude gradient (top edge warm) minus `elevation × lapse_rate`,
   normalized.
6. `_classify_biomes()` — the Whittaker priority lookup (see OVERVIEW §3.3).
7. `nutrients`, `plant_suitability` (`_plant_suitability()`), `cover` (forest only),
   `passable`.
8. Nearest-source fields for freshwater and cover, via `_nearest_source_fields`.

Key methods:
- `world_to_cell(x, y)` → clamped integer `(cx, cy)`. The universal continuous→grid mapping.
- `sample(field, x, y)` → field value at an entity's position.
- `is_freshwater / is_passable / in_cover(x, y)` → boolean lookups (accept arrays).
- `_nearest_source_fields(source_mask)` → `(dist, nearest_x, nearest_y)` via multi-source
  BFS over the 8-neighborhood (diagonal step = √2). Returns distances in cell units (inf
  where unreachable) and the nearest source's **cell-center coordinates** as floats — so
  perception/sleep can compute a direction vector with one array lookup. **Why precompute:**
  turns a per-agent map search into an O(1) field read each tick.
- `random_land_positions(n, rng, near_freshwater=False)` — weighted draw of passable land
  cells (moisture-weighted when biased to water).
- `clustered_land_positions(n, rng, n_clusters, spread, near_freshwater=False)` — draws
  cluster centers, then rejection-samples members in a Gaussian blob around each. This is
  what creates the founding herds/packs (see OVERVIEW §6.4).

Biome ids and `BIOME_COLORS` / `BIOME_NAMES` live here too; the colors are consumed only by
the renderer (the sim never uses them for logic).

---

## 5. `sim/hydrology.py` — water generation

Pure NumPy + `collections.deque` BFS (no SciPy). Single public entry:

```python
generate(elevation, cfg, rng) -> {
    "ocean", "river", "lake", "beach", "freshwater", "water_any", "moisture_boost"
}   # all boolean arrays except moisture_boost (float32 in [0,1])
```

Internal helpers and their roles:
- `_ocean_floodfill(elevation, sea_level)` — BFS from every border cell below sea level.
- `_carve_rivers(elevation, ocean, cfg, rng)` — picks sources from the top-quartile of land
  elevation, walks each downhill via `_lowest_neighbor`, marking river cells; on a local
  minimum it forms a lake.
- `_floodfill_basin(elevation, sy, sx, spill, ocean)` — floods a basin up to `spill` level,
  **capped at 400 cells** (a v1 simplification to avoid runaway lakes).
- `_find_spill(elevation, basin, ocean)` — the lowest rim cell adjacent to a basin (the
  overflow point the river continues from).
- `_distance_to(mask, max_dist)` — capped multi-source BFS distance field, used for the
  moisture boost falloff.

**Why standalone, dependency-free BFS:** avoids a SciPy build dependency on Windows and keeps
the algorithm fully transparent and deterministic. The known upgrade path (Priority-Flood +
D8 flow accumulation) is documented in the spec but deferred.

---

## 6. `sim/environment.py` — global time-varying state

`Environment(cfg: EnvConfig, rng)` holds only scalars. `update(dt)` advances `t`, recomputes
`time_of_day`, `season`, the three temperature offsets, and re-rolls weather.

Public API:
- `temperature_field(static_temp)` → full field = `static_temp + temp_offset`, clamped.
- `temp_offset` (property) = season + diurnal + weather offsets summed.
- `thirst_multiplier()` / `growth_multiplier()` — the derived climate scalars (OVERVIEW §4.4).
- Season controls for the viewer: `advance_season(amount)`, `toggle_season_pause()`. These
  use a separate `_season_phase` + `_season_shift` so that, with no manual input, a headless
  run is byte-for-byte identical to a pure `t`-derived season (determinism preserved).

Module-level helpers (label/cosmetic, used by the viewer and for readability):
- `season_name(season)`, `daytime_name(time_of_day)` — string labels anchored to the
  temperature model.
- `light_level(time_of_day)` — daylight in [0,1] via `_smoothstep`; **cosmetic only** (the
  viewer dims the scene by `1 − light_level`), explicitly *not* read by the sim.

Weather/season constants: `CLEAR, RAIN, HEAT = range(3)`, `WEATHER_NAMES`.

---

## 7. `sim/entities.py` — the SoA store

`Entities(cfg)` allocates every state array at capacity `max_entities`:

| Array | dtype | meaning |
|---|---|---|
| `pos_x, pos_y` | float32 | position |
| `heading_x, heading_y` | float32 | unit heading (movement momentum) |
| `energy, hunger, thirst, health` | float32 | needs/vitals |
| `age` | float32 | ticks alive |
| `sex` | int8 | `FEMALE=0`, `MALE=1` (random at birth, non-heritable) |
| `species` | int8 | `SHEEP`/`FOX`, `-1` when free |
| `genome` | float32 `(cap, N_GENES)` | the gene matrix |
| `repro_cooldown` | float32 | ticks until eligible again |
| `mating_glow` | float32 | cosmetic countdown (viewer only) |
| `asleep` | bool | circadian state (real sim state) |
| `alive` | bool | slot liveness mask |

A `_free` list (stack) holds available slots.

API:
- `n_alive`, `alive_indices()`, `species_mask(id)`, `count_species(id)`.
- `spawn(spec, genomes, pos, rng, energy=0.7, age=0.0)` — pops slots from the free list,
  fills all arrays (random heading, sex, starting needs), returns the slot indices. `age` may
  be a scalar or a per-entity array (used to seed founders as adults). Silently truncates if
  the pool is full.
- `kill(slots)` — marks dead, sets `species=-1`, returns slots to the free list **sorted
  descending** so the lowest indices are reused first. **Why:** keeping reuse biased to low
  indices keeps iteration order (ascending slot) stable across runs — part of determinism.

**Why a fixed pool + free list** rather than growing arrays: no reallocation churn, a hard
memory ceiling, and stable slot identity (a slot index is a usable handle within a tick).

---

## 8. `sim/genome.py` — genetics

The gene vector layout is **global and fixed** so any system can index a gene by name without
per-entity lookups:

```python
GENE_NAMES = [max_speed, sensory_range, metabolism_rate, size, max_age,
              repro_threshold, flee_distance, aggression, chronotype]
GENE_INDEX = {name: i}      # name → column
N_GENES = 9
```

All species share this physical layout; genes a species doesn't use are pinned to a
`_NEUTRAL` default and ignored.

API:
- `gene(genomes, name)` → a column view `genomes[:, GENE_INDEX[name]]`. The universal accessor.
- `_bounds(spec)` → `(lo, hi)` arrays for a species (neutral genes pin lo==hi).
- `random_genomes(spec, n, rng)` → uniform draw within ranges. Used for founders and (via the
  viewer) manual spawns.
- `mutate(genomes, spec, rng)` → per-gene Gaussian mutation with prob `mutation_rate`, width
  `mutation_strength × gene_span`, clamped. Operates on a copy.
- `crossover(parent_a, parent_b, spec, rng)` → uniform per-gene 50/50 pick, then `mutate`.
  Inputs are aligned `(M, N_GENES)` arrays (one row per child), so a whole generation is bred
  in one vectorized call.

**Why span-scaled mutation:** genes have wildly different units (`max_age` ~2000 vs
`repro_threshold` ~0.6); scaling the Gaussian by each gene's range gives them comparable
mutational pressure.

---

## 9. `sim/grid.py` — spatial hash

`SpatialGrid(width, height, cell_size)` is a uniform-bucket spatial hash rebuilt every tick.

- `rebuild(indices, pos_x, pos_y)` — buckets the given entity slots into cells using a
  CSR-style layout: entities sorted by flat cell id, with per-cell start offsets from a
  `bincount` cumulative sum. O(M log M) once per tick.
- `query_radius(x, y, radius)` → `(slot_indices, px, py)` for entities within `radius`.
  Scans the block of grid cells overlapping the query disk (using the CSR offsets to slice
  contiguous ranges), concatenates candidates, then does an exact distance filter.

`grid_cell_size` (28) is set ≈ the maximum sensory range, so a radius query touches roughly a
single ring of buckets. **Why a grid:** keeps perception's neighbor queries near O(N) instead
of O(N²). The simulation keeps **three** grids — one global (all animals) plus one per
species — rebuilt each tick, so "nearest sheep" / "nearest fox" / "nearest mate" queries hit
a pre-filtered grid.

---

## 10. `sim/perception.py` — the per-species grid builder

`Perception(cfg, world, entities, grid, env)`; `build(temp_field) -> (obs_by_species, idx)`
returns a `{species_id: Observation}` dict and the **global** alive-index array. Per-tick
context (`_species_grids`, `veg`) is wired in by `Simulation.step` before the call.

**`Observation`** (a `__slots__` class) carries `grids` `(N, C, K, K)` float32, `scalars`
`(N, SCALAR_DIM=10)` float32, `radius` (the window half-width `R`), `idx` (the global slot
ids of its rows, for scatter-back), and `species`.

**Window geometry, set once in `__init__`:** `R = ceil(largest sensory_range across all
species)`, `K = 2·R + 1`. A `_d_cell` `(K,K)` distance-from-centre stencil and a
`_mask_cache` of circular eye-masks (one per integer radius `0..R`) are precomputed — an
agent uses the mask for `round(sensory_range)`, so there's no per-agent disc recompute.
Static fields (terrain = normalized biome label, water = freshwater) are **zero-padded by
`R`** so each agent's `K×K` window is a plain `sliding_window_view` slice — no per-agent
index math.

**`build`** computes the global `idx`, pads the vegetation field once for the tick, then calls
`_build_species(sid, sp_idx, …)` for SHEEP and FOX. Output buffers are lazily grown per
species (`_ensure_buffers`). Each `_build_species`:
- **field channels** (`_field`): terrain (ch 0) and water (ch 1) for both species — the
  padded-field window times the agent's eye-mask.
- **food + entity channels** (species-specific):
  - **Sheep** — food (ch `SH_FOOD`) = the windowed **vegetation field**; then
    `_scatter_predators` marks in-range foxes into the threat channel (skipped entirely when
    no foxes are alive), and `_scatter_mates` marks opposite-sex adult conspecifics.
  - **Fox** — `_scatter_prey` marks in-range **exposed** sheep into the food channel
    (`world.in_cover` filters out sheep hidden in the refuge); `_scatter_mates` for mates. No
    threat channel.
- **scalars** — direct reads: hunger, thirst, energy, health, age/max_age, sex, own-cell
  temperature, time_of_day, season, and `sensory_range` (for distance normalization & the
  CNN).

Helpers:
- `_field(src_pad, cx, cy, masks)` — `sliding_window_view(src_pad,(K,K))[cy,cx] * masks`.
- `_scatter_predators / _scatter_prey / _scatter_mates` — per-agent `query_radius` on the
  relevant species grid, filtered (cover for prey; opposite-sex adult ≠ self for mates), then
  `_scatter` writes a `1.0` into the window cell of each in-window candidate. Juveniles skip
  the mate query (they can't breed).

Layout constants are the single source of truth: `SH_TERRAIN…SH_MATE` (5), `FX_TERRAIN…
FX_MATE` (4), `CHANNEL_NAMES`, `SPECIES_N_CHANNELS`, and the scalar indices `S_HUNGER…
S_SENSORY` (`SCALAR_DIM = 10`).

**Why per-species grids:** they are literally a CNN's channel-stack, and giving each species
only the channels it uses means a future per-species CNN has zero dead inputs. **Why strict
local gating with no global fallback:** it is the architectural promise that makes future
memory meaningful (OVERVIEW §8) — any code that "peeked" globally would quietly break it.

---

## 11. `sim/brain.py` — the decision contract

The spine. `ACT_DIM = 6`, action indices `A_DX, A_DY, A_EAT, A_DRINK, A_REPRO, A_SPEED`.
`A_SPEED` is a locomotion throttle in `[0,1]` (0 = hold, 1 = full `max_speed`): `movement`
scales the step by it and `metabolism` charges the locomotion burn in proportion.

```python
class Brain:
    def decide(self, obs_by_species, idx) -> np.ndarray:   # -> (len(idx), 6)
        raise NotImplementedError

class RuleBrain(Brain):
    def __init__(self, rng, food_threshold=_DEFAULT_FOOD_THR): ...
    def decide(self, obs_by_species, idx): ...   # per-species decode + arbitration
```

**Grid decoders** (module functions, since a rule brain can't convolve):
- `nearest_in_channel(chan)` → `(present, dx, dy, dist)` in cells: the nearest non-zero cell
  of a `(N,K,K)` channel relative to the window centre (threat / mate / water / prey).
- `best_in_channel(chan, thr)` → same shape, but the **highest-value** cell scored as
  `value − 0.02·dist` over cells above `thr` — the richest grass patch in sight (faithful to
  the old "best grass within sensory_range" rule). Both use a cached flat `_stencil(K)`.

`RuleBrain.decide` allocates the global `(len(idx), 6)` act, draws the explore-heading angles
**once over the global ordering**, then for each species calls `_decide_species` on its
`Observation` slice and scatters the result back via `np.searchsorted(idx, obs.idx)` (so
partitioning perception by species can't change the random stream). `_decide_species` decodes
that species' channels (sheep: `best_in_channel` food + `nearest_in_channel` water/mate/
threat; fox: nearest prey/water/mate, no threat), then overlays — in increasing priority —
reproduce, needs, and flee with `np.where` masks. Finally it sets `A_SPEED`: `0.0` for a
**content** animal holding on an adjacent resource (`(eat|drink|repro) & ~urgent`), else
`1.0` — so travellers, urgent foragers and fleers all sprint and only settled feeders stop
(keeping the fragile chase balance intact). Gate constants at module top: `_ADJ_NORM`
(0.25), `_NEED_URGENCY` (0.4), `_FLEE_TRIGGER` (0.45), `_DEFAULT_FOOD_THR` (0.15, overridden
by `cfg.sim.food_eat_threshold`). See OVERVIEW §9.

**Why a separate `Brain` base class for one implementation:** it *is* the contract. A
`TorchBrain(Brain)` later implements the same `decide(obs_by_species, idx)->act`, consuming
each species' `grids` as CNN channels, and `BrainSystem` calls it identically — the entire
neural roadmap hangs off this signature.

---

## 12. `sim/systems/` — the tick systems

Each is a module with one entry function. All mutate state in place and return summaries.

### 12.1 `brain_system.py`
`BrainSystem(brain)` with `decide(obs_by_species, idx) -> act`. A thin wrapper around
`brain.decide`. **Why it exists at all** (it's trivial for rules): it is the *global, batched*
seam — one call site that hands the brain every species' observations and returns one action
matrix aligned to the global `idx`. A neural brain needs exactly this batching, so it's built
now. `decide` is never a per-entity method.

### 12.2 `sleep.py`
`apply(cfg, world, ent, idx, act, obs_by_species, env) -> n_asleep`. Computes each animal's
personal night window (`sleep_onset + chronotype`, shared duration), classifies it as seeking
shelter / asleep / awake, steers seekers toward `world.cover_nearest_*`, zeroes the eat/
drink/repro gates for resting animals, and sets `ent.asleep`. It also overrides `A_SPEED`:
seekers get `1.0` (sprint to cover, ignoring any feed-in-place stop) and sleepers `0.0`. A
close predator overrides
sleep: it decodes the sheep observation's threat channel the same way the brain does
(`nearest_in_channel(sheep_obs.grids[:, SH_THREAT])`, distance as a fraction of `S_SENSORY`)
and scatters the wake flag back into the global ordering via `searchsorted` (foxes carry no
threat channel, so they never wake to flee). Runs **before movement** so its gating and
`asleep` flag are read downstream. See OVERVIEW §10.

### 12.3 `movement.py`
`apply(cfg, world, ent, idx, act, rng)`. Turn-rate-limited steering (`_MAX_TURN = 0.7` rad):
rotates the current heading toward the brain's desired heading by at most `_MAX_TURN`, which
is **where exploration momentum lives** — the stateless brain emits a fresh random heading
each tick and the turn limit smooths it into a directed wander. Speed = `max_speed` with a
slight size penalty, scaled by a terrain factor (`1 − 0.5·elevation`, clamped) **and by the
brain's `A_SPEED` throttle** (`0` = hold, `1` = full), so a settled feeder covers no ground.
Sleepers don't move. Moves into impassable cells (water / high mountain) are rejected and the
heading reflected so they wander off the wall.

### 12.4 `consumption.py`
`apply(cfg, world, ent, idx, act, veg, species_grids, rng) -> (killed, n_drink, n_graze,
n_pred)`. Three blocks, each gated by the brain and re-checked against the world:
- **Drink** — gate + on/adjacent freshwater (`fw_dist ≤ eat_radius`) → `thirst = 0`.
- **Graze** (sheep) — gate + veg in the current cell above `food_eat_threshold` → transfer
  `eat_value × take × size_factor` to energy, lower hunger, deplete the cell's veg + nutrients.
- **Predation** (fox) — gate + nearest adjacent **exposed** living sheep → kill with
  probability `aggression × hunt_success × scarcity`, where `scarcity = n²/(n²+halfsat²)` is
  the Type III response. On success the fox gains energy from prey size; the prey slot is
  killed immediately. Returns the killed slots so the caller drops them from the tick.

### 12.5 `metabolism.py`
`apply(cfg, world, ent, idx, act, temp_field, env, rng) -> causes`. Fully vectorized across
both species (per-species rates broadcast via `np.where(spec==FOX, ...)`). Burns energy —
basal `base_burn` plus locomotion `move_cost · A_SPEED · max_speed · size` (so the throttle
that slows a settled feeder also cuts its energy cost), reduced to basal only for sleepers —
accrues hunger and heat-scaled thirst (slower for sleepers), drains energy/
health under high hunger/thirst, recovers health when well-fed, ages, decrements cooldowns.
Deaths: energy≤0, thirst≥1.5, health≤0, or age (rising probability past 90% of `max_age`,
certain at `max_age`). Returns a death-cause tally with a fixed priority
(thirst > starve > health > age) so each death attributes to exactly one cause.

### 12.6 `reproduction.py`
`apply(cfg, world, ent, idx, act, species_grids, rng) -> births`. Per species: filters to
gate-raised eligible adults (adult, `energy ≥ repro_threshold` gene, cooldown elapsed,
hunger/thirst under limits), then **pairs greedily by ascending slot index** (deterministic),
querying the species grid within `repro_radius` for the nearest eligible opposite-sex
partner. Each pair breeds `litter_size` children via `genome.crossover`; offspring spawn near
the mother; both parents pay `repro_cost` and enter cooldown. Respects population cap and pool
capacity. **Why greedy-by-index pairing:** any mate-matching needs a deterministic tiebreak;
ascending slot index is the simplest one that reproduces identically across runs.

### 12.7 `vegetation.py`
- `initial_field(world, rng)` — seeds each cell at 20–60% of carrying capacity.
- `_carrying_capacity(world, growth_mult)` — `suitability × nutrients × (0.4+0.6·moisture) ×
  growth_mult`, zero on water.
- `grow(cfg, world, env, veg, dt)` — logistic growth toward capacity with a 1%-of-capacity
  seed floor (so fully grazed cells recover), then nutrient consumption + regen. Mutates
  `veg` and `world.nutrients` in place.

---

## 13. `sim/simulation.py` — the orchestrator

`Simulation(cfg=None)` constructs the world (`World(cfg.world)` — world seed only, no RNG),
environment, entity store, vegetation field, the three spatial grids, perception, the rule
brain (`RuleBrain(rng, cfg.sim.food_eat_threshold)`) + brain system, and seeds the founding
population. `make_rng()` resolves/records a random run seed if none was set. It owns the
canonical tick. `step` stashes this tick's per-species observations on `self.last_obs`
(captured **before** deaths are filtered, so each `Observation.idx` row still maps to its
slot) — a read-only handle the live viewer's perception inspector reads.

- `_seed_population()` — spawns each species in clustered herds/packs at **adult ages** near
  freshwater (the two founder mechanisms from OVERVIEW §6.4).
- `_rebuild_grids()` — rebuilds the global grid and both species grids from current positions.
- `step(dt=None) -> stats` — runs the 11-step tick (OVERVIEW §11), filtering dead entities out
  of the working set after predation and after metabolism, then assembles the `stats` dict
  (populations, biomass, births, deaths-by-cause, drink/graze/asleep counts).
- `spawn_at(species_id, x, y, n=1)` — viewer-only manual spawn (nudges to nearest passable
  land; draws from the master RNG → not reproducibility-safe).
- `trait_means(species_id)` — mean of each heritable gene over living members (the evolution
  signal for the logger).
- `populations` (property) — `{"sheep": ..., "fox": ...}`.

**Why the tick order is fixed and explicit here:** it is the readable spec of causality for
the whole sim. Anyone can see in one method that perception precedes the brain, that sleep
gates actions before movement, and that deaths are reconciled before reproduction.

---

## 14. `render/viewer.py` — the observer

The **only** file that imports `arcade`. `EcosystemViewer(arcade.Window)` holds a
`Simulation`, calls `sim.step()` in `on_update`, and draws it — **never mutating sim state**
(except via the explicit user-spawn control). Imported lazily by `run_live.py` so a headless
machine without OpenGL can still import the sim package.

Rendering approach:
- **Terrain** is baked **once** into a background texture from biome colors + freshwater.
- **Vegetation** is a live RGBA overlay re-uploaded in place each frame
  (`update_texture_image`, no per-frame allocation), tinting grazeable cells by veg level.
- **Entities** are drawn as batched point clouds — one `draw_points` call per (species ×
  state) group — with a male marker dot, a rose tint for recent breeders (`mating_glow`),
  and dimming for sleepers.
- A **night overlay** dims the scene by `1 − light_level(time_of_day)`.
- **Perception inspector** — left-click picks the nearest animal (`_pick_entity`, within ~14
  screen px, else deselect). A yellow ring (`_draw_selection_ring`) tracks it, and
  `_draw_perception_inspector` renders that agent's egocentric grid channels live in a
  top-right panel: it looks the slot up in `sim.last_obs` for the agent's species
  (`_selected_obs_grids`), then paints each channel into a lazily-built per-channel RGBA
  texture (`_ensure_grid_textures` / `_update_channel_texture`) — brightness ∝ value, a white
  marker at the centre (the agent), with the agent at the panel's centre cell. The channel
  list (`GRID_CHANNELS_BY_SPECIES`) adapts to the species (sheep: terrain/water/food/threat/
  mate; fox drops threat). The selection auto-clears when the inspected animal dies.
- Dual cameras: a world camera (zoom/pan) and a fixed GUI camera for the HUD.

Controls (pause, speed ×2/÷2, zoom/pan, veg overlay toggle, veg-freeze, season
fast-forward/pause, spawn sheep/fox at cursor, **left-click to inspect perception**) are
listed in the module docstring. All are cosmetic or explicitly out-of-band; none feed the
measurement path (the inspector only *reads* `sim.last_obs`).

**Why texture-baked terrain + point-cloud entities:** GPU does the heavy lifting; the CPU
per-frame cost is just the entity positions and the small veg overlay, so the viewer keeps up
even at large populations.

---

## 15. `analysis/` — logging & plots

### `logger.py`
`Logger(path, sim, log_every=None)` — context-manager CSV writer. `_build_fields()` defines
columns: tick, populations, biomass, births, deaths-by-cause, plus `{species}_{gene}` mean
columns for every heritable gene. `record()` writes a row every `log_every` ticks, pulling
populations/events from `sim.stats` and trait means from `sim.trait_means`.

### `plots.py`
Reads the CSV with pandas, renders a 2×2 matplotlib report (`make_report(csv, out_dir, show)`):
population, vegetation biomass, sheep trait drift, and the sheep–fox phase plot. Defaults to
the `Agg` backend (headless-safe), switching to `TkAgg` only for `--show`. Runnable as
`python -m analysis.plots <csv> [--out dir] [--show]`.

---

## 16. Entry points

- **`run_experiment.py`** — `run_experiment(ticks, out, world_seed=12345, seed=None,
  log_every, ...)`: builds a config via `make_config(world_seed=…, seed=…)`, prints the
  resolved `world_seed`/`run_seed`, runs the headless loop (`sim.step()` + `logger.record()`),
  prints progress and a final summary, detects total extinction and stops early. CLI:
  `--world-seed` (default 12345) fixes the map; `--seed` (default `None`) fixes the run, or
  omit for a random reproducible-after-the-fact run. `--plot` renders a report afterward.
- **`run_live.py`** — CLI `--world-seed` / `--seed` (both `None` by default → default world,
  random run), lazily imports `render.viewer.run`, opens the window. Shares the exact same
  `Simulation`.

---

## 17. How v2 (neural brains) slots in

The architecture's payoff. To add a neural brain:

1. Implement `class TorchBrain(Brain)` with `decide(obs_by_species, idx) -> np.ndarray` (or a
   torch tensor bridged to numpy). It consumes each species' `Observation.grids`
   `(N, C, K, K)` directly as CNN channel-stacks (the rule brain only decodes them because it
   can't convolve) plus the `(N, 10)` scalars, and returns the `(len(idx), 6)` act.
2. Construct it in `Simulation.__init__` instead of `RuleBrain` (one line), or make it a
   config switch.

Nothing else changes: perception still builds the per-species grids, `BrainSystem` still
batches, the systems still enforce world conditions. The local-only egocentric grids (already
a CNN's native input), the per-species channel split (no dead inputs), the SoA layout, the
batched brain seam, the `sensory_range` gene (first hook for evolvable perception), and the
two-seed determinism harness were all chosen specifically so this is a one-file change rather
than a rewrite. Two future learning loops are meant to stay separate and toggleable:
within-lifetime RL and across-generation neuroevolution.
