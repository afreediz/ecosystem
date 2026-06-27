# Ecosystem + Evolution Simulation — v1 Overview

> **What this document is.** A conceptual tour of everything v1 implements: the features,
> the algorithms behind them, *why* each algorithm was chosen, and the concrete metrics and
> thresholds that govern behaviour. It is the "what and why". For the "how it's coded"
> (file layout, classes, function signatures, code patterns) see [TECHNICAL.md](TECHNICAL.md).
> The original build spec is [../../v1.md](../../v1.md); calibration reasoning lives in
> [../../CLAUDE.md](../../CLAUDE.md).

---

## 1. The one-paragraph picture

v1 is a **headless, deterministic ecosystem simulation**. A noise-generated world with
biomes, rivers, lakes, an ocean, weather and seasons is populated by **plants** (a per-cell
food field), **sheep** (herbivores) and **foxes** (predators). Every animal acts through a
single contract — `brain.decide(observation) → action` — perceives **only its local
surroundings**, and carries a **heritable genome** that mutates and recombines across
sexual reproduction. The point is to *measure emergence*: population curves, predator–prey
oscillations, and trait drift over generations. The brain in v1 is hardcoded rules, but the
architecture is deliberately built so a PyTorch neural brain can drop in behind the same
contract with no rewrite (that's v2).

The research output is **data**, not a game: a CSV per run feeding population, biomass,
death-cause and trait-drift plots.

---

## 2. Design pillars (the non-negotiables)

These decisions shape everything else and exist to protect the long-term (neural) vision:

1. **Headless sim, separate renderer.** The `sim/` core is pure numbers and never imports
   `render/`. The Arcade window is an *observer* that can be attached or not. The same
   `sim/` runs both the live viewer and the headless experiment, so what you watch is
   exactly what you measure.

2. **The brain↔world contract is the spine.** Every decision flows through
   `decide(obs) → act` over batched `(N, 29)` observation / `(N, 5)` action matrices. The
   brain sees **only** the 29-dim observation — no hidden world access. This is the seam a
   neural network slots into.

3. **Structure-of-Arrays (SoA).** Entity state is parallel NumPy arrays indexed by slot,
   not one object per animal. This is what makes the batched brain, vectorized perception,
   and scaling to thousands of entities natural rather than a refactor.

4. **Determinism.** One seeded RNG, fixed timestep, fixed system order, iteration by slot
   index. Same seed + same config ⇒ byte-identical run. Essential for reproducible
   experiments.

---

## 3. The world

Generated **once** at startup from the master seed. The world is a grid of cells
(default **208 × 117**, a 16:9 aspect to match a widescreen display, with area roughly
equal to the old 160×160 so dynamics carry over). Arrays are indexed `[y, x]`.

### 3.1 Continuous fields

The world is built from **continuous** scalar fields, not biome labels. Animals read the
continuous fields, so biome transitions are gradual automatically; the discrete biome label
is used only for rendering colour and base plant suitability.

| Field | How it's built | Why |
|---|---|---|
| **elevation** | Fractal OpenSimplex noise: 5 octaves of decreasing amplitude (×0.5) and increasing frequency (×2), summed and normalized to [0,1] | Layered noise gives natural terrain (broad landmasses + fine detail) without external assets |
| **moisture** | Independent OpenSimplex field (4 octaves), then **boosted near freshwater** with distance falloff | A separate field decouples "wet" from "high", enabling the Whittaker biome matrix; the freshwater boost makes riverbanks green |
| **static temperature** | `latitude_gradient(y) − elevation × lapse_rate`, normalized | One map edge is colder (latitude); altitude cools (lapse rate 0.5). The *dynamic* part (day/season/weather) is added at runtime, not baked here |

**Why OpenSimplex:** pure-Python, no build/compile issues, and tile-free gradient noise that
avoids the directional artefacts of classic Perlin.

### 3.2 Hydrology — ocean, rivers, lakes

Water is a first-class feature with real consequences (freshwater is the limiting resource;
ocean is an impassable barrier). The algorithm:

1. **Ocean** = flood-fill (BFS) from the map borders across all cells below
   `sea_level_threshold` (0.38). Only border-connected low ground becomes sea, so inland
   basins are *not* automatically ocean.
2. **Rivers** = carved downhill. Pick `n_river_sources` (14) cells from the top quartile of
   land elevation; from each, repeatedly step to the **lowest 8-neighbour**, marking river
   cells, until reaching the ocean or merging into an existing river.
3. **Lakes** = when a river hits a **local minimum** (no lower neighbour), flood a small
   basin up to a spill level (`elevation + 0.015`, capped at 400 cells) and try to continue
   from the basin's lowest rim cell (the spill point).
4. **Beach** = land cells within one step of ocean.
5. Derived masks: `freshwater = river | lake` (**drinkable**), `water_any = freshwater |
   ocean` (blocks movement).

**Why this approach:** a simple, dependency-free hydrology (standard-library BFS, no SciPy)
that produces believable rivers connecting highlands to the sea and occasional lakes. The
known "proper" upgrade — Priority-Flood depression filling + D8 flow accumulation — is
deliberately deferred unless rivers look broken.

**Why freshwater vs ocean matters:** making inland freshwater the *limiting, drinkable*
resource (and ocean a salt barrier) forces animals to cluster near rivers and lakes,
creating spatial dynamics and giving rivers a genuine purpose.

### 3.3 Biomes (Whittaker-style lookup)

Biome is a label derived from `(elevation, temperature, moisture)` with a fixed priority
order (the first matching rule wins):

```
ocean  →  beach  →  mountain (elev > 0.80)  →  cold (temp < 0.30)
→  desert (moisture < 0.30 and temp > 0.65)  →  forest (moisture > 0.60)  →  plains
```

Seven biomes: **ocean, beach, mountain, cold, desert, forest, plains**. Each has a render
colour and a **plant suitability** multiplier (carrying-capacity factor):

| Biome | Suitability | Note |
|---|---|---|
| Plains | 1.00 | the prime grassland — best forage |
| Forest | 0.85 | good forage **and** the predator refuge (see §6.4) |
| Cold | 0.25 | tundra; sparse |
| Beach | 0.20 | |
| Desert | 0.15 | |
| Mountain | 0.10 | nearly barren |
| Water | 0.00 | no growth |

### 3.4 Derived world fields

Computed once and reused every tick (avoids per-agent searches):

- **soil nutrients** — a per-cell pool, `0.4 + 0.4·moisture − 0.3·elevation`, clamped to
  [0,1], zero on water. Richer in moist lowlands. Depleted by plant growth and grazing,
  regenerates slowly. This drives overgrazing/competition.
- **passability** — `not water_any and elevation ≤ 0.97`. Animals can't enter water or the
  highest peaks; they drink from an adjacent land cell.
- **cover** — `biome == forest` (~30% of land). The predator refuge (§6.4).
- **nearest-freshwater fields** (`fw_dist`, `fw_nearest_x/y`) and **nearest-cover fields**
  (`cover_dist`, `cover_nearest_x/y`) — precomputed by multi-source BFS so perception and
  the sleep system can read "direction & distance to the nearest water / safe spot" in O(1)
  per agent instead of searching the map each tick.

---

## 4. The atmosphere (environment & climate)

Global, time-varying state advanced every tick. The world stores spatial-only fields; the
environment supplies the additive offsets and clocks layered on top.

### 4.1 The clock

- **Day/night:** `time_of_day = (t mod day_length) / day_length`, `day_length = 240`
  sim-units. Labels: night / dawn / morning / noon / afternoon / dusk.
- **Season:** `season = (t mod year_length) / year_length`, `year_length = 4800` (~20 days).
  Anchored so summer is warmest at season 0.5, winter coldest at 0.0/1.0. Labels: winter /
  spring / summer / autumn. Season can be paused or nudged forward by the live viewer
  without affecting headless determinism.

### 4.2 Temperature

`temperature(cell, t) = static_temp[cell] + season_offset + diurnal_offset + weather_offset`,
clamped to [0,1].

- **Diurnal offset:** `diurnal_amp · sin(2π·(time_of_day − 0.25))`, amplitude 0.12 — coldest
  before dawn, warmest mid-afternoon.
- **Seasonal offset:** `seasonal_amp · sin(2π·(season − 0.25))`, amplitude 0.20 — peaks in
  summer.
- **Weather offset:** +0.10 in heat, −0.05 in rain, 0 in clear.

**Why sinusoidal offsets on a static field:** it cleanly separates the *spatial* climate
(latitude + altitude, computed once) from the *temporal* climate (cheap scalar offsets added
each tick), and keeps determinism trivial.

### 4.3 Weather

A simple stochastic state machine over **clear / rain / heat**. Each tick, with probability
`weather_change_rate` (0.02) the weather re-rolls uniformly. Effects:

- **Rain** → cooler, **+25% plant growth**.
- **Heat** → warmer, plant growth ×0.85, and **thirst rate ×1.8** (`heat_thirst_factor`).

Kept deliberately minimal in v1.

### 4.4 Derived climate multipliers

- **Thirst multiplier:** `1 + max(0, temp_offset)·1.5`, ×1.8 again under heat weather — hot
  days and summers make animals drink more.
- **Growth multiplier:** seasonal sine in [0.6, 1.0], ×1.25 in rain, ×0.85 in heat — winters
  suppress plant growth, rain boosts it.

---

## 5. Vegetation (the plant layer)

Plants are **a per-cell scalar field**, not thousands of plant entities. This is far cheaper
and means "food perception" is just sampling a field.

- **Carrying capacity** per cell = `plant_suitability × nutrients × (0.4 + 0.6·moisture) ×
  seasonal_growth`. So forage is richest in moist, nutrient-rich plains in summer.
- **Growth** is **logistic**: `ΔV = rate · V · (1 − V/capacity)` with `veg_regrow_rate =
  0.010`. A tiny seed floor (1% of capacity) lets fully-grazed suitable cells restart from
  zero — without it, a grazed-to-zero cell would never recover (logistic growth of 0 is 0).
- **Nutrient coupling:** growth consumes a little nutrient; nutrients regenerate toward 1.0
  at `nutrient_regen_rate = 0.0008`/dt on land. Grazing also depletes nutrients directly.

**Why logistic:** the textbook model for resource-limited growth — fast when sparse,
saturating near capacity — producing renewable-but-exhaustible forage that drives the
herbivore population and, through it, the predator.

---

## 6. Entities & species

### 6.1 Shared animal model

Both species share a per-entity state set: position, heading (for movement momentum),
energy, hunger, thirst, health, age, sex, species, a full genome, a reproduction cooldown,
and two flags (`asleep`, and a cosmetic `mating_glow`). All are columns in parallel arrays.

**Needs and survival:**
- **energy** — burned by metabolism and movement, refilled by eating. Hits 0 → death (starve).
- **hunger** — rises over time; eating lowers it. High hunger (>0.85) drains energy & health.
- **thirst** — rises over time, scaled by local heat; drinking zeroes it. ≥1.5 → death.
- **health** — drained by sustained hunger/thirst; recovers slowly when well-fed & watered.
  Hits 0 → death.
- **age** — rises each tick; death probability climbs past 90% of `max_age`, certain at
  `max_age`.

### 6.2 Sheep (herbivore, prey)

| Trait | Value / range |
|---|---|
| Initial count | 240, in 8 herds (spread 6.0) near freshwater |
| Population cap | 1400 |
| Maturity age | 110 |
| Repro cost / cooldown / litter | 0.25 energy / 90 ticks / 1 |
| hunger / thirst / base-burn / move-cost | 0.0040 / 0.0060 / 0.0020 / 0.0045 |
| eat_value | 0.9 (energy from a full veg cell) |

Sheep eat vegetation, flee foxes, need water, and use the **forest cover** refuge.

### 6.3 Fox (predator)

| Trait | Value / range |
|---|---|
| Initial count | 24, in 5 packs (spread 4.0) near freshwater |
| Population cap | 430 |
| Maturity age | 100 |
| Repro cost / cooldown / litter | 0.35 energy / 150 ticks / 2 |
| hunger / thirst / base-burn / move-cost | 0.0020 / 0.0050 / 0.0012 / 0.0020 |
| predation_gain | 0.72 (fraction of prey size → energy) |
| hunt_success | 0.5 base per-tick kill prob (× aggression × scarcity) |
| hunt_halfsat | 90 (Type III half-saturation) |

**Note the lean fox metabolism** — base-burn, move-cost and hunger rates run ~⅓ below the
sheep's. This is the single most important predator-persistence lever (see §6.4).

### 6.4 Predator–prey coexistence (the hard part)

Getting foxes and sheep to *coexist* rather than one crashing the other took several
stabilizing mechanisms, all biologically realistic. The failure mode is always the same: the
fox goes extinct at a prey trough, after which the sheep explode to their cap. The cures:

1. **Adult founders.** Initial animals are seeded at adult ages (maturity → ~3× maturity),
   not age 0 — otherwise the entire founding cohort is juvenile and dies before it can breed.
2. **Clustered spawning.** Animals start in a few tight herds/packs, which bootstraps
   mate-finding (a lone disperser can't find a mate → Allee extinction).
3. **Prey refuge (forest cover, ~30% of land).** Sheep in cover are invisible and
   uncatchable to foxes. This reservoir prevents total prey collapse. **Sizing is critical:**
   at ~40% (forest+mountain) foxes can never crop enough prey and starve; at ≤25% foxes
   over-crop and crash the prey. ~30% (forest only) is the sweet spot.
4. **Fear distance.** Sheep flee only when a fox is within 45% of their sensory range
   (`_FLEE_TRIGGER`), not for any fox in sight — constant fleeing would stop them
   eating/breeding entirely (a runaway "landscape of fear").
5. **Type III functional response.** Fox kill probability scales by
   `n_sheep² / (n_sheep² + hunt_halfsat²)`, so predation drops sharply when prey is scarce.
   This low-density refuge turns runaway crashes into a bounded limit cycle. Counter­
   intuitively, the *gentler* (higher half-sat) setting survives — lowering it lets foxes
   finish off prey in a trough and starve.
6. **Self-limited fox numbers.** Repro cost (0.35) + long cooldown (150) + a high repro
   threshold gene keep foxes a fraction of the prey so they can't over-crop.
7. **Lean predator metabolism.** Fox burn/hunger ~⅓ below the prey's lets a fox ride out
   lean periods between kills. Extremely sensitive: at base_burn 0.0015 the fox still goes
   extinct on the default seed; 0.0012 yields robust coexistence.

**Verified:** foxes persist and sheep stay bounded well below cap in sustained oscillations
to 8000 ticks across seeds 12345/7/99 (e.g. seed 12345: sheep ~150–350, fox ~30–90).

---

## 7. The genome & evolution

Each animal carries a fixed-length float gene vector. Genes are **heritable** (clamped to
per-species ranges); sex and all live state are not.

| Gene | Effect | Sheep range | Fox range |
|---|---|---|---|
| max_speed | top movement speed | 0.6–1.8 | 1.0–2.4 |
| sensory_range | perception radius (the local-vision trait) | 8–22 | 10–28 |
| metabolism_rate | multiplier on energy burn & need accrual | 0.7–1.3 | 0.7–1.3 |
| size | energy value when eaten, slight speed penalty | 0.7–1.4 | 0.9–1.8 |
| max_age | lifespan | 1400–2600 | 1600–3000 |
| repro_threshold | energy required to breed | 0.5–0.8 | 0.62–0.82 |
| flee_distance | (sheep) behavioural flee gene | 0.4–1.0 | — |
| aggression | (fox) predation probability multiplier | — | 0.4–1.0 |
| chronotype | per-individual sleep-time offset | −0.06–0.06 | −0.06–0.06 |

All species share the same physical gene layout; a gene a species doesn't use sits at a
neutral default and is ignored.

**Inheritance (sexual):**
- **Crossover** — uniform per-gene: each child gene is picked 50/50 from either parent.
  (Per-gene pick preserves more variance than averaging, which is why it's chosen.)
- **Mutation** — per gene, with probability `mutation_rate` (0.18), add a Gaussian of width
  `mutation_strength × gene_span` (0.08 × the gene's range), then clamp to the species range.
  Scaling by span keeps mutation pressure proportional across genes with different units.

**Why sexual reproduction:** a richer, more realistic evolutionary model than asexual
cloning, and the proximity-of-two-adults requirement ties directly into local perception
(you must *find* a mate, you can't teleport to one).

---

## 8. Perception — local only

Each tick, perception builds the `(N, 29)` observation matrix. The defining rule: **no
entity ever queries a global "nearest".** Every external category is gated by the agent's
own `sensory_range` gene; if nothing of that category is in range, the block is zeroed and
its `present` flag is 0. Relative offsets are normalized by sensory range so the vector is
scale-free.

The 29 dimensions:

| idx | content |
|---|---|
| 0–5 | internal: hunger, thirst, energy, health, age/max_age, sex |
| 6–9 | nearest **food**: dx/r, dy/r, dist/r, present |
| 10–13 | nearest **threat** (sheep see foxes; foxes none): dx, dy, dist, present |
| 14–17 | nearest **mate** (opposite sex, same species, adult): dx, dy, dist, present |
| 18–21 | nearest **freshwater**: dx, dy, dist, present |
| 22–25 | local temperature, nutrients, elevation, moisture |
| 26 | on/adjacent freshwater {0,1} |
| 27–28 | time of day, season |

- **Sheep food** = the **highest-vegetation cell** within the sheep's own sensory range
  (a faithful "best grass I can see" forage model, kept even though it's slightly slower
  than a coarser approximation — fidelity was chosen over speed).
- **Fox food** = the nearest **exposed** sheep (those in cover are hidden).
- **Threat / mate / fox-food** queries go through a uniform spatial hash for near-O(N)
  performance.

**Why local-only:** global knowledge is omniscience and contradicts reality. Local-only
perception is also what *creates the need for memory* later — it makes the future LSTM/memory
phase meaningful rather than redundant. Blind time becomes exploration in v1; informed
travel once there's memory.

---

## 9. The brain (hardcoded RuleBrain)

v1's brain is **vectorized priority arbitration** over the whole observation matrix. It is
stateless (no memory) and throwaway — its only job is to produce believable behaviour and
exercise the contract. Highest applicable priority wins:

```
1. FLEE     — a threat within 45% of sensory range → run directly away (overrides all)
2. NEEDS    — urgent hunger/thirst (need > 0.4) → steer toward food or water; if none in
              sight, explore. Opportunistically eat/drink whenever a resource is adjacent.
3. REPRODUCE— fit (energy>0.5, hunger<0.55, thirst<0.55) and a mate present → approach;
              raise the reproduce gate when adjacent
4. EXPLORE  — default: emit a fresh random heading (the movement system's turn-rate limit
              turns this into a smooth, momentum-carrying wander)
```

Two refinements that matter:
- **Food drive responds to both hunger and energy deficit** (`max(hunger, 1−energy)`), so an
  animal seeks food before its energy reserve runs dry — hunger alone rises too slowly to
  prevent starvation.
- **Opportunistic eat/drink** lets an animal top up any adjacent resource without entering
  full "need" mode, so it can still spend most of its time free to breed and explore.

The brain only *proposes* eat/drink/reproduce via gates; the consumption and reproduction
**systems** enforce the authoritative world conditions (true adjacency, true eligibility).
That's why the stateless brain needs no hidden world access.

---

## 10. Circadian rest (sleep)

Animals rest at night. As dusk falls (mean `sleep_onset = 0.80`, shifted per individual by
the **chronotype** gene), each animal heads for the nearest forest cover and beds down,
waking near dawn (`sleep_wake = 0.26`). Because onset is gene-shifted, the herd doesn't drop
unconscious all at once — and the timing can drift over generations.

- Within a short **grace window** (`sleep_shelter_window = 0.06` of a day) after onset, an
  animal still outside cover *seeks* cover (steered by the precomputed nearest-cover field).
  Past the grace window it collapses where it stands (exhaustion).
- Sleepers **hold position**, suppress eat/drink/mate, and burn energy at 45%
  (`sleep_burn_factor`); hunger/thirst accrue at 60% (`sleep_need_factor`).
- **A close predator overrides sleep** — you wake to flee — which keeps predator–prey
  dynamics intact at night.

Sleep is *real* sim state (it changes movement and metabolism), so it lives on the entity
store and is arbitrated by its own system, not the viewer.

---

## 11. The tick — order of operations

Each `Simulation.step(dt)` runs systems in this exact, deterministic order. The order is
load-bearing (e.g. sleep gates the action matrix before movement reads it):

```
1.  environment.update      — advance clock; recompute season/weather/temperature offsets
2.  grid.rebuild            — spatial hashes (global + per-species) from current positions
3.  perception.build        — the (N,29) observation matrix (local, radius-gated)
4.  brain_system.decide     — the (N,5) action matrix
5.  sleep.apply             — night sleepers steer to cover / bed down; gate their actions
6.  movement.apply          — headings → positions; terrain cost; water blocks
7.  consumption.apply       — graze / predation / drink (kills prey immediately)
8.  metabolism.apply        — energy/hunger/thirst/health/aging/death
9.  reproduction.apply      — pair mates → crossover+mutation → spawn offspring
10. vegetation.grow         — logistic regrowth from nutrients/moisture/season
11. logger.record           — append a CSV row (the caller does this)
```

Dead entities (from predation, then from metabolism) are dropped from the tick's working set
before later systems run, so nothing acts on a corpse.

---

## 12. Metrics, logging & analysis

A run logs one CSV row every `log_every` (10) ticks:

- **Populations:** `n_sheep`, `n_fox`, `veg_biomass`.
- **Vital events:** `births`, `deaths`, and deaths broken down by cause — `death_starve`,
  `death_thirst`, `death_age`, `death_health`, `death_predation`.
- **Evolution signal:** the mean of every heritable gene, per species (e.g.
  `sheep_sensory_range`, `fox_aggression`) — this is how trait drift under selection is
  measured.

`analysis/plots.py` turns a CSV into a 4-panel report:
1. **Population vs time** — the ecosystem signal; look for predator–prey oscillations.
2. **Vegetation biomass vs time** — the resource base.
3. **Trait drift vs time** — the evolution signal; look for directional drift under selection.
4. **Sheep-vs-fox phase plot** — the Lotka–Volterra loop.

---

## 13. Running it

```bash
# headless experiment (the measurement path) — deterministic given a seed
venv/Scripts/python.exe run_experiment.py --ticks 9000 --seed 12345 --out runs/run.csv --plot

# live viewer (needs an OpenGL display) — same sim core, observer only
venv/Scripts/python.exe run_live.py --seed 12345 --scale 5 --spf 2

# re-plot an existing CSV
venv/Scripts/python.exe -m analysis.plots runs/run.csv --out analysis/out
```

The live viewer adds inspection controls (pause, speed, zoom/pan, vegetation overlay,
season fast-forward/pause, manual spawning, a night-dimming overlay, male markers and a rose
mating tint) — all cosmetic; none of it feeds back into the sim, and manual spawning is the
only thing that breaks reproducibility (it draws from the master RNG), which is why the
headless path never touches it.

---

## 14. Key thresholds at a glance

| Constant | Value | Meaning |
|---|---|---|
| world size | 208 × 117 | grid cells |
| sea_level_threshold | 0.38 | ocean below this normalized elevation |
| mountain_threshold | 0.80 | mountain above this elevation |
| cover fraction | ~30% of land | forest-only prey refuge |
| day_length / year_length | 240 / 4800 | sim-units (~20 days/year) |
| weather_change_rate | 0.02 | per-tick re-roll probability |
| `_FLEE_TRIGGER` | 0.45 | flee only when fox within 45% of sensory range |
| `_NEED_URGENCY` | 0.4 | need level that overrides explore/reproduce |
| `_ADJ_NORM` | 0.25 | brain's "adjacent enough" gate (fraction of range) |
| `_MAX_TURN` | 0.7 rad | per-tick steering limit (exploration momentum) |
| fox hunt_success / hunt_halfsat | 0.5 / 90 | base kill prob / Type III half-saturation |
| mutation_rate / strength | 0.18 / 0.08 | per-gene mutation probability / Gaussian width (×span) |
| veg_regrow_rate | 0.010 | logistic growth rate |
| death thresholds | hunger>0.85, thirst≥1.5, energy≤0, health≤0, age≥max_age | |

---

## 15. What's deliberately *not* in v1

Recorded so they aren't lost — these are deferred, and v1 is built not to preclude them:

- **Neural brains** (the whole roadmap: MLP → LSTM/memory → CNN retina → GNN over body
  graph → RL + neuroevolution). The `decide(obs)→act` contract + batched BrainSystem is the
  seam they slot into.
- **Memory / learning** — made meaningful precisely by v1's local-only perception.
- **Evolvable morphology** — `sensory_range` as a gene is the first hook.
- **Cooperation / flocking / kin recognition.**
- **Field-of-view cones** (v1 uses an omnidirectional radius), and **pressure** (cut).

See [TECHNICAL.md](TECHNICAL.md) for how the code is structured to keep these doors open.
