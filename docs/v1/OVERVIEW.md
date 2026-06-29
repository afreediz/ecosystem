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
   `decide(obs_by_species, idx) → act`. Each species gets its own `Observation` — egocentric
   perception **grids** `(N, C, K, K)` (CNN-channel-ready) plus a small scalar vector
   `(N, 10)` — and the brain returns the `(len(idx), 5)` action matrix aligned to the
   **global** alive ordering `idx`. The brain sees **only** the observations — no hidden world
   access. This is the seam a neural network slots into (see §8).

3. **Structure-of-Arrays (SoA).** Entity state is parallel NumPy arrays indexed by slot,
   not one object per animal. This is what makes the batched brain, vectorized perception,
   and scaling to thousands of entities natural rather than a refactor.

4. **Determinism, two independent seeds.** The **world seed** drives world generation only
   (terrain noise *and* hydrology), so the same world seed always reproduces the same map.
   The **run seed** drives all stochastic *dynamics* through one threaded RNG (no global
   `np.random`); a run seed of `None` draws and records a fresh seed so each run differs.
   With a fixed timestep, fixed system order, and iteration by slot index: same world seed +
   same config + same run seed ⇒ byte-identical run; same world seed + a different run seed
   ⇒ a different run on the *same* world. Essential for reproducible experiments.

---

## 3. The world

Generated **once** at startup from the **world seed** (terrain noise + hydrology only — the
run seed never touches it, so a world seed always reproduces the same map). The world is a
grid of cells
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
| hunger / thirst / base-burn / move-cost | 0.0020 / 0.0050 / 0.0010 / 0.0020 |
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
   lean periods between kills — the single most important, most sensitive persistence lever.
   At base_burn 0.0015 the fox still goes extinct on the default seed. It was eased
   0.0012→**0.0010** when perception became egocentric **grids**: the grid's inherent cell
   quantization adds small noise to predator pursuit / prey fleeing that tipped the fragile
   balance to fox extinction (~t3000) on the default seed, and the leaner burn restores the
   endurance to ride it out (re-verified seeds 12345/7/99).

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

## 8. Perception — local, egocentric, per-species grids

Each tick, perception builds — **per species** — a stack of **egocentric grids** plus a
small scalar vector. This is the v1 → v2 hinge: the grids are literally CNN-channel-ready,
the whole point of the design.

**The grids** are `(N, C, K, K)`: for each agent, a square window of side `K = 2·R + 1`
cells centred on the agent, where `R = ceil(largest sensory_range across all species)` (so
the window is a single fixed, batchable canvas — here `R = 28`, `K = 57`). Each channel is a
raw local view of one thing. Cells beyond the agent's **own** heritable `sensory_range`
(a cached circular eye-mask) or off-world are zeroed, so an agent only ever perceives what
its eyes can reach. The defining rule still holds: **no entity ever queries a global
"nearest".**

Perception is **separated by species** — each carries only the channels it actually uses, so
a future per-species CNN has no dead inputs:

| Species | Channels (in order) |
|---|---|
| **Sheep** (5) | terrain, water, **food** (= grass field), threat (= foxes), mate |
| **Fox** (4) | terrain, water, **food** (= exposed prey), mate |

The `food` channel is unified in *position* but species-specific in *content* — vegetation
for the herbivore, prey entities for the carnivore. Foxes have no predators, so they carry
no threat channel.

- **terrain / water** — static world fields (biome label, freshwater), sliced as an
  egocentric window from a zero-padded copy (no per-agent index math).
- **Sheep food** = the vegetation field itself, windowed — the brain later picks the **best
  grass cell** within range (a faithful "best grass I can see" forage model, kept even though
  it's slightly slower than a coarser approximation — fidelity over speed).
- **Fox food** = **exposed** sheep marked into the window (sheep in forest cover are hidden
  from predators — the refuge, §6.4).
- **threat / mate** — in-range foxes / opposite-sex adult conspecifics marked into the
  window. Threat scatter is skipped entirely when no foxes are alive; juveniles skip the mate
  query (they can't breed) — both big savings. Entity queries go through per-species spatial
  hashes for near-O(N) performance.

**The scalars** are `(N, 10)`, identical layout for both species — the internal state and
global env that don't belong on a spatial grid:

| idx | content |
|---|---|
| 0–5 | internal: hunger, thirst, energy, health, age/max_age, sex |
| 6 | local temperature (own cell) |
| 7–8 | time of day, season |
| 9 | sensory_range (cells — for distance normalization & the CNN) |

`build` returns `(obs_by_species, idx)`: a dict mapping each species to its `Observation`
(grids + scalars + radius), and the **global** alive ordering `idx` so all downstream
systems stay aligned to one set. Each `Observation` also carries its own `idx` (the global
slot ids of its rows) so per-species results scatter back into the global ordering.

### What an Observation actually looks like

A concrete sheep, pulled live from a run (`R = 28`, so each grid is `K = 57` wide). Its
`Observation.grids` is `(5, 57, 57)` — five channels, each a 57×57 egocentric window with the
sheep at the centre cell `[28, 28]`. Here is the **7×7 patch around that centre** for three
of the channels (the full window extends 28 cells further out in every direction, masked to a
disc of the sheep's own `sensory_range ≈ 20`):

```
terrain (normalized biome id)          food (= grass field)              mate (1 = a mateable sheep)
 .86 .86 .86 .86 .86 .86 .86           .05 .05 .06 .06 .04 .05 .05        0  0  0  1  0  0  0
 .86 .86 .86 .86 .86 .86 .86           .19 .04 .05 .03 .05 .05 .05        0  0  0  0  0  0  0
 .86 .86 .86 .86 .86 .86 .86           .14 .05 .04 .05 .06 .04 .05        0  0  1  0  0  0  0
1.0  .86 .86 [.86] .86 .86 .86         .06 .04 .06 [.06] .06 .04 .05      0  0  0 [0] 0  0  0
1.0 1.0  .86 .86 .86 .86 .86           .15 .04 .04 .15 .04 .05 .05        0  0  0  0  0  0  0
1.0 1.0 1.0  .86 .86 .86 .29           .03 .04 .05 .05 .05 .05 .04        0  0  0  0  0  0  0
1.0 1.0 1.0 1.0  .86 .86 .29           .04 .05 .07 .06 .04 .05 .05        0  0  0  0  0  0  0

circular vision (1 = visible)
0  0  0  0  0  0  0  0  0
0  0  0  0  1  0  0  0  0
0  0  0  1  1  1  0  0  0
0  0  1  1  1  1  1  0  0
0  1  1  1 [1]  1 1  1  0
0  0  1  1  1  1  1  0  0
0  0  0  1  1  1  0  0  0
0  0  0  0  1  0  0  0  0
0  0  0  0  0  0  0  0  0

```

(`[·]` marks the agent's own cell.) Reading it: the sheep stands in **forest** (`.86`) with a
strip of a different biome (`1.0`) to the lower-left and a sliver of another (`.29`) at the
edge; **grass** is thin all around with a couple of richer patches up-left (`.19`); the
**water** and **threat** channels are all-zero (no river or fox in sight); and **two other
sheep** are visible — one is an adjacent mateable conspecific just above-left. A fox would
have a 4-channel `(4, 57, 57)` window instead (no threat channel, and `food` = exposed prey).

Its **scalars** (length 10) for the same tick:

```
[0.06, 0.53, 0.86, 1.0, 0.16, 1.0, 0.25, 0.5, 0.03, 19.83]
  hung  thir  enrg  hlth  age   sex  temp  tod  seas  sensory_range
```

So: barely hungry (0.06), half-thirsty (0.53), well-fuelled (energy 0.86, health 1.0), young
(16% of lifespan), male, in a temperate cell at midday in early spring, with a vision radius
of ~20 cells.

**Why local-only:** global knowledge is omniscience and contradicts reality. Local-only
perception is also what *creates the need for memory* later — it makes the future LSTM/memory
phase meaningful rather than redundant. Blind time becomes exploration in v1; informed
travel once there's memory.

---

## 9. The brain (hardcoded RuleBrain)

v1's brain is **vectorized priority arbitration**, run **per species** over that species'
observation. It is stateless (no memory) and throwaway — its only job is to produce
believable behaviour and exercise the contract.

A *rule* brain can't run a convolution, so it first **decodes** each species' grid channels
back into the simple targets it reasons over — `nearest_in_channel` (nearest present cell →
the nearest threat / mate / water / prey) and `best_in_channel` (highest-value cell minus a
mild distance pull → the best grass patch). A neural brain would skip the decode and learn
straight off the channels. Each species is decoded separately, but the explore-heading RNG
is drawn **once over the global ordering** so partitioning perception by species doesn't
change the run. Highest applicable priority wins:

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

### What the brain outputs

For a batch of `N` agents the brain returns an `(N, 5)` action matrix — one row per agent,
columns `[A_DX, A_DY, A_EAT, A_DRINK, A_REPRO]`. Two real rows from the run above:

```
[ 0.00, -1.00,  1.0, 1.0, 1.0 ]   # heading straight "up"; eat + drink + reproduce all gated
[ 0.83,  0.56,  1.0, 0.0, 0.0 ]   # heading up-right; eat gated, no drink, no reproduce
```

- **`A_DX, A_DY`** are a unit heading vector (the movement system smooths it via a turn-rate
  limit into momentum-carrying motion). The first agent points straight along −y toward its
  adjacent mate; the second wanders/forages along a diagonal.
- **`A_EAT, A_DRINK, A_REPRO`** are gates in `[0,1]` (here either `1.0` or `0.0`). The first
  agent is adjacent to grass, water *and* a mate, so it raises all three; the second only
  raises eat.

Remember these are **proposals**: a raised gate just means "a target read close enough"
(within 25% of sensory range). The consumption/reproduction systems still re-check true world
adjacency and eligibility, so raising `A_REPRO` doesn't guarantee a birth. A future neural
brain would emit the same five numbers, but with *continuous* gate values that the systems
threshold at `> 0.5`.

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
3.  perception.build        — per-species observations (grids + scalars) + global idx
4.  brain_system.decide     — the (len(idx),5) action matrix, aligned to the global idx
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
# headless experiment (the measurement path) — reproducible given both seeds
venv/Scripts/python.exe run_experiment.py --ticks 9000 --world-seed 12345 --seed 7 --out runs/run.csv --plot

# same fixed world, but a fresh random run (the resolved seed is printed so you can replay it)
venv/Scripts/python.exe run_experiment.py --ticks 9000 --world-seed 12345

# live viewer (needs an OpenGL display) — same sim core, observer only
venv/Scripts/python.exe run_live.py --world-seed 12345 --seed 7 --scale 5 --spf 2

# re-plot an existing CSV
venv/Scripts/python.exe -m analysis.plots runs/run.csv --out analysis/out
```

`--world-seed` fixes terrain + rivers; `--seed` fixes the run dynamics (omit it for a random,
non-reproducible run — the resolved seed is printed at startup).

The live viewer adds inspection controls (pause, speed, zoom/pan, vegetation overlay,
season fast-forward/pause, manual spawning, a night-dimming overlay, male markers and a rose
mating tint). New: **left-click any animal to inspect its perception** — a ring marks the
selected agent and a top-right panel renders its egocentric grid channels live (terrain /
water / food / threat / mate, adapting to the species), the agent at the centre. All of it is
cosmetic; none feeds back into the sim, and manual spawning is the only thing that breaks
reproducibility (it draws from the run RNG), which is why the headless path never touches it.

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
  graph → RL + neuroevolution). The `decide(obs_by_species, idx)→act` contract + batched
  BrainSystem is the seam they slot into — and the per-species `(N, C, K, K)` perception
  grids are already exactly a CNN's channel-stack input (the rule brain only decodes them
  because it can't convolve).
- **Memory / learning** — made meaningful precisely by v1's local-only perception.
- **Evolvable morphology** — `sensory_range` as a gene is the first hook.
- **Cooperation / flocking / kin recognition.**
- **Field-of-view cones** (v1 uses an omnidirectional radius), and **pressure** (cut).

See [TECHNICAL.md](TECHNICAL.md) for how the code is structured to keep these doors open.
