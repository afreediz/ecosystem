# darwinism — an ecosystem + evolution simulation framework
<img src="highlights/demo.gif" alt="Demo" width="800">

A headless, deterministic ecosystem simulation on a noise-generated world with biomes,
hydrology, weather and seasons. Animals act through a `brain.decide(observation) -> action`
contract, carry a heritable genome, and perceive only their **local** surroundings — so
evolution and predator–prey dynamics emerge and can be measured.

`darwinism` is a **framework, not just an app**: `import darwinism`, compose a `Config`, and
build around four extension points — **species**, **brains**, **tick-systems**, and heritable
**traits** — without editing the core. The default world ships two species (sheep + fox) on a
hardcoded rule brain, but the architecture around it (batched brain system, per-species
egocentric perception **grids** + scalar schemas, SoA entity store) is built so a PyTorch
neural brain drops in behind the same contract with **zero sim rewrite**.

## Quickstart

```python
import darwinism as dw

cfg = dw.make_config(world_seed=12345, seed=7)   # world seed + run seed (both reproducible)
sim = dw.Simulation(cfg)                          # default RuleBrain drives every species
for _ in range(9000):
    stats = sim.step()
print(sim.populations)                            # {'sheep': ..., 'fox': ...}
```

**Extending it** — add a species, a tick-system, a trait, or a brain, all as composition. See
**[EXTENDING.md](EXTENDING.md)** and runnable **[`examples/`](examples/)**:

```python
RABBIT = 2
cfg.species[RABBIT] = dw.SpeciesConfig(
    name="rabbit", species_id=RABBIT, init_count=90,
    diet=[dw.FieldFood("vegetation", eat_value=0.7)],       # herbivore
    gene_ranges={"max_speed": dw.GeneRange(0.8, 2.2), "burrow_depth": dw.GeneRange(0, 1), ...},
)
sim = dw.Simulation(cfg, systems=[*dw.default_pipeline(cfg), MyDiseaseSystem()],
                    brain={RABBIT: MyBrain()})
```

## Documentation

- **[EXTENDING.md](EXTENDING.md)** — the framework guide: add species / brains / tick-systems /
  traits, and the determinism rules to respect. Runnable versions in **[`examples/`](examples/)**.
- **[docs/v1/OVERVIEW.md](docs/v1/OVERVIEW.md)** — the "what & why": features, algorithms,
  the reasoning behind each choice, and all metrics/thresholds (world, atmosphere,
  vegetation, entities, genome, perception, brain, sleep).
- **[docs/v1/TECHNICAL.md](docs/v1/TECHNICAL.md)** — the "how it's coded": file structure,
  recurring code patterns, and the per-module API (classes, functions, signatures).
- **[docs/v1/v1.md](docs/v1/v1.md)** — the original authoritative build spec.
- **[CLAUDE.md](CLAUDE.md)** — working guidance + predator–prey calibration notes.

## Install

`darwinism` is GitHub-installable. The core (headless simulator) needs only `numpy` +
`opensimplex`; heavy pieces are optional extras that mirror the code's lazy-import boundaries.

```bash
pip install "git+https://github.com/afreediz/darwinism.git"           # core, headless
pip install "darwinism[render] @ git+https://github.com/afreediz/darwinism.git"   # + Arcade viewer
pip install "darwinism[torch]  @ git+https://github.com/afreediz/darwinism.git"   # + learned PolicyBrain
```

Extras: `analysis` (pandas + matplotlib for the CSV report), `render` (Arcade viewer),
`torch` (learned policies), `dev` (pytest, ruff, import-linter), `all`. For local development:

```bash
python -m venv venv
venv/Scripts/activate          # Windows;  source venv/bin/activate on Unix
pip install -e ".[all,dev]"
```

Installing adds two console scripts, `darwinism-run` (headless) and `darwinism-live` (viewer);
`python -m darwinism [run|live]` and the root `run_experiment.py` / `run_live.py` shims are
equivalent.

## Run

**Watch live** (Arcade observer window):

```bash
# default world, random run
darwinism-live

# with world configs
darwinism-live --world-seed 12345 --seed 7 --scale 5 --spf 4

# with trained pytorch brains
darwinism-live --world-seed 1 --seed 7 \
 --sheep-brain notebooks/imitation_learning/sheep.pt \
 --fox-brain notebooks/live_learning/offline/fox_offline_ppo.pt \
 --device cuda
```

CLI flags: `--world-seed N` (terrain/rivers; same world-seed ⇒ identical map) · `--seed N`
(run dynamics; omit for a random run on that world) · `--scale N` (pixels per world cell) ·
`--spf N` (sim steps per rendered frame; fractional ok, e.g. `0.25` = 1 step every 4 frames) ·
`--log-csv PATH` (also log this live run to a CSV) · `--monitor` (open a separate live analysis
window — see [Live monitor](#live-monitor); defaults `--log-csv` to `runs/live.csv`).

### Live viewer controls

| Key / input | Action |
|---|---|
| `SPACE` | pause / resume |
| `↑` / `↓` | sim speed up / slow down (×2 / ÷2 steps-per-frame) |
| `+` / `-` / `=` | zoom in / out (centered on screen) |
| mouse wheel | zoom in / out (centered on cursor) |
| middle-drag | pan the map |
| `0` | reset view (refit the whole map) |
| `V` | toggle the vegetation overlay |
| `Ctrl+V` | freeze / unfreeze vegetation regrowth (grazing still depletes it) |
| `S` | fast-forward the season (+0.1 year) |
| `Ctrl+S` | pause / resume seasonal progression (day & weather keep running) |
| `Shift+S` | spawn a sheep at the cursor |
| `Shift+F` | spawn a fox at the cursor |
| left-click | inspect an animal's perception — ring it and show its egocentric grid channels |
| `ESC` | quit |

The viewer is an **observer only** — these controls never feed back into the measured
simulation, except manual spawning (`Shift+S` / `Shift+F`), which draws from the run RNG
and so breaks run reproducibility (the headless path never spawns manually).

On-screen markers: a small black dot = male · a rose tint = bred in the last few ticks ·
dimmed = asleep · the whole scene darkens at night. Left-click any animal to open a
top-right panel showing its live perception grids (terrain / water / food / threat / mate).

**Headless experiment** (fast-forward, writes CSV):

```bash
darwinism-run --ticks 20000 --world-seed 12345 --seed 7 --out runs/run.csv
darwinism-run --ticks 20000 --world-seed 12345    # random run on a fixed world
darwinism-run --ticks 20000 --plot                # also render a PNG report
darwinism-run --ticks 20000 --monitor             # watch the plots update live
```

`--world-seed` fixes the map; `--seed` fixes the run (omit for a random run — the resolved
seed is printed at startup so you can replay it). `--log-every N` sets how often a CSV row is
written (default every 10 ticks). `--monitor` opens a separate live analysis window (below).

**Analysis** (population curves, trait drift, phase plot):

```bash
python -m darwinism.analysis.plots runs/run.csv --out analysis/out
```

Each run produces a 4-panel report — population vs time, vegetation biomass, sheep trait
drift (the evolution signal), and the sheep–fox phase plot (the Lotka–Volterra loop):

![Sample analysis report](analysis/out/demo_report.png)

### Live monitor

`analysis.monitor` opens a **separate window** (independent of the sim) that tails a run CSV
and re-draws the same 4-panel report on an interval — so you can watch population curves,
trait drift and the phase plot evolve while a run is still going.

```bash
# automatic: launch the monitor alongside a run (spawned as its own process)
darwinism-run --ticks 20000 --world-seed 12345 --seed 7 --monitor
darwinism-live --world-seed 12345 --seed 7 --monitor

# standalone: point it at ANY run CSV — one still being written, or a finished one
python -m darwinism.analysis.monitor runs/run.csv --interval 1.0
```

`--interval` sets the refresh period in seconds (default `1.0`). The monitor tolerates an
empty/header-only file, so it can start before the first row is written. It needs a display
(matplotlib `TkAgg` backend), so it can't run in a headless shell. With `--monitor`, the
sim window closes independently and the monitor window stays open showing the final data.

## Architecture (the non-negotiables)

- **`sim/` is pure numbers and never imports `render/`.** Both entry points share the
  exact same `sim/` core. The Arcade renderer is an optional, read-only observer.
- **The brain↔world contract is the spine.** Every decision flows through
  `Brain.decide(obs_by_species, idx) -> act`: each species gets egocentric perception grids
  `(N, C, K, K)` + a scalar vector, the brain returns the `(len(idx), ACT_DIM)` action matrix
  aligned to the global alive ordering. v1's `RuleBrain` is throwaway; the per-species grid/
  scalar schemas (`sim/perception.py`, `sim/brain.py`) are the real design.
- **Entity state is Structure-of-Arrays** (`sim/entities.py`) — parallel NumPy arrays, not
  one object per entity.
- **Perception is local-only** — each agent perceives food/threats/mates/water only within
  its heritable `sensory_range`, as masked egocentric grids; never a global "nearest". Blind
  time becomes exploration.
- **Determinism, two seeds**: the **world seed** drives world generation (terrain + rivers);
  the **run seed** drives all dynamics via one `numpy` Generator (from `config.py`) threaded
  into every system. Fixed `dt`, iteration by slot index. Same world seed + config + run seed
  ⇒ identical run; a different run seed ⇒ a different run on the same world.

## Layout

```
pyproject.toml       packaging (hatchling; core + [render]/[torch]/[analysis]/[dev] extras)
darwinism/           the package (flat layout)
  __init__.py        public API (Simulation, Config, Brain, System, ...) + __version__
  config.py          all tunables + world seed + run seed + declarative SpeciesConfig/diet
  sim/               headless core (world, hydrology, environment, entities, genome,
                     perception, brain, grid, systems/ incl. the pipeline registry)
  render/viewer.py   Arcade observer (never mutates the sim)
  analysis/          CSV logger + matplotlib plots + live monitor window
  cli/               console-script entry points (experiment, live)
examples/            runnable extension examples (species, system, brain)
tests/               golden-master determinism suite + extension tests
run_experiment.py / run_live.py   thin back-compat shims -> darwinism.cli
```

## Tick order (`Simulation.step`)

The tick is an ordered list of `System` objects sharing a `StepContext` (see
`darwinism.sim.systems.pipeline` and `dw.default_pipeline`):

`environment → grid rebuild → perception → brain → sleep → movement → consumption →
metabolism → reproduction → vegetation → stats`.

Insert/replace/reorder systems via `Simulation(systems=...)` — but keep the RNG-drawing
systems (movement, consumption, metabolism, reproduction) in their relative order, or the run
changes. See [EXTENDING.md](EXTENDING.md).
