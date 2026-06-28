# Ecosystem + Evolution Simulation (v1)

A headless, deterministic ecosystem simulation on a noise-generated world with biomes,
hydrology, weather and seasons. Plants, sheep and foxes act through a
`brain.decide(observation) -> action` contract, carry a heritable genome, and perceive
only their **local** surroundings — so evolution and predator–prey dynamics emerge and
can be measured.

This is **v1**: the brain is hardcoded rules, but the architecture around it (batched
brain system, per-species egocentric perception **grids** + scalar schemas, SoA entity
store) is built so a PyTorch neural brain — the grids are already a CNN's channel-stack —
can be dropped in later with **zero sim rewrite**.

## Documentation

- **[docs/v1/OVERVIEW.md](docs/v1/OVERVIEW.md)** — the "what & why": features, algorithms,
  the reasoning behind each choice, and all metrics/thresholds (world, atmosphere,
  vegetation, entities, genome, perception, brain, sleep).
- **[docs/v1/TECHNICAL.md](docs/v1/TECHNICAL.md)** — the "how it's coded": file structure,
  recurring code patterns, and the per-module API (classes, functions, signatures).
- **[v1.md](v1.md)** — the original authoritative build spec.
- **[CLAUDE.md](CLAUDE.md)** — working guidance + predator–prey calibration notes.

## Install

```bash
python -m venv venv
venv/Scripts/activate          # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt
```

## Run

**Watch live** (Arcade observer window):

```bash
python run_live.py                              # default world, random run
python run_live.py --world-seed 12345 --seed 7 --scale 5 --spf 4
```

CLI flags: `--world-seed N` (terrain/rivers; same world-seed ⇒ identical map) · `--seed N`
(run dynamics; omit for a random run on that world) · `--scale N` (pixels per world cell) ·
`--spf N` (sim steps per rendered frame; fractional ok, e.g. `0.25` = 1 step every 4 frames).

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
python run_experiment.py --ticks 20000 --world-seed 12345 --seed 7 --out runs/run.csv
python run_experiment.py --ticks 20000 --world-seed 12345    # random run on a fixed world
python run_experiment.py --ticks 20000 --plot                # also render a PNG report
```

`--world-seed` fixes the map; `--seed` fixes the run (omit for a random run — the resolved
seed is printed at startup so you can replay it).

**Analysis** (population curves, trait drift, phase plot):

```bash
python -m analysis.plots runs/run.csv --out analysis/out
```

Each run produces a 4-panel report — population vs time, vegetation biomass, sheep trait
drift (the evolution signal), and the sheep–fox phase plot (the Lotka–Volterra loop):

![Sample analysis report](analysis/out/demo_report.png)

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
config.py            all tunables + world seed + run seed
run_live.py          Arcade window entry point
run_experiment.py    headless CSV entry point
sim/                 headless core (world, hydrology, environment, entities,
                     genome, perception, brain, grid, systems/)
render/viewer.py     Arcade observer (never mutates the sim)
analysis/            CSV logger + matplotlib plots
```

## Tick order (`Simulation.step`)

`environment → grid rebuild → perception → brain → movement → consumption → metabolism →
reproduction → vegetation → log`.
