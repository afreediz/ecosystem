# Ecosystem + Evolution Simulation (v1)

A headless, deterministic ecosystem simulation on a noise-generated world with biomes,
hydrology, weather and seasons. Plants, sheep and foxes act through a
`brain.decide(observation) -> action` contract, carry a heritable genome, and perceive
only their **local** surroundings — so evolution and predator–prey dynamics emerge and
can be measured.

This is **v1**: the brain is hardcoded rules, but the architecture around it (batched
brain system, observation/action vector schemas, SoA entity store) is built so a PyTorch
neural brain can be dropped in later with **zero sim rewrite**.

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
python run_live.py                 # default seed
python run_live.py --seed 7 --scale 5 --spf 4
```

CLI flags: `--seed N` (master seed) · `--scale N` (pixels per world cell) · `--spf N`
(sim steps per rendered frame; fractional ok, e.g. `0.25` = 1 step every 4 frames).

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
| `ESC` | quit |

The viewer is an **observer only** — these controls never feed back into the measured
simulation, except manual spawning (`Shift+S` / `Shift+F`), which draws from the master RNG
and so breaks run reproducibility (the headless path never spawns manually).

On-screen markers: a small black dot = male · a rose tint = bred in the last few ticks ·
dimmed = asleep · the whole scene darkens at night.

**Headless experiment** (fast-forward, writes CSV):

```bash
python run_experiment.py --ticks 20000 --seed 12345 --out runs/run.csv
python run_experiment.py --ticks 20000 --plot      # also render a PNG report
```

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
  `Brain.decide(obs) -> act` over batched `(N, OBS_DIM)` / `(N, ACT_DIM)` matrices. v1's
  `RuleBrain` is throwaway; the obs/act schemas (`sim/perception.py`, `sim/brain.py`) are
  the real design.
- **Entity state is Structure-of-Arrays** (`sim/entities.py`) — parallel NumPy arrays, not
  one object per entity.
- **Perception is local-only** — animals see food/threats/mates/water only within their
  heritable `sensory_range`; never a global "nearest". Blind time becomes exploration.
- **Determinism**: one seeded `numpy` Generator (from `config.py`) threaded into every
  system; fixed `dt`; iteration by slot index. Same seed + config ⇒ identical run.

## Layout

```
config.py            all tunables + master seed
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
