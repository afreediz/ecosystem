# Ecosystem + Evolution Simulation (v1)

A headless, deterministic ecosystem simulation on a noise-generated world with biomes,
hydrology, weather and seasons. Plants, sheep and foxes act through a
`brain.decide(observation) -> action` contract, carry a heritable genome, and perceive
only their **local** surroundings — so evolution and predator–prey dynamics emerge and
can be measured.

This is **v1**: the brain is hardcoded rules, but the architecture around it (batched
brain system, observation/action vector schemas, SoA entity store) is built so a PyTorch
neural brain can be dropped in later with **zero sim rewrite**. See [v1.md](v1.md) for the
full spec.

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

Controls: `SPACE` pause · `↑/↓` sim-steps-per-frame · `ESC` quit.

**Headless experiment** (fast-forward, writes CSV):

```bash
python run_experiment.py --ticks 20000 --seed 12345 --out runs/run.csv
python run_experiment.py --ticks 20000 --plot      # also render a PNG report
```

**Analysis** (population curves, trait drift, phase plot):

```bash
python -m analysis.plots runs/run.csv --out analysis/out
```

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
