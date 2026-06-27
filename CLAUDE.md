# CLAUDE.md — Ecosystem + Evolution Simulation

Guidance for working in this repo. The authoritative design spec is [v1.md](v1.md); read
it before making structural changes.

## What this is

A headless, deterministic ecosystem + evolution simulation. Plants (a per-cell field),
sheep, and foxes live on a noise-generated world with biomes, hydrology, weather and
seasons. Animals act through a `brain.decide(obs) -> act` contract, carry a heritable
genome, and perceive only their **local** surroundings. The research output is data:
population curves, predator–prey oscillations, and trait drift over generations.

v1's brain is hardcoded rules; the architecture is built so a PyTorch neural brain drops
in behind the same contract with no sim rewrite.

## Hard rules (do not break)

- **`sim/` never imports `render/`.** The sim core is pure numbers. Verify with:
  `grep -rn --include=*.py "^\s*\(import render\|from render\)" sim/` → must be empty.
- **Brain↔world contract is the spine.** All decisions go through
  `Brain.decide(obs) -> act` over batched `(N, OBS_DIM=29)` / `(N, ACT_DIM=5)` matrices
  (`sim/perception.py` defines obs, `sim/brain.py` defines act). The brain sees ONLY the
  29-dim observation — no hidden state. Adjacency / reproduction eligibility are proxied
  from obs in the brain; the consumption/reproduction **systems** enforce the authoritative
  world conditions.
- **Structure-of-Arrays.** Entity state is parallel NumPy arrays in `sim/entities.py`,
  indexed by slot, with an `alive` mask + free list. Never one-object-per-entity.
- **Determinism.** One seeded `numpy` Generator from `config.py`, threaded into every
  system. No global `np.random`. Fixed `dt`, iterate by slot index. Same seed+config ⇒
  identical run (checked: position/energy/genome hash is stable across runs).
- **Fixed tick order** (`Simulation.step`): environment → grid rebuild → perception →
  brain → movement → consumption → metabolism → reproduction → vegetation → log.

## Run

```bash
venv/Scripts/python.exe run_experiment.py --ticks 9000 --seed 12345 --out runs/run.csv --plot
venv/Scripts/python.exe run_live.py --seed 12345 --scale 5 --spf 2     # needs a display
venv/Scripts/python.exe -m analysis.plots runs/run.csv --out analysis/out
```

Use `venv/Scripts/python.exe` (deps live in `./venv`). Live viewer needs an OpenGL display
and can't run in a headless shell; `run_experiment.py` is the headless path.

## Calibration notes (predator–prey is fragile — see v1.md §18)

Getting sheep + foxes to coexist took several stabilizing mechanisms, all realistic.
Removing any one tends to collapse the predator. Keep them in mind before retuning:

1. **Adult founders** — initial animals are seeded at adult ages (`Simulation._seed_population`),
   not age 0. Otherwise the whole founding population is juvenile and dies before it can
   breed.
2. **Clustered spawning** — animals start in a few tight herds/packs
   (`World.clustered_land_positions`), which bootstraps mate-finding (a lone disperser
   can't breed → Allee extinction).
3. **Prey refuge** — `World.cover` (forest+mountain): sheep there are invisible/uncatchable
   to foxes (`perception` fox-food + `consumption` predation both skip covered sheep). This
   is the prey reservoir that prevents total prey collapse.
4. **Fear distance** — sheep flee only when a fox is within `_FLEE_TRIGGER` of sensory
   range (`brain.py`), not for any fox in sight. Constant fleeing would stop prey
   eating/breeding entirely.
5. **Type III functional response** — `consumption.py` scales fox kill probability by
   `n_sheep² / (n_sheep² + hunt_halfsat²)`, so predation drops sharply when prey is scarce
   (a low-density refuge). This converts runaway crashes into a bounded cycle.
6. **Self-limited fox numbers** — high fox `repro_threshold` gene + cost + cooldown keep
   the predator a small fraction of the prey, so prey isn't over-cropped.

With the default seed (12345) this yields rich predator–prey oscillations (sheep ~50–600,
fox ~3–80) for ~10k ticks, then the predator typically goes extinct at a deep trough — a
realistic outcome. The food-perception "best grass cell within sensory_range" is preserved
(faithful, slightly slower) per an explicit decision to favor fidelity over speed.

## Layout

`config.py` (all tunables + seed), `sim/` (world, hydrology, environment, entities,
genome, perception, brain, grid, systems/), `render/viewer.py` (Arcade observer),
`analysis/` (logger + plots). Milestones M0–M7 in v1.md §19 are all implemented.
