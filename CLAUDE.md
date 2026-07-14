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
  `Brain.decide(obs_by_species, idx) -> act` where `obs_by_species` maps each species to its
  `Observation` (egocentric perception **grids** + a scalar vector) and `act` is the batched
  `(len(idx), ACT_DIM=6)` action matrix aligned to the **global** alive ordering `idx`
  (`sim/perception.py` defines obs, `sim/brain.py` defines act). Action columns:
  `A_DX, A_DY` (unit heading), `A_EAT, A_DRINK, A_REPRO` (gates in [0,1]), `A_SPEED`
  (locomotion throttle in [0,1]: 0=hold, 1=full max_speed — `movement` scales the step by it
  and `metabolism` charges locomotion burn in proportion; the RuleBrain stops a *content*
  feeder-in-place and sprints everything else, so travellers/fleers/hunters are unaffected and
  the fragile chase balance is untouched). The brain sees ONLY the
  observations — no hidden state. Perception is **per-species**: each carries only the
  channels it uses, so a future per-species CNN has no dead inputs. `obs.grids` is
  `(N, C, K, K)` egocentric channels centred on each agent, with `K = 2*R+1` and
  `R = ceil(max sensory_range)`; cells beyond an agent's own `sensory_range` or off-world
  are zeroed. Layouts: **sheep** = `terrain, water, food(=grass field), threat(=foxes),
  mate` (5); **fox** = `terrain, water, food(=exposed prey), mate` (4). The `food` channel
  is species-specific in content (grass for herbivores, prey for carnivores). `obs.scalars`
  is `(N, SCALAR_DIM=10)` internal state + global env. The grids are CNN-channel-ready (the
  whole point of the grid design); the `RuleBrain` decodes each species separately into
  nearest/best targets (`nearest_in_channel` / `best_in_channel` in `sim/brain.py`) since a
  rule brain can't convolve, but draws explore headings ONCE over the global ordering so
  partitioning perception by species does not change the run. Adjacency / reproduction
  eligibility are proxied from obs in the brain; the
  consumption/reproduction **systems** enforce the authoritative world conditions.
- **Structure-of-Arrays.** Entity state is parallel NumPy arrays in `sim/entities.py`,
  indexed by slot, with an `alive` mask + free list. Never one-object-per-entity.
- **Two independent seeds.** `world.seed` (the **world seed**) drives world generation only
  — terrain noise *and* hydrology (rivers use a generator derived solely from it), so the
  same world seed always reproduces the same map. `Config.seed` (the **run seed**) drives all
  stochastic *dynamics* via the single run `Generator` from `config.py` (`make_rng`), threaded
  into every system; no global `np.random`. `make_config(world_seed=…, seed=…)` sets them
  separately. Run seed `None` ⇒ a fresh random seed is drawn + recorded (each run differs);
  an explicit value makes it reproducible. So **same world_seed + config + run seed ⇒
  identical run**; same world_seed + different run seed ⇒ a different run on the *same* world.
  Fixed `dt`, iterate by slot index.
- **Fixed tick order** (`Simulation.step`): environment → grid rebuild → perception →
  brain → movement → consumption → metabolism → reproduction → vegetation → log.

## Run

```bash
venv/Scripts/python.exe run_experiment.py --ticks 9000 --world-seed 12345 --seed 7 --out runs/run.csv --plot
venv/Scripts/python.exe run_experiment.py --ticks 9000 --world-seed 12345   # random run, fixed world
venv/Scripts/python.exe run_live.py --world-seed 12345 --seed 7 --scale 5 --spf 2  # needs a display
venv/Scripts/python.exe -m analysis.plots runs/run.csv --out analysis/out
```

Use `venv/Scripts/python.exe` (deps live in `./venv`). Live viewer needs an OpenGL display
and can't run in a headless shell; `run_experiment.py` is the headless path.

## Neural brain (learned, PyTorch) — `sim/policy_brain.py`

A `PolicyBrain` implements the **same `Brain` contract** as `RuleBrain` and is a drop-in.
Brains are selected **per species** on `run_experiment.py` / `run_live.py` via
`--sheep-brain PATH` / `--fox-brain PATH` (a species with no path uses the rule brain); the
checkpoint is a memoryless imitation-learning policy (`notebooks/imitation_learning/*.pt`, a
`.pt` with a `state_dict` key). Under the hood the per-species brains are threaded through a
`CompositeBrain` that routes each species to its own brain and fills unspecified species with a
shared `RuleBrain` on the run RNG (so an all-rule run is byte-identical). Per species the
`PolicyBrain` is: **CNN** over `obs.grids` → concat with an **MLP** over `obs.scalars`
(health/hunger/thirst/energy/age/…) → feed-forward trunk → action heads (heading mean,
eat/drink/repro gate logits, speed logit). It is **memoryless** (no LSTM, no critic) — each
decision is a pure function of the current observation. The CNN ends in an adaptive pool so it
accepts any window `K`. Torch is imported lazily, so the rule-brain path never needs it.
**Deployment acts deterministically (head means / gate thresholds) → draws zero randomness →
runs stay reproducible** and does not perturb the numpy run-RNG the other systems consume.

Training is **behavioural cloning** in `notebooks/imitation_learning/` (`collect` records the
RuleBrain teacher across worlds → `train_sheep` / `train_fox` clone it into the per-species
`SpeciesPolicy` → `evaluate` drops the clones into the real `Simulation`). Deploy:
`run_experiment.py --sheep-brain notebooks/imitation_learning/sheep.pt --ticks 8000` (drive one
or both species).

> **Archived:** the older recurrent CNN+MLP+**LSTM** actor-critic (`NeuralBrain`) and its RL
> trainer (imitation warm-start → recurrent PPO, `train_neural_brain.py`) are detached under
> `backup/` and no longer deployable — see `backup/README.md` to restore them.

## Calibration notes (predator–prey is fragile — see v1.md §18)

Getting sheep + foxes to coexist took several stabilizing mechanisms, all realistic.
Removing any one tends to collapse the predator. Keep them in mind before retuning:

1. **Adult founders** — initial animals are seeded at adult ages (`Simulation._seed_population`),
   not age 0. Otherwise the whole founding population is juvenile and dies before it can
   breed.
2. **Clustered spawning** — animals start in a few tight herds/packs
   (`World.clustered_land_positions`), which bootstraps mate-finding (a lone disperser
   can't breed → Allee extinction).
3. **Prey refuge** — `World.cover` (**forest only, ~30% of land**): sheep there are
   invisible/uncatchable to foxes (`perception` fox-food + `consumption` predation both skip
   covered sheep). This reservoir prevents total prey collapse. **Sizing matters a lot:** it
   was forest+mountain (~40%), which made the refuge so large that foxes could never crop
   enough prey — the predator starved to extinction and the prey then exploded to the cap.
   Dropping bare/rocky mountain (now forest only) shrank it to ~30%, the level that keeps
   prey safe from extinction *and* leaves foxes enough huntable range to persist. Too small
   (≤25%) instead lets foxes over-crop the prey, crash it, and starve. ~30% is the sweet spot.
4. **Fear distance** — sheep flee only when a fox is within `_FLEE_TRIGGER` of sensory
   range (`brain.py`), not for any fox in sight. Constant fleeing would stop prey
   eating/breeding entirely.
5. **Type III functional response** — `consumption.py` scales fox kill probability by
   `n_sheep² / (n_sheep² + hunt_halfsat²)`, so predation drops sharply when prey is scarce
   (a low-density refuge). This is **stabilizing — do not over-weaken it.** It seems like
   "buffing the fox" to lower `hunt_halfsat` (90) so foxes hunt better at low prey, but that
   backfires: foxes then finish off the prey during a trough and starve. Higher half-sat =
   gentler, more persistent cycle. Counterintuitively the *least* aggressive setting survives.
6. **Self-limited fox numbers** — fox `repro_cost` (0.35) + cooldown (150) + `repro_threshold`
   gene (0.62–0.82) keep the predator a fraction of the prey so it can't over-crop. These are
   *eased* from the original (cooldown 180, threshold 0.68–0.85, cap 350→430) so foxes can
   mount a numerical response and recover from troughs — but `repro_cost` is kept as the brake.
7. **Lean predator metabolism** — fox `base_burn`/`move_cost`/`hunger_rate` run ~⅓ below the
   prey's (0.0010/0.0020/0.0020) so a fox can ride out lean periods between kills instead of
   starving the moment prey dips. **This is the single most important — and most sensitive —
   persistence lever:** at `base_burn` 0.0015 the predator still goes extinct on the default
   seed. `base_burn` was eased 0.0012→**0.0010** when perception became egocentric **grids**:
   the grid's inherent cell-quantization adds small noise to predator pursuit / prey fleeing
   that tipped the (chaotic, fragile) balance to fox extinction (~t3000) on the default seed —
   the leaner burn restores the endurance to ride it out (re-verified seeds 12345/7/99).
   Retune it gently and always re-run before trusting.

With these, foxes **persist** (they no longer go extinct at the first deep trough) and the
sheep stay bounded well below their cap, in sustained predator–prey oscillations. Verified to
8000 ticks across seeds 12345/7/99: e.g. seed 12345 sheep ~150–350 & fox ~30–90; seed 99
sheep ~300–400 & fox ~50–70 (exact levels are seed-dependent — wetter worlds carry more of
both). The earlier symptom — sheep exploding to the cap while foxes stayed tiny — was the
*downstream* effect of the predator going extinct; the cure was fox persistence (3/6/7), not
a bigger fox boom (which only deepens the overshoot-and-crash). The food-perception "best
grass cell within sensory_range" is preserved
(faithful, slightly slower) per an explicit decision to favor fidelity over speed.

## Layout

`config.py` (all tunables + seed), `sim/` (world, hydrology, environment, entities,
genome, perception, brain, grid, systems/), `render/viewer.py` (Arcade observer),
`analysis/` (logger + plots). Milestones M0–M7 in v1.md §19 are all implemented.
