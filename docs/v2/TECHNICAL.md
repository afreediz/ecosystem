# Ecosystem + Evolution Simulation — v2 Technical Reference (Neural Brain)

> **What this document is.** The code-level companion to [OVERVIEW.md](OVERVIEW.md): the new
> files, the classes and their public API, the integration points into the v1 core, and *why*
> the code is written the way it is. Read OVERVIEW.md first for the model. For the sim core this
> plugs into, see [../v1/TECHNICAL.md](../v1/TECHNICAL.md); calibration notes are in
> [../../CLAUDE.md](../../CLAUDE.md).

---

## 1. What changed, at a glance

```
ecosystem/
├── sim/
│   ├── neural_brain.py     ★ NEW  — NeuralBrain + SpeciesActorCritic (CNN+MLP+LSTM actor-critic)
│   ├── entities.py         ~ EDIT — added birth_id identity token + _next_birth_id
│   └── simulation.py       ~ EDIT — Simulation(cfg, brain=None): pluggable brain + bind()
├── train_neural_brain.py   ★ NEW  — RL trainer: imitation warm-start + recurrent PPO, checkpoints
├── run_experiment.py       ~ EDIT — _make_brain(), --brain/--weights/--device flags (headless)
├── run_live.py             ~ EDIT — reuses _make_brain(), same flags (live viewer)
└── render/viewer.py        ~ EDIT — plumbs a `brain=` through; viewer still an OBSERVER
```

Net: **~75 lines changed across 5 existing files** (all additive plumbing) + **2 new modules**.
No sim rule, system, or the `Brain`/`Observation` contract was modified.

---

## 2. `sim/neural_brain.py`

### 2.1 `SpeciesActorCritic(nn.Module)`

One per species. Constructed with the species' channel count so its first conv layer matches
the perception layout:

```python
SpeciesActorCritic(n_channels, scalar_dim=SCALAR_DIM, hidden=128, cnn_feat=128, scalar_feat=32)
```

| Stage        | Layers | Output |
|--------------|--------|--------|
| `conv`       | `Conv2d(C→16,k3,s2,p1)→ReLU → Conv2d(16→32,…)→ReLU → Conv2d(32→32,…)→ReLU → AdaptiveAvgPool2d(4,4)` | `(32,4,4)` for **any** `K` |
| `cnn_fc`     | `Linear(32·4·4 → cnn_feat)` + ReLU | `(cnn_feat=128,)` |
| `scalar_mlp` | `Linear(10→32)→ReLU → Linear(32→32)→ReLU` | `(scalar_feat=32,)` |
| `lstm`       | `LSTMCell(cnn_feat+scalar_feat → hidden)` | `(hidden=128,)` |
| `head_mean`  | `Linear(hidden→2)` | heading gaussian mean `(dx,dy)` |
| `head_logstd`| `Parameter((2,), init −0.5)`, clamped `[−2,1]` | state-independent log-std |
| `gate_logits`| `Linear(hidden→3)` | eat / drink / repro bernoulli logits |
| `speed_ab`   | `Linear(hidden→2)` then `softplus(·)+1` | Beta `(α,β) ≥ 1` (unimodal) |
| `value_head` | `Linear(hidden→1)` | state value |

- **`features(grids, scalars)`** → concatenated `[cnn_feat | scalar_feat]`.
- **`forward(grids, scalars, h, c)`** → `(params, value, h_next, c_next)` where
  `params = (mean, logstd, gate_logits, speed_ab)`.
- **`AdaptiveAvgPool2d((4,4))` is the K-independence trick:** the config sets `K = 2·R+1` from
  the largest sensory range, so it can differ between runs (e.g. a small training world vs. the
  full world). The pool collapses any spatial size to `4×4`, so one architecture and one
  checkpoint work across window sizes.

### 2.2 Action distribution helpers (module-level)

The action space is **hybrid**; these three functions keep the sampling / log-prob / entropy
math in one place, matching the `A_DX,A_DY,A_EAT,A_DRINK,A_REPRO,A_SPEED` column layout from
`sim/brain.py`:

- **`_dists(params)`** → `(Normal heading, Bernoulli gates, Beta speed)`.
- **`action_logp_entropy(params, actions)`** → `(logp, entropy)`, summed across the three
  factors. Speed is clamped to `[ε, 1−ε]` for a finite Beta log-prob.
- **`sample_action(params)`** → `(actions (N,6), logp)` — **training** rollouts (stochastic).
- **`mode_action(params)`** → `actions (N,6)` — **eval** (deterministic): mean heading,
  `gate = logit>0` (i.e. sigmoid > ½), Beta-mean speed `α/(α+β)`. **Draws no randomness.**

### 2.3 `NeuralBrain(Brain)`

Implements the v1 `Brain` contract. Constructor:

```python
NeuralBrain(cfg, device="cpu", hidden=128, training=False)
```

Holds `self.nets = {SHEEP: SpeciesActorCritic(5,…), FOX: SpeciesActorCritic(4,…)}` and per-slot
LSTM tables `self.h[sid]`, `self.c[sid]` of shape `(cap, hidden)` (one table per species; a slot
only ever holds one species at a time, so tables never collide).

**Contract method:**

- **`decide(obs_by_species, idx) → act (n_global, 6)`** — allocates the global action matrix,
  then dispatches per species to `_decide_species`, which:
  1. `_sync_memory(sid, slots)` — reset recycled slots' LSTM state (see §2.4);
  2. moves `grids`/`scalars` to torch, gathers this species' `h0,c0` rows;
  3. runs `net(...)` under `torch.no_grad()`;
  4. **training** → `sample_action` + records the rollout; **eval** → `mode_action`;
  5. carries `h1,c1` back into the tables;
  6. scatters the actions into the global rows via `np.searchsorted(idx, slots)`.

**Lifecycle / wiring:**

- **`bind(entities)`** — hands the brain a read-only handle on the entity store, used **only**
  to read `birth_id` for slot-recycle detection. Called once by `Simulation`.
- **`eval()` / `train_mode()`** — toggle sampling-vs-mode and put the torch modules in the
  matching mode; return `self`.
- **`reset()`** — zero **all** LSTM memory (used when a training episode restarts the world).
- **`parameters(sid)`** — the optimizer's parameter iterator for one species.
- **`recorder`** — an optional hook (set by the trainer) that captures each training decision.

**Persistence:** `state_dict()` → `{hidden, sheep, fox}`; `save(path)` / `load(path, strict=True)`.

### 2.4 Memory lifecycle — `_sync_memory` and `birth_id`

```python
def _sync_memory(self, sid, slots):
    cur = self.entities.birth_id[slots]
    changed = cur != self._occupant[slots]      # slot now holds a DIFFERENT animal
    if changed.any():
        # zero h/c for those slots, then remember the new occupants
```

`self._occupant` (int64, length `cap`) records which `birth_id` currently owns each slot's
memory. A mismatch against the live `birth_id` means the free list recycled the slot into a new
animal, so its LSTM state is zeroed before use. This is the mechanism that keeps memory from
leaking across the birth/death boundary while still honouring the "decide reads only the
observation" rule — the identity token is never fed into the network.

---

## 3. Integration points into the v1 core

### 3.1 `sim/entities.py` — the identity token

```python
self.birth_id = np.zeros(cap, dtype=np.int64)   # 0 == slot never held an animal
self._next_birth_id = 1
# in spawn():
self.birth_id[slots_k] = np.arange(self._next_birth_id, self._next_birth_id + k, dtype=np.int64)
self._next_birth_id += k
```

Monotonic, unique per spawn, `0` reserved for "never spawned". **Draws no RNG** → does not
affect run determinism. Anything needing per-agent state across ticks (the LSTM here; future
brains too) can detect slot recycling by a changed id.

### 3.2 `sim/simulation.py` — the pluggable brain

```python
def __init__(self, cfg=None, brain=None):
    ...
    self.brain = brain if brain is not None else RuleBrain(self.rng, self.cfg.sim.food_eat_threshold)
    if hasattr(self.brain, "bind"):
        self.brain.bind(self.entities)
    self.brain_system = BrainSystem(self.brain)
```

Default behaviour is **identical to v1** (a `RuleBrain`). Any object honouring the contract can
be injected; if it exposes `bind`, it gets the entity handle. The tick order and every system
are unchanged.

### 3.3 Runners & viewer

- **`run_experiment.py`** — `_make_brain(kind, weights, cfg, device)` returns `None` for
  `"rule"` (Simulation builds its own) or a weight-loaded `NeuralBrain` in eval mode for
  `"neural"`. **torch is imported lazily inside** so the rule path never requires it. The brain
  is sized to the checkpoint's `hidden` before loading. New CLI: `--brain {rule,neural}`,
  `--weights PATH`, `--device`.
- **`run_live.py`** — imports and reuses `_make_brain`; same three flags; passes the brain into
  `render.viewer.run(..., brain=...)`.
- **`render/viewer.py`** — `EcosystemViewer(..., brain=None)` and `run(..., brain=None)` just
  forward the brain into `Simulation(cfg, brain=brain)`. The viewer **never constructs a brain**;
  it stays a pure observer, preserving the `sim/`↔`render/` boundary.

---

## 4. `train_neural_brain.py` — the RL trainer

### 4.1 Config dataclasses

- **`RewardConfig`** — the reward/pain weights and `reward_pain(d_energy, d_health, hunger,
  thirst, bred, died) → (reward, pain)`. Defaults: `survive 0.01`, `energy_gain 1.0`,
  `health_gain 0.5`, `reproduce 1.0` (reward); `energy_loss 1.0`, `health_loss 1.0`,
  `hunger 0.03`, `thirst 0.03`, `death 1.0` (pain). On death, only the death penalty applies.
- **`PPOConfig`** — `gamma 0.99`, `lam 0.95`, `clip 0.2`, `epochs 4`, `lr 3e-4`,
  `ent_coef 0.005`, `vf_coef 0.5`, `max_grad_norm 0.5`, `horizon 256` (ticks/iter, a *cap* under
  sleep consolidation), `max_seq 64` (BPTT window), `seq_batch 8`, `hidden 128`,
  `max_agents 96` (memory bound), `night_training True` + `night_hi/lo`, `min_cycle`, and the
  warm-start knobs `bc_iters/bc_horizon/bc_epochs`.

### 4.2 Rollout collection

- **`RolloutCollector`** — the brain calls `record(...)` once per species per tick during
  `decide`; the trainer calls `commit(ent, snap)` right after `sim.step()` with the pre-step
  `_snapshot(ent)`, computing reward/pain/terminal from the state delta. Works for both the
  neural brain (supplies logp+value+LSTM state) and the rule teacher (those default to zero).
  Key correctness details:
  - Trajectories keyed by `(sid, birth_id)`; a window full at `max_seq` **closes and bootstraps
    GAE with the next step's value**, then a fresh window opens carrying the true LSTM state
    (`init_h/init_c`). Terminal → bootstrap `0`. Rollout cut → self-bootstrap `V(s_last)`.
  - At most `max_agents` living animals per species tracked at once; grids stored `float16`.
  - `reward_pain_stats()` aggregates mean reward/pain/net + birth/death counts for logging.
- **`RecordingRuleBrain(Brain)`** — wraps a `RuleBrain` so the teacher's `(obs → act)` streams
  into the collector for imitation.
- **`_snapshot(ent)`** copies `energy, health, hunger, thirst, repro_cooldown, alive, birth_id`.
  `bred` is detected by a jump in `repro_cooldown`; the `repro_cost` is added back into
  `d_energy` so breeding isn't double-charged.

### 4.3 Returns

- **`compute_gae(signal, value, done, last_val, gamma, lam)`** → `(adv, returns)` — GAE(λ) over
  the reward−pain `signal`, honouring per-step `done`.
- **`compute_mc_returns(signal, done, gamma)`** → discounted returns, used to warm-start the
  critic during imitation.

### 4.4 `PPOTrainer`

- **`imitation_pretrain()`** — collect a fixed teacher rollout, then `_imitation_update`:
  behavioural cloning (MSE on heading + Beta-mean speed, BCE on gates) plus critic regression
  onto MC returns. Skipped when resuming or `--bc-iters 0`.
- **`collect()` / `update()`** — one PPO cycle: `_run_collect` steps the sim to the rollout
  boundary (sleep edge via `_night_edge`, or the `horizon` cap; rebuilds the world with a new
  episode seed if a species dies out), then `_ppo_update` runs clipped PPO with **stored-state
  truncated BPTT** — `_pad_batch` packs variable-length windows into `(Tm, B, …)` tensors and the
  loss loop replays the LSTM from each window's real `init_h/init_c`.
- **Masking:** policy + entropy losses train only on **controlled** steps (`ctrl` mask — the
  animal chose the action; sleep steps are excluded); the **critic trains on every valid step**.
  Advantages are normalized over controlled steps only.
- **`_night_edge`** — edge-triggered consolidation: arm when the asleep fraction drops below
  `night_lo`, fire when it crosses `night_hi` (hysteresis), after `min_cycle` ticks.
- **`save` / `try_resume`** — atomic checkpoint (`torch.save` to `.tmp` then `os.replace`) storing
  weights + `{iters_done, bootstrapped}` + `hidden`; resume reloads progress, rebuilding at the
  checkpoint's hidden size if it differs, and skips the warm-start. Unreadable checkpoints fall
  back to training fresh rather than crashing.
- **`run(iters, out, save_every)`** — warm-start (once) → PPO cycles → periodic + final atomic
  save, including on `KeyboardInterrupt`.

### 4.5 CLI (`build_train_config`, `main`)

Small world by default (`96×54`, ~90 sheep / 14 fox, caps 300/90) for fast iteration; `--full-world`
uses the full size. Flags mirror `PPOConfig`: `--iters --horizon --out --world-seed --seed
--device --hidden --max-seq --seq-batch --max-agents --lr --epochs --bc-iters --bc-horizon
--save-every --fixed-horizon --no-resume --full-world`.

---

## 5. Invariants a v2 change must preserve

1. **`sim/` never imports `render/`.** `neural_brain.py` imports only `config`, `torch`, `numpy`,
   and sibling `sim` modules. Re-verify:
   `grep -rn --include=*.py "^\s*\(import render\|from render\)" sim/` → empty.
2. **Contract unchanged.** `decide(obs_by_species, idx) → act (len(idx), ACT_DIM)`; the network
   consumes only `obs.grids`/`obs.scalars`. The identity token is lifecycle bookkeeping only and
   must never enter the policy input.
3. **Eval determinism.** Deployment must use `mode_action` (no sampling) and draw no RNG, so a
   `(world seed, run seed, weights)` triple reproduces a run and does not perturb the NumPy run
   stream. Only the trainer samples (seeded via `torch.manual_seed`).
4. **`birth_id` draws no RNG** and is monotonic — recycled slots must always reset dependent
   per-agent state.
5. **Default path is byte-for-byte v1.** `Simulation(cfg)` with no brain, and both runners
   without `--brain neural`, behave exactly as before; the neural/torch dependency stays lazy.
