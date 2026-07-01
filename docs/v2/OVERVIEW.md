# Ecosystem + Evolution Simulation — v2 Overview (Neural Brain)

> **What this document is.** A conceptual tour of v2: the **learned neural brain** that drops
> in behind the v1 `Brain.decide` contract, and the reinforcement-learning trainer that teaches
> it. It is the "what and why". For the code-level companion (files, classes, signatures) see
> [TECHNICAL.md](TECHNICAL.md). v2 changes **no sim rules** — it adds a second brain and the
> plumbing to train, save, load and deploy it. For the world/systems it runs inside, read the
> v1 docs first: [../v1/OVERVIEW.md](../v1/OVERVIEW.md) and [../v1/TECHNICAL.md](../v1/TECHNICAL.md).

---

## 1. The one-paragraph picture

v1 proved the seam: every animal decides through `brain.decide(obs_by_species, idx) → act`,
perceiving only egocentric **grids** `(N, C, K, K)` + a scalar vector `(N, 10)` and returning
the `(len(idx), 6)` action matrix. v1's brain was **hardcoded rules**. **v2 slots a learned
PyTorch brain into that exact seam** — no sim rewrite, as designed. The neural brain reads the
raw perception channels through a **CNN**, folds in interoceptive scalars through an **MLP**,
carries **per-agent LSTM memory** across ticks, and emits the same 6-column action through
learned actor heads. It is trained by a two-phase pipeline — **imitate the rule brain**, then
**improve with reinforcement learning** — driven entirely by a biologically framed **reward vs
pain** signal. No behaviour is scripted in the network: foraging, drinking, fleeing and mating
all emerge as ways to earn reward and dodge pain.

The deliverable is still **data**: a trained brain (`.pt`) you can deploy headless or watch
live, producing the same population/trait CSVs as v1 so learned and rule-driven ecosystems are
directly comparable.

---

## 2. What v2 adds (and what it deliberately doesn't touch)

**Added:**

- `sim/neural_brain.py` — `NeuralBrain`, a drop-in `Brain` with a per-species CNN+MLP+LSTM
  actor-critic network.
- `train_neural_brain.py` — the RL trainer (imitation warm-start + recurrent PPO), with
  checkpointing and resume.
- A **pluggable brain** on `Simulation` and both runners (`run_experiment.py`, `run_live.py`),
  so `--brain neural --weights PATH` swaps the learned brain in anywhere the rule brain ran.
- A per-animal **identity token** (`entities.birth_id`) so a brain can keep memory across ticks
  without a recycled slot leaking one animal's memory into the next.

**Untouched (the invariants v1 established still hold):**

- **`sim/` never imports `render/`.** The neural brain lives in `sim/`, is pure numbers + torch,
  and the viewer stays an observer that merely *receives* a brain.
- **The brain↔world contract is unchanged.** Same `Observation`, same `ACT_DIM=6` action, same
  per-species grid layouts. The network sees **only** the observation — no hidden world access.
- **Determinism.** In eval mode the brain acts deterministically and draws no randomness, and
  `birth_id` consumes no RNG, so **same world seed + config + run seed + weights ⇒ identical
  run**, exactly as v1 promised (see §7).
- **The calibration.** Every predator–prey stabilizer in [../../CLAUDE.md](../../CLAUDE.md)
  (refuge, Type III response, adult founders, lean predator metabolism, …) is a *sim* mechanism
  and is untouched. The brain only chooses actions within that world.

---

## 3. The network — one policy per species

Two independent networks are held, one for **sheep** (5 perception channels) and one for
**fox** (4), because the two species carry different perception layouts — exactly the
per-species CNN the grid design was built for. Each is a **recurrent actor-critic**
(`SpeciesActorCritic`) with four stages:

```
  grids (N, C, K, K)  ── CNN ──▶ spatial feature (128)
  scalars (N, 10)     ── MLP ──▶ interoceptive feature (32)     health, hunger, thirst, energy,
                                                                age, sex, temperature, time-of-day, …
  [spatial | intero]  ── LSTMCell ──▶ recurrent memory (128, per-agent, carried across ticks)
  memory ── actor heads ──▶ action        heading (gaussian), eat/drink/repro (bernoulli), speed (beta)
  memory ── critic head ──▶ state value    (used only by the trainer)
```

- **CNN** — three stride-2 conv layers (`C→16→32→32`) then an **`AdaptiveAvgPool2d(4,4)`**. The
  adaptive pool is deliberate: it makes the network accept **any window size `K`**, which the
  config sets from the largest sensory range and can differ between runs. So a brain trained on
  a small world still loads and runs on the full world.
- **Scalar MLP** — a small 2-layer MLP over the 10-D interoceptive/global vector (internal state
  + global environment).
- **LSTM** — an `LSTMCell` over the concatenated features gives each agent **memory** (see §6).
- **Actor heads** — a **hybrid action space**: a 2-D **Gaussian** heading (movement normalizes
  it to a unit direction, so magnitude is a free exploration dimension), three independent
  **Bernoulli** gates (eat/drink/repro), and a **Beta** speed throttle in `(0,1)`. Log-probs and
  entropies sum across the three factors.
- **Critic head** — a scalar state-value, used only during training for advantage estimation.

---

## 4. Reward vs pain — the only teaching signal

There is **no scripted behaviour** in the network. It is shaped entirely by a per-tick signal,
framed as two opposing drives whose difference is what the policy maximizes:

```
  net learning signal  =  reward − pain      (per acting animal, per tick)

  REWARD (worth seeking)                     PAIN (worth avoiding)
  ─────────────────────────                  ─────────────────────────
  + survive      small tick bonus            − hunger       ∝ how hungry it is
  + energy_gain  ate / made a kill           − thirst       ∝ how thirsty it is
  + health_gain  recovering                  − energy_loss  burned more than it gained
  + reproduce    produced offspring          − health_loss  starving / parched / hurt
                 (the fitness win)           − death        the tick it dies (terminal)
```

Two subtleties that keep the signal honest:

- **Death** is a terminal pain-only event. Because the slot is recycled, its post-step state
  belongs to a *different* animal, so state deltas are meaningless there — the death penalty is
  the whole signal for that final transition.
- **Reproduction** pays an energy `repro_cost`. That cost is added back into the energy delta on
  the tick an animal breeds, so the parent isn't *double-charged* (once as `reproduce` reward
  earned, once as `energy_loss` pain) for the same event.

Foraging, drinking, predator evasion and mate-seeking are never coded — they are simply the
strategies that maximize reward and minimize pain in this world.

---

## 5. Training — imitate, then improve

Learning an ecosystem from scratch is a hard multi-agent RL problem: hundreds of animals share
one policy per species, are born and die mid-episode, and each sees only its local patch. Every
alive animal is an independent, parallel experience source for its species' network. Two phases
make it tractable:

1. **Imitation warm-start (behavioural cloning).** Run the hardcoded `RuleBrain` as a *teacher*,
   recording `(perception → action)` and the reward/pain it earns. The network is pretrained to
   **copy the teacher's actions** (MSE on heading and speed, cross-entropy on the gates) while
   its **critic regresses onto the teacher's reward−pain returns**. This hands the brain a
   competent starting point instead of thousands of ticks of random flailing.
2. **Continuous recurrent PPO.** The network then takes over, **samples its own actions**, and
   improves from its own reward/pain via PPO. Because the policy is recurrent, updates use
   **stored-state truncated BPTT** (R2D2-style): each training window replays from the LSTM
   state the agent *actually had* entering it, not a phantom zero state.

**Credit assignment across birth and death.** Trajectories are keyed by an animal's unique
`birth_id`, not its recycled slot. A trajectory closes (terminal) the tick its animal dies. Long
lives are cut into fixed-length windows for truncated BPTT.

**Sleep consolidation — the training rhythm.** v1's sleep system overrides actions at night
(animals bed down in cover), so night ticks are ones the animal *wasn't in control* of — they're
masked out of the policy loss. And the world is largely idle then. So by default the trainer
**collects experience through the day and runs the PPO update when the population falls asleep**
— one consolidation per day/night cycle, edge-triggered on the sleeping fraction (with
hysteresis). It's a satisfying mirror of biological memory consolidation during sleep. Pass
`--fixed-horizon` to instead cut rollouts at a fixed tick count.

**Bounded memory.** Storing every animal's perception grid for a whole rollout is the dominant
cost, so the collector tracks at most `--max-agents` living animals per species at once (each
with full LSTM continuity), and stores grids as `float16`. RAM stays bounded no matter how big
the population grows.

**Checkpoints & resume.** Weights (plus hidden size and cumulative iterations) are saved to
`--out` **atomically** on exit — including on `Ctrl+C` — and reloaded on the next run, so
training resumes where it left off (and skips the warm-start). A checkpoint saved at a different
hidden size is honoured by rebuilding the network at that size.

---

## 6. Memory that respects the contract

The LSTM hidden state is the brain's **own** internal memory, not a back door into the world.
Each decision still reads **only** the observation — the contract is intact. Memory lives in a
per-slot table (`h`/`c` of shape `(cap, hidden)`, one pair per species).

The problem this creates: entity slots are recycled by the free list, so slot 42 might hold a
sheep this tick and its great-grandchild 500 ticks later. Without care, the newborn would
inherit the dead animal's memory. The fix is `entities.birth_id` — a **monotonic identity
token** stamped uniquely on every spawn. The brain remembers which token owns each slot's memory;
when it sees a *changed* token, it knows the slot was recycled and **zeros that slot's LSTM
state** before use. Reading the token is pure lifecycle bookkeeping — **it never enters the
policy input**, and stamping it **draws no RNG**, so determinism is untouched.

---

## 7. Determinism — still guaranteed at deploy time

v1's reproducibility promise survives intact when deploying a trained brain:

- **Eval mode acts by the distribution's mode** (mean heading, gate = probability > ½, Beta-mean
  speed) — it **samples nothing**, so it draws no randomness at all and does **not** perturb the
  NumPy run-RNG stream the other systems consume.
- `birth_id` stamping consumes no RNG.

So **same world seed + config + run seed + weights ⇒ byte-identical run**, exactly like the rule
brain — and a trained brain is a reproducible experiment. (During *training* the brain samples
via the global torch RNG, which the trainer seeds with `torch.manual_seed`.)

---

## 8. Using it

```bash
# 1. Train (small world by default; resumes automatically if runs/brain.pt exists)
venv/Scripts/python.exe train_neural_brain.py --iters 200 --out runs/brain.pt

#    skip the imitation warm-start, or train on the full-size world:
venv/Scripts/python.exe train_neural_brain.py --iters 50 --bc-iters 0
venv/Scripts/python.exe train_neural_brain.py --iters 200 --full-world

# 2. Deploy headless — produces the same CSV as a rule-brain run, for comparison
venv/Scripts/python.exe run_experiment.py --brain neural --weights runs/brain.pt --ticks 8000 --plot

# 3. Watch it live (needs a display)
venv/Scripts/python.exe run_live.py --brain neural --weights runs/brain.pt --scale 5 --spf 2
```

Omit `--brain neural` anywhere and you get the v1 rule brain unchanged — the default is always
`rule`, and the neural path imports torch lazily so the rule path never needs it installed.

---

## 9. Where this sits on the roadmap

v2 delivers the first rung of the long-term fully-neural vision (v1.md §7.1, §13, §21): a
**CNN + scalar-MLP + LSTM actor-critic per species**, trained by imitation + RL on an emergent
reward/pain signal, deployable deterministically. The seam it plugs into was v1's whole reason
for existing. Natural next rungs — richer memory (transformer/GNN perception), neuroevolution of
architecture alongside the genome, and replay-as-sleep at larger scale — all reuse this same
contract and trainer scaffold.
