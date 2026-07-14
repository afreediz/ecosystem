"""Reinforcement-learning trainer for the neural brain (imitation warm-start + recurrent PPO).

Runs the ecosystem forward, turns each animal's per-tick experience into a REWARD and a
PAIN signal, and improves the two species policies. It is a *multi-agent* RL problem:
hundreds of animals share one policy per species, are born and die mid-episode, and each
sees only its local egocentric perception. Every alive animal is an independent, parallel
experience source for its species' network.

REWARD vs PAIN (the drives; net learning signal = reward - pain, per acting animal per tick):

    REWARD (things worth seeking)          PAIN (things worth avoiding)
    -----------------------------          ----------------------------
    + survive           small tick bonus   - hunger        proportional to how hungry it is
    + energy_gain       ate / made a kill  - thirst        proportional to how thirsty it is
    + health_gain       recovering         - energy_loss   burned more than it gained
    + reproduce         produced offspring - health_loss   starving / parched / hurt
                        (the fitness win)  - death         the tick it dies (terminal)

There is NO scripted behaviour in the network. Foraging, drinking, fleeing predators and
seeking mates all emerge as the ways to earn reward and dodge pain.

TWO-PHASE TRAINING:
  1. IMITATION WARM-START.  We first run the hardcoded RuleBrain as a *teacher*, recording
     (perception -> action) and the reward/pain it earns. The network is pretrained to copy
     the teacher's actions (behavioural cloning) while its critic regresses onto the teacher's
     reward-minus-pain returns. This gives the brain a competent starting point instead of
     flailing randomly for thousands of ticks.
  2. CONTINUOUS PPO.  The network then takes over and improves itself with recurrent PPO,
     sampling its own actions and learning from its own reward/pain.

CREDIT ASSIGNMENT ACROSS BIRTH & DEATH.  Slots are recycled, so trajectories are keyed by an
animal's unique ``birth_id`` (in the entity store), not its slot. A trajectory closes
(terminal) the tick its animal dies -- detected because the slot is no longer alive or is
already re-occupied by a different id. Long lives are cut into ``max_seq`` windows for
truncated back-prop-through-time; each window stores the LSTM state that ENTERED it and is
replayed from that stored state during the update (R2D2-style stored-state recurrent PPO), so
the update sees the memory the agent actually had -- not a phantom zero state.

SLEEP CONSOLIDATION (the training rhythm).  Two facts make night the natural time to learn:
the sleep system overrides actions at night, so those ticks are masked out of the policy loss
(the animal wasn't in control), and the world is largely idle. So by default the trainer
COLLECTS experience through the day and CONSOLIDATES (runs the PPO update) when the population
falls asleep -- one update per day/night cycle, edge-triggered on the sleeping fraction. Pass
``--fixed-horizon`` to instead cut rollouts at a fixed tick count.

MEMORY.  Storing every animal's perception grid for a whole rollout is the dominant cost, so
the collector tracks at most ``max_agents`` living animals per species at once (keeping each
tracked animal's full LSTM continuity) -- RAM stays bounded regardless of the population.

CHECKPOINTS.  Weights (+ hidden size + iterations seen) are saved to ``--out`` atomically
(temp file + os.replace) on exit -- including Ctrl+C -- and reloaded on the next run, so
training resumes where it left off. Resuming skips the imitation warm-start.

Usage:
    venv/Scripts/python.exe train_neural_brain.py --iters 200 --out runs/brain.pt
    venv/Scripts/python.exe train_neural_brain.py --iters 200 --out runs/brain.pt   # resumes
    venv/Scripts/python.exe train_neural_brain.py --iters 50 --bc-iters 0           # skip warm-start
    # then deploy the trained brain headless (per species; drive one or both):
    venv/Scripts/python.exe run_experiment.py --sheep-brain runs/brain.pt --fox-brain runs/brain.pt --ticks 8000
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import make_config, SHEEP, FOX, SPECIES_NAMES
from sim.brain import Brain, RuleBrain
from sim.simulation import Simulation
from sim.neural_brain import NeuralBrain, action_logp_entropy


# --------------------------------------------------------------------------- configs
@dataclass
class RewardConfig:
    """Weights turning per-tick state changes into REWARD and PAIN (see module docstring)."""
    # reward (positive drives)
    survive: float = 0.01
    energy_gain: float = 1.0
    health_gain: float = 0.5
    reproduce: float = 1.0
    # pain (negative drives)
    energy_loss: float = 1.0
    health_loss: float = 1.0
    hunger: float = 0.03
    thirst: float = 0.03
    death: float = 1.0

    def reward_pain(self, d_energy, d_health, hunger, thirst, bred, died):
        """Return (reward, pain) floats for one transition. On death only ``pain`` applies
        (the recycled slot's post-step state belongs to a different animal, so deltas are
        meaningless -- the death penalty is the whole signal)."""
        if died:
            return 0.0, self.death
        reward = (self.survive
                  + self.energy_gain * max(d_energy, 0.0)
                  + self.health_gain * max(d_health, 0.0)
                  + (self.reproduce if bred else 0.0))
        pain = (self.energy_loss * max(-d_energy, 0.0)
                + self.health_loss * max(-d_health, 0.0)
                + self.hunger * hunger
                + self.thirst * thirst)
        return reward, pain


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    lr: float = 3e-4
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    horizon: int = 256            # sim ticks collected per PPO iteration
    max_seq: int = 64             # truncated-BPTT window length (one training sequence)
    seq_batch: int = 8            # sequences per gradient step
    hidden: int = 128
    # memory bound: track at most this many living animals per species for experience at once
    # (preserves each tracked animal's full LSTM continuity while capping RAM regardless of the
    # population -- the stored grid buffer is ~ max_agents * horizon * grid_size per species).
    max_agents: int = 96
    # sleep consolidation: when True, a rollout ends (and the PPO update runs) the moment the
    # population falls asleep, so learning happens while the world rests. ``horizon`` is then a
    # SAFETY CAP on ticks-per-cycle. When False, rollouts are exactly ``horizon`` ticks.
    night_training: bool = True
    night_hi: float = 0.5         # asleep fraction that triggers consolidation ...
    night_lo: float = 0.2         # ... after the population has been mostly awake (hysteresis)
    min_cycle: int = 60           # minimum ticks collected before a night trigger is honoured
    # imitation warm-start
    bc_iters: int = 6             # imitation passes over freshly-collected teacher data
    bc_horizon: int = 512         # teacher sim ticks collected per imitation pass
    bc_epochs: int = 3            # gradient epochs per imitation pass


# --------------------------------------------------------------------------- reward snapshot
_SNAP_KEYS = ("energy", "health", "hunger", "thirst", "repro_cooldown", "alive", "birth_id")


def _snapshot(ent):
    """Copy the pre-step per-slot state the reward/pain functions difference against."""
    return {k: getattr(ent, k).copy() for k in _SNAP_KEYS}


# --------------------------------------------------------------------------- collector
class _Traj:
    __slots__ = ("sid", "grids", "scalars", "actions", "logp", "value", "signal",
                 "done", "ctrl", "init_h", "init_c")

    def __init__(self, sid, init_h, init_c):
        self.sid = sid
        self.init_h = init_h        # LSTM state that ENTERED this window's first step
        self.init_c = init_c
        self.grids, self.scalars, self.actions = [], [], []
        self.logp, self.value, self.signal, self.done, self.ctrl = [], [], [], [], []

    def __len__(self):
        return len(self.signal)


class RolloutCollector:
    """Captures decisions from ANY recording brain and assembles per-agent training sequences.

    The brain calls ``record`` once per species per tick (during ``decide``). The trainer calls
    ``commit`` right after the sim step with the pre-step snapshot, so reward, pain and
    terminals are computed from the state delta. Works for both the neural brain (which
    supplies logp + value + LSTM state) and the rule teacher (which does not -- those default
    to zero). Finished sequences accumulate in ``self.finished[species]``.

    Two correctness details:
      * Each window stores the LSTM state that entered its first step (``init_h``/``init_c``),
        and a window cut mid-life bootstraps its GAE with the NEXT step's value (looked up when
        the following step arrives), not a self-bootstrap.
      * To bound memory, at most ``max_agents`` living animals per species are tracked at once;
        a tracked animal keeps its full continuity, and untracked ones are dropped entirely.
    """

    def __init__(self, rcfg: RewardConfig, max_seq: int, max_agents: int,
                 repro_cost: dict, hidden: int):
        self.rcfg = rcfg
        self.max_seq = max_seq
        self.max_agents = max_agents
        self.repro_cost = repro_cost          # {species_id: repro_cost}
        self.hidden = hidden
        self._pending = []
        self._open = {}
        self._tracked = {SHEEP: set(), FOX: set()}   # birth_ids we currently record
        self.finished = {SHEEP: [], FOX: []}
        self._reset_stats()

    def _reset_stats(self):
        self.sum_reward = 0.0
        self.sum_pain = 0.0
        self.n_signals = 0
        self.n_births = 0
        self.n_deaths = 0

    # -- called by the brain, mid-step --
    def record(self, sid, slots, grids, scalars, action, logp=None, value=None,
               h0=None, c0=None):
        n = np.asarray(slots).shape[0]
        z1 = np.zeros(n, np.float32)
        zh = np.zeros((n, self.hidden), np.float32)
        self._pending.append((
            sid, np.asarray(slots),
            grids.astype(np.float16, copy=True),     # views into perception buffer -> copy
            scalars.astype(np.float32, copy=True),
            action.astype(np.float32, copy=True),
            z1 if logp is None else logp.astype(np.float32, copy=True),
            z1 if value is None else value.astype(np.float32, copy=True),
            zh if h0 is None else h0.astype(np.float32, copy=True),
            zh if c0 is None else c0.astype(np.float32, copy=True),
        ))

    def begin_tick(self):
        self._pending = []

    # -- called by the trainer, right after the sim step --
    def commit(self, ent, snap):
        rc = self.rcfg
        for sid, slots, grids, scalars, action, logp, value, h0, c0 in self._pending:
            tracked = self._tracked[sid]
            for k in range(slots.shape[0]):
                slot = int(slots[k])
                bid = int(snap["birth_id"][slot])
                alive_after = bool(ent.alive[slot]) and int(ent.birth_id[slot]) == bid

                if bid not in tracked:
                    # only START tracking a living animal, and only within the memory budget
                    if not alive_after or len(tracked) >= self.max_agents:
                        continue
                    tracked.add(bid)
                    self.n_births += 1

                if alive_after:
                    d_energy = float(ent.energy[slot] - snap["energy"][slot])
                    d_health = float(ent.health[slot] - snap["health"][slot])
                    hunger = float(ent.hunger[slot])
                    thirst = float(ent.thirst[slot])
                    bred = ent.repro_cooldown[slot] > snap["repro_cooldown"][slot] + 1.5
                    if bred:
                        # the repro_cost the parent just paid is the PRICE of the reproduce
                        # reward, not a separate pain -- add it back so it isn't double-charged
                        d_energy += self.repro_cost[sid]
                    reward, pain = rc.reward_pain(d_energy, d_health, hunger, thirst, bred, False)
                    # exclude steps whose action the sleep system overrode (asleep OR dashing to
                    # cover) from the policy gradient -- the emitted action drove no outcome
                    controlled = not bool(ent.action_overridden[slot])
                    done = False
                else:
                    reward, pain = rc.reward_pain(0, 0, 0, 0, False, True)
                    controlled, done = True, True
                    self.n_deaths += 1
                    tracked.discard(bid)

                self.sum_reward += reward
                self.sum_pain += pain
                self.n_signals += 1
                self._append(sid, bid, grids[k], scalars[k], action[k], float(logp[k]),
                             float(value[k]), reward - pain, done, controlled, h0[k], c0[k])
        self._pending = []

    def _append(self, sid, bid, grids, scalars, action, logp, value, signal, done, ctrl, h0, c0):
        key = (sid, bid)
        traj = self._open.get(key)
        if traj is not None and len(traj) >= self.max_seq and not done:
            # window full and the animal lives on: close it, bootstrapping GAE with THIS step's
            # value (= V of the state entering this step, the correct V(s_next) for the window),
            # then open a fresh window that BEGINS at this step (carrying its true LSTM state).
            self._finalize(key, traj, bootstrap=value)
            traj = None
        if traj is None:
            traj = _Traj(sid, h0, c0)
            self._open[key] = traj
        traj.grids.append(grids)
        traj.scalars.append(scalars)
        traj.actions.append(action)
        traj.logp.append(logp)
        traj.value.append(value)
        traj.signal.append(signal)
        traj.done.append(done)
        traj.ctrl.append(ctrl)
        if done:
            self._finalize(key, traj, bootstrap=0.0)     # terminal: no future value

    def _finalize(self, key, traj, bootstrap):
        self.finished[traj.sid].append(_pack(traj, bootstrap))
        del self._open[key]

    def flush_open(self):
        # rollout / episode cut: no next state was recorded, so self-bootstrap V(s_last)
        for key, traj in list(self._open.items()):
            self._finalize(key, traj, bootstrap=float(traj.value[-1]))

    def reset_tracking(self):
        """Forget which birth_ids are tracked WITHOUT dropping finished sequences. Called on a
        mid-rollout world rebuild: the new world restarts birth_ids from 1, so stale ids would
        otherwise linger, never be death-discarded, and fill the max_agents cap."""
        self._tracked = {SHEEP: set(), FOX: set()}

    def reward_pain_stats(self):
        n = max(self.n_signals, 1)
        return {
            "reward": self.sum_reward / n,
            "pain": self.sum_pain / n,
            "net": (self.sum_reward - self.sum_pain) / n,
            "births": self.n_births,
            "deaths": self.n_deaths,
        }

    def clear(self):
        self.finished = {SHEEP: [], FOX: []}
        self._open = {}
        self._tracked = {SHEEP: set(), FOX: set()}
        self._reset_stats()


def _pack(traj: _Traj, bootstrap: float) -> dict:
    return {
        "grids": np.stack(traj.grids).astype(np.float16),      # (T, C, K, K)
        "scalars": np.stack(traj.scalars).astype(np.float32),  # (T, S)
        "actions": np.stack(traj.actions).astype(np.float32),  # (T, 6)
        "logp": np.asarray(traj.logp, dtype=np.float32),       # (T,)
        "value": np.asarray(traj.value, dtype=np.float32),     # (T,)
        "signal": np.asarray(traj.signal, dtype=np.float32),   # (T,)  reward - pain
        "done": np.asarray(traj.done, dtype=bool),             # (T,)
        "ctrl": np.asarray(traj.ctrl, dtype=np.float32),       # (T,)  1 = the animal was in control
        "init_h": traj.init_h.astype(np.float32),              # (hidden,)
        "init_c": traj.init_c.astype(np.float32),
        "last_val": float(bootstrap),
    }


# --------------------------------------------------------------------------- returns
def compute_gae(signal, value, done, last_val, gamma, lam):
    T = signal.shape[0]
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        nonterminal = 0.0 if done[t] else 1.0
        next_val = value[t + 1] if t + 1 < T else last_val
        delta = signal[t] + gamma * next_val * nonterminal - value[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + value


def compute_mc_returns(signal, done, gamma):
    """Discounted return of the reward-minus-pain signal (for critic warm-start)."""
    T = signal.shape[0]
    ret = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = signal[t] + gamma * running * (0.0 if done[t] else 1.0)
        ret[t] = running
    return ret


# --------------------------------------------------------------- rule-brain teacher
class RecordingRuleBrain(Brain):
    """Wraps a RuleBrain so its decisions stream into a collector for imitation learning."""

    def __init__(self, rule: RuleBrain, collector: RolloutCollector):
        self.rule = rule
        self.collector = collector

    def decide(self, obs_by_species, idx):
        act = self.rule.decide(obs_by_species, idx)
        for sid in (SHEEP, FOX):
            obs = obs_by_species.get(sid)
            if obs is None or obs.grids.shape[0] == 0:
                continue
            pos = np.searchsorted(idx, obs.idx)
            self.collector.record(sid, obs.idx, obs.grids, obs.scalars, act[pos])
        return act


# --------------------------------------------------------------------------- trainer
class PPOTrainer:
    def __init__(self, cfg, ppo: PPOConfig, rcfg: RewardConfig, device: str = "cpu",
                 seed: int | None = 0):
        self.cfg = cfg
        self.ppo = ppo
        self.rcfg = rcfg
        self.device = torch.device(device)
        self.seed = seed
        if seed is not None:
            torch.manual_seed(int(seed))
        self._np_rng = np.random.default_rng(0)

        self._build_brain(ppo.hidden)
        self.sim = self._build_neural_sim()
        self.iters_done = 0          # cumulative PPO iters (persisted in the checkpoint)
        self.bootstrapped = False    # whether the imitation warm-start has run
        self._armed = False          # night-trigger arming state (see _night_edge)

    def _build_brain(self, hidden):
        """(Re)build the brain, optimizers and collector at a given LSTM hidden size. Kept
        separate so try_resume() can rebuild at a checkpoint's size."""
        self.ppo.hidden = hidden
        self.brain = NeuralBrain(self.cfg, device=str(self.device), hidden=hidden, training=True)
        self.brain.train_mode()
        self.opts = {sid: torch.optim.Adam(self.brain.parameters(sid), lr=self.ppo.lr)
                     for sid in (SHEEP, FOX)}
        repro_cost = {sid: self.cfg.species[sid].repro_cost for sid in (SHEEP, FOX)}
        self.collector = RolloutCollector(self.rcfg, self.ppo.max_seq, self.ppo.max_agents,
                                          repro_cost, hidden)
        self.brain.recorder = self.collector

    # ------------------------------------------------------------------ sim builders
    def _build_neural_sim(self):
        self.brain.reset()
        return Simulation(self.cfg, brain=self.brain)

    def _build_teacher_sim(self):
        # a RuleBrain with its OWN rng (decoupled from run determinism -- this is just data
        # collection); wrapped so its decisions are recorded for imitation.
        rule = RuleBrain(np.random.default_rng(self.seed or 0), self.cfg.sim.food_eat_threshold)
        teacher = RecordingRuleBrain(rule, self.collector)
        return Simulation(self.cfg, brain=teacher)

    def _next_episode_seed(self):
        base = self.cfg.seed if self.cfg.seed is not None else 0
        self.cfg.seed = int((base * 2654435761 + 12345) % (2 ** 31 - 1))

    # ------------------------------------------------------------------ collection
    def _night_edge(self, sim, collected) -> bool:
        """Edge-triggered 'the world has fallen asleep' detector (one fire per day/night cycle).
        Arms once the population is mostly awake, fires when the asleep fraction crosses the
        high threshold -- i.e. at dusk, when it is time to consolidate."""
        if collected < self.ppo.min_cycle:
            return False
        pop = sim.populations["sheep"] + sim.populations["fox"]
        if pop == 0:
            return False
        frac = sim.stats.get("n_asleep", 0) / pop
        if frac < self.ppo.night_lo:
            self._armed = True
        if self._armed and frac >= self.ppo.night_hi:
            self._armed = False
            return True
        return False

    def _run_collect(self, sim, rebuild_fn, night_trigger: bool):
        """Step ``sim``, recording via the shared collector, until the rollout boundary:
        either the population falls asleep (``night_trigger``) or the ``horizon`` cap is hit;
        with ``night_trigger`` off it is exactly ``horizon`` ticks. Rebuilds the world (via
        ``rebuild_fn``) if a species dies out. Returns (stats, final_sim)."""
        coll = self.collector
        pops = {"sheep": [], "fox": []}
        collected = 0
        while collected < self.ppo.horizon:
            ent = sim.entities
            snap = _snapshot(ent)
            coll.begin_tick()
            sim.step()
            coll.commit(ent, snap)
            collected += 1
            p = sim.populations
            pops["sheep"].append(p["sheep"])
            pops["fox"].append(p["fox"])
            if p["sheep"] == 0 or p["fox"] == 0:
                coll.flush_open()
                coll.reset_tracking()      # new world restarts birth_ids -> drop stale tracking
                self._next_episode_seed()
                sim = rebuild_fn()
                self._armed = False
                continue
            if night_trigger and self._night_edge(sim, collected):
                break
        coll.flush_open()
        stats = {"sheep_pop": float(np.mean(pops["sheep"]) if pops["sheep"] else 0.0),
                 "fox_pop": float(np.mean(pops["fox"]) if pops["fox"] else 0.0),
                 "ticks": collected}
        return stats, sim

    # ------------------------------------------------------------------ imitation warm-start
    def imitation_pretrain(self):
        p = self.ppo
        if p.bc_iters <= 0:
            return
        print("[warm-start] imitation: learning from the rule brain (behavioural cloning "
              "+ critic regression on reward-pain returns)")
        saved_horizon = p.horizon
        p.horizon = p.bc_horizon                       # BC collects a fixed teacher rollout
        for it in range(1, p.bc_iters + 1):
            self.collector.clear()
            self._armed = False
            sim = self._build_teacher_sim()
            cstats, _ = self._run_collect(sim, self._build_teacher_sim, night_trigger=False)
            rp = self.collector.reward_pain_stats()
            losses = {}
            for sid in (SHEEP, FOX):
                seqs = self.collector.finished[sid]
                if seqs:
                    losses[SPECIES_NAMES[sid]] = self._imitation_update(sid, seqs)
            self._log_bc(it, p.bc_iters, cstats, rp, losses)
        p.horizon = saved_horizon
        self.bootstrapped = True
        print("[warm-start] done -- the neural brain now imitates the rule brain; "
              "handing control to it for continuous RL")

    def _imitation_update(self, sid, seqs):
        p = self.ppo
        net, opt = self.brain.nets[sid], self.opts[sid]
        net.train()
        for s in seqs:
            s["ret"] = compute_mc_returns(s["signal"], s["done"], p.gamma)
        order = np.arange(len(seqs))
        agg = {"act": 0.0, "val": 0.0, "n": 0}
        for _ in range(p.bc_epochs):
            self._np_rng.shuffle(order)
            for i in range(0, len(seqs), p.seq_batch):
                batch = [seqs[j] for j in order[i:i + p.seq_batch]]
                loss, info = self._imitation_loss(net, batch)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), p.max_grad_norm)
                opt.step()
                agg["act"] += info["act"]
                agg["val"] += info["val"]
                agg["n"] += 1
        m = max(agg["n"], 1)
        return {"act_loss": agg["act"] / m, "val_loss": agg["val"] / m, "n_seqs": len(seqs)}

    def _imitation_loss(self, net, batch):
        d = self._pad_batch(batch)
        # the teacher is stateless, so its windows carry a zero init state (stored as zeros);
        # the LSTM is rolled from there and learns the obs->action map in-context.
        h, c = d["init_h"], d["init_c"]
        mask = d["mask"]
        head_l = gate_l = speed_l = val_l = 0.0
        Tm = d["grids"].shape[0]
        for t in range(Tm):
            params, value, h, c = net(d["grids"][t], d["scalars"][t], h, c)
            mean, _, gate_logits, speed_ab = params
            m_t = mask[t]
            tgt = d["actions"][t]
            # behavioural cloning: match the teacher's heading, gates and speed
            head_l = head_l + (((mean - tgt[:, 0:2]) ** 2).sum(-1) * m_t).sum()
            gate_l = gate_l + (F.binary_cross_entropy_with_logits(
                gate_logits, tgt[:, 2:5], reduction="none").sum(-1) * m_t).sum()
            speed_mean = speed_ab[:, 0] / (speed_ab[:, 0] + speed_ab[:, 1])
            speed_l = speed_l + (((speed_mean - tgt[:, 5]) ** 2) * m_t).sum()
            # critic warm-start: regress value onto the reward-pain return
            val_l = val_l + (((value - d["ret"][t]) ** 2) * m_t).sum()
        denom = mask.sum().clamp(min=1.0)
        act_loss = (head_l + gate_l + speed_l) / denom
        val_loss = val_l / denom
        loss = act_loss + self.ppo.vf_coef * val_loss
        return loss, {"act": float(act_loss.detach()), "val": float(val_loss.detach())}

    # ------------------------------------------------------------------ PPO
    def collect(self):
        self.collector.clear()
        cstats, self.sim = self._run_collect(self.sim, self._build_neural_sim,
                                             night_trigger=self.ppo.night_training)
        cstats.update(self.collector.reward_pain_stats())
        return cstats

    def update(self):
        stats = {}
        for sid in (SHEEP, FOX):
            seqs = self.collector.finished[sid]
            if seqs:
                stats[SPECIES_NAMES[sid]] = self._ppo_update(sid, seqs)
        return stats

    def _ppo_update(self, sid, seqs):
        p = self.ppo
        net, opt = self.brain.nets[sid], self.opts[sid]
        net.train()
        for s in seqs:
            s["adv"], s["ret"] = compute_gae(s["signal"], s["value"], s["done"],
                                             s["last_val"], p.gamma, p.lam)
        # normalize advantages over the CONTROLLED steps only (the ones the policy is trained
        # on); sleep/rest steps don't contribute a policy gradient so shouldn't skew the stats
        ctrl_advs = [s["adv"][s["ctrl"].astype(bool)] for s in seqs if s["ctrl"].any()]
        if ctrl_advs:
            ctrl_adv = np.concatenate(ctrl_advs)
            a_mean, a_std = float(ctrl_adv.mean()), float(ctrl_adv.std() + 1e-8)
        else:
            a_mean, a_std = 0.0, 1.0    # whole rollout uncontrolled (e.g. collected at night)
        for s in seqs:
            s["adv"] = (s["adv"] - a_mean) / a_std

        order = np.arange(len(seqs))
        agg = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "n": 0}
        for _ in range(p.epochs):
            self._np_rng.shuffle(order)
            for i in range(0, len(seqs), p.seq_batch):
                batch = [seqs[j] for j in order[i:i + p.seq_batch]]
                loss, info = self._ppo_loss(net, batch)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), p.max_grad_norm)
                opt.step()
                for k in ("policy", "value", "entropy"):
                    agg[k] += info[k]
                agg["n"] += 1
        m = max(agg["n"], 1)
        return {"policy_loss": agg["policy"] / m, "value_loss": agg["value"] / m,
                "entropy": agg["entropy"] / m, "n_seqs": len(seqs),
                "n_steps": int(sum(len(s["signal"]) for s in seqs))}

    def _ppo_loss(self, net, batch):
        p = self.ppo
        d = self._pad_batch(batch)
        h, c = d["init_h"], d["init_c"]           # replay from the REAL rollout LSTM state
        logps, ents, vals = [], [], []
        Tm = d["grids"].shape[0]
        for t in range(Tm):
            params, value, h, c = net(d["grids"][t], d["scalars"][t], h, c)
            lp, ent = action_logp_entropy(params, d["actions"][t])
            logps.append(lp)
            ents.append(ent)
            vals.append(value)
        logp_new = torch.stack(logps)
        entropy = torch.stack(ents)
        values = torch.stack(vals)

        ratio = torch.exp(logp_new - d["logp"])
        surr1 = ratio * d["adv"]
        surr2 = torch.clamp(ratio, 1.0 - p.clip, 1.0 + p.clip) * d["adv"]
        policy_term = -torch.min(surr1, surr2)
        value_term = (values - d["ret"]) ** 2
        ent_term = -entropy

        # policy + entropy train only on CONTROLLED steps (the animal chose the action); the
        # critic trains on every valid step (a state's value is well-defined even when asleep)
        pad_mask = d["mask"]
        ctrl_mask = pad_mask * d["ctrl"]
        pad_den = pad_mask.sum().clamp(min=1.0)
        ctrl_den = ctrl_mask.sum().clamp(min=1.0)
        policy_loss = (policy_term * ctrl_mask).sum() / ctrl_den
        ent_loss = (ent_term * ctrl_mask).sum() / ctrl_den
        value_loss = (value_term * pad_mask).sum() / pad_den
        loss = policy_loss + p.vf_coef * value_loss + p.ent_coef * ent_loss
        return loss, {"policy": float(policy_loss.detach()), "value": float(value_loss.detach()),
                      "entropy": float(-ent_loss.detach())}

    # ------------------------------------------------------------------ batching
    def _pad_batch(self, batch):
        """Pad variable-length sequences into a dict of (Tm, B, ...) device tensors plus the
        per-sequence initial LSTM state (B, hidden). Fields: grids, scalars, actions, mask
        (padding validity), ctrl (agent-in-control), logp, adv, ret, init_h, init_c. ``adv`` is
        zero when absent (imitation batches don't compute advantages)."""
        dev = self.device
        B = len(batch)
        lengths = [len(s["signal"]) for s in batch]
        Tm = max(lengths)
        C, K = batch[0]["grids"].shape[1], batch[0]["grids"].shape[2]
        S = batch[0]["scalars"].shape[1]
        A = batch[0]["actions"].shape[1]
        H = batch[0]["init_h"].shape[0]

        grids = np.zeros((Tm, B, C, K, K), dtype=np.float32)
        scalars = np.zeros((Tm, B, S), dtype=np.float32)
        actions = np.zeros((Tm, B, A), dtype=np.float32)
        mask = np.zeros((Tm, B), dtype=np.float32)
        ctrl = np.zeros((Tm, B), dtype=np.float32)
        ret = np.zeros((Tm, B), dtype=np.float32)
        logp = np.zeros((Tm, B), dtype=np.float32)
        adv = np.zeros((Tm, B), dtype=np.float32)
        init_h = np.zeros((B, H), dtype=np.float32)
        init_c = np.zeros((B, H), dtype=np.float32)
        for b, s in enumerate(batch):
            T = lengths[b]
            grids[:T, b] = s["grids"].astype(np.float32)
            scalars[:T, b] = s["scalars"]
            actions[:T, b] = s["actions"]
            mask[:T, b] = 1.0
            ctrl[:T, b] = s["ctrl"]
            ret[:T, b] = s["ret"]
            logp[:T, b] = s["logp"]
            if "adv" in s:
                adv[:T, b] = s["adv"]
            init_h[b] = s["init_h"]
            init_c[b] = s["init_c"]

        t = lambda x: torch.from_numpy(x).to(dev)
        return {"grids": t(grids), "scalars": t(scalars), "actions": t(actions),
                "mask": t(mask), "ctrl": t(ctrl), "ret": t(ret), "logp": t(logp),
                "adv": t(adv), "init_h": t(init_h), "init_c": t(init_c)}

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        blob = self.brain.state_dict()
        blob["meta"] = {"iters_done": self.iters_done, "bootstrapped": self.bootstrapped}
        # atomic write: a Ctrl+C / crash mid-write must not corrupt the existing checkpoint
        tmp = f"{path}.tmp"
        torch.save(blob, tmp)
        os.replace(tmp, path)

    def try_resume(self, path):
        """Load weights + progress from ``path`` if it exists. Returns True on resume. A
        corrupt/unreadable checkpoint is skipped (train fresh) rather than crashing, and a
        checkpoint saved at a different --hidden is honoured by rebuilding at that size."""
        if not Path(path).exists():
            return False
        try:
            blob = torch.load(path, map_location=self.device)
        except Exception as e:
            print(f"[resume] checkpoint at {path} is unreadable ({e}); starting fresh")
            return False
        ckpt_hidden = int(blob.get("hidden", self.ppo.hidden))
        if ckpt_hidden != self.ppo.hidden:
            print(f"[resume] checkpoint hidden={ckpt_hidden} != requested {self.ppo.hidden}; "
                  f"rebuilding at {ckpt_hidden}")
            self._build_brain(ckpt_hidden)
            self.sim = self._build_neural_sim()
        self.brain.nets[SHEEP].load_state_dict(blob["sheep"])
        self.brain.nets[FOX].load_state_dict(blob["fox"])
        meta = blob.get("meta", {})
        self.iters_done = int(meta.get("iters_done", 0))
        self.bootstrapped = bool(meta.get("bootstrapped", True))
        print(f"[resume] loaded {Path(path).resolve()} "
              f"({self.iters_done} iters already trained, hidden={ckpt_hidden})")
        return True

    # ------------------------------------------------------------------ driver
    def run(self, iters, out, save_every=10):
        t0 = time.time()
        try:
            if not self.bootstrapped:
                self.imitation_pretrain()      # a no-op if bc_iters == 0
                self.bootstrapped = True        # warm-start phase is behind us either way
                self.save(out)
            mode = ("sleep-consolidation (update when the world sleeps; horizon "
                    f"{self.ppo.horizon} = safety cap)" if self.ppo.night_training
                    else f"fixed horizon {self.ppo.horizon} ticks")
            print(f"[RL] continuous PPO -- {mode}: {iters} cycles")
            for it in range(1, iters + 1):
                cstats = self.collect()
                ustats = self.update()
                self.iters_done += 1
                self._log_ppo(it, iters, cstats, ustats, time.time() - t0)
                if it % save_every == 0:
                    self.save(out)
                    print(f"[save] checkpoint -> {out} (cycle {self.iters_done})")
        except KeyboardInterrupt:
            print("\n[interrupt] saving before exit ...")
        finally:
            self.save(out)
            print(f"[save] final -> {Path(out).resolve()} "
                  f"({self.iters_done} total cycles)")
        return self.brain

    # ------------------------------------------------------------------ logging
    @staticmethod
    def _log_bc(it, iters, cstats, rp, losses):
        parts = [f"  [BC {it}/{iters}]",
                 f"sheep~{cstats['sheep_pop']:.0f} fox~{cstats['fox_pop']:.0f}",
                 f"reward={rp['reward']:.3f} pain={rp['pain']:.3f}"]
        for name in ("sheep", "fox"):
            u = losses.get(name)
            if u:
                parts.append(f"{name[:2]}: act={u['act_loss']:.3f} val={u['val_loss']:.3f}")
        print("  ".join(parts))

    @staticmethod
    def _log_ppo(it, iters, cstats, ustats, elapsed):
        parts = [f"  [{it:>4}/{iters}]",
                 f"{cstats.get('ticks', 0):>3}t",
                 f"sheep~{cstats['sheep_pop']:.0f} fox~{cstats['fox_pop']:.0f}",
                 f"R={cstats['reward']:.3f} P={cstats['pain']:.3f} net={cstats['net']:+.3f}",
                 f"births={cstats['births']} deaths={cstats['deaths']}"]
        for name in ("sheep", "fox"):
            u = ustats.get(name)
            if u:
                parts.append(f"{name[:2]}:pi={u['policy_loss']:+.3f} "
                             f"v={u['value_loss']:.2f} H={u['entropy']:.2f}")
        parts.append(f"{elapsed:.0f}s")
        print("  ".join(parts))


# --------------------------------------------------------------------------- CLI
def build_train_config(world_seed, seed, small):
    cfg = make_config(world_seed=world_seed, seed=seed)
    if small:
        cfg.world.width = 96
        cfg.world.height = 54
        cfg.species[SHEEP].init_count = 90
        cfg.species[FOX].init_count = 14
        cfg.species[SHEEP].population_cap = 300
        cfg.species[FOX].population_cap = 90
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Train the neural ecosystem brain (BC + PPO).")
    ap.add_argument("--iters", type=int, default=200,
                    help="training cycles this run (one day/night cycle each by default)")
    ap.add_argument("--horizon", type=int, default=300,
                    help="ticks per cycle (a hard cap when sleep-consolidation is on)")
    ap.add_argument("--out", type=str, default="runs/brain.pt")
    ap.add_argument("--world-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max-seq", type=int, default=128)
    ap.add_argument("--seq-batch", type=int, default=32)
    ap.add_argument("--max-agents", type=int, default=96,
                    help="max living animals tracked per species (memory bound)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bc-iters", type=int, default=6,
                    help="imitation warm-start passes (0 to skip); ignored when resuming")
    ap.add_argument("--bc-horizon", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--fixed-horizon", action="store_true",
                    help="disable sleep-consolidation; use exactly --horizon ticks per cycle")
    ap.add_argument("--no-resume", action="store_true",
                    help="start fresh even if a checkpoint exists at --out")
    ap.add_argument("--small", action="store_true",
                    help="train on a shrunken world/populations (faster, less memory); "
                         "default is the full-size world")
    args = ap.parse_args()

    cfg = build_train_config(args.world_seed, args.seed, small=args.small)
    ppo = PPOConfig(horizon=args.horizon, hidden=args.hidden, max_seq=args.max_seq,
                    seq_batch=args.seq_batch, max_agents=args.max_agents, lr=args.lr,
                    epochs=args.epochs, bc_iters=args.bc_iters, bc_horizon=args.bc_horizon,
                    night_training=not args.fixed_horizon)
    trainer = PPOTrainer(cfg, ppo, RewardConfig(), device=args.device, seed=args.seed)

    if not args.no_resume:
        trainer.try_resume(args.out)

    print(f"world={'small' if args.small else 'FULL'} "
          f"(K={trainer.sim.perception.K}, hidden={trainer.ppo.hidden}) device={args.device} "
          f"| {'sleep-consolidation' if ppo.night_training else 'fixed-horizon'}")
    trainer.run(args.iters, args.out, save_every=args.save_every)


if __name__ == "__main__":
    main()
