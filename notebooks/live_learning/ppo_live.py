"""Live reinforcement-learning engine for the ecosystem sim (critic-free PPO).

The imitation-learning notebooks (``notebooks/imitation_learning/``) distil the ``RuleBrain``
into a memoryless per-species policy by behavioural cloning.  Cloning can only *copy* the
teacher -- the evaluate notebook found the cloned FOX dies out, because a single-step clone
cannot hold a predator through prey troughs.  This module lets the animals keep learning *as
they live*: each individual collects on-policy experience while awake, and when the population
falls asleep at night the simulation pauses and a PPO update runs.  On waking, collection
resumes.  The pretrained clones are the warm start; there is NO ``RuleBrain`` in the loop.

DESIGN (matches the user's two decisions)
-----------------------------------------
* **Critic-free PPO.**  There is no value network.  The advantage of each step is its
  discounted return-to-go along that agent's own trajectory, whitened to a per-species
  baseline (REINFORCE-with-baseline optimised under the clipped PPO surrogate).  This keeps the
  deployed brain literally "actor only, no critic".
* **Heuristic old-age death.**  We never modify ``sim/``.  A death is detected by an agent's
  ``birth_id`` vanishing from its slot; whether it was *avoidable* is inferred from its age vs
  its ``max_age`` gene at the moment it acted (old age -> no pain).

CONTRACT PRESERVED
------------------
The trainable policy adds only a learnable heading log-std on top of the imitation actor.  Its
``forward(grids, scalars)`` still returns the exact ``(head_mean, gate_logits, speed_logit)``
3-tuple, so ``export_policy`` writes a drop-in TorchScript archive that ``sim.policy_brain``
loads unchanged for ``run_experiment.py`` / ``run_live.py``.

DETERMINISM
-----------
Action sampling draws from torch's global RNG (seed it with ``torch.manual_seed``), which is
independent of the sim's numpy ``Generator``.  An RL run is therefore not bit-reproducible like
a rule run -- expected for on-policy RL -- but the sim systems (movement/consumption/
metabolism/reproduction) keep their own RNG stream intact because the brain never draws from it.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- repo wiring: reach the shared notebooks toolkit (notebooks/common.py), which in turn puts
#     the repo root on sys.path (via common.find_repo()), so ``import config`` / ``import sim``
#     work too. ---
_HERE = Path(__file__).resolve().parent
_NOTEBOOKS = _HERE.parent
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))

import common as C                                       # noqa: E402  (shared notebooks toolkit)
from config import SHEEP, FOX                            # noqa: E402
from sim import genome as gn                             # noqa: E402
from sim.brain import (                                  # noqa: E402
    ACT_DIM, Brain, A_DX, A_DY, A_EAT, A_DRINK, A_REPRO, A_SPEED,
)

import torch                                             # noqa: E402
import torch.nn as nn                                    # noqa: E402
import torch.nn.functional as F                          # noqa: E402
from torch.distributions import Normal, Bernoulli        # noqa: E402

SPECIES_IDS = (SHEEP, FOX)
# heading columns are contiguous (A_DX, A_DY == 0, 1); gate columns contiguous (2,3,4)
_HEAD = slice(A_DX, A_DY + 1)          # 0:2
_GATES = slice(A_EAT, A_REPRO + 1)     # 2:5


# =========================================================================== POLICY
class PPOPolicy(nn.Module):
    """The imitation actor made stochastic for PPO, with NO critic.

    Wraps a ``common.build_policy`` module (the exact CNN+MLP+soft-argmax architecture the
    clones use -- so warm-start weights load by name and export is a drop-in) and adds a single
    learnable heading log-std.  Action distribution:

      * heading (dx, dy) ~ Normal(head_mean, exp(head_logstd))   -- 2-D, magnitude is free
                                                                    (movement normalises it)
      * eat / drink / repro ~ Bernoulli(gate_logits)             -- three 0/1 gates
      * speed ~ Bernoulli(speed_logit)                           -- 0/1 throttle (kept Bernoulli
                                                                    so the deploy contract holds)
    """

    def __init__(self, sid, pool="softargmax"):
        super().__init__()
        self.actor = C.build_policy(sid, pool=pool)     # head_mean/head_gates/head_speed inside
        # state-independent heading exploration std (log space); clamped in use.
        self.head_logstd = nn.Parameter(torch.full((2,), -0.5))

    # -- deploy contract: identical 3-tuple, so save_model -> drop-in TorchScript archive --
    def forward(self, grids, scalars):
        return self.actor(grids, scalars)

    def _heading_std(self, mean):
        return self.head_logstd.clamp(-2.0, 1.0).exp().expand_as(mean)

    @torch.no_grad()
    def act(self, grids, scalars):
        """Sample an action + its log-prob for on-policy collection. Returns (actions(N,6), logp(N,))."""
        mean, gate_logits, speed_logit = self.actor(grids, scalars)
        head_d = Normal(mean, self._heading_std(mean))
        gate_d = Bernoulli(logits=gate_logits)
        speed_d = Bernoulli(logits=speed_logit.squeeze(-1))
        head = head_d.sample()
        gate = gate_d.sample()
        speed = speed_d.sample()
        logp = (head_d.log_prob(head).sum(-1)
                + gate_d.log_prob(gate).sum(-1)
                + speed_d.log_prob(speed))
        actions = torch.cat([head, gate, speed.unsqueeze(-1)], dim=-1)
        return actions, logp

    @torch.no_grad()
    def act_greedy(self, grids, scalars):
        """Deterministic action (head mean / logit>0 thresholds) -- matches ``PolicyBrain``."""
        mean, gate_logits, speed_logit = self.actor(grids, scalars)
        gate = (gate_logits > 0.0).float()
        speed = (speed_logit.squeeze(-1) > 0.0).float()
        return torch.cat([mean, gate, speed.unsqueeze(-1)], dim=-1)

    def eval_actions(self, grids, scalars, actions):
        """Log-prob + entropy of stored actions under the CURRENT policy (grads on). For PPO."""
        mean, gate_logits, speed_logit = self.actor(grids, scalars)
        head_d = Normal(mean, self._heading_std(mean))
        gate_d = Bernoulli(logits=gate_logits)
        speed_d = Bernoulli(logits=speed_logit.squeeze(-1))
        logp = (head_d.log_prob(actions[:, _HEAD]).sum(-1)
                + gate_d.log_prob(actions[:, _GATES]).sum(-1)
                + speed_d.log_prob(actions[:, A_SPEED]))
        ent = head_d.entropy().sum(-1) + gate_d.entropy().sum(-1) + speed_d.entropy()
        return logp, ent

    def load_warm_start(self, path):
        """Load imitation weights into the actor. Returns True on success, False (with a warning)
        if the checkpoint is missing/incompatible -- training then starts from fresh init."""
        path = Path(path)
        if not path.exists():
            warnings.warn(f"warm-start checkpoint not found: {path} -- starting from random init")
            return False
        try:
            sd = torch.jit.load(str(path), map_location="cpu").state_dict()
            missing, unexpected = self.actor.load_state_dict(sd, strict=False)
            if unexpected:
                warnings.warn(f"warm-start had unexpected keys (ignored): {list(unexpected)[:5]}")
            if missing:
                warnings.warn(f"warm-start missing keys (kept at init): {list(missing)[:5]}")
            return True
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"warm-start failed to load {path}: {e} -- starting from random init")
            return False


def build_ppo_policy(sid, warm_start=None, device="cpu", pool="softargmax"):
    """Construct a species' trainable policy, optionally warm-started from a clone checkpoint."""
    policy = PPOPolicy(sid, pool=pool)
    if warm_start is not None:
        policy.load_warm_start(warm_start)
    return policy.to(device)


def export_policy(sid, policy, path, meta=None):
    """Write the actor as a self-contained TorchScript archive loadable by ``sim.policy_brain``.

    The actor IS a ``common.build_policy`` module (the exact class the clones script), so this
    yields a clean ``(grids, scalars) -> 3-tuple`` drop-in with none of the training-only params
    (``head_logstd``) attached.
    """
    return C.save_model(sid, policy.actor, path=path, meta=meta)


# =========================================================================== REWARD
@dataclass
class RewardConfig:
    """Per-tick reward shaping. Positive terms reward living well; ``death_penalty`` is the
    'pain' of an avoidable death. Death by old age is unavoidable and carries NO pain."""
    survive_bonus: float = 0.02        # small living wage each controlled awake tick
    eat_gain_scale: float = 4.0        # x max(0, delta energy) -- grazing / a successful hunt
    drink_bonus: float = 0.15          # thirst quenched (dropped to ~0 from a real need)
    drink_thr: float = 0.25            # only reward drinking that relieved a real thirst
    repro_bonus: float = 2.0           # produced offspring this tick (repro_cooldown jumped up)
    health_gain_scale: float = 1.0     # x signed delta health (recovery rewarded, damage penalised)
    need_penalty: float = 0.02         # x (hunger + thirst) after -- gentle discomfort pressure
    death_penalty: float = 2.0         # pain of an AVOIDABLE death (starve/thirst/predation/...)
    age_death_frac: float = 0.9        # age >= frac * max_age when it died -> treat as old age (no pain)


def snapshot(ent):
    """Copy the per-slot state PPO needs to diff across a step (indexed by global slot id).

    Taken BEFORE ``sim.step()``; slots recorded in ``pending`` are still valid indices into
    these arrays because perception built ``idx`` from the same pre-step alive set."""
    return {
        "energy": ent.energy.copy(),
        "health": ent.health.copy(),
        "hunger": ent.hunger.copy(),
        "thirst": ent.thirst.copy(),
        "repro_cooldown": ent.repro_cooldown.copy(),
        "age": ent.age.copy(),
        "max_age": gn.gene(ent.genome, "max_age").copy(),
        "birth_id": ent.birth_id.copy(),
    }


def compute_rewards(rcfg, snap, ent, pend):
    """Reward for each agent that acted this tick (``pend``), by snapshot-and-diff.

    Returns (reward(N,), done(N,) bool, controlled(N,) bool).  ``done`` marks agents that died
    this tick; ``controlled`` is False for survivors whose action the sleep system overrode
    (excluded from the policy gradient, per ``ent.action_overridden``)."""
    sl = pend["slot"]                       # global slot ids this agent occupied when it acted
    bid = pend["birth_id"]
    n = sl.shape[0]

    # who is still the same living animal after the step?  (slot may have been recycled)
    alive_now = ent.alive[sl] & (ent.birth_id[sl] == bid)
    died = ~alive_now

    # before-state (valid for everyone -- taken pre-step)
    e0 = snap["energy"][sl]; h0 = snap["health"][sl]
    hu0 = snap["hunger"][sl]; th0 = snap["thirst"][sl]
    rc0 = snap["repro_cooldown"][sl]
    age0 = snap["age"][sl]; maxage = snap["max_age"][sl]

    # after-state (only meaningful where the animal survived; keep = before elsewhere so deltas=0)
    e1 = np.where(alive_now, ent.energy[sl], e0)
    h1 = np.where(alive_now, ent.health[sl], h0)
    hu1 = np.where(alive_now, ent.hunger[sl], hu0)
    th1 = np.where(alive_now, ent.thirst[sl], th0)
    rc1 = np.where(alive_now, ent.repro_cooldown[sl], rc0)

    r = np.full(n, rcfg.survive_bonus, dtype=np.float32)
    r += rcfg.eat_gain_scale * np.maximum(0.0, e1 - e0)          # ate / hunted
    drank = (th0 > rcfg.drink_thr) & (th1 <= 1e-3)
    r += rcfg.drink_bonus * drank
    bred = rc1 > (rc0 + 1.5)                                     # cooldown jump beats the -dt decay
    r += rcfg.repro_bonus * bred
    r += rcfg.health_gain_scale * (h1 - h0)                     # signed
    r -= rcfg.need_penalty * (hu1 + th1)

    # death: dead animals get NO shaping; avoidable death gets pain; old age gets nothing.
    old_age = age0 >= rcfg.age_death_frac * np.maximum(maxage, 1e-6)
    r = np.where(died, 0.0, r).astype(np.float32)
    r -= (rcfg.death_penalty * (died & ~old_age)).astype(np.float32)

    # controlled mask: only trust action_overridden for survivors (dead slots may be recycled).
    controlled = np.ones(n, dtype=bool)
    if alive_now.any():
        controlled[alive_now] = ~ent.action_overridden[sl[alive_now]]
    return r, died, controlled


# =========================================================================== ROLLOUT BUFFER
class RolloutBuffer:
    """On-policy experience for ONE species over the current day, keyed by ``birth_id``.

    Tracks at most ``max_agents`` individuals' full trajectories (bounds memory regardless of
    population -- grids are the memory hog).  Grids are stored float16 on CPU; minibatches are
    cast to float32 on the training device.  At night, ``build_batch`` turns each trajectory
    into discounted returns-to-go and whitens them into critic-free advantages."""

    def __init__(self, gamma=0.99, max_agents=128):
        self.gamma = float(gamma)
        self.max_agents = int(max_agents)
        self.traj = {}          # birth_id -> dict of per-step lists
        self.skipped = 0        # agents dropped this cycle because the tracker was full

    def add(self, pend, reward, done, controlled):
        bid = pend["birth_id"]; grids = pend["grids"]; scalars = pend["scalars"]
        action = pend["action"]; logp = pend["logp"]
        for i in range(bid.shape[0]):
            b = int(bid[i])
            t = self.traj.get(b)
            if t is None:
                if len(self.traj) >= self.max_agents:
                    self.skipped += 1
                    continue
                t = {"grids": [], "scalars": [], "action": [], "logp": [],
                     "reward": [], "done": [], "controlled": []}
                self.traj[b] = t
            t["grids"].append(grids[i])
            t["scalars"].append(scalars[i])
            t["action"].append(action[i])
            t["logp"].append(logp[i])
            t["reward"].append(float(reward[i]))
            t["done"].append(bool(done[i]))
            t["controlled"].append(bool(controlled[i]))

    def n_transitions(self):
        return sum(len(t["reward"]) for t in self.traj.values())

    def build_batch(self):
        """Flatten trajectories into a training batch with critic-free (whitened-return) advantages.

        Returns a dict of numpy arrays, or None if nothing was collected."""
        if not self.traj:
            return None
        grids_all, scalars_all, act_all, logp_all, ret_all, ctrl_all = [], [], [], [], [], []
        for t in self.traj.values():
            rew = np.asarray(t["reward"], dtype=np.float32)
            done = np.asarray(t["done"], dtype=bool)
            G = np.zeros_like(rew)
            running = 0.0
            for k in range(len(rew) - 1, -1, -1):
                if done[k]:                      # death is terminal -> no bootstrap past it
                    running = 0.0
                running = rew[k] + self.gamma * running
                G[k] = running
            grids_all.append(np.asarray(t["grids"], dtype=np.float16))
            scalars_all.append(np.asarray(t["scalars"], dtype=np.float32))
            act_all.append(np.asarray(t["action"], dtype=np.float32))
            logp_all.append(np.asarray(t["logp"], dtype=np.float32))
            ret_all.append(G)
            ctrl_all.append(np.asarray(t["controlled"], dtype=bool))

        grids = np.concatenate(grids_all, axis=0)
        scalars = np.concatenate(scalars_all, axis=0)
        actions = np.concatenate(act_all, axis=0)
        old_logp = np.concatenate(logp_all, axis=0)
        returns = np.concatenate(ret_all, axis=0)
        controlled = np.concatenate(ctrl_all, axis=0)
        # critic-free advantage: whiten returns to a baseline (over controlled steps only).
        base = returns[controlled] if controlled.any() else returns
        adv = (returns - base.mean()) / (base.std() + 1e-8)
        return {
            "grids": grids, "scalars": scalars, "actions": actions,
            "old_logp": old_logp, "returns": returns.astype(np.float32),
            "adv": adv.astype(np.float32), "controlled": controlled,
        }

    def clear(self):
        self.traj = {}
        self.skipped = 0


# =========================================================================== PPO UPDATE
@dataclass
class PPOConfig:
    gamma: float = 0.99
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 1024
    lr: float = 1e-4
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    target_kl: float = 0.03            # stop the update early once the policy has moved this far
    max_agents: int = 128              # trajectories tracked per species per day (memory bound)
    # night detection (asleep-fraction hysteresis) + safety cap
    night_hi: float = 0.5              # >= this fraction asleep -> pause + train
    night_lo: float = 0.2              # <= this fraction asleep -> daytime, resume collecting
    min_cycle: int = 60                # min ticks of collection before a night trigger is allowed
    horizon: int = 400                 # force a train if a day runs longer than this (safety)


def ppo_update(policy, optimizer, batch, cfg, device="cpu"):
    """One PPO update (several epochs of clipped-surrogate minibatches). Critic-free: no value
    loss.  The policy loss and entropy bonus are averaged over CONTROLLED steps only (sleep-
    overridden actions drove no outcome, so they must not push the gradient)."""
    grids = torch.from_numpy(batch["grids"])            # f16, moved+cast per minibatch
    scalars = torch.from_numpy(batch["scalars"])
    actions = torch.from_numpy(batch["actions"])
    old_logp = torch.from_numpy(batch["old_logp"])
    adv = torch.from_numpy(batch["adv"])
    controlled = torch.from_numpy(batch["controlled"])
    n = grids.shape[0]

    logs = {"policy_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
    stop = False
    for _ in range(cfg.epochs):
        if stop:
            break
        perm = torch.randperm(n)
        for s in range(0, n, cfg.minibatch):
            mb = perm[s:s + cfg.minibatch]
            g = grids[mb].to(device).float()
            sc = scalars[mb].to(device).float()
            a = actions[mb].to(device).float()
            olp = old_logp[mb].to(device).float()
            ad = adv[mb].to(device).float()
            m = controlled[mb].to(device).float()
            denom = m.sum().clamp(min=1.0)

            new_logp, ent = policy.eval_actions(g, sc, a)
            ratio = torch.exp(new_logp - olp)
            surr1 = ratio * ad
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * ad
            pg = -torch.min(surr1, surr2)
            policy_loss = (pg * m).sum() / denom
            entropy = (ent * m).sum() / denom
            loss = policy_loss - cfg.entropy_coef * entropy

            with torch.no_grad():
                approx_kl = float(((olp - new_logp) * m).sum() / denom)
                clipfrac = float((((ratio - 1.0).abs() > cfg.clip).float() * m).sum() / denom)
            logs["policy_loss"].append(float(policy_loss.detach()))
            logs["entropy"].append(float(entropy.detach()))
            logs["approx_kl"].append(approx_kl)
            logs["clipfrac"].append(clipfrac)

            # PPO trust region: bail out of the update once the policy has moved far enough,
            # BEFORE applying the step that broke it (critic-free updates can be aggressive).
            if approx_kl > 1.5 * cfg.target_kl:
                stop = True
                break

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in logs.items()}


# =========================================================================== LIVE BRAIN
class LivePPOBrain(Brain):
    """A single learning ``Brain`` that drives BOTH species (no RuleBrain, no CompositeBrain).

    In ``training`` + ``collecting`` mode it samples stochastically and records the (obs, action,
    log-prob) of every agent that acted, keyed by ``birth_id``, for the driver to reward and file
    into the per-species ``RolloutBuffer`` after the step.  Otherwise it acts deterministically
    (head means / gate thresholds), exactly like the deployed ``PolicyBrain``.

    Sampling draws from torch's global RNG only -- it never touches the sim's numpy Generator, so
    the other systems' RNG stream (and their determinism) is untouched."""

    def __init__(self, policies, device="cpu"):
        self.policies = policies            # {species_id: PPOPolicy}
        self.device = device
        self.training = True
        self.collecting = True
        self.ent = None
        self.pending = {}                   # {sid: dict(...)}  -- overwritten each decide()

    def bind(self, entities):
        self.ent = entities

    def decide(self, obs_by_species, idx):
        act = np.zeros((idx.shape[0], ACT_DIM), dtype=np.float32)
        self.pending = {}
        if idx.shape[0] == 0:
            return act
        record = self.training and self.collecting
        for sid, policy in self.policies.items():
            obs = obs_by_species.get(sid)
            if obs is None or obs.grids.shape[0] == 0:
                continue
            # Perception REUSES its grid/scalar buffers each tick (obs.grids / obs.scalars are
            # views into persistent arrays), so anything we keep must be a private COPY. The f16
            # dtype conversion copies grids; scalars need an explicit copy. We store exactly what
            # the update re-reads (grids as f16) so old_logp matches new_logp on epoch 0.
            grids16 = np.array(obs.grids, dtype=np.float16)
            scalars32 = np.array(obs.scalars, dtype=np.float32)
            g = torch.from_numpy(grids16.astype(np.float32)).to(self.device)
            s = torch.from_numpy(scalars32).to(self.device)
            if record:
                actions_t, logp_t = policy.act(g, s)
                logp = logp_t.cpu().numpy().astype(np.float32)
            else:
                actions_t = policy.act_greedy(g, s)
                logp = None
            a = actions_t.cpu().numpy().astype(np.float32)
            pos = np.searchsorted(idx, obs.idx)
            act[pos] = a
            if record:
                self.pending[sid] = {
                    "slot": obs.idx.copy(),
                    "birth_id": self.ent.birth_id[obs.idx].copy(),
                    "grids": grids16,
                    "scalars": scalars32,
                    "action": a,
                    "logp": logp,
                }
        return act
