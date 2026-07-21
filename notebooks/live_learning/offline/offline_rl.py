"""Offline reinforcement-learning engine for the ecosystem sim (critic-free PPO).

This is the OFFLINE sibling of ``../ppo_live.py``. The live engine collects on-policy
experience *while the sim runs* and pauses each night to update; here we split those two halves
in time. We collect a fixed dataset of transitions ONCE (running the sim with a behaviour
policy), write it to disk, and then train the policy from that frozen dataset -- as many passes
as we like -- without ever touching the sim again. That is the whole point of "offline" RL:
learning from pre-collected data.

WHY NOT ``imitation_learning/fox.npz``?
--------------------------------------
The behavioural-cloning dataset (``notebooks/imitation_learning/fox.npz``) is a *reservoir
sample* of independent ``(observation -> teacher action)`` rows (see ``imitation.py``). It has
no rewards, no next-states, no ``birth_id`` and no tick -- so its rows cannot be grouped into
per-agent trajectories, and a discounted return-to-go cannot be accumulated along anything.
Offline RL needs transitions ``(s, a, r, done)`` in temporal order per agent. So this module
RE-COLLECTS a trajectory-structured dataset (keeping every tick's rows, in order, tagged with
episode + birth_id + tick), reusing the *identical* reward function the live trainer uses
(``ppo_live.compute_rewards``). Everything downstream -- returns-to-go, whitening, the clipped
PPO update -- is then the faithful offline analog of the live loop's per-night update.

DESIGN (inherited from ``ppo_live``)
------------------------------------
* **Critic-free PPO.** No value network. Each step's advantage is its discounted return-to-go
  along that agent's own trajectory, whitened to a per-batch baseline, optimised under the
  clipped PPO surrogate. The deployed brain stays "actor only, no critic".
* **Behaviour policy = the warm-started clone.** Collection runs the imitation clone
  (``imitation_learning/fox.pt``) made stochastic (the ``PPOPolicy`` heading log-std), sampling
  actions and logging their log-probs. Those logged log-probs are the PPO ``old_logp``; because
  offline training rebuilds the SAME warm-started policy, round 0's importance ratio is ~1, so
  the first update starts exactly on-policy and then improves under the KL trust region.
* **No sim writes.** Deaths / rewards are inferred by snapshot-and-diff exactly as in
  ``ppo_live`` (we never modify ``sim/``).

DETERMINISM
-----------
Collection samples from torch's global RNG (seed with ``torch.manual_seed``), independent of the
sim's numpy ``Generator`` -- so the sim systems keep their own RNG stream and the deployed,
greedy policy stays reproducible, same as the live engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- repo wiring: reach ../ppo_live.py (the live engine we reuse) and, through it,
#     notebooks/common.py + the repo root (config / sim). ---
_HERE = Path(__file__).resolve().parent          # notebooks/live_learning/offline
_LIVE = _HERE.parent                             # notebooks/live_learning  (ppo_live.py)
_NOTEBOOKS = _LIVE.parent                         # notebooks                (common.py)
for _p in (_LIVE, _NOTEBOOKS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ppo_live as P                              # noqa: E402  the live PPO engine (reused wholesale)
from darwinism.config import make_config, SHEEP, FOX, SPECIES_NAMES  # noqa: E402
from darwinism.sim.simulation import Simulation             # noqa: E402

import torch                                       # noqa: E402

# transition columns written to / read from the offline dataset. ``grids`` is the memory hog and
# is kept float16 (as perception hands it to us); everything else is small.
_ARRAY_KEYS = ("grids", "scalars", "actions", "old_logp",
               "reward", "done", "controlled", "episode", "birth_id", "tick")


# =========================================================================== COLLECTION
def collect_offline_dataset(world_seeds, behavior_policy, opponent, rcfg, *,
                            warmup=100, record_ticks=600, max_transitions=80_000,
                            train_species=FOX, other_species=SHEEP,
                            run_seed=7, vary_run_seed=True, device="cpu", verbose=True):
    """Run the sim across several worlds and log every ``train_species`` transition to memory.

    Each world in ``world_seeds`` is one EPISODE: the ``Simulation`` is rebuilt from that world
    seed, warmed up (recording off) so the founder seeding relaxes, then stepped with the fox
    driven by ``behavior_policy`` sampling stochastically. After every step we reward each fox
    that acted (via ``ppo_live.compute_rewards`` -- the exact live reward) and append its row.
    Rows are appended in tick order and tagged with (episode, birth_id, tick), so the loader can
    regroup them into per-agent trajectories. Collection stops once ``max_transitions`` rows are
    gathered (the last trajectory is simply truncated -- its final ``done`` stays False).

    ``behavior_policy`` is a ``ppo_live.PPOPolicy`` (typically warm-started from the clone).
    ``opponent`` drives ``other_species`` (a frozen ``PolicyBrain``, or ``None`` -> RuleBrain).

    Returns ``(data, meta)``: ``data`` is ``{key: ndarray}`` over ``_ARRAY_KEYS``; ``meta`` holds
    collection stats + a per-tick population log for plotting.
    """
    brain = P.LivePPOBrain({train_species: behavior_policy}, device=device)
    brain.training = True                       # sample stochastically + record ``pending``

    store = {k: [] for k in _ARRAY_KEYS}
    pop_log = []                                # (episode, tick, n_sheep, n_fox)
    n_deaths = 0
    total = 0
    nm = SPECIES_NAMES[train_species]

    for epi, ws in enumerate(world_seeds):
        rs = (run_seed + epi) if vary_run_seed else run_seed
        cfg = make_config(world_seed=ws, seed=rs)
        sim = Simulation(cfg, brain={train_species: brain, other_species: opponent})
        ent = sim.entities

        # ---- warm-up: recording OFF, act greedily so the founders settle into dynamics ----
        brain.collecting = False
        for _ in range(warmup):
            sim.step()
            if ent.count_species(train_species) == 0 or ent.n_alive == 0:
                break

        # ---- record: stochastic actions + snapshot/diff rewards, filed every tick ----
        brain.collecting = True
        got0 = total
        for _ in range(record_ticks):
            snap = P.snapshot(ent)               # BEFORE the step (see ppo_live.snapshot)
            stats = sim.step()                   # brain.decide fills brain.pending
            pop_log.append((epi, sim.tick, stats["n_sheep"], stats["n_fox"]))

            for sid, pend in brain.pending.items():   # holds only the train species
                r, done, controlled = P.compute_rewards(rcfg, snap, ent, pend)
                n = pend["slot"].shape[0]
                store["grids"].append(pend["grids"])                       # (n,C,K,K) f16
                store["scalars"].append(pend["scalars"])                   # (n,10) f32
                store["actions"].append(pend["action"])                    # (n,6) f32
                store["old_logp"].append(pend["logp"].astype(np.float32))  # (n,) behaviour logp
                store["reward"].append(r.astype(np.float32))
                store["done"].append(done.astype(bool))
                store["controlled"].append(controlled.astype(bool))
                store["episode"].append(np.full(n, epi, dtype=np.int32))
                store["birth_id"].append(pend["birth_id"].astype(np.int64))
                store["tick"].append(np.full(n, sim.tick, dtype=np.int32))
                n_deaths += int(done.sum())
                total += n

            if total >= max_transitions:
                break
            if ent.count_species(train_species) == 0 or ent.n_alive == 0:
                break

        if verbose:
            print(f"  world {ws:>5} (epi {epi:>2}): pops {sim.populations} "
                  f"-> +{total - got0:>6} rows (total {total})")
        if total >= max_transitions:
            if verbose:
                print(f"  reached max_transitions={max_transitions} -- stopping collection")
            break

    if total == 0:
        raise RuntimeError("collected zero transitions -- check warmup / population survival")

    data = {k: np.concatenate(store[k], axis=0) for k in _ARRAY_KEYS}
    n_traj = np.unique(np.stack([data["episode"], data["birth_id"]], axis=1), axis=0).shape[0]
    meta = {
        "species": SPECIES_NAMES[train_species], "n_transitions": int(total),
        "n_trajectories": int(n_traj), "n_deaths": int(n_deaths),
        "n_episodes": int(data["episode"].max()) + 1, "pop_log": pop_log,
        "reward_mean": float(data["reward"].mean()), "controlled_frac": float(data["controlled"].mean()),
    }
    if verbose:
        print(f"\ncollected {total} transitions across {n_traj} trajectories "
              f"({n_deaths} deaths, {meta['controlled_frac']*100:.1f}% controlled), "
              f"mean per-tick reward {meta['reward_mean']:+.3f}")
    return data, meta


def save_offline_dataset(path, data):
    """Write the offline transition dataset to a single compressed ``.npz`` (grids stay f16)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: data[k] for k in _ARRAY_KEYS})
    return path


def load_offline_dataset(path):
    """Reload a dataset written by ``save_offline_dataset`` as ``{key: ndarray}``."""
    z = np.load(path)
    return {k: z[k] for k in _ARRAY_KEYS}


# =========================================================================== RETURNS / BATCH
def compute_returns(reward, done, episode, birth_id, tick, gamma=0.99):
    """Discounted return-to-go for every row, accumulated per agent trajectory.

    Rows are grouped by ``(episode, birth_id)`` (one agent's life within one world) and ordered
    by ``tick``; within each group the return-to-go is the SAME backward recurrence the live
    buffer uses (``ppo_live.RolloutBuffer.build_batch``): a death (``done``) is terminal, so the
    discounted sum does not bootstrap across it. Returns ``(G, n_trajectories)`` with ``G``
    aligned to the input row order."""
    n = reward.shape[0]
    G = np.zeros(n, dtype=np.float32)
    # a stable trajectory key per row (episode is small, birth_id can be large)
    key = episode.astype(np.int64) * (int(birth_id.max()) + 1) + birth_id.astype(np.int64)
    order = np.lexsort((tick, key))             # sort by key, then tick within key
    ukeys, starts = np.unique(key[order], return_index=True)
    bounds = np.append(starts, n)
    for gi in range(len(ukeys)):
        idx = order[bounds[gi]:bounds[gi + 1]]  # this trajectory's rows, in tick order
        rew = reward[idx]; dn = done[idx]
        running = 0.0
        g = np.empty(idx.shape[0], dtype=np.float32)
        for k in range(idx.shape[0] - 1, -1, -1):
            if dn[k]:
                running = 0.0                    # death terminates the trajectory
            running = float(rew[k]) + gamma * running
            g[k] = running
        G[idx] = g
    return G, len(ukeys)


def build_offline_batch(data, gamma=0.99):
    """Turn the flat transition dataset into a training batch with critic-free advantages.

    Computes per-trajectory discounted returns-to-go, then whitens them to a baseline over the
    CONTROLLED rows only (sleep-overridden actions drove no outcome). The returned dict has the
    exact keys ``ppo_live.ppo_update`` consumes, so the offline update reuses it unchanged."""
    returns, n_traj = compute_returns(
        data["reward"], data["done"], data["episode"], data["birth_id"],
        data["tick"], gamma=gamma)
    controlled = data["controlled"].astype(bool)
    base = returns[controlled] if controlled.any() else returns
    adv = (returns - base.mean()) / (base.std() + 1e-8)
    return {
        "grids": data["grids"], "scalars": data["scalars"], "actions": data["actions"],
        "old_logp": data["old_logp"].astype(np.float32),
        "returns": returns.astype(np.float32), "adv": adv.astype(np.float32),
        "controlled": controlled, "n_trajectories": n_traj,
    }


# =========================================================================== OFFLINE PPO
@torch.no_grad()
def refresh_old_logp(policy, batch, device="cpu", minibatch=4096):
    """Recompute ``batch["old_logp"]`` as the CURRENT policy's log-prob of the stored actions.

    Between offline rounds this re-centres the PPO trust region on the current policy (so the
    KL early-stop measures each round's own movement). Round 0 keeps the behaviour log-probs
    from the dataset, so the first update starts on-policy (ratio ~ 1)."""
    grids = torch.from_numpy(batch["grids"])
    scalars = torch.from_numpy(batch["scalars"])
    actions = torch.from_numpy(batch["actions"])
    n = grids.shape[0]
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, minibatch):
        sl = slice(s, s + minibatch)
        g = grids[sl].to(device).float()
        sc = scalars[sl].to(device).float()
        a = actions[sl].to(device).float()
        lp, _ = policy.eval_actions(g, sc, a)
        out[sl] = lp.detach().cpu().numpy()
    batch["old_logp"] = out
    return batch


def offline_ppo_train(policy, optimizer, batch, cfg, *, n_rounds=6, refresh=True,
                      device="cpu", verbose=True):
    """Improve ``policy`` from the frozen ``batch`` over several offline PPO rounds.

    Each round is one ``ppo_live.ppo_update`` (its own ``cfg.epochs`` clipped-surrogate passes
    with KL early-stop). The returns/advantages are fixed (they come from the logged rewards --
    critic-free, so they do not depend on the current policy); only the trust-region centre
    moves. With ``refresh=True`` every round after the first re-centres ``old_logp`` on the
    current policy so successive rounds keep improving under KL control rather than immediately
    tripping the early-stop against the stale behaviour policy.

    Returns a per-round metrics history (list of dicts)."""
    history = []
    meanR = float(batch["returns"].mean())
    for rd in range(n_rounds):
        if refresh and rd > 0:
            refresh_old_logp(policy, batch, device=device)
        m = P.ppo_update(policy, optimizer, batch, cfg, device=device)
        rec = {"round": rd, "meanR": meanR, **m}
        history.append(rec)
        if verbose:
            print(f"  round {rd:>2}/{n_rounds} | meanR={meanR:+.3f} "
                  f"ploss={m['policy_loss']:+.4f} ent={m['entropy']:.3f} "
                  f"kl={m['approx_kl']:.4f} clip={m['clipfrac']:.3f}")
    return history


# re-export the live engine's building blocks so a notebook reaches everything through ``O.``
build_ppo_policy = P.build_ppo_policy
export_policy = P.export_policy
PPOConfig = P.PPOConfig
RewardConfig = P.RewardConfig
