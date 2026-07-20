"""Behavioural-cloning helpers for the imitation-learning notebooks (no memory).

This is the toolkit the four notebooks in this folder call into (imported as ``IL``), so each
notebook stays a thin, readable driver:

    collect.ipynb      -> records the RuleBrain teacher's (perception -> action) decisions
                          across several *different worlds* into sheep.npz / fox.npz
    train_sheep.ipynb  -> clones the sheep teacher into a memoryless CNN+MLP policy
    train_fox.ipynb    -> same for the fox
    evaluate.ipynb     -> scores the clones and drops them into the real Simulation

The POLICY NETWORK and its (de)serialization live in the shared ``notebooks/common.py`` (also
used by ``live_learning/ppo_live.py``); this module imports them and adds the parts that ONLY
behavioural cloning needs -- dataset collection, the BC loss/metrics/training loop, and a
notebook-local eval brain.  The shared names are re-exported below, so a notebook needs only
``import imitation as IL`` and reaches everything through the one ``IL.`` namespace.

WHY NO MEMORY.  These clones are memoryless: each decision is a pure function of the *current*
observation, so there is no per-agent recurrent state to carry, no ``birth_id`` bookkeeping, and
every (obs, action) row is an independent training example.  That makes the dataset a plain
shuffled table and the model a feed-forward net.  The clones deploy through
``sim.policy_brain.PolicyBrain`` (a fuller CNN+MLP+LSTM recurrent brain and its RL trainer are
archived under ``backup/``).

THE DATA.  Perception is a stack of egocentric grids ``(C, K, K)`` (K = 57 in the default
world) plus a ``(10,)`` scalar vector, exactly what ``Brain.decide`` receives.  A 57x57x5
grid is ~16k floats per sheep-row, so a flat CSV would run to many GB and be slow to load;
we store the tensors in a single compressed ``.npz`` per species instead (float16 grids),
which loads in one shot and moves straight onto the GPU.

THE TARGET.  The RuleBrain emits a 6-D action per animal:
``[dx, dy, eat, drink, repro, speed]`` -- a (near) unit heading, three 0/1 gates, and a 0/1
speed throttle (see ``sim/brain.py``).  The clone regresses the heading (MSE) and classifies
the four binary channels (BCE), which is all behavioural cloning needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- reach the shared toolkit (notebooks/common.py) one level up, which also puts the repo root
#     on sys.path (via common.find_repo()) so ``import config`` / ``import sim`` work too. ---
_HERE = Path(__file__).resolve().parent
_NOTEBOOKS = _HERE.parent
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))

import common as C                                              # noqa: E402  (shared toolkit)
# re-export the shared surface so a notebook reaches everything through the single ``IL.`` alias
from common import (                                            # noqa: E402,F401
    REPO, DATA_DIR, build_policy, save_model, load_model, MODEL_PATHS,
)
from config import make_config, SHEEP, FOX, SPECIES_NAMES       # noqa: E402
from sim.brain import ACT_DIM, Brain, RuleBrain                 # noqa: E402
from sim.perception import SCALAR_DIM, SPECIES_N_CHANNELS       # noqa: E402
from sim.simulation import Simulation                           # noqa: E402

# action-column layout of the 6-D action (shared with sim/brain.py)
A_HEAD = slice(0, 2)      # dx, dy   (regressed)
A_GATES = slice(2, 5)     # eat, drink, repro  (classified)
A_SPEED = 5               # speed throttle 0/1 (classified)

SPECIES_IDS = (SHEEP, FOX)
DATA_PATHS = {SHEEP: DATA_DIR / "sheep.npz", FOX: DATA_DIR / "fox.npz"}


# ======================================================================= COLLECTION
class _ReservoirRecorder(Brain):
    """Wraps a RuleBrain teacher; keeps a uniform random sample of its decisions per species.

    A ``decide`` call still just forwards to the real RuleBrain (so the sim runs exactly as it
    would with the rule brain), stashing each alive animal's (grids, scalars, action) row as
    *pending*.  The trainer then calls ``commit`` right after the sim step, which keeps only
    the rows the animal was actually IN CONTROL of and offers them to a per-species reservoir.

    SLEEP FILTER.  The sleep system runs after the brain each tick and, for any animal that is
    asleep (or dashing to cover at dusk), OVERRIDES the emitted action and sets
    ``ent.action_overridden``.  Cloning those rows would teach the network the sleep system's
    behaviour, not the policy -- so ``commit`` drops every row where ``action_overridden`` is
    True.  It also drops rows whose slot died or was recycled during the step (checked via
    ``birth_id``), so a slot reused by a newborn mid-tick can't corrupt the filter.

    Reservoir sampling keeps a uniform sample of a fixed ``cap`` rows out of an arbitrarily
    long stream, so memory is bounded and the sample is spread evenly across the whole
    recording window rather than front-loaded onto the first few ticks.
    """

    def __init__(self, rule: RuleBrain, caps: dict, sub_rng: np.random.Generator):
        self.rule = rule
        self.caps = caps                       # {species_id: reservoir capacity}
        self.rng = sub_rng
        self.enabled = False                   # off during warm-up, on while recording
        self.buf = {}                          # species_id -> dict of preallocated arrays
        self.filled = {sid: 0 for sid in SPECIES_IDS}
        self.seen = {sid: 0 for sid in SPECIES_IDS}
        self.ent = None                        # bound by Simulation; used for the sleep filter
        self._pending = []                     # rows captured this tick, awaiting commit()

    def bind(self, entities):
        """Simulation hands us the entity store so ``commit`` can read ``action_overridden``
        and ``birth_id`` after the step (the decision still reads only the observation)."""
        self.ent = entities

    def _alloc(self, sid, grids, scalars):
        cap = self.caps[sid]
        C, K = grids.shape[1], grids.shape[2]
        self.buf[sid] = {
            "grids": np.zeros((cap, C, K, K), dtype=np.float16),
            "scalars": np.zeros((cap, scalars.shape[1]), dtype=np.float32),
            "actions": np.zeros((cap, ACT_DIM), dtype=np.float32),
        }

    def _offer(self, sid, grids, scalars, actions):
        cap = self.caps[sid]
        if cap <= 0:
            return
        if sid not in self.buf:
            self._alloc(sid, grids, scalars)
        b = self.buf[sid]
        for r in range(grids.shape[0]):
            n = self.seen[sid]
            if self.filled[sid] < cap:
                j = self.filled[sid]
                self.filled[sid] += 1
            else:
                j = int(self.rng.integers(0, n + 1))
                if j >= cap:
                    self.seen[sid] = n + 1
                    continue
            b["grids"][j] = grids[r]
            b["scalars"][j] = scalars[r]
            b["actions"][j] = actions[r]
            self.seen[sid] = n + 1

    def begin_tick(self):
        self._pending = []

    def decide(self, obs_by_species, idx):
        act = self.rule.decide(obs_by_species, idx)
        if self.enabled:
            for sid in SPECIES_IDS:
                obs = obs_by_species.get(sid)
                if obs is None or obs.grids.shape[0] == 0:
                    continue
                pos = np.searchsorted(idx, obs.idx)     # this species' rows in the global act
                slots = np.asarray(obs.idx)
                bid = self.ent.birth_id[slots].copy() if self.ent is not None else None
                self._pending.append((
                    sid, slots,
                    obs.grids.astype(np.float16, copy=True),
                    obs.scalars.astype(np.float32, copy=True),
                    act[pos].astype(np.float32, copy=True),
                    bid,
                ))
        return act

    def commit(self):
        """After the sim step: keep only rows the animal controlled (awake, not overridden,
        still the same living animal) and offer them to the reservoir."""
        ent = self.ent
        for sid, slots, grids, scalars, actions, bid in self._pending:
            if ent is None:
                keep = np.ones(slots.shape[0], dtype=bool)
            else:
                keep = (ent.alive[slots] & (ent.birth_id[slots] == bid)
                        & ~ent.action_overridden[slots])
            if keep.any():
                self._offer(sid, grids[keep], scalars[keep], actions[keep])
        self._pending = []

    def collected(self, sid):
        """Return the filled slice of one species' reservoir as (grids, scalars, actions)."""
        if sid not in self.buf:
            C, K = SPECIES_N_CHANNELS[sid], 1
            return (np.zeros((0, C, K, K), np.float16),
                    np.zeros((0, SCALAR_DIM), np.float32),
                    np.zeros((0, ACT_DIM), np.float32))
        n = self.filled[sid]
        b = self.buf[sid]
        return b["grids"][:n], b["scalars"][:n], b["actions"][:n]


def collect_from_world(world_seed, run_seed, sub_seed, warmup, record_ticks, caps,
                       food_thr=None, verbose=True):
    """Run the RuleBrain teacher on ONE world and return a uniform sample of its decisions.

    Steps the real ``Simulation`` (with the teacher brain injected) for ``warmup`` ticks with
    recording off -- so the founder seeding relaxes into natural predator-prey dynamics before
    we look -- then records for up to ``record_ticks`` ticks, stopping early once every
    species' reservoir is full or the world collapses.

    Returns ``{species_id: (grids, scalars, actions)}`` for this world.
    """
    cfg = make_config(world_seed=world_seed, seed=run_seed)
    thr = cfg.sim.food_eat_threshold if food_thr is None else food_thr
    rule = RuleBrain(np.random.default_rng(run_seed), thr)
    rec = _ReservoirRecorder(rule, caps, np.random.default_rng(sub_seed))
    sim = Simulation(cfg, brain=rec)

    for _ in range(warmup):
        sim.step()
        if sim.populations["sheep"] == 0 or sim.populations["fox"] == 0:
            break

    rec.enabled = True
    for t in range(record_ticks):
        rec.begin_tick()
        sim.step()                 # calls rec.decide -> stashes this tick's rows
        rec.commit()               # drops sleep-overridden / dead rows, offers the rest
        if all(rec.filled[sid] >= caps[sid] for sid in SPECIES_IDS):
            break
        if sim.populations["sheep"] == 0 and sim.populations["fox"] == 0:
            break

    out = {sid: rec.collected(sid) for sid in SPECIES_IDS}
    if verbose:
        got = {SPECIES_NAMES[sid]: out[sid][0].shape[0] for sid in SPECIES_IDS}
        print(f"  world {world_seed}: pops {sim.populations}  ->  rows {got}")
    return out


def collect_dataset(world_seeds, per_world_caps, warmup=300, record_ticks=400,
                    run_seed_base=1000, sub_seed_base=5000, verbose=True):
    """Collect + concatenate teacher decisions across several DIFFERENT worlds.

    ``per_world_caps`` is ``{species_id: rows}`` sampled per world, so every world contributes
    an equal share (better generalization than letting one big-population world dominate).
    Returns ``{species_id: dict(grids, scalars, actions, world)}`` where ``world`` tags each
    row with the index of the world it came from (used to hold a world out for validation).
    """
    parts = {sid: {"grids": [], "scalars": [], "actions": [], "world": []} for sid in SPECIES_IDS}
    for wi, ws in enumerate(world_seeds):
        got = collect_from_world(ws, run_seed_base + wi, sub_seed_base + wi,
                                 warmup, record_ticks, per_world_caps, verbose=verbose)
        for sid in SPECIES_IDS:
            g, s, a = got[sid]
            parts[sid]["grids"].append(g)
            parts[sid]["scalars"].append(s)
            parts[sid]["actions"].append(a)
            parts[sid]["world"].append(np.full(g.shape[0], wi, dtype=np.int16))

    data = {}
    for sid in SPECIES_IDS:
        data[sid] = {
            "grids": np.concatenate(parts[sid]["grids"], axis=0),
            "scalars": np.concatenate(parts[sid]["scalars"], axis=0),
            "actions": np.concatenate(parts[sid]["actions"], axis=0),
            "world": np.concatenate(parts[sid]["world"], axis=0),
        }
    return data


def save_dataset(sid, d, path=None):
    path = DATA_PATHS[sid] if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, grids=d["grids"], scalars=d["scalars"],
                        actions=d["actions"], world=d["world"])
    return path


def load_dataset(sid, path=None):
    path = DATA_PATHS[sid] if path is None else Path(path)
    z = np.load(path)
    return {k: z[k] for k in ("grids", "scalars", "actions", "world")}


# ======================================================================= TRAINING
def bc_loss(out, target, w_head=1.0, w_gate=1.0, w_speed=0.5,
            pos_weight_gates=None, pos_weight_speed=None):
    """Behavioural-cloning loss: MSE on the heading + BCE on the gate/speed classes.

    The teacher's gates are strongly imbalanced (a fox raises ``eat`` ~7% of ticks, ``repro``
    ~20%), so unweighted BCE lets the clone win the loss by almost always predicting "off" --
    which in the sim means foxes never breed and sheep under-eat, and the population collapses.
    ``pos_weight_*`` (n_neg/n_pos per channel, computed from the data in ``train_policy``)
    up-weights the positive class so the clone fires each gate at the teacher's true rate."""
    torch, nn, F = C._make_torch()
    mean, gate_logits, speed_logit = out
    head = ((mean - target[:, A_HEAD]) ** 2).sum(-1).mean()
    gate = F.binary_cross_entropy_with_logits(gate_logits, target[:, A_GATES],
                                              pos_weight=pos_weight_gates)
    speed = F.binary_cross_entropy_with_logits(speed_logit.squeeze(-1), target[:, A_SPEED],
                                               pos_weight=pos_weight_speed)
    loss = w_head * head + w_gate * gate + w_speed * speed
    return loss, {"head": float(head.detach()), "gate": float(gate.detach()),
                  "speed": float(speed.detach())}


def bc_metrics(out, target):
    """Interpretable clone-quality metrics: heading cosine + per-channel classification acc."""
    torch, nn, F = C._make_torch()
    mean, gate_logits, speed_logit = (t.detach() for t in out)
    tgt_h = target[:, A_HEAD]
    tnorm = tgt_h.norm(dim=-1)
    valid = tnorm > 1e-6
    cos = (mean * tgt_h).sum(-1) / (mean.norm(dim=-1) * tnorm + 1e-8)
    heading_cos = float(cos[valid].mean()) if bool(valid.any()) else float("nan")
    # The teacher's heading is a *random explore angle* whenever no target is urgent
    # (~unit vector, unlearnable direction), which dilutes the overall cosine. On the rows
    # where the clone commits to a direction (|pred| > 0.3) it is pursuing a perceived
    # target -- ``heading_cos_conf`` is the honest "did it point the right way" number.
    conf = valid & (mean.norm(dim=-1) > 0.3)
    heading_cos_conf = float(cos[conf].mean()) if bool(conf.any()) else float("nan")
    conf_frac = float(conf.float().mean())
    gate_pred = (gate_logits > 0).float()
    gate_acc = (gate_pred == target[:, A_GATES]).float().mean(0)      # per-channel
    speed_acc = float(((speed_logit.squeeze(-1) > 0).float() == target[:, A_SPEED]).float().mean())
    return {"heading_cos": heading_cos, "heading_cos_conf": heading_cos_conf,
            "conf_frac": conf_frac,
            "eat_acc": float(gate_acc[0]), "drink_acc": float(gate_acc[1]),
            "repro_acc": float(gate_acc[2]), "speed_acc": speed_acc}


def split_by_world(d, val_world):
    """Hold out every row from ``val_world`` for validation (generalization to an unseen map);
    the rest is training. Falls back to a random 90/10 split if ``val_world`` is None."""
    w = d["world"]
    if val_world is None:
        rng = np.random.default_rng(0)
        perm = rng.permutation(w.shape[0])
        n_val = max(1, int(0.1 * w.shape[0]))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
    else:
        val_idx = np.nonzero(w == val_world)[0]
        tr_idx = np.nonzero(w != val_world)[0]
    return tr_idx, val_idx


def train_policy(sid, d, device="cuda", epochs=25, batch_size=512, lr=1e-3,
                 val_world=None, seed=0, pool="softargmax", verbose=True):
    """GPU-batched behavioural cloning of one species' teacher.

    The whole dataset lives in CPU RAM as tensors (grids kept float16); each minibatch is
    sliced by a shuffled index and moved to the GPU (cast to float32) — no Python-side
    per-sample DataLoader overhead. Returns ``(model, history)``.
    """
    torch, nn, F = C._make_torch()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    grids = torch.from_numpy(d["grids"])                 # (N,C,K,K) float16, stays on CPU
    scalars = torch.from_numpy(d["scalars"])             # (N,10) float32
    actions = torch.from_numpy(d["actions"])             # (N,6) float32
    tr_idx, val_idx = split_by_world(d, val_world)
    tr_idx = torch.from_numpy(tr_idx.astype(np.int64))
    val_idx = torch.from_numpy(val_idx.astype(np.int64))

    model = C.build_policy(sid, hidden=128, pool=pool).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    # class-balance weights (n_neg/n_pos per channel) from the TRAIN split, capped so a very
    # rare gate can't dominate the loss; counters the majority-"off" bias (see bc_loss)
    tr_act = actions[tr_idx]
    gate_pos = tr_act[:, A_GATES].float().mean(0).clamp(1e-3, 1 - 1e-3)
    speed_pos = tr_act[:, A_SPEED].float().mean().clamp(1e-3, 1 - 1e-3)
    pw_gates = ((1 - gate_pos) / gate_pos).clamp(max=20.0).to(dev)
    pw_speed = ((1 - speed_pos) / speed_pos).clamp(max=20.0).to(dev)
    if verbose:
        print(f"  gate +rates {np.round(gate_pos.numpy(), 2)} "
              f"speed +rate {float(speed_pos):.2f} -> pos_weight "
              f"gates {np.round(pw_gates.cpu().numpy(), 1)} speed {float(pw_speed):.1f}")

    def run_batches(idx, train):
        model.train(train)
        tot = {"loss": 0.0, "head": 0.0, "gate": 0.0, "speed": 0.0, "n": 0}
        order = idx[torch.from_numpy(rng.permutation(idx.shape[0]))] if train else idx
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i in range(0, order.shape[0], batch_size):
                bi = order[i:i + batch_size]
                g = grids[bi].to(dev, non_blocking=True).float()
                s = scalars[bi].to(dev, non_blocking=True)
                a = actions[bi].to(dev, non_blocking=True)
                out = model(g, s)
                loss, info = bc_loss(out, a, pos_weight_gates=pw_gates, pos_weight_speed=pw_speed)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                bs = bi.shape[0]
                tot["loss"] += float(loss.detach()) * bs
                for k in ("head", "gate", "speed"):
                    tot[k] += info[k] * bs
                tot["n"] += bs
        n = max(tot["n"], 1)
        return {k: tot[k] / n for k in ("loss", "head", "gate", "speed")}

    @torch.no_grad()
    def val_metrics():
        model.eval()
        g = grids[val_idx].to(dev).float()
        s = scalars[val_idx].to(dev)
        a = actions[val_idx].to(dev)
        return bc_metrics(model(g, s), a)

    history = []
    for ep in range(1, epochs + 1):
        tr = run_batches(tr_idx, train=True)
        vm = val_metrics()
        history.append({"epoch": ep, **{f"tr_{k}": v for k, v in tr.items()}, **vm})
        if verbose:
            print(f"  [{SPECIES_NAMES[sid]} {ep:>2}/{epochs}] "
                  f"loss={tr['loss']:.4f} (head={tr['head']:.4f} gate={tr['gate']:.4f} "
                  f"speed={tr['speed']:.4f}) | val cos={vm['heading_cos']:.3f} "
                  f"eat={vm['eat_acc']:.2f} drink={vm['drink_acc']:.2f} "
                  f"repro={vm['repro_acc']:.2f} speed={vm['speed_acc']:.2f}")
    return model, history


# ======================================================================= DEPLOYMENT
class LearnedPolicyBrain(Brain):
    """Drops the cloned per-species policies into the real ``Brain.decide`` contract, so the
    learned behaviour can be run inside the full ``Simulation`` (headless) exactly where the
    RuleBrain would go. Memoryless + deterministic (acts by the head means / gate thresholds),
    so it draws no randomness and a run stays reproducible."""

    def __init__(self, models: dict, device="cpu", explore_seed=None, explore_thr=0.3):
        torch, _, _ = C._make_torch()
        self.torch = torch
        self.dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.models = {sid: m.to(self.dev).eval() for sid, m in models.items()}
        # The teacher explores by picking a fresh RANDOM heading whenever nothing is urgent --
        # a stochastic signal a memoryless deterministic clone cannot reproduce (it emits the
        # same heading for the same view every tick, so it walks in straight lines instead of
        # searching). With ``explore_seed`` set, we supply that missing wander: on ticks where
        # the clone does NOT commit to a perceived target (|heading| < ``explore_thr``) we
        # substitute a random full-speed heading. Off (None) => pure greedy clone.
        self.explore = None if explore_seed is None else np.random.default_rng(explore_seed)
        self.explore_thr = explore_thr

    def decide(self, obs_by_species, idx):
        torch = self.torch
        act = np.zeros((idx.shape[0], ACT_DIM), dtype=np.float32)
        if idx.shape[0] == 0:
            return act
        for sid in SPECIES_IDS:
            obs = obs_by_species.get(sid)
            model = self.models.get(sid)
            if obs is None or model is None or obs.grids.shape[0] == 0:
                continue
            g = torch.from_numpy(np.ascontiguousarray(obs.grids)).to(self.dev).float()
            s = torch.from_numpy(np.ascontiguousarray(obs.scalars)).to(self.dev).float()
            with torch.no_grad():
                mean, gate_logits, speed_logit = model(g, s)
            n = obs.grids.shape[0]
            a = np.zeros((n, ACT_DIM), dtype=np.float32)
            a[:, A_HEAD] = mean.cpu().numpy()
            a[:, A_GATES] = (gate_logits > 0).float().cpu().numpy()
            a[:, A_SPEED] = (speed_logit.squeeze(-1) > 0).float().cpu().numpy()
            if self.explore is not None:
                mag = np.linalg.norm(a[:, A_HEAD], axis=1)
                unc = mag < self.explore_thr
                if unc.any():
                    ang = self.explore.uniform(0.0, 2 * np.pi, size=int(unc.sum())).astype(np.float32)
                    a[unc, 0] = np.cos(ang)
                    a[unc, 1] = np.sin(ang)
                    a[unc, A_SPEED] = 1.0          # travel while searching
            pos = np.searchsorted(idx, obs.idx)
            act[pos] = a
        return act


def run_headless(brain_factory, world_seed, run_seed, ticks, log_every=200, verbose=True):
    """Run a full Simulation with a freshly built brain and return the population history.

    ``brain_factory(cfg) -> Brain`` builds the brain (so RuleBrain and the learned brain are
    driven through identical code). Returns ``{"sheep": [...], "fox": [...]}`` per-tick counts.
    """
    cfg = make_config(world_seed=world_seed, seed=run_seed)
    brain = brain_factory(cfg)
    sim = Simulation(cfg, brain=brain)
    hist = {"sheep": [], "fox": []}
    for t in range(ticks):
        sim.step()
        hist["sheep"].append(sim.populations["sheep"])
        hist["fox"].append(sim.populations["fox"])
        if verbose and (t + 1) % log_every == 0:
            print(f"  t={t + 1:>5}  sheep={sim.populations['sheep']:>4}  fox={sim.populations['fox']:>4}")
    return hist
