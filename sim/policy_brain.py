"""Memoryless deployment brain: runs the imitation-learning policies (§7.1, §13 of v1.md).

A drop-in ``Brain`` (``decide(obs_by_species, idx) -> act``) that runs the *memoryless*
behavioural-cloning policies produced by ``notebooks/imitation_learning/`` (``sheep.pt`` /
``fox.pt``). Each decision is a pure function of the *current* observation -- a memoryless
feed-forward policy, so there is no per-agent recurrent state and no ``birth_id`` bookkeeping.
(A fuller recurrent CNN+MLP+**LSTM** actor-critic and its RL trainer were archived under
``backup/`` when deployment moved to these imitation-learning policies.)

Checkpoints are SELF-CONTAINED TorchScript archives (saved by the notebook's
``common.save_model`` via ``torch.jit.script``): the file carries the network code + weights,
so ``torch.jit.load`` reconstructs the policy without any architecture class here. The only
contract is the interface ``model(grids, scalars) -> (head_mean, gate_logits, speed_logit)``
-- the architecture itself is defined once, in the training notebook, and can change freely
without touching ``sim/``. Torch is imported lazily by the callers, so the rule path never
needs it.

Each policy is per-species and independent: build one ``PolicyBrain`` per checkpoint (holding a
single species' network) and route species to it via ``CompositeBrain`` (see ``sim/brain.py``),
so e.g. sheep can run a learned policy while foxes stay on the rule brain.

DETERMINISM. Deployment acts by the head means / gate thresholds (no sampling), so it draws no
randomness at all: a run is reproducible from (world seed, run seed, weights) just like the rule
brain, and it does not perturb the numpy run-RNG stream the other systems consume.
"""
from __future__ import annotations

import numpy as np
import torch

from sim.brain import ACT_DIM, Brain

# action columns (shared with sim/brain.py): heading (0:2), gates eat/drink/repro (2:5), speed (5)
_A_HEAD = slice(0, 2)
_A_GATES = slice(2, 5)
_A_SPEED = 5


class PolicyBrain(Brain):
    """Runs one or more memoryless per-species policies behind the ``Brain`` contract.

    ``models`` maps species id -> any torch module (typically a loaded ``ScriptModule``)
    implementing ``(grids, scalars) -> (head_mean, gate_logits, speed_logit)``. Only species
    present are decided; rows of any absent species are left zero (compose with a ``RuleBrain``
    via ``CompositeBrain`` to fill them). Deterministic + memoryless, so no RNG is drawn."""

    def __init__(self, models: dict, device: str = "cpu"):
        self.device = torch.device(device)
        self.models = {sid: m.to(self.device).eval() for sid, m in models.items()}

    def decide(self, obs_by_species, idx) -> np.ndarray:
        n_global = idx.shape[0]
        act = np.zeros((n_global, ACT_DIM), dtype=np.float32)
        if n_global == 0:
            return act
        for sid, model in self.models.items():
            obs = obs_by_species.get(sid)
            if obs is None or obs.grids.shape[0] == 0:
                continue
            grids = torch.from_numpy(np.ascontiguousarray(obs.grids)).to(self.device).float()
            scalars = torch.from_numpy(np.ascontiguousarray(obs.scalars)).to(self.device).float()
            with torch.no_grad():
                mean, gate_logits, speed_logit = model(grids, scalars)
            n = obs.grids.shape[0]
            a = np.zeros((n, ACT_DIM), dtype=np.float32)
            a[:, _A_HEAD] = mean.cpu().numpy()
            a[:, _A_GATES] = (gate_logits > 0.0).float().cpu().numpy()   # sigmoid>0.5 <=> logit>0
            a[:, _A_SPEED] = (speed_logit.squeeze(-1) > 0.0).float().cpu().numpy()
            pos = np.searchsorted(idx, obs.idx)          # rows of this species in the global act
            act[pos] = a
        return act


def policy_brain_from_path(path: str, species_id: int, device: str = "cpu") -> PolicyBrain:
    """Build a single-species ``PolicyBrain`` from a self-contained TorchScript checkpoint
    (as saved by ``notebooks/common.save_model``). The archive carries its
    own network code, so no architecture class is needed -- ``torch.jit.load`` raises
    ``RuntimeError`` on a non-TorchScript file (e.g. a legacy ``state_dict`` blob)."""
    module = torch.jit.load(str(path), map_location=torch.device(device))
    return PolicyBrain({species_id: module}, device=device)
