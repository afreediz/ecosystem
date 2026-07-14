"""Memoryless deployment brain: runs the imitation-learning policies (§7.1, §13 of v1.md).

A drop-in ``Brain`` (``decide(obs_by_species, idx) -> act``) that runs the *memoryless*
behavioural-cloning policies produced by ``notebooks/imitation_learning/`` (``sheep.pt`` /
``fox.pt``). Each decision is a pure function of the *current* observation -- a memoryless
feed-forward policy, so there is no per-agent recurrent state and no ``birth_id`` bookkeeping.
(A fuller recurrent CNN+MLP+**LSTM** actor-critic and its RL trainer were archived under
``backup/`` when deployment moved to these imitation-learning policies.)

The network here mirrors the notebook's ``SpeciesPolicy`` front-end exactly (same conv stack +
adaptive pool + trunk + heads and the same submodule names), so a checkpoint saved by the
notebook loads straight into it by ``state_dict``. Torch is imported lazily by the callers, so
the rule path never needs it.

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
import torch.nn as nn
import torch.nn.functional as F

from sim.brain import ACT_DIM, Brain
from sim.perception import SCALAR_DIM, SPECIES_N_CHANNELS

# action columns (shared with sim/brain.py): heading (0:2), gates eat/drink/repro (2:5), speed (5)
_A_HEAD = slice(0, 2)
_A_GATES = slice(2, 5)
_A_SPEED = 5


class SpeciesPolicy(nn.Module):
    """Memoryless behavioural-cloning policy: CNN(grids) + MLP(scalars) -> action heads.

    A CNN over the egocentric grids + MLP over the scalar vector feed a feed-forward trunk (the
    adaptive pool lets it accept any window ``K``); there is no LSTM and no critic. Heads: a 2-D
    heading mean (regressed), 3 gate logits
    + 1 speed logit (classified). Layer sizes/names match the notebook so checkpoints load."""

    def __init__(self, n_channels: int, hidden: int = 128, cnn_feat: int = 128,
                 scalar_feat: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.cnn_fc = nn.Linear(32 * 4 * 4, cnn_feat)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(SCALAR_DIM, scalar_feat), nn.ReLU(inplace=True),
            nn.Linear(scalar_feat, scalar_feat), nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(cnn_feat + scalar_feat, hidden), nn.ReLU(inplace=True),
        )
        self.head_mean = nn.Linear(hidden, 2)
        self.head_gates = nn.Linear(hidden, 3)
        self.head_speed = nn.Linear(hidden, 1)

    def forward(self, grids, scalars):
        c = self.conv(grids).flatten(1)
        c = F.relu(self.cnn_fc(c))
        s = self.scalar_mlp(scalars)
        z = self.trunk(torch.cat([c, s], dim=1))
        return self.head_mean(z), self.head_gates(z), self.head_speed(z)


class PolicyBrain(Brain):
    """Runs one or more memoryless per-species policies behind the ``Brain`` contract.

    ``models`` maps species id -> ``SpeciesPolicy``. Only species present are decided; rows of
    any absent species are left zero (compose with a ``RuleBrain`` via ``CompositeBrain`` to fill
    them). Deterministic + memoryless, so no RNG is drawn."""

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


def policy_brain_from_blob(blob: dict, species_id: int, device: str = "cpu") -> PolicyBrain:
    """Build a single-species ``PolicyBrain`` from an already-loaded imitation-learning blob
    (as saved by ``notebooks/imitation_learning/common.save_model``: keys ``n_channels`` /
    ``state_dict``)."""
    n_channels = int(blob.get("n_channels", SPECIES_N_CHANNELS[species_id]))
    model = SpeciesPolicy(n_channels)
    model.load_state_dict(blob["state_dict"])
    return PolicyBrain({species_id: model}, device=device)
