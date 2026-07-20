"""Shared toolkit for the learning notebooks: the policy network + its persistence + repo wiring.

This holds ONLY what more than one notebook family needs, so it stays a small, stable core that
both ``imitation_learning/`` and ``live_learning/`` depend on:

  * ``build_policy`` / ``SpeciesPolicy`` -- the memoryless CNN+MLP+soft-argmax policy network.
    The imitation-learning notebooks clone the ``RuleBrain`` into it; ``live_learning/ppo_live.py``
    wraps the *same* module for critic-free PPO. This is the ONLY definition of the architecture.
  * ``save_model`` / ``load_model`` -- self-contained TorchScript (de)serialization, so a
    checkpoint carries its own network code and ``sim.policy_brain`` loads it without any class.
  * ``MODEL_PATHS`` / ``DATA_DIR`` -- default checkpoint locations (kept under
    ``imitation_learning/`` beside the datasets that produced them).
  * repo wiring (``find_repo`` / ``REPO``) -- puts the repo root on ``sys.path`` so ``import
    config`` / ``import sim`` work inside the notebooks.

The behavioural-cloning-only helpers (dataset collection, the BC loss/metrics/training loop, and
the notebook eval brain) live in ``imitation_learning/imitation.py``, which imports this module.
Torch is imported lazily (``_make_torch``), so a caller that only needs the repo wiring never
pulls it in.
"""
from __future__ import annotations

import sys
from pathlib import Path


# --------------------------------------------------------------------------- repo wiring
def find_repo() -> Path:
    """Locate the ecosystem repo root (holds ``config.py`` + ``sim/``) from any cwd, and put
    it on ``sys.path`` so ``import config`` / ``import sim`` work inside the notebooks."""
    here = Path.cwd()
    for c in [here, *here.parents]:
        if (c / "config.py").exists() and (c / "sim").is_dir():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    raise RuntimeError("could not find the ecosystem repo root from " + str(here))


REPO = find_repo()
# this shared module lives at notebooks/, but the default datasets + checkpoints stay under
# imitation_learning/ (beside the collection code that writes them)
DATA_DIR = Path(__file__).resolve().parent / "imitation_learning"

from config import SHEEP, FOX                                   # noqa: E402
from sim.perception import SCALAR_DIM, SPECIES_N_CHANNELS       # noqa: E402

MODEL_PATHS = {SHEEP: DATA_DIR / "sheep.pt", FOX: DATA_DIR / "fox.pt"}


# ======================================================================= MODEL
def _make_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_policy(sid, hidden=128, cnn_feat=128, scalar_feat=32, pool="softargmax"):
    """Construct the memoryless per-species policy network (see ``SpeciesPolicy``).

    ``pool`` selects how the conv feature map is reduced to a vector:
      * ``"softargmax"`` (default) -- a spatial soft-argmax that reads out, per feature map,
        the EXPECTED (x, y) location of its activation (in [-1,1]) plus a presence score. This
        is the differentiable analog of "argmax over cells -> coordinates" -- the very thing
        the RuleBrain computes when it decodes the nearest/best cell -- and it pairs with the
        radial-distance positional channel perception now provides.
      * ``"avg"`` -- the legacy ``AdaptiveAvgPool2d((4,4))`` + flatten (kept for A/B).
    Both are window-size (K) agnostic.
    """
    torch, nn, F = _make_torch()

    class SpeciesPolicy(nn.Module):
        """Memoryless behavioural-cloning policy: CNN(grids) + MLP(scalars) -> action heads.

        A CNN over the egocentric grids is reduced to a feature vector (see ``pool``), concat
        with an MLP over the scalars, then a plain feed-forward trunk -> action heads; no LSTM
        and no critic -- this is supervised imitation, not RL.  This is the ONLY definition of
        the architecture: ``save_model`` exports it as a self-contained TorchScript archive, so
        deployment (``sim.policy_brain``) never rebuilds the class.
        Heads: a 2-D heading mean (regressed), 3 gate logits + 1 speed logit (classified).

        SOFT-ARGMAX HEAD.  The conv stack ends WITHOUT a pool; ``_soft_argmax`` takes a spatial
        softmax over each of the 32 feature maps and returns its expected (x, y) coordinate
        plus the map's peak activation as a presence score -> a 3*32 vector. A learnable
        inverse-temperature (softplus-positive) sharpens the softmax toward a hard argmax as
        training proceeds. Average pooling smears a sparse target and loses its position, which
        is why it is the wrong reduction for a "nearest target" policy; soft-argmax preserves it.
        """

        def __init__(self, n_channels, pool):
            super().__init__()
            # bool drives the reduction branch; TorchScript infers its type from this
            # assignment (a class-level annotation would be stringified by ``from __future__
            # import annotations`` at module top and break jit.script).
            self.use_avg = (pool == "avg")
            self.conv = nn.Sequential(
                nn.Conv2d(n_channels, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            )
            self._feat_ch = 32
            self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))       # used only when pool == "avg"
            # learnable inverse-temperature for the soft-argmax (softplus keeps it > 0; the
            # init value gives softplus(x) ~ 1.0, a neutral starting sharpness)
            self.inv_temp_raw = nn.Parameter(torch.tensor([0.5413]))
            cnn_in = self._feat_ch * 4 * 4 if self.use_avg else self._feat_ch * 3
            self.cnn_fc = nn.Linear(cnn_in, cnn_feat)
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

        def _soft_argmax(self, f):
            # f: (N, C, H, W) -> (N, 3C): expected x, expected y (in [-1,1]) + peak activation.
            # Normalized coordinates make this window-size (K) agnostic, just like the old pool.
            N = f.size(0); C = f.size(1); H = f.size(2); W = f.size(3)
            flat = f.reshape(N, C, H * W)
            conf = torch.max(flat, dim=2)[0]                          # (N, C) presence proxy
            beta = F.softplus(self.inv_temp_raw)                      # > 0, sharpens the softmax
            p = torch.softmax(flat * beta, dim=2)                     # (N, C, H*W)
            xs = torch.linspace(-1.0, 1.0, W, device=f.device, dtype=f.dtype)
            ys = torch.linspace(-1.0, 1.0, H, device=f.device, dtype=f.dtype)
            gx = xs.reshape(1, W).expand(H, W).reshape(1, 1, H * W)   # x varies fastest (row-major)
            gy = ys.reshape(H, 1).expand(H, W).reshape(1, 1, H * W)
            ex = (p * gx).sum(dim=2)                                  # (N, C)
            ey = (p * gy).sum(dim=2)                                  # (N, C)
            return torch.cat([ex, ey, conf], dim=1)                   # (N, 3C)

        def forward(self, grids, scalars):
            f = self.conv(grids)
            if self.use_avg:
                c = F.relu(self.cnn_fc(self.avg_pool(f).flatten(1)))
            else:
                c = F.relu(self.cnn_fc(self._soft_argmax(f)))
            s = self.scalar_mlp(scalars)
            z = self.trunk(torch.cat([c, s], dim=1))
            return self.head_mean(z), self.head_gates(z), self.head_speed(z)

    return SpeciesPolicy(SPECIES_N_CHANNELS[sid], pool)


def save_model(sid, model, path=None, meta=None):
    """Save one species' policy as a SELF-CONTAINED TorchScript archive (code + weights).

    ``torch.jit.script`` (not trace) is used so variable batch size and window ``K`` keep
    working. Deployment (``sim.policy_brain``) just ``torch.jit.load``s the file -- no
    architecture class needed anywhere outside ``build_policy``. Species / channel info
    rides along as ``meta.json`` inside the archive."""
    torch, _, _ = _make_torch()
    import json
    path = MODEL_PATHS[sid] if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.eval())
    extra = {"meta.json": json.dumps({"species": sid,
                                      "n_channels": SPECIES_N_CHANNELS[sid],
                                      "meta": meta or {}})}
    tmp = f"{path}.tmp"
    torch.jit.save(scripted, tmp, _extra_files=extra)
    import os
    os.replace(tmp, path)
    return path


def load_model(sid, path=None, device="cpu"):
    """Load a TorchScript policy archive; the file carries its own network code, so this
    works for any architecture ``save_model`` wrote (no ``build_policy`` call)."""
    torch, _, _ = _make_torch()
    path = MODEL_PATHS[sid] if path is None else Path(path)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = torch.jit.load(str(path), map_location=dev)
    model.eval()
    return model
