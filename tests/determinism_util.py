"""Shared helpers for the golden-master determinism tests.

The whole framework refactor is gated on one property: for the DEFAULT config
(sheep + fox) the run must stay byte-identical. These helpers turn a short run into
a compact, comparable fingerprint:

  * ``run_fingerprint`` runs a fresh ``Simulation`` for a few hundred ticks, logs every
    tick through the real ``Logger`` (so the fingerprint is tied to the exact CSV bytes),
    and hashes both the CSV output and the raw entity Structure-of-Arrays state.
  * ``state_hash`` hashes every dynamically-meaningful SoA array at full float32 precision
    -- this catches divergences that the coarser ``log_every`` CSV sampling would miss.

Baselines are captured ONCE on the pre-refactor code (see ``capture_baselines.py``) and
frozen in ``baselines/golden.json``; ``test_determinism.py`` regenerates and compares.

NOTE (pre-packaging): this inserts the repo root on sys.path so the flat top-level
modules (``config``/``sim``/``analysis``) import. In Phase 1 the imports below become
``darwinism.*`` and this shim is removed (the package is installed with ``pip install -e .``).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from darwinism.config import make_config              # noqa: E402
from darwinism.sim.simulation import Simulation       # noqa: E402
from darwinism.analysis.logger import Logger          # noqa: E402

# SoA arrays that carry real (RNG-load-bearing) dynamics. ``mating_glow`` /
# ``action_overridden`` are cosmetic/diagnostic per the entity store's own docs, but they
# are still deterministic, so including them only makes the fingerprint stricter.
_STATE_ARRAYS = (
    "pos_x", "pos_y", "heading_x", "heading_y", "energy", "hunger", "thirst",
    "health", "age", "sex", "species", "repro_cooldown", "birth_id",
    "mating_glow", "asleep", "action_overridden", "alive",
)


def state_hash(sim) -> str:
    """SHA-256 over the full entity SoA state (all slots, incl. free ones)."""
    ent = sim.entities
    h = hashlib.sha256()
    for name in _STATE_ARRAYS:
        h.update(np.ascontiguousarray(getattr(ent, name)).tobytes())
    h.update(np.ascontiguousarray(ent.genome).tobytes())
    return h.hexdigest()


def run_fingerprint(seed: int, world_seed: int = 12345, ticks: int = 200,
                    log_every: int = 1) -> dict:
    """Run the default-config sim and return a compact, comparable fingerprint.

    Mirrors ``run_experiment``'s core loop (default RuleBrain, no early-extinction stop,
    no prints). Returns CSV + entity-state hashes plus a few human-readable final values
    for debuggability when a comparison fails.
    """
    cfg = make_config(world_seed=world_seed, seed=seed)
    cfg.sim.log_every = log_every
    sim = Simulation(cfg)   # default RuleBrain for every species

    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        logger = Logger(path, sim)
        logger.open()
        for _ in range(ticks):
            sim.step()
            logger.record()
        logger.close()
        csv_bytes = Path(path).read_bytes()
    finally:
        os.remove(path)

    return {
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "state_sha256": state_hash(sim),
        "final": {
            "tick": int(sim.tick),
            "n_sheep": int(sim.populations["sheep"]),
            "n_fox": int(sim.populations["fox"]),
            "veg_biomass": round(float(sim.stats["veg_biomass"]), 3),
        },
    }
