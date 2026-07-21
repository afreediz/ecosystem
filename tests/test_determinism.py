"""Golden-master determinism tests -- the safety net for the framework refactor.

These assert that the DEFAULT (sheep + fox) config stays byte-identical to the frozen
baseline (``baselines/golden.json``, captured on pre-refactor code). Run after every
refactor step:

    venv/Scripts/python.exe -m pytest tests/test_determinism.py -q

A failure means a change perturbed the RNG stream / dynamics for the default config -- the
one thing the refactor must never do. (Adding a NEW species is a new config and is expected
to differ; that path is covered by its own smoke test, not the golden-master.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from determinism_util import run_fingerprint

_GOLDEN = json.loads((Path(__file__).resolve().parent / "baselines" / "golden.json").read_text())
_META = _GOLDEN["meta"]
_SEEDS = _META["seeds"]


@pytest.mark.parametrize("seed", _SEEDS)
def test_matches_golden(seed):
    """Full byte-identical match (CSV + entity state) to the frozen baseline."""
    fp = run_fingerprint(seed, world_seed=_META["world_seed"],
                         ticks=_META["ticks"], log_every=_META["log_every"])
    want = _GOLDEN["runs"][str(seed)]
    assert fp["csv_sha256"] == want["csv_sha256"], (
        f"CSV drift on seed {seed}: got final {fp['final']}, baseline {want['final']}")
    assert fp["state_sha256"] == want["state_sha256"], (
        f"entity-state drift on seed {seed} (CSV may match but full state diverged)")


def test_reproducible():
    """Same config run twice in-process is identical (pure determinism, no baseline)."""
    a = run_fingerprint(7, world_seed=_META["world_seed"], ticks=40, log_every=1)
    b = run_fingerprint(7, world_seed=_META["world_seed"], ticks=40, log_every=1)
    assert a == b


def test_seeds_distinct():
    """Different run seeds on the same world produce different runs (checks the frozen
    baseline itself, so it costs no extra sim runs)."""
    hashes = {s: _GOLDEN["runs"][str(s)]["state_sha256"] for s in _SEEDS}
    assert len(set(hashes.values())) == len(_SEEDS), f"seeds collided: {hashes}"
