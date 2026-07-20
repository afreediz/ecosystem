"""Capture golden-master baselines for the default (sheep + fox) config.

Run this ONCE on trusted (pre-refactor) code:

    venv/Scripts/python.exe tests/capture_baselines.py

It writes ``tests/baselines/golden.json``, which ``test_determinism.py`` compares against
after every refactor step. Re-run it ONLY when you have deliberately, knowingly changed the
default-config dynamics (then the diff in the committed JSON is the audit trail).
"""
from __future__ import annotations

import json
from pathlib import Path

from determinism_util import run_fingerprint

WORLD_SEED = 12345
SEEDS = [7, 99, 12345]
TICKS = 200
LOG_EVERY = 1

OUT = Path(__file__).resolve().parent / "baselines" / "golden.json"


def main() -> None:
    runs = {}
    for seed in SEEDS:
        print(f"capturing seed={seed} ...", flush=True)
        runs[str(seed)] = run_fingerprint(seed, world_seed=WORLD_SEED,
                                          ticks=TICKS, log_every=LOG_EVERY)
    payload = {
        "meta": {
            "world_seed": WORLD_SEED,
            "seeds": SEEDS,
            "ticks": TICKS,
            "log_every": LOG_EVERY,
            "note": "default sheep+fox config; frozen on pre-refactor code as the "
                    "byte-identical baseline for the framework refactor",
        },
        "runs": runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    for seed, fp in runs.items():
        print(f"  seed {seed}: {fp['final']}  csv={fp['csv_sha256'][:12]}  "
              f"state={fp['state_sha256'][:12]}")


if __name__ == "__main__":
    main()
