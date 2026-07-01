"""Entry point: watch the simulation live in an Arcade window (observer only).

Shares the exact same sim/ core as run_experiment.py. Usage:
    python run_live.py [--world-seed N] [--seed N] [--scale N] [--spf N]

``--world-seed`` fixes the terrain/rivers; ``--seed`` fixes the run dynamics (omit for a
random run on that world).
"""
from __future__ import annotations

import argparse

from config import make_config


def main():
    ap = argparse.ArgumentParser(description="Watch the ecosystem simulation live.")
    ap.add_argument("--world-seed", type=int, default=None,
                    help="world seed (terrain/rivers); omit for the default world")
    ap.add_argument("--seed", type=int, default=None,
                    help="run/determinism seed; omit for a random run on that world")
    ap.add_argument("--scale", type=int, default=3,
                    help="pixels per world cell (larger => bigger sim window)")
    ap.add_argument("--spf", type=float, default=1.0,
                    help="sim steps per rendered frame (fractional ok, e.g. 0.25 = "
                         "1 step every 4 frames, for slow observation)")
    ap.add_argument("--log-csv", type=str, default=None,
                    help="also log the live run to this CSV (default runs/live.csv "
                         "when --monitor is set)")
    ap.add_argument("--monitor", action="store_true",
                    help="open a separate live window that plots the CSV as it is written")
    args = ap.parse_args()

    cfg = make_config(world_seed=args.world_seed, seed=args.seed)
    # import arcade lazily so headless environments without a display can still import
    # the sim package without pulling in OpenGL.
    from render.viewer import run
    run(cfg, scale=args.scale, steps_per_frame=args.spf,
        log_csv=args.log_csv, monitor=args.monitor)


if __name__ == "__main__":
    main()
