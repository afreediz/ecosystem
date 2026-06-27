"""Entry point: watch the simulation live in an Arcade window (observer only).

Shares the exact same sim/ core as run_experiment.py. Usage:
    python run_live.py [--seed N] [--scale N] [--spf N]
"""
from __future__ import annotations

import argparse

from config import make_config


def main():
    ap = argparse.ArgumentParser(description="Watch the ecosystem simulation live.")
    ap.add_argument("--seed", type=int, default=None, help="master seed override")
    ap.add_argument("--scale", type=int, default=4, help="pixels per world cell")
    ap.add_argument("--spf", type=int, default=1, help="sim steps per rendered frame")
    args = ap.parse_args()

    cfg = make_config(seed=args.seed) if args.seed is not None else None
    # import arcade lazily so headless environments without a display can still import
    # the sim package without pulling in OpenGL.
    from render.viewer import run
    run(cfg, scale=args.scale, steps_per_frame=args.spf)


if __name__ == "__main__":
    main()
