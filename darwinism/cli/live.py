"""Entry point: watch the simulation live in an Arcade window (observer only).

Shares the exact same sim core as the headless experiment. Usage (installed console script;
``python -m darwinism live`` and the ``run_live.py`` root shim are equivalent):
    darwinism-live [--world-seed N] [--seed N] [--scale N] [--spf N]
    darwinism-live --sheep-brain notebooks/imitation_learning/sheep.pt   # learned sheep

``--world-seed`` fixes the terrain/rivers; ``--seed`` fixes the run dynamics (omit for a
random run on that world). ``--sheep-brain PATH`` / ``--fox-brain PATH`` drive that species
with a trained brain; a species with no path stays on the hardcoded rule brain. ``--save-gif
PATH`` records the visual run to an animated GIF, written when the window closes.
"""
from __future__ import annotations

import argparse

from darwinism.cli.experiment import build_brain
from darwinism.config import make_config


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
    ap.add_argument("--sheep-brain", type=str, default=None,
                    help="path to a trained sheep-brain checkpoint (.pt), e.g. "
                         "notebooks/imitation_learning/sheep.pt; omit to drive sheep with the "
                         "rule brain")
    ap.add_argument("--fox-brain", type=str, default=None,
                    help="path to a trained fox-brain checkpoint (.pt); omit to drive foxes "
                         "with the rule brain")
    ap.add_argument("--device", type=str, default="cpu",
                    help="torch device for any neural brain (default cpu)")
    ap.add_argument("--save-gif", type=str, default=None,
                    help="record the visual run to this animated GIF (written on window "
                         "close), e.g. runs/live.gif")
    args = ap.parse_args()

    cfg = make_config(world_seed=args.world_seed, seed=args.seed)
    # resolve + record the run seed now (mutates cfg.seed in place) so we can report it before
    # the viewer creates the Simulation; the Simulation reuses the same cfg => identical run.
    cfg.make_rng()
    print(f"world_seed={cfg.world.seed}  run_seed={cfg.seed}  "
          f"sheep_brain={args.sheep_brain or 'rule'}  fox_brain={args.fox_brain or 'rule'}")
    # build the per-species brain spec (None => Simulation uses its default RuleBrain for both);
    # any learned brain is imported lazily inside build_brain so the rule path never needs torch.
    brain = build_brain(args.sheep_brain, args.fox_brain, args.device)
    # import arcade lazily so headless environments without a display can still import
    # the sim package without pulling in OpenGL.
    from darwinism.render.viewer import run
    run(cfg, scale=args.scale, steps_per_frame=args.spf,
        log_csv=args.log_csv, monitor=args.monitor, brain=brain, save_gif=args.save_gif)


if __name__ == "__main__":
    main()
