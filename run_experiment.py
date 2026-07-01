"""Entry point: headless, fast-forward experiment that writes a CSV (§17, §19 of v1.md).

No rendering, no display required. Usage:
    python run_experiment.py --ticks 20000 --world-seed 12345 --seed 7 --out runs/run.csv
    python run_experiment.py --ticks 20000 --world-seed 12345   # random run on a fixed world
    python run_experiment.py --ticks 20000 --plot               # also render a PNG report

``--world-seed`` fixes the terrain/rivers; ``--seed`` fixes the run dynamics (omit it for a
random, non-reproducible run -- the resolved seed is printed so you can reproduce it later).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import make_config, SHEEP, FOX
from sim.simulation import Simulation
from analysis.logger import Logger
from analysis.monitor import launch as _launch_monitor


def run_experiment(ticks: int, out: str, world_seed: int | None = 12345,
                   seed: int | None = None, log_every: int | None = None,
                   progress_every: int = 2000, quiet: bool = False,
                   monitor: bool = False):
    cfg = make_config(world_seed=world_seed, seed=seed)
    if log_every is not None:
        cfg.sim.log_every = log_every
    sim = Simulation(cfg)   # make_rng resolves + records the run seed (random if unset)
    if not quiet:
        print(f"world_seed={cfg.world.seed}  run_seed={sim.cfg.seed}")
    logger = Logger(out, sim)
    logger.open()   # writes the header now, so the monitor has a file to tail
    mon_proc = _launch_monitor(out) if monitor else None

    t0 = time.time()
    extinct_at = None
    for i in range(ticks):
        sim.step()
        logger.record()
        if not quiet and progress_every and (i + 1) % progress_every == 0:
            pops = sim.populations
            print(f"  tick {i+1:>7}  sheep {pops['sheep']:>5}  fox {pops['fox']:>4}  "
                  f"veg {sim.stats['veg_biomass']:.0f}")
        # stop as soon as either species dies out -- with one gone the predator-prey
        # dynamics are over (no fox => sheep run to the cap; no sheep => foxes starve).
        if sim.populations["sheep"] == 0 or sim.populations["fox"] == 0:
            extinct_at = i + 1
            break
    logger.close()
    dt = time.time() - t0

    final = sim.populations
    if not quiet:
        print(f"\ndone: {sim.tick} ticks in {dt:.1f}s "
              f"({sim.tick / max(dt, 1e-9):.0f} ticks/s)")
        print(f"final populations: sheep={final['sheep']}  fox={final['fox']}")
        if extinct_at:
            gone = [s for s in ("sheep", "fox") if final[s] == 0]
            print(f"** {' & '.join(gone)} extinct at tick {extinct_at} -- stopping **")
        print(f"CSV: {Path(out).resolve()}")
    if mon_proc is not None and mon_proc.poll() is None and not quiet:
        print("monitor window still open (close it to exit); "
              "showing the final data")
    return sim, out


def main():
    ap = argparse.ArgumentParser(description="Headless ecosystem experiment.")
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--world-seed", type=int, default=12345,
                    help="seeds the terrain/rivers; same world-seed => identical world")
    ap.add_argument("--seed", type=int, default=None,
                    help="run/determinism seed; omit for a random (non-reproducible) run")
    ap.add_argument("--out", type=str, default="runs/run.csv")
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--plot", action="store_true", help="render a PNG report after the run")
    ap.add_argument("--monitor", action="store_true",
                    help="open a separate live window that plots the CSV as it is written")
    args = ap.parse_args()

    sim, out = run_experiment(args.ticks, args.out, world_seed=args.world_seed,
                              seed=args.seed, log_every=args.log_every,
                              monitor=args.monitor)

    if args.plot:
        from analysis.plots import make_report
        make_report(out, out_dir="analysis/out")


if __name__ == "__main__":
    main()
