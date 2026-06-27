"""Entry point: headless, fast-forward experiment that writes a CSV (§17, §19 of v1.md).

No rendering, no display required. Deterministic given a seed. Usage:
    python run_experiment.py --ticks 20000 --seed 12345 --out runs/run.csv
    python run_experiment.py --ticks 20000 --plot          # also render a PNG report
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import make_config, SHEEP, FOX
from sim.simulation import Simulation
from analysis.logger import Logger


def run_experiment(ticks: int, seed: int, out: str, log_every: int | None = None,
                   progress_every: int = 2000, quiet: bool = False):
    cfg = make_config(seed=seed)
    if log_every is not None:
        cfg.sim.log_every = log_every
    sim = Simulation(cfg)
    logger = Logger(out, sim)
    logger.open()

    t0 = time.time()
    extinct_at = None
    for i in range(ticks):
        sim.step()
        logger.record()
        if not quiet and progress_every and (i + 1) % progress_every == 0:
            pops = sim.populations
            print(f"  tick {i+1:>7}  sheep {pops['sheep']:>5}  fox {pops['fox']:>4}  "
                  f"veg {sim.stats['veg_biomass']:.0f}")
        if sim.populations["sheep"] == 0 and sim.populations["fox"] == 0:
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
            print(f"** total extinction at tick {extinct_at} **")
        print(f"CSV: {Path(out).resolve()}")
    return sim, out


def main():
    ap = argparse.ArgumentParser(description="Headless ecosystem experiment.")
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default="runs/run.csv")
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--plot", action="store_true", help="render a PNG report after the run")
    args = ap.parse_args()

    sim, out = run_experiment(args.ticks, args.seed, args.out, log_every=args.log_every)

    if args.plot:
        from analysis.plots import make_report
        make_report(out, out_dir="analysis/out")


if __name__ == "__main__":
    main()
