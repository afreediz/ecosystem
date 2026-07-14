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


def _load_species_brain(path: str, species_id: int, device: str = "cpu"):
    """Load a per-species deployment brain from an imitation-learning checkpoint.

    The checkpoint is a memoryless behavioural-cloning policy from
    ``notebooks/imitation_learning/`` -- a SELF-CONTAINED TorchScript archive (code +
    weights, saved by ``common.save_model``), so no architecture class is needed to load
    it -> a ``sim.policy_brain.PolicyBrain`` for this species. It runs in eval /
    deterministic mode, so it draws no randomness and keeps the run reproducible. Torch is
    imported lazily (inside ``sim.policy_brain``) so the rule path never needs it installed.

    (The older recurrent CNN+MLP+LSTM ``NeuralBrain`` and its RL trainer are archived under
    ``backup/`` and no longer deployable -- see ``backup/README.md``.)
    """
    from sim.policy_brain import policy_brain_from_path
    try:
        return policy_brain_from_path(path, species_id, device=device)
    except RuntimeError as e:
        raise ValueError(
            f"could not load brain checkpoint {path!r} as a TorchScript archive "
            f"(expected e.g. notebooks/imitation_learning/"
            f"{('sheep' if species_id == SHEEP else 'fox')}.pt). If this is a legacy "
            f"state_dict checkpoint, re-export it with "
            f"notebooks/imitation_learning/convert_to_jit.py. Original error: {e}") from e


def build_brain(sheep_weights: str | None, fox_weights: str | None, device: str = "cpu"):
    """Build the per-species brain spec for ``Simulation``.

    A species with a checkpoint path gets a learned brain loaded from it; a species with no path
    uses the RuleBrain (``Simulation`` fills it in on the run RNG). Returns ``None`` when NEITHER
    path is set, so ``Simulation`` uses its default rule brain for both species.
    """
    if not sheep_weights and not fox_weights:
        return None
    return {
        SHEEP: _load_species_brain(sheep_weights, SHEEP, device) if sheep_weights else None,
        FOX: _load_species_brain(fox_weights, FOX, device) if fox_weights else None,
    }


def run_experiment(ticks: int, out: str, world_seed: int | None = 12345,
                   seed: int | None = None, log_every: int | None = None,
                   progress_every: int = 2000, quiet: bool = False,
                   monitor: bool = False, sheep_brain: str | None = None,
                   fox_brain: str | None = None, device: str = "cpu"):
    cfg = make_config(world_seed=world_seed, seed=seed)
    if log_every is not None:
        cfg.sim.log_every = log_every
    brain_spec = build_brain(sheep_brain, fox_brain, device)
    sim = Simulation(cfg, brain=brain_spec)   # make_rng resolves + records the run seed
    if not quiet:
        print(f"world_seed={cfg.world.seed}  run_seed={sim.cfg.seed}  "
              f"sheep_brain={sheep_brain or 'rule'}  fox_brain={fox_brain or 'rule'}")
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
    ap.add_argument("--sheep-brain", type=str, default=None,
                    help="path to a trained sheep-brain checkpoint (.pt), e.g. "
                         "notebooks/imitation_learning/sheep.pt; omit to drive sheep with the "
                         "rule brain")
    ap.add_argument("--fox-brain", type=str, default=None,
                    help="path to a trained fox-brain checkpoint (.pt); omit to drive foxes "
                         "with the rule brain")
    ap.add_argument("--device", type=str, default="cpu",
                    help="torch device for any neural brain (default cpu)")
    args = ap.parse_args()

    sim, out = run_experiment(args.ticks, args.out, world_seed=args.world_seed,
                              seed=args.seed, log_every=args.log_every,
                              monitor=args.monitor, sheep_brain=args.sheep_brain,
                              fox_brain=args.fox_brain, device=args.device)

    if args.plot:
        from analysis.plots import make_report
        make_report(out, out_dir="analysis/out")


if __name__ == "__main__":
    main()
