"""Live CSV monitor: a standalone window that re-plots a run CSV as it grows.

This is completely independent of the simulation -- it just tails the CSV file that
``analysis.logger.Logger`` writes (the logger flushes every row) and redraws the same
4-panel report as ``analysis.plots`` on a fixed interval. Run it in its own process so it
never blocks or couples to the sim loop:

    python -m analysis.monitor runs/run.csv [--interval 1.0]

``run_experiment.py --monitor`` spawns exactly this as a subprocess.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib

from analysis.plots import population_plot, biomass_plot, trait_plot, phase_plot


def launch(csv_path, interval=1.0):
    """Spawn this monitor in its own process/window, fully decoupled from the caller.
    Returns the Popen handle, or None if it could not start."""
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "analysis.monitor", str(csv_path),
             "--interval", str(interval)])
    except OSError as e:
        print(f"** could not launch monitor: {e} **")
        return None


def _read(csv_path):
    """Read the CSV, tolerating an empty/header-only/mid-write file. Returns None if
    there is nothing plottable yet."""
    try:
        df = pd.read_csv(csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    if df.empty or "tick" not in df.columns:
        return None
    return df


def _redraw(df, axes, csv_path):
    for row in axes:
        for ax in row:
            ax.clear()
    population_plot(df, axes[0, 0])
    biomass_plot(df, axes[0, 1])
    trait_plot(df, axes[1, 0], species="sheep")
    phase_plot(df, axes[1, 1])
    last = int(df["tick"].iloc[-1])
    axes[0, 0].figure.suptitle(
        f"Live monitor: {Path(csv_path).name}  (tick {last})", fontsize=14)


def monitor(csv_path, interval=1.0):
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show(block=False)

    while plt.fignum_exists(fig.number):
        df = _read(csv_path)
        if df is not None:
            _redraw(df, axes, csv_path)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.canvas.draw_idle()
        # pause services the GUI event loop and sleeps; also our redraw cadence
        plt.pause(interval)


def main():
    ap = argparse.ArgumentParser(description="Live monitor window for a run CSV.")
    ap.add_argument("csv", help="path to the run CSV being written")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between refreshes (default 1.0)")
    args = ap.parse_args()
    monitor(args.csv, interval=args.interval)


if __name__ == "__main__":
    main()
