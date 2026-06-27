"""matplotlib analysis of a logged run (§17 of v1.md).

Reads the CSV produced by ``analysis.logger.Logger`` with pandas and renders:
  - population vs time (ecosystem signal; look for predator-prey oscillations)
  - mean trait vs time (evolution signal; look for drift under selection)
  - sheep-vs-fox phase plot (Lotka-Volterra loop)

Usage:  python -m analysis.plots <run.csv> [--out plots_dir] [--show]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe default; overridden to a GUI backend if --show
import matplotlib.pyplot as plt

from sim import genome as gn


def population_plot(df, ax):
    ax.plot(df["tick"], df["n_sheep"], label="sheep", color="#3a7d2c")
    ax.plot(df["tick"], df["n_fox"], label="fox", color="#b03a2e")
    ax.set_xlabel("tick")
    ax.set_ylabel("population")
    ax.set_title("Population vs time")
    ax.legend()


def biomass_plot(df, ax):
    ax.plot(df["tick"], df["veg_biomass"], color="#4a9d3a")
    ax.set_xlabel("tick")
    ax.set_ylabel("vegetation biomass")
    ax.set_title("Vegetation biomass vs time")


def trait_plot(df, ax, species="sheep", traits=("max_speed", "sensory_range", "size")):
    for t in traits:
        col = f"{species}_{t}"
        if col in df.columns:
            ax.plot(df["tick"], df[col], label=t)
    ax.set_xlabel("tick")
    ax.set_ylabel("mean gene value")
    ax.set_title(f"{species} trait drift (evolution signal)")
    ax.legend()


def phase_plot(df, ax):
    ax.plot(df["n_sheep"], df["n_fox"], color="#555", lw=0.8)
    ax.scatter(df["n_sheep"].iloc[:1], df["n_fox"].iloc[:1], c="green", label="start", zorder=5)
    ax.scatter(df["n_sheep"].iloc[-1:], df["n_fox"].iloc[-1:], c="red", label="end", zorder=5)
    ax.set_xlabel("sheep")
    ax.set_ylabel("fox")
    ax.set_title("Phase plot (sheep vs fox)")
    ax.legend()


def make_report(csv_path, out_dir=None, show=False):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    population_plot(df, axes[0, 0])
    biomass_plot(df, axes[0, 1])
    trait_plot(df, axes[1, 0], species="sheep")
    phase_plot(df, axes[1, 1])
    fig.suptitle(f"Ecosystem run: {Path(csv_path).name}", fontsize=14)
    fig.tight_layout()

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / (Path(csv_path).stem + "_report.png")
        fig.savefig(dest, dpi=110)
        print(f"wrote {dest}")
    if show:
        plt.show()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot an ecosystem run CSV.")
    ap.add_argument("csv", help="path to the logged run CSV")
    ap.add_argument("--out", default="analysis/out", help="output directory for PNGs")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args()
    if args.show:
        matplotlib.use("TkAgg", force=True)
    make_report(args.csv, out_dir=args.out, show=args.show)


if __name__ == "__main__":
    main()
