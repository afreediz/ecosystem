"""Play the fox-vision DATASET: step through the CSV and show each row's actual vision.

No simulation. This just loads ``fox_vision_dataset.csv`` and, one row at a time, displays
exactly what was recorded for that sample:

  * the fox's **terrain** channel   (left panel)
  * the fox's **entity** channel     (right panel, the sheep blip)
  * the stored heading ``(dx, dy)``  (red arrow from the fox at the centre)
  * the label: TRUE (toward the sheep) or FALSE (a wrong heading -- sideways, oblique or
    opposite), with its angular offset from the true direction to the sheep

It animates through the rows like a flip-book. Nothing in ``sim/`` is touched.

Run (needs a display):
    venv/Scripts/python.exe notebooks/fvision_play.py
    venv/Scripts/python.exe notebooks/fvision_play.py --true-only
    venv/Scripts/python.exe notebooks/fvision_play.py --save vision.gif --ticks 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

_here = Path(__file__).resolve().parent
DATA_PATH = _here / "fox_vision_dataset.csv"


def main():
    ap = argparse.ArgumentParser(description="Replay the fox-vision dataset.")
    ap.add_argument("--data", type=str, default=str(DATA_PATH), help="path to the CSV")
    ap.add_argument("--true-only", action="store_true", help="only show TRUE (toward) rows")
    ap.add_argument("--false-only", action="store_true", help="only show FALSE (away) rows")
    ap.add_argument("--start", type=int, default=0, help="first row to show")
    ap.add_argument("--interval", type=int, default=300, help="ms between rows")
    ap.add_argument("--ticks", type=int, default=None, help="number of rows to play (for --save)")
    ap.add_argument("--save", type=str, default=None, help="write an animated GIF here")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"dataset not found: {data}\nRun generate_fox_vision_ds.ipynb first.")

    df = pd.read_csv(data)
    WIN = int(round(int(df.columns.str.startswith("t_").sum()) ** 0.5))
    t_cols = [f"t_{i}" for i in range(WIN * WIN)]
    e_cols = [f"e_{i}" for i in range(WIN * WIN)]
    print(f"loaded {len(df)} rows from {data.name}; vision window {WIN}x{WIN}")

    # pick which rows to play
    if args.true_only:
        df = df[df["label"] == 1]
    elif args.false_only:
        df = df[df["label"] == 0]
    df = df.reset_index(drop=True)
    order = list(range(args.start, len(df)))
    if args.ticks is not None:
        order = order[:args.ticks]
    if not order:
        raise SystemExit("no rows to show with the given filters/start.")

    # pre-extract the channels + headings we will page through
    terr = df[t_cols].to_numpy(np.float32).reshape(-1, WIN, WIN)
    enti = df[e_cols].to_numpy(np.float32).reshape(-1, WIN, WIN)
    dxdy = df[["dx", "dy"]].to_numpy(np.float32)
    label = df["label"].to_numpy(int)
    scen = df["scenario_id"].to_numpy(int)
    dist = df["dist"].to_numpy(np.float32)
    # cosine of this row's heading vs the true direction to the sheep (+1 toward, 0 sideways,
    # -1 opposite); older CSVs without the column fall back to computing it from the offset.
    if "cos" in df.columns:
        cosv = df["cos"].to_numpy(np.float32)
    else:
        sdx, sdy = df["sheep_dx"].to_numpy(np.float32), df["sheep_dy"].to_numpy(np.float32)
        dx_, dy_ = df["dx"].to_numpy(np.float32), df["dy"].to_numpy(np.float32)
        cosv = (dx_ * sdx + dy_ * sdy) / (np.hypot(dx_, dy_) * np.hypot(sdx, sdy) + 1e-9)

    c = WIN // 2                                   # the fox sits at the window centre

    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(9.5, 5))
    im_t = ax_t.imshow(terr[order[0]], origin="upper", cmap="viridis", vmin=0, vmax=1)
    im_e = ax_e.imshow(enti[order[0]], origin="upper", cmap="magma", vmin=0, vmax=1)
    for ax, title in ((ax_t, "terrain channel"), (ax_e, "entity channel (sheep)")):
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(c, c, "o", color="cyan", ms=9, mec="k")     # the fox
    (arr_t,) = ax_t.plot([], [], "-", color="red", lw=2.4)
    (arr_e,) = ax_e.plot([], [], "-", color="red", lw=2.4)
    suptitle = fig.suptitle("", fontsize=13)

    def draw(i):
        r = order[i]
        dx, dy = float(dxdy[r, 0]), float(dxdy[r, 1])
        im_t.set_data(terr[r]); im_e.set_data(enti[r])
        L = 6.0
        arr_t.set_data([c, c + dx * L], [c, c + dy * L])
        arr_e.set_data([c, c + dx * L], [c, c + dy * L])
        off = float(np.degrees(np.arccos(np.clip(cosv[r], -1.0, 1.0))))
        if label[r] == 1:
            kind = "TRUE  → toward sheep"
        else:
            name = "sideways" if off < 115 else ("oblique" if off < 160 else "opposite")
            kind = f"FALSE → {name} ({off:.0f}° off)"
        col = "#1a7f37" if label[r] == 1 else "#b3261e"
        suptitle.set_text(f"row {r}/{len(df)-1}   scenario {scen[r]}   [{kind}]\n"
                          f"heading (dx, dy) = ({dx:+.2f}, {dy:+.2f})   "
                          f"cos={cosv[r]:+.2f}   sheep dist = {dist[r]:.1f} cells")
        suptitle.set_color(col)
        return im_t, im_e, arr_t, arr_e, suptitle

    draw(0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if args.save:
        anim = FuncAnimation(fig, draw, frames=len(order), interval=args.interval, blit=False)
        out = Path(args.save)
        anim.save(str(out), writer=PillowWriter(fps=max(1, int(1000 / args.interval))))
        print(f"saved {len(order)} rows -> {out.resolve()}")
    else:
        anim = FuncAnimation(fig, draw, frames=len(order), interval=args.interval,
                             blit=False, cache_frame_data=False)
        globals()["_anim"] = anim          # keep a reference so it is not garbage-collected
        plt.show()


if __name__ == "__main__":
    main()
