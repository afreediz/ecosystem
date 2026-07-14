"""Play the sheep-vision DATASET: step through the CSV and show each row's actual vision.

No simulation. This just loads ``sheep_vision_dataset.csv`` and, one row at a time, displays
exactly what was recorded for that sample:

  * the sheep's **terrain** channel   (left panel)
  * the sheep's **food** channel       (middle panel, the grass field)
  * the sheep's **threat** channel     (right panel, the fox blip -- empty in forage scenes)
  * the stored heading ``(dx, dy)``    (red arrow from the sheep at the centre)
  * the label + mode: whether this is a FLEE scene (true heading points AWAY from the fox) or a
    FORAGE scene (true heading points TOWARD the grass), and whether this row's candidate
    heading is TRUE or a wrong one (sideways / oblique / opposite), with its angular offset.

It animates through the rows like a flip-book. Nothing in ``sim/`` is touched.

Run (needs a display):
    venv/Scripts/python.exe notebooks/vision/sheep/show_dataset.py
    venv/Scripts/python.exe notebooks/vision/sheep/show_dataset.py --true-only
    venv/Scripts/python.exe notebooks/vision/sheep/show_dataset.py --mode flee --true-only
    venv/Scripts/python.exe notebooks/vision/sheep/show_dataset.py --save vision.gif --ticks 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

_here = Path(__file__).resolve().parent
DATA_PATH = _here / "sheep_vision_dataset.csv"


def main():
    ap = argparse.ArgumentParser(description="Replay the sheep-vision dataset.")
    ap.add_argument("--data", type=str, default=str(DATA_PATH), help="path to the CSV")
    ap.add_argument("--true-only", action="store_true", help="only show TRUE-heading rows")
    ap.add_argument("--false-only", action="store_true", help="only show wrong-heading rows")
    ap.add_argument("--mode", choices=["flee", "forage"], default=None,
                    help="only show scenes of this behaviour")
    ap.add_argument("--start", type=int, default=0, help="first row to show")
    ap.add_argument("--interval", type=int, default=300, help="ms between rows")
    ap.add_argument("--ticks", type=int, default=None, help="number of rows to play (for --save)")
    ap.add_argument("--save", type=str, default=None, help="write an animated GIF here")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"dataset not found: {data}\nRun generate_dataset.ipynb first.")

    df = pd.read_csv(data)
    WIN = int(round(int(df.columns.str.startswith("t_").sum()) ** 0.5))
    t_cols = [f"t_{i}" for i in range(WIN * WIN)]
    f_cols = [f"f_{i}" for i in range(WIN * WIN)]
    x_cols = [f"x_{i}" for i in range(WIN * WIN)]
    print(f"loaded {len(df)} rows from {data.name}; vision window {WIN}x{WIN}")

    # pick which rows to play
    if args.true_only:
        df = df[df["label"] == 1]
    elif args.false_only:
        df = df[df["label"] == 0]
    if args.mode is not None:
        df = df[df["mode"] == (1 if args.mode == "flee" else 0)]
    df = df.reset_index(drop=True)
    order = list(range(args.start, len(df)))
    if args.ticks is not None:
        order = order[:args.ticks]
    if not order:
        raise SystemExit("no rows to show with the given filters/start.")

    # pre-extract the channels + headings we will page through
    terr = df[t_cols].to_numpy(np.float32).reshape(-1, WIN, WIN)
    food = df[f_cols].to_numpy(np.float32).reshape(-1, WIN, WIN)
    thre = df[x_cols].to_numpy(np.float32).reshape(-1, WIN, WIN)
    dxdy = df[["dx", "dy"]].to_numpy(np.float32)
    label = df["label"].to_numpy(int)
    mode = df["mode"].to_numpy(int)
    scen = df["scenario_id"].to_numpy(int)
    dist = df["dist"].to_numpy(np.float32)
    # cosine of this row's heading vs the TRUE heading (+1 toward, 0 sideways, -1 opposite);
    # older CSVs without the column fall back to computing it from the stored true heading.
    if "cos" in df.columns:
        cosv = df["cos"].to_numpy(np.float32)
    else:
        tdx, tdy = df["tgt_dx"].to_numpy(np.float32), df["tgt_dy"].to_numpy(np.float32)
        dx_, dy_ = df["dx"].to_numpy(np.float32), df["dy"].to_numpy(np.float32)
        cosv = (dx_ * tdx + dy_ * tdy) / (np.hypot(dx_, dy_) * np.hypot(tdx, tdy) + 1e-9)

    c = WIN // 2                                   # the sheep sits at the window centre

    fig, (ax_t, ax_f, ax_x) = plt.subplots(1, 3, figsize=(13, 5))
    im_t = ax_t.imshow(terr[order[0]], origin="upper", cmap="viridis", vmin=0, vmax=1)
    im_f = ax_f.imshow(food[order[0]], origin="upper", cmap="YlGn", vmin=0, vmax=1)
    im_x = ax_x.imshow(thre[order[0]], origin="upper", cmap="magma", vmin=0, vmax=1)
    panels = ((ax_t, "terrain channel"), (ax_f, "food channel (grass)"),
              (ax_x, "threat channel (fox)"))
    for ax, title in panels:
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(c, c, "o", color="cyan", ms=9, mec="k")     # the sheep
    arrows = [ax.plot([], [], "-", color="red", lw=2.4)[0] for ax, _ in panels]
    suptitle = fig.suptitle("", fontsize=13)

    def draw(i):
        r = order[i]
        dx, dy = float(dxdy[r, 0]), float(dxdy[r, 1])
        im_t.set_data(terr[r]); im_f.set_data(food[r]); im_x.set_data(thre[r])
        L = 6.0
        for arr in arrows:
            arr.set_data([c, c + dx * L], [c, c + dy * L])
        off = float(np.degrees(np.arccos(np.clip(cosv[r], -1.0, 1.0))))
        mode_name = "FLEE" if mode[r] == 1 else "FORAGE"
        target = "away from fox" if mode[r] == 1 else "toward grass"
        if label[r] == 1:
            kind = f"TRUE  → {target}"
        else:
            name = "sideways" if off < 115 else ("oblique" if off < 160 else "opposite")
            kind = f"FALSE → {name} ({off:.0f}° off)"
        col = "#1a7f37" if label[r] == 1 else "#b3261e"
        suptitle.set_text(f"row {r}/{len(df)-1}   scenario {scen[r]}   [{mode_name}: {kind}]\n"
                          f"heading (dx, dy) = ({dx:+.2f}, {dy:+.2f})   "
                          f"cos={cosv[r]:+.2f}   target dist = {dist[r]:.1f} cells")
        suptitle.set_color(col)
        return (im_t, im_f, im_x, *arrows, suptitle)

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
