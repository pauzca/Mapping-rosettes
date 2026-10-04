"""Build train / val / test splits for the detection ablation.

- train + val: the Kaggle train tiles. ~20% of the plants of each source mosaic (agosto, dic1cm, dic3cm)
  goes to val, chosen as whole spatial blocks of BLOCK x BLOCK tile positions; train tiles whose footprint
  overlaps any val tile are dropped so no ground is shared between train and val. Large blocks keep that
  buffer small (8x8 blocks lose ~7% of the plants, 4x4 blocks ~21%).
- test: the Kaggle val tiles of Diciembre 1 cm / 3.6 cm (cut from the evaluation orthomosaics), the Agosto
  evaluation set restricted to its annotated area (the Kaggle Agosto val tiles also covered an unannotated
  half of that orthomosaic), plus Junio 2025.

Both regimes get the same tile assignment: `mixed` uses the original-resolution tiles and `lowres`
the degraded ones (agosto / dic1cm) for train and val. The test set is the same for both regimes.
Splits are built with symlinks under OUT, and the assignment is written to ablations/splits.csv.

Example:
    python -m ablations.make_splits
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/disk/guille/mapping-rosettes/data")
OUT = DATA / "splits"
MIXED, LOWRES, JUNIO = DATA / "mixed-resolution/dataset", DATA / "low-resolution/dataset", DATA / "junio_2025_val/dataset"
AGOSTO_TEST = DATA / "dataset_agosto_val/dataset"
OVERLAP = 0.2  # tiles were cut with 20% overlap: footprint = stride / (1 - overlap)


def lowres_name(name):
    src, x, y = re.match(r"([a-z0-9]+)_tile_(\d+)_(\d+)\.png", name).groups()
    return f"3cmdic_tile_{x}_{y}.png" if src == "dic3cm" else f"tile_{x}_{y}_{src}.png"


def assign_val(df, frac, block, rng):
    """Mark whole blocks of tiles as val until `frac` of the source's plants is reached."""
    df = df.copy()
    stride = np.bincount(np.diff(np.unique(df.x))).argmax()
    df["block"] = list(zip(df.x // (stride * block), df.y // (stride * block)))
    w = df.groupby("block").boxes.sum()
    blocks = list(w.index)
    rng.shuffle(blocks)
    val, n = set(), 0
    for b in blocks:
        if n + w[b] / 2 > frac * w.sum():  # skip blocks that would overshoot the target
            continue
        val.add(b)
        n += w[b]
        if n >= frac * w.sum():
            break
    df["split"] = np.where(df.block.isin(val), "val", "train")
    # drop train tiles whose footprint intersects a val tile (shared ground)
    size = stride / (1 - OVERLAP)
    v = df[df.split == "val"][["x", "y"]].to_numpy()
    for i in df.index[df.split == "train"]:
        if ((np.abs(v[:, 0] - df.at[i, "x"]) < size) & (np.abs(v[:, 1] - df.at[i, "y"]) < size)).any():
            df.at[i, "split"] = "dropped"
    return df


def n_boxes(path):
    return sum(1 for _ in open(path)) if path.exists() else 0


def link(src_img, src_lbl, dst_dir, name):
    for kind, src in (("images", src_img), ("labels", src_lbl)):
        if src is None or not src.exists():
            continue
        dst = dst_dir / kind / (name if kind == "images" else Path(name).with_suffix(".txt").name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src, dst)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--block", type=int, default=8, help="block side in tile positions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.out
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; remove it to rebuild")
    rng = np.random.default_rng(args.seed)

    rows = []
    for f in sorted(os.listdir(MIXED / "train/images")):
        src, x, y = re.match(r"([a-z0-9]+)_tile_(\d+)_(\d+)\.png", f).groups()
        rows.append(dict(name=f, source=src, x=int(x), y=int(y),
                         boxes=n_boxes(MIXED / "train/labels" / f"{Path(f).stem}.txt")))
    tv = pd.concat([assign_val(g, args.frac, args.block, rng) for _, g in pd.DataFrame(rows).groupby("source")])
    test = [dict(name=f, source=f.split("_")[0], split="test") for f in sorted(os.listdir(MIXED / "val/images"))
            if not f.startswith("agosto")]
    test += [dict(name=f"agosto_{f}", source="agosto", split="test") for f in sorted(os.listdir(AGOSTO_TEST / "images"))]
    test += [dict(name=f"junio2025_{f}", source="junio2025", split="test") for f in sorted(os.listdir(JUNIO / "images"))]
    man = pd.concat([tv, pd.DataFrame(test)], ignore_index=True)

    for _, r in man.iterrows():
        if r.split == "dropped":
            continue
        stem = Path(r["name"]).stem
        if r.split == "test":
            if r.source in ("junio2025", "agosto"):
                root = JUNIO if r.source == "junio2025" else AGOSTO_TEST
                orig = r["name"].removeprefix(f"{r.source}_")
                img, lbl = root / "images" / orig, root / "labels" / Path(orig).with_suffix(".txt").name
            else:
                img, lbl = MIXED / "val/images" / r["name"], MIXED / "val/labels" / f"{stem}.txt"
            for regime in ("mixed", "lowres"):
                link(img, lbl, OUT / regime / "test", r["name"])
            man.loc[_, "boxes"] = n_boxes(lbl)
            continue
        link(MIXED / "train/images" / r["name"], MIXED / "train/labels" / f"{stem}.txt", OUT / "mixed" / r.split, r["name"])
        low = lowres_name(r["name"])
        link(LOWRES / "train/images" / low, LOWRES / "train/labels" / Path(low).with_suffix(".txt").name,
             OUT / "lowres" / r.split, r["name"])

    for regime, note in (("mixed", "original resolutions"), ("lowres", "agosto/dic1cm degraded to low resolution")):
        (OUT / regime / "data.yaml").write_text(
            f"# Detection ablation splits ({note}); built by ablations/make_splits.py\n"
            f"path: {OUT / regime}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: frailejon\n")
    man["boxes"] = man["boxes"].fillna(0).astype(int)
    man.drop(columns="block").to_csv(Path(__file__).with_name("splits.csv"), index=False)
    print(man.groupby(["source", "split"]).agg(tiles=("name", "size"), boxes=("boxes", "sum")).to_string())


if __name__ == "__main__":
    main()
