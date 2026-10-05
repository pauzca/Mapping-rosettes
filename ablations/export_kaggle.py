"""Export the train / val / test splits as two self-contained Kaggle datasets.

Copies real files (no symlinks) from data/splits/{mixed,lowres} into data/kaggle_export/<slug>/, with
one naming scheme (<source>_tile_<x>_<y>.png), a data.yaml with relative paths, the per-tile split
manifest, a README with counts and a dataset-metadata.json for `kaggle datasets version`.

Example:
    python -m ablations.export_kaggle
"""
import json
import shutil
from pathlib import Path

import pandas as pd

from ablations.make_splits import DATA, OUT as SPLITS

EXPORT = DATA / "kaggle_export"
MANIFEST = Path(__file__).with_name("splits.csv")
DATASETS = {
    "mixed": dict(slug="mixed-resolution-frailejones", title="Mixed resolution frailejones detection",
                  note="Agosto and Diciembre 1 cm tiles at their original (high) resolution, Diciembre 3.6 cm at low "
                       "resolution. Used to train the main models of the paper."),
    "lowres": dict(slug="low-resolution-frailejones-detection", title="Low resolution frailejones detection",
                   note="Same tiles and labels as the mixed dataset, but in train and val the Agosto and Diciembre 1 cm "
                        "tiles are degraded to low resolution, so the model only sees low-resolution imagery (models "
                        "marked * in the paper). The test split is identical to the mixed dataset (original resolution)."),
}
SOURCES = {"agosto": "Agosto", "dic1cm": "Diciembre, GSD ~1 cm", "dic3cm": "Diciembre, GSD 3.6 cm",
           "junio2025": "Junio 2025 (test only)"}


def readme(info, man):
    t = man[man.split != "dropped"].groupby(["split", "source"]).agg(tiles=("name", "size"), plants=("boxes", "sum"))
    rows = "\n".join(f"| {sp} | {SOURCES[src]} | {r.tiles} | {r.plants} |" for (sp, src), r in t.iterrows())
    tot = man[man.split != "dropped"].groupby("split").agg(tiles=("name", "size"), plants=("boxes", "sum"))
    totals = "\n".join(f"| **{sp}** | all | **{r.tiles}** | **{r.plants}** |" for sp, r in tot.iterrows())
    return f"""# {info['title']}

Drone orthomosaic tiles of páramo giant rosettes (frailejones, *Espeletia*) for object detection, from the
paper *Mapping post-fire mortality of paramo giant rosette plants using artificial intelligence on
multispectral drone imagery*.

{info['note']}

## Format
- YOLO format: `dataset/<split>/images/*.png` (512 x 512 px RGB) and `dataset/<split>/labels/*.txt`
  (`class cx cy w h`, normalised). One class: `0 = frailejon`. Tiles without a label file contain no plants.
- All tiles were cut at a ground resolution of 1.36 cm/px (geodataset, 512 px tiles, 20% overlap), so plants
  have a similar size in pixels across sources.
- File names: `<source>_tile_<x>_<y>.png`, where x, y are the tile origin in pixels of the source orthomosaic.
- `data.yaml` works directly with Ultralytics; `splits.csv` lists the split of every original tile.

## Checkpoints and mosaics
- `checkpoints/<model>/weights/best.pt`: the detectors of the ablation trained on this dataset (best
  validation mAP50-95). Ultralytics checkpoints (yolo11s, yolov8s, yolo11m, rtdetr_l) load with
  `ultralytics`; the others (rtdetr_r18, frcnn_mnv3, frcnn_r50, deepforest) with `ablations.models.load_run`
  from the code repository, which also reads `args.json`.
- `mosaics/` (mixed-resolution dataset only): the Diciembre 1 cm and 3.6 cm evaluation orthomosaics used
  for the mosaic-level table.

## Splits
- **train / val**: the original training tiles. About 20% of the plants of each source orthomosaic were
  moved to val as whole spatial blocks (8 x 8 tile positions); training tiles overlapping a val tile were
  removed (split `dropped` in `splits.csv`, ~7.5% of the plants), so train and val share no ground.
- **test**: tiles cut from the evaluation orthomosaics (Agosto, restricted to its annotated area, and
  Diciembre 1 cm and 3.6 cm) plus Junio 2025.
  No test tile comes from the areas used for train or val. Use val for model and threshold selection and
  report on test.

| split | source | tiles | plants |
|---|---|---|---|
{rows}
{totals}
"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-test", action="store_true", help="rewrite only the test split, README and manifest")
    ap.add_argument("--checkpoints", type=Path, default=None,
                    help="training root with <regime>/<model>_s123/ runs; adds checkpoints/<model>/ to each dataset")
    ap.add_argument("--mosaics", type=Path, nargs="*", default=[],
                    help="evaluation orthomosaics to add under mosaics/ (mixed-resolution dataset only)")
    args = ap.parse_args()
    man = pd.read_csv(MANIFEST)
    for regime, info in DATASETS.items():
        root = EXPORT / info["slug"]
        splits = ("test",) if args.only_test else ("train", "val", "test")
        if args.only_test:
            shutil.rmtree(root / "dataset" / "test")
        elif root.exists():
            shutil.rmtree(root)
        for split in splits:
            for kind in ("images", "labels"):
                src_dir, dst_dir = SPLITS / regime / split / kind, root / "dataset" / split / kind
                dst_dir.mkdir(parents=True)
                for f in sorted(src_dir.iterdir()):
                    if f.suffix in (".png", ".txt"):
                        shutil.copyfile(f.resolve(), dst_dir / f.name)
        (root / "dataset" / "data.yaml").write_text(
            "train: train/images\nval: val/images\ntest: test/images\nnames:\n  0: frailejon\n")
        man.to_csv(root / "splits.csv", index=False)
        (root / "README.md").write_text(readme(info, man))
        if args.checkpoints:
            for run in sorted((args.checkpoints / regime).glob("*_s123")):
                dst = root / "checkpoints" / run.name.removesuffix("_s123")
                (dst / "weights").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(run / "weights" / "best.pt", dst / "weights" / "best.pt")
                if (run / "results.csv").exists() and not (run / "args.yaml").exists():  # ablations.train_torch run
                    shutil.copyfile(run / "args.json", dst / "args.json")
        if args.mosaics and regime == "mixed":
            (root / "mosaics").mkdir(exist_ok=True)
            for m in args.mosaics:
                shutil.copyfile(m, root / "mosaics" / m.name)
        (root / "dataset-metadata.json").write_text(json.dumps(
            {"id": f"paulauzcategui/{info['slug']}", "title": info["title"], "licenses": [{"name": "unknown"}]},
            indent=2))
        n = {s: len(list((root / "dataset" / s / "images").iterdir())) for s in ("train", "val", "test")}
        print(f"{info['slug']}: {n} -> {root}")


if __name__ == "__main__":
    main()
