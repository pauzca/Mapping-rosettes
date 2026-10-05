"""Compare evaluation outputs with the published results (ablations/expected_results.json).

Every count (TP / FP / FN) and metric of the tile-level and mosaic-level tables must match exactly.

Example:
    python -m ablations.check --eval-root outputs/eval
"""
import argparse
import json
from pathlib import Path

EXPECTED = Path(__file__).with_name("expected_results.json")


def compare(eval_root):
    expected, diffs, n = json.loads(EXPECTED.read_text()), [], 0
    for regime, models in expected.items():
        for model, exp in models.items():
            d = Path(eval_root) / regime / model
            got_tiles = json.loads((d / "test_tiles.json").read_text())["prf"] if (d / "test_tiles.json").exists() else {}
            for src, vals in exp["tiles"].items():
                for k, v in vals.items():
                    n += 1
                    g = got_tiles.get(src, {}).get(k)
                    if g != v:
                        diffs.append(f"{regime}/{model} tiles {src} {k}: expected {v}, got {g}")
            for mosaic, vals in exp["mosaics"].items():
                f = d / f"{mosaic}_metrics.json"
                got = json.loads(f.read_text())["at_iou50"][0] if f.exists() else {}
                for k, v in vals.items():
                    n += 1
                    if got.get(k) != v:
                        diffs.append(f"{regime}/{model} mosaic {mosaic} {k}: expected {v}, got {got.get(k)}")
    return n, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/eval")
    n, diffs = compare(ap.parse_args().eval_root)
    print(f"{n - len(diffs)}/{n} values match the published tables")
    for d in diffs:
        print("  " + d)
    raise SystemExit(1 if diffs else 0)


if __name__ == "__main__":
    main()
