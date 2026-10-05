"""Compare evaluation outputs with the published results (ablations/expected_results.json).

With --tol 0 every count (TP / FP / FN) and metric must match exactly, as in the environment of
requirements.txt. With another PyTorch / CUDA build a detection right at a threshold can flip, so
--tol compares the metrics (F1, P, R, RF1, RP, RR) with that absolute tolerance and only reports counts.

Example:
    python -m ablations.check --eval-root outputs/eval --tol 0.005
"""
import argparse
import json
from pathlib import Path

EXPECTED = Path(__file__).with_name("expected_results.json")
COUNTS = {"tp", "fp", "fn"}


def compare(eval_root, tol=0.0):
    expected = json.loads(EXPECTED.read_text())
    n = exact = 0
    worst, diffs = 0.0, []
    for regime, models in expected.items():
        for model, exp in models.items():
            d = Path(eval_root) / regime / model
            got = {}
            if (d / "test_tiles.json").exists():
                got.update({("tiles", s): v for s, v in json.loads((d / "test_tiles.json").read_text())["prf"].items()})
            for mosaic in exp["mosaics"]:
                f = d / f"{mosaic}_metrics.json"
                if f.exists():
                    got[("mosaics", mosaic)] = json.loads(f.read_text())["at_iou50"][0]
            for level in ("tiles", "mosaics"):
                for key, vals in exp[level].items():
                    for k, v in vals.items():
                        g = got.get((level, key), {}).get(k)
                        n += 1
                        exact += g == v
                        if g is None:
                            diffs.append(f"{regime}/{model} {level} {key}: missing")
                        elif k not in COUNTS:
                            worst = max(worst, abs(g - v))
                            if abs(g - v) > tol:
                                diffs.append(f"{regime}/{model} {level} {key} {k}: expected {v}, got {g}")
                        elif tol == 0 and g != v:
                            diffs.append(f"{regime}/{model} {level} {key} {k}: expected {v}, got {g}")
    return n, exact, worst, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/eval")
    ap.add_argument("--tol", type=float, default=0.0, help="absolute tolerance on metrics (0 = exact match)")
    args = ap.parse_args()
    n, exact, worst, diffs = compare(args.eval_root, args.tol)
    print(f"{exact}/{n} values identical to the published tables; largest metric difference {worst:.4f}")
    print("OK" if not diffs else f"{len(diffs)} values outside the tolerance ({args.tol}):")
    for d in diffs:
        print("  " + d)
    raise SystemExit(1 if diffs else 0)


if __name__ == "__main__":
    main()
