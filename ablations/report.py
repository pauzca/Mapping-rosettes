"""Collect evaluation results into the format of the paper's results table.

Resolución | Modelo | Mosaicos | RF1 | RP | RR, mosaic-level metrics at IoU 0.5. A TOTAL row per
resolution averages the per-mosaic metrics weighted by the number of plants in each mosaic, as
in the paper. Models trained on the low-resolution set are marked with *, like in the paper.
Writes Markdown and CSV next to the evaluation folders.

Example:
    python -m ablations.report --eval-root outputs/eval --rule fixed --ckpt last
"""
import argparse
import json
from pathlib import Path

import pandas as pd

LABELS = {
    "paper_yolov11": "YOLOv11 (paper checkpoint)",
    "yolo11s": "YOLO11s", "yolov8s": "YOLOv8s", "yolo11m": "YOLO11m",
    "rtdetr_l": "RT-DETR-L", "rtdetr_r18": "RT-DETR-R18",
    "frcnn_mnv3": "Faster R-CNN MobileNetV3-FPN", "frcnn_r50": "Faster R-CNN R50-FPN",
    "deepforest": "DeepForest (RetinaNet R50-FPN)",
}
ORDER = list(LABELS)
METRICS = ["RF1", "RP", "RR", "AP50"]


def parse_run(run):
    """'yolo11s_last' -> ('yolo11s', 'finetune', 'last'); 'zs_frcnn_r50' -> ('frcnn_r50', 'zeroshot', '-')."""
    if run.startswith("zs_"):
        return run[3:], "zeroshot", "-"
    if run.startswith("paper_"):
        return run, "paper", "-"
    model, _, ckpt = run.rpartition("_")
    return model, "finetune", ckpt


def load(eval_root, rule):
    rows = []
    for f in sorted(Path(eval_root).glob("*/*/*_metrics.json")):
        r = json.loads(f.read_text())
        regime, run = f.parts[-3], f.parts[-2]
        model, kind, ckpt = parse_run(run)
        m = r["at_iou50"][0 if rule == "fixed" else 1]
        rows.append(dict(regime=regime, kind=kind, ckpt=ckpt, model=model,
                         Modelo=LABELS.get(model, model) + ("*" if regime == "lowres" and kind == "finetune" else ""),
                         Resolución=r["resolution"], Mosaicos=r["name"], n_gt=r["n_gt"], thr=m["score_thr"],
                         RF1=m["RF1"], RP=m["RP"], RR=m["RR"], AP50=r["AP50"]))
    return pd.DataFrame(rows)


def with_totals(df):
    out = []
    for _, g in df.groupby(["Resolución", "Modelo"], sort=False):
        out.append(g)
        if len(g) > 1:
            w = g.n_gt / g.n_gt.sum()
            tot = g.iloc[0].to_dict() | {"Mosaicos": "TOTAL", "n_gt": g.n_gt.sum(), "thr": None}
            tot |= {k: round(float((g[k] * w).sum()), 3) for k in METRICS}
            out.append(pd.DataFrame([tot]))
    return pd.concat(out, ignore_index=True) if out else df


def to_markdown(df, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


TEX_NAMES = {
    "yolo11s": "YOLOv11s", "yolov8s": "YOLOv8s", "yolo11m": "YOLOv11m", "rtdetr_l": "RT-DETR-L",
    "rtdetr_r18": "RT-DETR-R18", "frcnn_mnv3": "Faster R-CNN (MobileNetV3)", "frcnn_r50": "Faster R-CNN (ResNet-50)",
    "deepforest": "DeepForest",
}
BLOCKS = [  # (mosaic key, label in the Resolución column, label in the Mosaicos column, regimes shown)
    ("dic1cm", r"\textbf{Alta} \\ (GSD $\approx$ 1 cm)", "Diciembre", ["mixed"]),
    ("dic3cm", r"\textbf{Baja} \\ (GSD $\approx$ 3 cm)", r"Diciembre \\ GSD 3.6 cm", ["mixed", "lowres"]),
]


def latex_table(eval_root, rule="fixed", ckpt="last"):
    """Fine-tuned results in the layout of the paper's Cuadro 4, plus parameters and throughput."""
    i = 0 if rule == "fixed" else 1
    bench = {}
    for f in Path(eval_root).glob("*/benchmark.json"):
        bench |= json.loads(f.read_text())

    def metrics(regime, model, mosaic):
        f = Path(eval_root) / regime / f"{model}_{ckpt}" / f"{mosaic}_metrics.json"
        if not f.exists():
            return None
        m = json.loads(f.read_text())["at_iou50"][i]
        return [m["RF1"], m["RP"], m["RR"]]

    def mr(n, text):
        text = rf"\shortstack{{{text}}}" if "\\" in text else text
        return rf"\multirow{{{n}}}{{*}}{{{text}}}" if n > 1 else text

    lines = [r"\begin{table}[ht]", r"\centering", r"\small", r"\renewcommand{\arraystretch}{1.3}",
             r"\begin{tabular}{|l|l|l|c|c|c|c|c|}", r"\hline",
             r"\textbf{Resolución} & \textbf{Modelo} & \textbf{Mosaicos} & \textbf{RF1} & \textbf{RP} & \textbf{RR}"
             r" & \textbf{Parám. (M)} & \textbf{Teselas/s} \\", r"\hline"]
    for mosaic, res_label, mos_label, regimes in BLOCKS:
        rows = [(model, regime, metrics(regime, model, mosaic)) for model in TEX_NAMES for regime in regimes]
        best = [max((r[2][k] for r in rows if r[2]), default=None) for k in range(3)]
        n = len(rows)
        for j, (model, regime, m) in enumerate(rows):
            first_of_model = regime == regimes[0]
            name = TEX_NAMES[model] + ("*" if regime == "lowres" else "")
            cells = ["--"] * 3 if m is None else [
                rf"\textbf{{{v:.3f}}}" if v == best[k] else f"{v:.3f}" for k, v in enumerate(m)]
            b = bench.get(model)
            span = len(regimes)
            params = (mr(span, f"{b['params_M']:.1f}") if b else "--") if first_of_model else ""
            tps = (mr(span, f"{1000 / b['ms_fp32']:.0f}") if b else "--") if first_of_model else ""
            lead = [mr(n, res_label), name, mr(n, mos_label)] if j == 0 else ["", name, ""]
            lines.append(" & ".join(lead + cells + [params, tps]) + r" \\")
            last_of_model = regime == regimes[-1]
            if j == n - 1:
                lines.append(r"\hline")
            elif last_of_model:
                lines.append(r"\cline{2-2} \cline{4-8}")
            else:
                lines.append(r"\cline{2-2} \cline{4-6}")
    lines += [r"\end{tabular}",
              r"\caption{Ablación de modelos de detección de frailejones. Todos los modelos se ajustaron con los mismos"
              r" datos, aumentaciones y número de iteraciones (22\,400 iteraciones, lote de 16, teselas de 512\,px) y se"
              r" reporta el checkpoint final. RF1, RP y RR se calculan a nivel de mosaico con el protocolo del Cuadro~4"
              r" (IoU $\geq$ 0.5, puntaje $\geq$ 0.5). * simboliza que el conjunto de datos de entrenamiento contenía"
              r" únicamente imágenes de baja resolución. Parám.: millones de parámetros. Teselas/s: teselas de"
              r" 512$\times$512\,px procesadas por segundo en una NVIDIA T4 (lote 1, FP32, incluye pre y"
              r" posprocesamiento). YOLOv11s y RT-DETR-L corresponden a YOLOv11 y RT-DETR del Cuadro~4.}",
              r"\label{tab:ablacion_deteccion}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/eval")
    ap.add_argument("--rule", choices=["fixed", "val"], default="fixed",
                    help="fixed = score 0.5 (paper protocol); val = F1-optimal confidence on validation tiles")
    ap.add_argument("--ckpt", choices=["last", "best"], default="last")
    args = ap.parse_args()

    df = load(args.eval_root, args.rule)
    if df.empty:
        raise SystemExit("no results yet")
    df["order"] = df.model.map({m: i for i, m in enumerate(ORDER)}).fillna(len(ORDER))
    df = df.sort_values(["Resolución", "regime", "order", "Mosaicos"])
    cols = ["Resolución", "Modelo", "Mosaicos", "RF1", "RP", "RR", "AP50", "thr"]

    bench = {}
    for f in Path(args.eval_root).glob("*/benchmark.json"):
        bench |= json.loads(f.read_text())
    sections = []
    for title, sel in [(f"Ajuste fino (22,400 iteraciones, checkpoint {args.ckpt})",
                        (df.kind == "finetune") & (df.ckpt == args.ckpt)),
                       ("Inferencia sin ajuste fino (zero-shot)", df.kind.isin(["zeroshot", "paper"]))]:
        part = with_totals(df[sel])
        if not part.empty:
            sections.append(f"## {title}\n\n{to_markdown(part, cols)}")
    if bench:
        b = pd.DataFrame([dict(Modelo=LABELS.get(k, k), **v) for k, v in bench.items()])
        sections.append("## Costo de inferencia (tile 512 px, Tesla T4, batch 1)\n\n"
                        + to_markdown(b, ["Modelo", "params_M", "gflops_512", "ms_fp16", "ms_fp32"]))

    text = (f"# Ablación de detección — umbral: {'0.5 fijo' if args.rule == 'fixed' else 'óptimo F1 en validación'}\n\n"
            + "\n\n".join(sections) + "\n")
    stem = Path(args.eval_root) / f"report_{args.rule}_{args.ckpt}"
    stem.with_suffix(".md").write_text(text)
    stem.with_suffix(".tex").write_text(latex_table(args.eval_root, args.rule, args.ckpt))
    df.drop(columns="order").to_csv(stem.with_suffix(".csv"), index=False)
    print(text)


if __name__ == "__main__":
    main()
