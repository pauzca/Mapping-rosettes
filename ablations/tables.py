"""LaTeX tables of the detection ablation in the layout of the paper's Cuadro 4.

Table 1 (main): test tiles, P / R / F1 per source at each model's validation-selected threshold.
  Alta: Agosto, Diciembre (1 cm), Junio 2025 and a TOTAL weighted by the number of plants per source
  (as in the paper); Baja: Diciembre 3.6 cm for every model and its low-resolution (*) version.
Table 2 (complementary): mosaic-level RP / RR / RF1 on the Diciembre 1 cm and 3.6 cm orthomosaics.
Both add parameters and throughput (512 px tiles per second, T4, batch 1, FP32).

Example:
    python -m ablations.tables --eval-root outputs/eval_v2 --bench outputs/eval/mixed/benchmark.json
"""
import argparse
import json
from pathlib import Path

NAMES = {
    "yolo11s": "YOLOv11s", "yolov8s": "YOLOv8s", "yolo11m": "YOLOv11m", "rtdetr_l": "RT-DETR-L",
    "rtdetr_r18": "RT-DETR-R18", "frcnn_mnv3": "Faster R-CNN (MobileNetV3)", "frcnn_r50": "Faster R-CNN (ResNet-50)",
    "deepforest": "DeepForest",
}
ALTA = [("agosto", "Agosto"), ("dic1cm", "Diciembre"), ("junio2025", "Junio 2025")]
RES_ALTA = r"\textbf{Alta} \\ (GSD $\approx$ 1 cm)"
RES_BAJA = r"\textbf{Baja} \\ (GSD $\approx$ 3 cm)"
HEAD = [r"\begin{table}[ht]", r"\centering", r"\small", r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{|l|l|l|c|c|c|c|c|}", r"\hline"]


def mr(n, text):
    text = rf"\shortstack{{{text}}}" if "\\\\" in text else text
    return rf"\multirow{{{n}}}{{*}}{{{text}}}" if n > 1 else text


def fmt(vals, best):
    return [rf"\textbf{{{v:.3f}}}" if best and abs(v - b) < 1e-9 else f"{v:.3f}" for v, b in zip(vals, best or vals)]


class Results:
    def __init__(self, root, bench):
        self.root, self.bench = Path(root), json.loads(Path(bench).read_text())

    def tiles(self, regime, model):
        f = self.root / regime / model / "test_tiles.json"
        return json.loads(f.read_text())["prf"] if f.exists() else None

    def mosaic(self, regime, model, key):
        f = self.root / regime / model / f"{key}_metrics.json"
        if not f.exists():
            return None
        m = json.loads(f.read_text())["at_iou50"][0]
        return [m["RF1"], m["RP"], m["RR"]]

    def cost(self, model, span):
        b = self.bench.get(model)
        return [mr(span, f"{b['params_M']:.1f}"), mr(span, f"{1000 / b['ms_fp32']:.0f}")] if b else ["--", "--"]


def tile_table(R):
    lines = HEAD + [r"\textbf{Resolución} & \textbf{Modelo} & \textbf{Mosaicos} & \textbf{F1} & \textbf{P} & \textbf{R}"
                    r" & \textbf{Parám. (M)} & \textbf{Teselas/s} \\", r"\hline"]
    # Alta: per source + TOTAL weighted by plants; bold the best TOTAL
    alta = {}
    for model in NAMES:
        t = R.tiles("mixed", model)
        if t is None:
            continue
        rows = [[t[s]["F1"], t[s]["P"], t[s]["R"]] for s, _ in ALTA]
        w = [t[s]["n_gt"] for s, _ in ALTA]
        rows.append([sum(r[k] * n for r, n in zip(rows, w)) / sum(w) for k in range(3)])
        alta[model] = rows
    best_total = [max(r[3][k] for r in alta.values()) for k in range(3)]
    n_rows = 4 * len(alta)
    for i, (model, rows) in enumerate(alta.items()):
        for j, (vals, label) in enumerate(zip(rows, [l for _, l in ALTA] + ["TOTAL"])):
            lead = mr(n_rows, RES_ALTA) if i == 0 and j == 0 else ""
            name = mr(4, NAMES[model]) if j == 0 else ""
            cells = fmt(vals, best_total if j == 3 else None)
            cost = R.cost(model, 4) if j == 0 else ["", ""]
            lines.append(" & ".join([lead, name, label] + cells + cost) + r" \\")
            lines.append(r"\cline{3-6}" if j < 3 else (r"\cline{2-8}" if i < len(alta) - 1 else r"\hline"))
    lines += baja_rows(R, lambda reg, m: (lambda t: t and [t["dic3cm"]["F1"], t["dic3cm"]["P"], t["dic3cm"]["R"]])(
        R.tiles(reg, m)), r"Diciembre \\ GSD 3.6 cm")
    lines += [r"\end{tabular}",
              r"\caption{Ablación de modelos de detección de frailejones a nivel de tesela en el conjunto de prueba."
              r" Todos los modelos se ajustaron con los mismos datos, aumentaciones y número de iteraciones (14\,000"
              r" iteraciones, lote de 16, teselas de 512\,px); se usa el checkpoint con mejor mAP50-95 en validación y"
              r" el umbral de puntaje que maximiza F1 en validación. P, R y F1 con IoU $\geq$ 0.5. El total se calcula"
              r" ponderando las métricas de cada mosaico por su número de frailejones. * simboliza que el conjunto de"
              r" datos de entrenamiento contenía únicamente imágenes de baja resolución. Parám.: millones de parámetros."
              r" Teselas/s: teselas de 512$\times$512\,px procesadas por segundo en una NVIDIA T4 (lote 1, FP32, incluye"
              r" pre y posprocesamiento).}",
              r"\label{tab:ablacion_deteccion_teselas}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def baja_rows(R, get, mos_label):
    """Baja block: every model and its * version on Diciembre 3.6 cm; bold the best of the block."""
    rows = [(m, reg, get(reg, m)) for m in NAMES for reg in ("mixed", "lowres")]
    best = [max(r[2][k] for r in rows if r[2]) for k in range(3)]
    out = []
    for j, (model, reg, vals) in enumerate(rows):
        lead = [mr(len(rows), RES_BAJA), "", mr(len(rows), mos_label)] if j == 0 else ["", "", ""]
        lead[1] = NAMES[model] + ("*" if reg == "lowres" else "")
        cells = fmt(vals, best) if vals else ["--"] * 3
        cost = R.cost(model, 2) if reg == "mixed" else ["", ""]
        out.append(" & ".join(lead + cells + cost) + r" \\")
        out.append(r"\cline{2-2} \cline{4-6}" if reg == "mixed" else (r"\cline{2-2} \cline{4-8}" if j < len(rows) - 1 else r"\hline"))
    return out


def mosaic_table(R):
    lines = HEAD + [r"\textbf{Resolución} & \textbf{Modelo} & \textbf{Mosaicos} & \textbf{RF1} & \textbf{RP} & \textbf{RR}"
                    r" & \textbf{Parám. (M)} & \textbf{Teselas/s} \\", r"\hline"]
    alta = [(m, R.mosaic("mixed", m, "dic1cm")) for m in NAMES]
    best = [max(v[k] for _, v in alta if v) for k in range(3)]
    for i, (model, vals) in enumerate(alta):
        lead = [mr(len(alta), RES_ALTA), NAMES[model], mr(len(alta), "Diciembre")] if i == 0 else ["", NAMES[model], ""]
        lines.append(" & ".join(lead + (fmt(vals, best) if vals else ["--"] * 3) + R.cost(model, 1)) + r" \\")
        lines.append(r"\cline{2-2} \cline{4-8}" if i < len(alta) - 1 else r"\hline")
    lines += baja_rows(R, lambda reg, m: R.mosaic(reg, m, "dic3cm"), r"Diciembre \\ GSD 3.6 cm")
    lines += [r"\end{tabular}",
              r"\caption{Ablación de modelos de detección de frailejones a nivel de mosaico (protocolo del Cuadro~4:"
              r" teselas a 1.36\,cm/px con 20\% de solapamiento, NMS global con IoU 0.2 e IoU $\geq$ 0.5 para el"
              r" emparejamiento), con el mismo checkpoint y umbral de puntaje seleccionados en validación que en el"
              r" cuadro a nivel de tesela. * simboliza que el conjunto de datos de entrenamiento contenía únicamente"
              r" imágenes de baja resolución. Parám.: millones de parámetros. Teselas/s: teselas de 512$\times$512\,px"
              r" procesadas por segundo en una NVIDIA T4 (lote 1, FP32).}",
              r"\label{tab:ablacion_deteccion_mosaicos}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/eval_v2")
    ap.add_argument("--bench", default="outputs/eval/mixed/benchmark.json")
    args = ap.parse_args()
    R = Results(args.eval_root, args.bench)
    for name, tex in (("tabla_teselas", tile_table(R)), ("tabla_mosaicos", mosaic_table(R))):
        (Path(args.eval_root) / f"{name}.tex").write_text(tex)
        print(f"wrote {Path(args.eval_root) / name}.tex")


if __name__ == "__main__":
    main()
