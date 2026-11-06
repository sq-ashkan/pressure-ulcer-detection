"""Thesis figures (vector PDF) and LaTeX number macros, generated from results/.

Every number printed in the thesis comes from a macro written here, so text and
results cannot drift apart (thesis rule: no hand-copied numbers).

  python scripts/make_figures.py      -> docs/thesis/figures/*.pdf, docs/thesis/generated/*.tex
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "docs" / "thesis" / "figures"
GEN = ROOT / "docs" / "thesis" / "generated"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
LABEL = {"unet": "U-Net", "yoloseg": "YOLOv8-seg"}
COL = {"unet": OKABE[0], "yoloseg": OKABE[1]}
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "figure.dpi": 150,
})
W = 5.9   # text width in inches (A4, 3 cm + 2.5 cm margins)

macros: dict[str, str] = {}


def mac(name, value, fmt="{:.3f}"):
    macros[name] = fmt.format(value) if isinstance(value, (float, np.floating)) else str(value)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def per_image(model, split):
    p = RES / "comparison" / f"{model}_{split}_per_image.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def fig_dice_distribution():
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.4))
    data = []
    for i, m in enumerate(("unet", "yoloseg")):
        d = [float(r["dice"]) for r in per_image(m, "test") if r["gt_positive"] == "True"]
        data.append(d)
        x = np.random.default_rng(i).normal(i + 1, 0.06, len(d))
        ax.scatter(x, d, s=6, color=COL[m], alpha=0.45, linewidths=0)
    bp = ax.boxplot(data, widths=0.35, showfliers=False, medianprops={"color": "black"})
    ax.set_xticks([1, 2], [LABEL["unet"], LABEL["yoloseg"]])
    ax.set_ylabel("per-image Dice (test, ulcer present)")
    ax.set_ylim(-0.02, 1.02)
    save(fig, "comparison_dice_distribution")


def fig_stage_dice(summary):
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.4))
    stages = ["stage1", "stage2", "stage3", "stage4"]
    for i, m in enumerate(("unet", "yoloseg")):
        v = [summary[m]["test"]["stage_dice"][s] for s in stages]
        ax.bar(np.arange(4) + (i - 0.5) * 0.36, v, 0.36, color=COL[m], label=LABEL[m])
    ax.set_xticks(range(4), ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
    ax.set_ylabel("pooled Dice (test)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save(fig, "comparison_stage_dice")


def fig_false_positives(evals):
    order = [("BG", "background"), ("N", "healthy skin"), ("D", "diabetic"), ("V", "venous"), ("S", "surgical")]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for i, m in enumerate(("unet", "yoloseg")):
        e = evals[m]["extra_negatives"]
        v = [e[k]["false_positive_rate"] for k, _ in order]
        lo = [e[k]["false_positive_rate"] - e[k]["ci95"][0] for k, _ in order]
        hi = [e[k]["ci95"][1] - e[k]["false_positive_rate"] for k, _ in order]
        ax.bar(np.arange(len(order)) + (i - 0.5) * 0.36, v, 0.36, yerr=[lo, hi], capsize=2,
               color=COL[m], label=LABEL[m], error_kw={"elinewidth": 0.8})
    ax.set_xticks(range(len(order)), [f"{t}\n(n = {evals['unet']['extra_negatives'][k]['n']})" for k, t in order])
    ax.set_ylabel("false-positive rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "comparison_false_positives_by_wound_type")


def fig_threshold_curves(evals):
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.3))
    for m in ("unet", "yoloseg"):
        t = evals[m]["val_threshold_table"]
        xs = sorted(float(k) for k in t)
        ax.plot(xs, [t[str(x)] if str(x) in t else t[f"{x}"] for x in xs], color=COL[m], label=LABEL[m])
        ax.axvline(evals[m]["threshold"], color=COL[m], ls=":", lw=0.9)
    ax.set_xlabel("operating threshold")
    ax.set_ylabel("mean per-image Dice (val)")
    ax.legend(frameon=False)
    save(fig, "threshold_selection_val")


def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2))
    h = list(csv.DictReader(open(ROOT / "runs" / "compare" / "unet" / "history.csv")))
    axes[0].plot([int(r["epoch"]) for r in h], [float(r["val_dice"]) for r in h], color=COL["unet"])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val binary Dice (thr. 0.5)"); axes[0].set_title("U-Net")
    sel = json.load(open(ROOT / "runs" / "compare" / "yoloseg" / "selection.json"))
    pts = sorted((int(k[5:-3]), v) for k, v in sel["scores"].items())
    axes[1].plot([p[0] + 1 for p in pts], [p[1] for p in pts], color=COL["yoloseg"])
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val binary Dice (conf. 0.25)"); axes[1].set_title("YOLOv8-seg")
    for a in axes:
        a.set_ylim(0, 1)
    save(fig, "training_curves")


def dataset_macros():
    st = json.load(open(RES / "seg_dataset_stats.json"))
    mac("dsPhotographs", st["photographs"], "{}")
    mac("dsMerges", st["perceptual_merges"], "{}")
    for sp in ("train", "val", "test"):
        mac(f"dsPhotos{sp.capitalize()}", st["photographs_per_split"].get(sp, 0), "{}")
        mac(f"dsImages{sp.capitalize()}", st["images_per_split"].get(sp, 0), "{}")
    mac("dsAuditTwins", st["eval_with_train_twin"], "{}")
    mac("dsExtraNegatives", sum(st["extra_negatives"].values()), "{}")


def comparison_macros(summary, evals):
    for m, p in (("unet", "Unet"), ("yoloseg", "Yolo")):
        t = summary[m]["test"]
        mac(f"cmp{p}Dice", t["dice_mean"]); mac(f"cmp{p}DiceLo", t["dice_ci95"][0]); mac(f"cmp{p}DiceHi", t["dice_ci95"][1])
        mac(f"cmp{p}IoU", t["iou_mean"]); mac(f"cmp{p}Sens", t["sensitivity"]); mac(f"cmp{p}FPR", t["false_positive_rate"])
        mac(f"cmp{p}StageMacro", t["stage_dice_macro"]); mac(f"cmp{p}StageAcc", t["dominant_stage_accuracy"])
        mac(f"cmp{p}Thr", summary[m]["threshold"], "{:.2f}")
        mac(f"cmp{p}Ms", summary[m]["inference_ms_per_image"], "{:.1f}")
        mac(f"cmp{p}ExtraFPR", evals[m]["extra_negatives"]["all"]["false_positive_rate"])
        mac(f"cmp{p}NPos", t["n_positive"], "{}"); mac(f"cmp{p}NNeg", t["n_negative"], "{}")
    d = summary["paired_test"]
    mac("cmpDiff", d["mean_diff"]); mac("cmpDiffLo", d["ci95"][0]); mac("cmpDiffHi", d["ci95"][1])
    mac("cmpP", d["p_value"]); mac("cmpN", d["n"], "{}")
    mac("cmpWinner", LABEL[summary["winner"]], "{}")


def growth():
    reg = json.load(open(ROOT / "system_ws" / "registry.json"))
    vs = reg["versions"]
    photos = [v["train_photographs"] for v in vs]
    dice = [v["metrics"]["test_dice_mean"] for v in vs]
    lo = [v["metrics"]["test_dice_ci95"][0] for v in vs]
    hi = [v["metrics"]["test_dice_ci95"][1] for v in vs]
    prom = [bool(v["gate"].get("promoted")) for v in vs]
    fig, ax = plt.subplots(figsize=(W * 0.7, 2.5))
    ax.errorbar(photos, dice, yerr=[np.subtract(dice, lo), np.subtract(hi, dice)], color=OKABE[0],
                fmt="-", capsize=3, lw=1)
    for x, y, pr, v in zip(photos, dice, prom, vs):
        ax.scatter([x], [y], s=36, zorder=3, color=OKABE[0] if pr else "white", edgecolors=OKABE[0])
        ax.annotate(f"v{v['version']}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel("training photographs (independent)")
    ax.set_ylabel("test Dice (frozen test, 95 % CI)")
    ax.set_ylim(0.6, 0.9)
    save(fig, "growth_learning_curve")
    for v in vs:
        n = v["version"]; m = v["metrics"]
        mac(f"gV{chr(65 + n)}Photos", v["train_photographs"], "{}")
        mac(f"gV{chr(65 + n)}Dice", m["test_dice_mean"]); mac(f"gV{chr(65 + n)}Lo", m["test_dice_ci95"][0])
        mac(f"gV{chr(65 + n)}Hi", m["test_dice_ci95"][1])
        if v["gate"].get("ci95"):
            mac(f"gV{chr(65 + n)}Diff", v["gate"]["mean_diff"], "{:+.3f}")
            mac(f"gV{chr(65 + n)}DiffLo", v["gate"]["ci95"][0], "{:+.3f}")
            mac(f"gV{chr(65 + n)}DiffHi", v["gate"]["ci95"][1], "{:+.3f}")
    mac("gCurrent", f"v{reg['current']}", "{}")


def piid():
    for m, p in (("unet", "Unet"), ("yoloseg", "Yolo")):
        r = json.load(open(RES / "external" / f"external_piid_{m}.json"))
        mac(f"piid{p}N", r["external_n"], "{}"); mac(f"piid{p}Removed", r["overlap_removed"], "{}")
        mac(f"piid{p}Sens", r["sensitivity"]); mac(f"piid{p}SensLo", r["sensitivity_ci95"][0])
        mac(f"piid{p}SensHi", r["sensitivity_ci95"][1]); mac(f"piid{p}StageAcc", r["stage_accuracy_detected"])
        mac(f"piid{p}Kappa", r["quadratic_kappa"])
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4))
    for ax, m in zip(axes, ("unet", "yoloseg")):
        cm = np.array(json.load(open(RES / "external" / f"external_piid_{m}.json"))["confusion_rows_piid_stage_cols_pred0to4"])
        rown = cm / cm.sum(1, keepdims=True)
        ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if rown[i, j] > 0.55 else "black")
        ax.set_xticks(range(5), ["none", "1", "2", "3", "4"]); ax.set_yticks(range(4), ["1", "2", "3", "4"])
        ax.set_xlabel("predicted stage"); ax.set_ylabel("PIID stage"); ax.set_title(LABEL[m])
    save(fig, "external_piid_confusion")


def agreement():
    a = json.load(open(RES / "annotator_agreement.json"))
    mac("agrN", a["matched_photographs"], "{}"); mac("agrDice", a["binary_dice_mean"])
    mac("agrDiceLo", a["binary_dice_ci95"][0]); mac("agrDiceHi", a["binary_dice_ci95"][1])
    mac("agrDiceMedian", a["binary_dice_median"]); mac("agrStagePairs", a["stage_pairs"], "{}")
    mac("agrStageAgree", a["dominant_stage_agreement"])


def seed1():
    s = json.load(open(RES / "comparison_seed1" / "summary.json"))
    d = s["paired_test"]
    mac("sOneDiff", d["mean_diff"]); mac("sOneDiffLo", d["ci95"][0]); mac("sOneDiffHi", d["ci95"][1])
    mac("sOneP", d["p_value"]); mac("sOneUnetDice", s["unet"]["test"]["dice_mean"])
    mac("sOneYoloDice", s["yoloseg"]["test"]["dice_mean"])


def write_macros():
    GEN.mkdir(parents=True, exist_ok=True)
    lines = ["% generated by scripts/make_figures.py -- do not edit"]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (GEN / "macros.tex").write_text("\n".join(lines) + "\n")


def main():
    summary = json.load(open(RES / "comparison" / "summary.json"))
    evals = {m: json.load(open(RES / "comparison" / f"{m}_evaluation.json")) for m in ("unet", "yoloseg")}
    fig_dice_distribution(); fig_stage_dice(summary); fig_false_positives(evals)
    fig_threshold_curves(evals); fig_training_curves()
    dataset_macros(); comparison_macros(summary, evals)
    growth(); piid(); agreement(); seed1()
    write_macros()
    print(f"{len(list(FIG.glob('*.pdf')))} figures, {len(macros)} macros")


if __name__ == "__main__":
    main()
