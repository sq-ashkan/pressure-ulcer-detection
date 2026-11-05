"""
Prototype interface for the pressure ulcer detector.

Two ways in: upload a photograph, or use the webcam. Both run the same model and
render the same summary, so what the demo shows is what the detector reports.

Usage:
    python app/main.py
    python app/main.py --weights runs/detect/pu_yolov8s_v4/weights/best.pt --share
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gradio as gr
import numpy as np

from detector import (
    DEFAULT_WEIGHTS,
    STAGE_DESCRIPTIONS,
    STAGE_LABELS,
    PressureUlcerDetector,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DISCLAIMER = """
### Research prototype

This tool was built for a bachelor thesis within the NaMeKI research project at
HAWK. It is **not a medical device** and must not be used to make clinical
decisions. Its predictions come from a model trained on a small set of public
photographs and have not been validated on clinical data.
"""

STAGE_REFERENCE = "\n".join(
    f"- **{STAGE_LABELS[i]}** — {STAGE_DESCRIPTIONS[i]}" for i in sorted(STAGE_LABELS)
)

CSS = """
.gradio-container { max-width: 1180px !important; }
footer { display: none !important; }
"""


def build_ui(detector: PressureUlcerDetector, model_info: str) -> gr.Blocks:

    def run(image: np.ndarray | None, conf: float, iou: float):
        if image is None:
            return None, "Upload a photograph or take one with the camera.", None
        detections, elapsed = detector.predict(image, conf=conf, iou=iou)
        annotated = detector.draw(image, detections)
        summary = detector.describe(detections, elapsed)
        payload = {
            "count": len(detections),
            "inference_ms": round(elapsed, 1),
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "detections": [d.as_dict() for d in detections],
        }
        return annotated, summary, json.dumps(payload, indent=2)

    with gr.Blocks(title="Pressure Ulcer Detection", css=CSS,
                   theme=gr.themes.Soft(primary_hue="red")) as demo:
        gr.Markdown("# Pressure Ulcer Detection\n"
                    "Automatic detection and NPUAP/EPUAP staging of pressure ulcers "
                    "in photographs.")

        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Image(type="numpy", label="Photograph", height=380,
                                  sources=["upload", "webcam", "clipboard"])
                with gr.Accordion("Detection settings", open=False):
                    conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                     label="Confidence threshold",
                                     info="Lower finds more, with more false positives.")
                    iou = gr.Slider(0.10, 0.90, value=0.45, step=0.05,
                                    label="Overlap threshold (NMS IoU)",
                                    info="How much two boxes may overlap before "
                                         "one is suppressed.")
                run_btn = gr.Button("Detect", variant="primary")
                clear_btn = gr.ClearButton(value="Clear")

            with gr.Column(scale=1):
                out_image = gr.Image(label="Detections", height=380)
                out_summary = gr.Markdown()

        with gr.Accordion("Raw output (JSON)", open=False):
            out_json = gr.Code(language="json", label="")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Staging reference\n" + STAGE_REFERENCE)
            with gr.Column():
                gr.Markdown("### Model\n" + model_info)

        gr.Markdown(DISCLAIMER)

        sample = stage_examples()
        if sample:
            gr.Examples(examples=[[str(p)] for p in sample], inputs=[source],
                        label="Examples from the held-out test split")

        run_btn.click(run, [source, conf, iou], [out_image, out_summary, out_json])
        source.change(run, [source, conf, iou], [out_image, out_summary, out_json])
        clear_btn.add([source, out_image, out_summary, out_json])

    return demo


EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def stage_examples(limit: int = 8) -> list[Path]:
    """Copy a few held-out test photographs next to the app.

    Gradio only serves files from the working directory or the system temp
    directory, so the dataset cannot be referenced in place.
    """
    source_dir = PROJECT_ROOT / "data" / "unified" / "images" / "test"
    if not source_dir.exists():
        return []
    EXAMPLES_DIR.mkdir(exist_ok=True)
    for stale in EXAMPLES_DIR.iterdir():
        stale.unlink()

    # Spread the sample across the split rather than taking the first few, which
    # would all come from whichever source sorts first.
    candidates = sorted(p for p in source_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not candidates:
        return []
    step = max(1, len(candidates) // limit)
    picked = candidates[::step][:limit]

    staged = []
    for i, src in enumerate(picked, 1):
        dest = EXAMPLES_DIR / f"example_{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        staged.append(dest)
    return staged


def load_model_info(weights: Path) -> str:
    run_dir = weights.parent.parent
    lines = [f"- Architecture: YOLOv8s", f"- Weights: `{weights.name}`"]
    ev = PROJECT_ROOT / "results" / "evaluation_test.json"
    if ev.exists():
        try:
            data = json.loads(ev.read_text())
            head = data.get("headline", {})
            if "mAP50" in head:
                ci = head.get("mAP50_ci95")
                ci_txt = f" (95% CI {ci[0]:.3f}–{ci[1]:.3f})" if ci else ""
                lines.append(f"- Test mAP@50: {head['mAP50']:.3f}{ci_txt}")
            if "mAP50_95" in head:
                lines.append(f"- Test mAP@50-95: {head['mAP50_95']:.3f}")
            lines.append(f"- Held-out test photographs: {data.get('images', 'n/a')}")
        except Exception:
            pass
    cfg = run_dir / "run_config.json"
    if cfg.exists():
        try:
            counts = json.loads(cfg.read_text()).get("dataset", {}).get("counts", {})
            if counts:
                lines.append(f"- Training images: {counts.get('train', 'n/a')}")
        except Exception:
            pass
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="Expose a temporary public URL via Gradio")
    args = ap.parse_args()

    weights = Path(args.weights)
    detector = PressureUlcerDetector(weights)
    print(f"loaded {weights} on {detector.device}")

    demo = build_ui(detector, load_model_info(weights))
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, inbrowser=False,
                allowed_paths=[str(EXAMPLES_DIR)])


if __name__ == "__main__":
    main()
