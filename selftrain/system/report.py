"""Cycle report (SPEC §3 step 9, feature F8): one HTML and one PDF document per cycle.

Content, in this order
----------------------
1. Header: cycle number, date, the model version trained in the cycle (the
   *candidate*), the version it was compared with (the *baseline*, i.e. the model in
   service before the cycle) and the version in service after the cycle.
2. Summary box: the gate decision in one sentence (`gate_sentence`) and key numbers.
3. Cycle steps: status and one-line summary of every step that ran before the report.
4. Data: per source the labelled images received, rejected by quality control and
   added to the pool (train/val, distinct photographs, licence). The added images are
   read from the pool manifest (rows with ``added_cycle`` = cycle), i.e. what was
   actually written, not what a step intended to write; pool totals after the cycle.
5. Quality control: rejections grouped by reason category (`reason_category`).
6. Duplicates: images whose photograph is already in the pool or occurs twice in the
   cycle (dedupe), with the matched photograph, its split and the action taken.
7. Drift: per image feature the two-sample Kolmogorov-Smirnov statistic D, its
   p-value, the Bonferroni-adjusted p-value min(1, k p) over the k features, and the
   flag "shifted" (adjusted p < drift_alpha) -- a warning, never a blocker.
8. Metrics of candidate and baseline on the frozen test split with 95 % confidence
   intervals (percentile bootstrap for Dice/IoU, Wilson score for rates) and the gate's
   paired-bootstrap comparison.
9. Figures (below), notes, and as appendix the candidate's model card (F12).

Sources: persistent state is read from the workspace -- registry.json (versions,
gate decisions, metrics), models/v{N}/evaluation.json with its per-image CSVs,
pool/manifest.csv and models/v{N}/model_card.md. The transient results of the cycle
(step summaries, rejections, duplicates, drift) come from `context_data`, whose
schema is documented in `build_report`; `context_from_cycle` derives it from a
running cycle. The cycle stores that context in its summary reports/cycle_{N}.json
(`steps.cycle_summary`), and `write_report` renders a report from such a summary --
during the cycle and again later (``python -m selftrain report --version N``).

Figures
-------
matplotlib (object-oriented API, no pyplot state), one style for all figures
(`STYLE`: Okabe-Ito colour-blind-safe palette, no titles -- the captions carry the
text -- axis labels with units), each saved as vector PDF for LaTeX and as 200 dpi
PNG for HTML:

* version history: mean test Dice with its 95 % CI per registered version; promoted
  versions filled, not promoted hollow, the version in service ringed; distinct
  training photographs per version as grey bars on a secondary axis;
* per-stage Dice of the candidate next to the baseline (pooled over all pixels of
  the frozen test split, 2TP / (2TP + FP + FN) per stage);
* false-positive rate per wound type on the extra ulcer-free photographs and on the
  ulcer-free test photographs, with Wilson 95 % CIs;
* images added in the cycle per source (train/val) and rejections per reason;
* example overlays from the frozen test split (`select_examples`): best, median and
  worst per-image Dice among ulcer-containing images and the ulcer-free image with
  the largest predicted area. The prediction is recomputed with the evaluated
  weights at the evaluated operating threshold and the evaluation's post-processing
  (`selftrain.metrics.remove_specks`); ground truth is drawn as contour, the
  prediction as fill.

Typesetting
-----------
The PDF is rendered from templates/report.tex.j2 (jinja2 with the delimiters
``((* *))``, ``((( )))`` and ``((= =))``, so LaTeX braces need no escaping). Every
inserted value passes through `latex_escape` unless it is a `Latex` string built by
this module. The source is compiled with ``latexmk -pdf`` in a temporary directory
(no rc files, no shell escape, timeout 120 s). If that fails for any reason --
latexmk missing, a LaTeX error, the timeout -- a plain-layout PDF is written with
matplotlib's PdfPages instead, and the reason is returned, emitted as an event and
printed in both reports. The HTML report (templates/report.html.j2) is standalone:
inline CSS, figures embedded as base64 PNG, no external resources, for the dashboard.
"""
from __future__ import annotations

import base64
import csv
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import textwrap
import unicodedata
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import cv2
import jinja2
import markupsafe
import matplotlib
import matplotlib.image
import numpy as np
from matplotlib import cycler
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ..config import MIN_COMPONENT_FRAC, STAGE_NAMES
from . import STEP_NAMES

if TYPE_CHECKING:                      # type hints only: no import-time coupling
    from .registry import Registry, VersionRecord
    from .steps import CycleContext
    from .workspace import Workspace

__all__ = [
    "BarSeries", "DataRow", "DriftRow", "DriftSummary", "DuplicateRow", "Example",
    "ExampleSource", "FigureFile", "Latex", "MetricRow", "RateBar", "ReasonRow",
    "ReportData", "TableLayout", "VersionPoint", "build_report", "collect", "compile_latex",
    "context_from_cycle", "draw_data_added", "draw_examples", "draw_fpr",
    "draw_stage_dice", "draw_version_history", "duplicates_from_dedupe", "fmt_num",
    "fmt_pct", "fmt_signed", "gate_sentence", "latex_breakable", "latex_escape",
    "make_figures", "markdown_to_html", "markdown_to_latex", "normalise_drift",
    "normalise_metrics", "parse_markdown", "pdf_page_count", "reason_category",
    "rejection_rows", "render_html", "render_latex", "select_examples",
    "table_layout", "write_fallback_pdf", "write_report",
]

log = logging.getLogger(__name__)

STEP = "report"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
LATEX_TEMPLATE = "report.tex.j2"
HTML_TEMPLATE = "report.html.j2"
LATEX_TIMEOUT_S = 120.0
PNG_DPI = 200
MAX_LATEX_ROWS = 12            # rows of a long table (duplicates, rejections) in the PDF
MAX_HTML_ROWS = 500            # ... in the HTML report
MODEL_CARD_MAX_LINES = 400     # markdown lines of the model card typeset in the PDF appendix
TABLE_LINE_CHARS = 105         # characters of a \small table row at \linewidth (A4, 2 cm margins: ~110)
TABLE_COLSEP_CHARS = 3         # ... taken by the space between two columns (2 \tabcolsep)
TABLE_WRAP_MIN_CHARS = 12      # width reserved for a wrapping column when fitting a table
TABLE_WRAP_CAP_CHARS = 60      # longest-cell length beyond which a wrapping column gets no wider
TABLE_HEADER_SCALE = 1.1       # bold header text is about this much wider than body text
BREAK_CHUNK = 8                # a long letter/digit run may break after every BREAK_CHUNK characters
DISPLAY_MAX_SIDE = 640         # px, long side of a photograph in the overlay figure
TEXT_WIDTH_IN = 6.69           # A4 width minus 2 x 2 cm margins, in inches
HALF_WIDTH_IN = 3.3
A4_IN = (8.27, 11.69)
FALLBACK_LINE_CHARS = 100      # characters per line of a text page of the fallback PDF
FALLBACK_LINES_PER_PAGE = 68

DASH = "–"                # shown for a missing value
MINUS = "−"
NNBSP = " "               # narrow no-break space, between a number and "%"

# Okabe & Ito (2008), "Color universal design": distinguishable with all common
# forms of colour-vision deficiency.
OKABE_ITO: Mapping[str, str] = MappingProxyType({
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"})
GREY = "#8C8C8C"
LIGHT_GREY = "#D9D9D9"
INK = "#333333"
CANDIDATE_COLOR = OKABE_ITO["blue"]
BASELINE_COLOR = OKABE_ITO["orange"]
TRAIN_COLOR = OKABE_ITO["blue"]
VAL_COLOR = OKABE_ITO["sky_blue"]
REJECT_COLOR = OKABE_ITO["vermillion"]
IN_SERVICE_COLOR = OKABE_ITO["vermillion"]
PRED_COLOR = OKABE_ITO["blue"]         # prediction fill: blue/yellow stay distinct on skin
PRED_ALPHA = 0.45
GT_COLOR = OKABE_ITO["yellow"]         # ground-truth contour, drawn over a black halo

STYLE: Mapping[str, Any] = MappingProxyType({
    "font.family": "sans-serif", "font.sans-serif": "DejaVu Sans", "font.size": 8.0,
    "axes.labelsize": 8.0, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0, "axes.titlesize": 8.0,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler(color=[OKABE_ITO[k] for k in
                                     ("blue", "orange", "green", "vermillion", "purple",
                                      "sky_blue", "yellow", "black")]),
    "lines.linewidth": 1.2, "lines.markersize": 5.0, "errorbar.capsize": 2.5,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,                # embed TrueType: text stays text in the PDF
    "image.interpolation": "antialiased",
})

# Wound classes of the AZH wound dataset (source of the extra ulcer-free photographs).
WOUND_TYPES: Mapping[str, str] = MappingProxyType({
    "BG": "background", "N": "normal skin", "D": "diabetic ulcer",
    "S": "surgical wound", "V": "venous ulcer", "P": "pressure ulcer"})

# Names under which the cycle and the registry store metrics (steps.summary_metrics =
# registry.metrics_from_evaluation) -> names of the evaluation's test block, used here.
METRIC_ALIASES: Mapping[str, str] = MappingProxyType({
    "test_dice_mean": "dice_mean", "test_dice_ci95": "dice_ci95",
    "test_iou_mean": "iou_mean", "test_iou_ci95": "iou_ci95",
    "test_sensitivity": "sensitivity", "test_sensitivity_ci95": "sensitivity_ci95",
    "test_fpr": "false_positive_rate", "test_fpr_ci95": "false_positive_rate_ci95",
    "test_stage_dice": "stage_dice", "test_stage_dice_macro": "stage_dice_macro"})

# (label, kind, value key, CI key) of the metrics table; kind selects the format.
METRIC_TABLE: tuple[tuple[str, str, str, str | None], ...] = (
    ("Dice, ulcer images (mean, 95 % bootstrap CI)", "ratio", "dice_mean", "dice_ci95"),
    ("IoU, ulcer images (mean, 95 % bootstrap CI)", "ratio", "iou_mean", "iou_ci95"),
    ("Sensitivity, image level (Wilson 95 % CI)", "pct", "sensitivity", "sensitivity_ci95"),
    ("False-positive rate, ulcer-free test photographs (Wilson 95 % CI)", "pct",
     "false_positive_rate", "false_positive_rate_ci95"),
    ("False-positive rate, extra ulcer-free photographs (Wilson 95 % CI)", "pct",
     "extra_negatives_fpr", "extra_negatives_fpr_ci95"),
    ("Stage Dice, macro mean of stages 1–4 (pooled pixels)", "ratio",
     "stage_dice_macro", None),
    ("Dice, external split (mean, 95 % bootstrap CI)", "ratio", "external_dice_mean",
     "external_dice_ci95"),
    ("Operating threshold (chosen on validation)", "threshold", "threshold", None),
    ("Inference time (ms per image)", "ms", "inference_ms_per_image", None),
)

Emit = Callable[..., None]


# ============================================================== number formatting

def finite(value: Any) -> float | None:
    """`value` as float if it is a finite real number, else None (None, NaN, inf,
    booleans and text are 'missing')."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ci_pair(value: Any) -> tuple[float, float] | None:
    """A confidence interval given as a 2-sequence of finite numbers, else None."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        return None
    lo, hi = finite(value[0]), finite(value[1])
    return None if lo is None or hi is None else (lo, hi)


def _int_or_none(value: Any) -> int | None:
    number = finite(value)
    return None if number is None or number != int(number) else int(number)


def fmt_num(value: Any, digits: int = 3) -> str:
    """Fixed-point number, or an en dash when missing."""
    number = finite(value)
    return DASH if number is None else f"{number:.{digits}f}"


def fmt_signed(value: Any, digits: int = 3) -> str:
    """Number with explicit sign (+ or the minus sign U+2212), or an en dash."""
    number = finite(value)
    if number is None:
        return DASH
    return f"{MINUS if number < 0 else '+'}{abs(number):.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    """Proportion in [0, 1] as a percentage ('12.5 %'), or an en dash."""
    number = finite(value)
    return DASH if number is None else f"{100 * number:.{digits}f}{NNBSP}%"


def fmt_count(value: Any) -> str:
    """Integer count, or an en dash when missing."""
    number = _int_or_none(value)
    return DASH if number is None else str(number)


def fmt_interval(ci: Any, digits: int = 3, scale: float = 1.0, signed: bool = False) -> str:
    """Interval '[lo, hi]' (values multiplied by `scale`), or an en dash."""
    pair = ci_pair(ci)
    if pair is None:
        return DASH
    show = fmt_signed if signed else fmt_num
    return f"[{show(pair[0] * scale, digits)}, {show(pair[1] * scale, digits)}]"


def metric_text(value: Any, ci: Any, kind: str) -> str:
    """One table cell: the value with its CI, formatted by kind -- 'ratio' (0.812
    [0.780, 0.840]), 'pct' (12.5 % [8.1, 18.0]), 'threshold' (0.45) or 'ms' (35.2)."""
    if finite(value) is None:
        return DASH
    if kind == "pct":
        text = fmt_pct(value)
        return text if ci_pair(ci) is None else f"{text} {fmt_interval(ci, 1, 100.0)}"
    if kind == "threshold":
        return fmt_num(value, 2)
    if kind == "ms":
        return fmt_num(value, 1)
    text = fmt_num(value)
    return text if ci_pair(ci) is None else f"{text} {fmt_interval(ci)}"


def shorten(text: Any, width: int = 40) -> str:
    """`text` cut in the middle with an ellipsis if longer than `width` characters
    (file names keep their distinctive start and end)."""
    text = str(text)
    if len(text) <= width:
        return text
    keep = width - 1
    return f"{text[:(keep + 1) // 2]}…{text[len(text) - keep // 2:]}"


# =================================================================== LaTeX escaping

class Latex(str):
    """A string that already is LaTeX source: the template inserts it unescaped."""
    __slots__ = ()


_LATEX_CHARS: Mapping[str, str] = MappingProxyType({
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
    "#": r"\#", "^": r"\textasciicircum{}", "_": r"\_", "%": r"\%",
    "~": r"\textasciitilde{}", "<": r"\textless{}", ">": r"\textgreater{}",
    "|": r"\textbar{}", '"': r"\textquotedbl{}",
    " ": "~", "­": r"\-", " ": r"\,", " ": r"\,",
    "−": r"\ensuremath{-}", "–": r"\textendash{}", "—": r"\textemdash{}",
    "…": r"\ldots{}", "‘": "`", "’": "'", "“": "``", "”": "''",
    "•": r"\textbullet{}", "€": r"\texteuro{}",
    "≤": r"\ensuremath{\leq}", "≥": r"\ensuremath{\geq}",
    "≠": r"\ensuremath{\neq}", "≈": r"\ensuremath{\approx}",
    "→": r"\ensuremath{\rightarrow}", "←": r"\ensuremath{\leftarrow}",
    "Δ": r"\ensuremath{\Delta}", "α": r"\ensuremath{\alpha}",
    "β": r"\ensuremath{\beta}", "μ": r"\ensuremath{\mu}",
    "σ": r"\ensuremath{\sigma}", "✓": "yes", "✗": "no",
})


def latex_escape(text: Any) -> str:
    """Escape arbitrary text for LaTeX (pdfLaTeX, T1 font encoding, UTF-8 input).

    The ten LaTeX special characters and the T1 characters that would print wrongly
    (< > | ") become commands; a few frequent typographic and mathematical Unicode
    characters are mapped explicitly; ASCII and Latin-1 (U+00A0-U+00FF, which the
    LaTeX kernel defines for T1/TS1) pass through. Any other character is replaced by
    its ASCII compatibility decomposition (NFKD, combining marks dropped) or by '?',
    so no input can stop the compilation. Control characters become spaces and the
    ligature '--' is broken, so the output shows exactly the input characters."""
    out: list[str] = []
    text = str(text)
    for i, ch in enumerate(text):
        if ch in _LATEX_CHARS:
            out.append(_LATEX_CHARS[ch])
        elif ch == "-":
            out.append("-{}" if text[i + 1:i + 2] == "-" else "-")
        elif ord(ch) < 128:
            out.append(ch if ch.isprintable() else " ")
        elif 0xA0 <= ord(ch) <= 0xFF:
            out.append(ch)
        else:
            base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
            out.append(latex_escape(base) if base.strip() and base.isprintable() else "?")
    return "".join(out)


_LONG_RUN = re.compile(r"[A-Za-z0-9]{16,}")


def _chunk_long_runs(piece: str) -> list[str]:
    """`piece` split after every BREAK_CHUNK characters of each run of 16 or more letters
    and digits that contains a digit or is at least 24 characters long (a hash, an
    export id); ordinary words keep their hyphenation."""
    parts: list[str] = []
    pos = 0
    for m in _LONG_RUN.finditer(piece):
        run = m.group(0)
        if not (any(c.isdigit() for c in run) or len(run) >= 24):
            continue
        cuts = list(range(m.start() + BREAK_CHUNK, m.end(), BREAK_CHUNK))
        for cut in cuts:
            parts.append(piece[pos:cut])
            pos = cut
    parts.append(piece[pos:])
    return [part for part in parts if part]


def latex_breakable(text: Any) -> Latex:
    """Escaped text in which a line may break after '_', '-', '.' and '/', and inside a
    long run of letters and digits after every BREAK_CHUNK characters (_chunk_long_runs:
    a SHA-256 digest or a Roboflow export hash has no other break point) -- for file
    names, identifiers and digests in narrow table columns."""
    pieces = [chunk for piece in re.split(r"(?<=[_\-./])", str(text))
              for chunk in _chunk_long_runs(piece)]
    return Latex(r"\allowbreak{}".join(latex_escape(p) for p in pieces if p))


def _latex_finalize(value: Any) -> str:
    """Jinja finalize hook of the LaTeX template: escape every inserted value unless
    it is a Latex string."""
    if isinstance(value, Latex):
        return value
    if value is None:
        return ""
    return latex_escape(value)


# ======================================================== markdown (model card) subset

@dataclass(frozen=True)
class MdBlock:
    """One block of the markdown subset used by model cards: kind 'heading' (text,
    level), 'paragraph' (text), 'list' (items, ordered), 'table' (rows, first = header;
    align: 'l', 'r', 'c' or '' per column, from the separator row) or 'code' (text)."""
    kind: str
    text: str = ""
    level: int = 0
    items: tuple[str, ...] = ()
    ordered: bool = False
    rows: tuple[tuple[str, ...], ...] = ()
    align: tuple[str, ...] = ()


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_MD_ORDERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_MD_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_MD_FENCE = re.compile(r"^\s*(```|~~~)")
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_MD_INLINE = re.compile(
    r"`([^`]+)`"                                    # 1 code
    r"|\*\*(.+?)\*\*"                               # 2 bold
    r"|\[([^\]]+)\]\(([^)\s]+)\)"                   # 3 link text, 4 link target
    r"|(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])"   # 5 italic (asterisks only)
    r"|(https?://[^\s<>()\[\]]*[^\s<>()\[\].,;:!?'\"])")  # 6 bare URL


def _table_cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _table_alignment(separator: str) -> tuple[str, ...]:
    """Column alignment of a pipe-table separator row: ':---' 'l', '---:' 'r',
    ':---:' 'c', '---' '' (not specified)."""
    out = []
    for cell in _table_cells(separator):
        left, right = cell.startswith(":"), cell.endswith(":")
        out.append("c" if left and right else "r" if right else "l" if left else "")
    return tuple(out)


def parse_markdown(text: str) -> list[MdBlock]:
    """Parse the markdown subset of a model card into blocks: ATX headings, paragraphs,
    bullet and numbered lists (one level; indented lines continue an item), pipe
    tables (the separator row below the header gives the column alignment, MdBlock.align),
    fenced code and horizontal rules (dropped).
    Inline markup is kept in the block text (see `inline_runs`)."""
    blocks: list[MdBlock] = []
    para: list[str] = []
    items: list[str] = []
    ordered = False
    table: list[tuple[str, ...]] = []
    align: tuple[str, ...] = ()
    code: list[str] | None = None

    def flush() -> None:
        nonlocal ordered, align
        if para:
            blocks.append(MdBlock("paragraph", text=" ".join(para)))
            para.clear()
        if items:
            blocks.append(MdBlock("list", items=tuple(items), ordered=ordered))
            items.clear()
        if table:
            blocks.append(MdBlock("table", rows=tuple(table), align=align))
            table.clear()
            align = ()

    for raw in str(text).splitlines():
        line = raw.rstrip()
        if code is not None:
            if _MD_FENCE.match(line):
                blocks.append(MdBlock("code", text="\n".join(code)))
                code = None
            else:
                code.append(raw)
            continue
        if _MD_FENCE.match(line):
            flush()
            code = []
            continue
        if not line.strip():
            flush()
            continue
        if line.lstrip().startswith("|"):
            if para or items:
                flush()
            if not _MD_TABLE_SEP.match(line):
                table.append(_table_cells(line))
            elif len(table) == 1:            # the separator below the header row
                align = _table_alignment(line)
            continue
        if table:
            flush()
        heading = _MD_HEADING.match(line)
        if heading:
            flush()
            blocks.append(MdBlock("heading", text=heading.group(2), level=len(heading.group(1))))
            continue
        if _MD_RULE.match(line):
            flush()
            continue
        item = _MD_BULLET.match(line) or _MD_ORDERED.match(line)
        if item:
            is_ordered = item.re is _MD_ORDERED
            if para or (items and is_ordered != ordered):
                flush()
            ordered = is_ordered
            items.append(item.group(1).strip())
            continue
        if items and raw[:1] in (" ", "\t"):
            items[-1] = f"{items[-1]} {line.strip()}"
            continue
        if items:
            flush()
        para.append(line.strip())
    flush()
    if code is not None:                 # unterminated fence: keep its content
        blocks.append(MdBlock("code", text="\n".join(code)))
    return blocks


def inline_runs(text: str) -> list[tuple[str, str, str]]:
    """Split text into (kind, text, url) runs; kind is 'text', 'code', 'bold',
    'italic', 'link' or 'url'. Emphasis with underscores is not recognised, so
    identifiers such as dice_ci95 stay intact."""
    runs: list[tuple[str, str, str]] = []
    pos = 0
    for m in _MD_INLINE.finditer(text):
        if m.start() > pos:
            runs.append(("text", text[pos:m.start()], ""))
        if m.group(1) is not None:
            runs.append(("code", m.group(1), ""))
        elif m.group(2) is not None:
            runs.append(("bold", m.group(2), ""))
        elif m.group(3) is not None:
            runs.append(("link", m.group(3), m.group(4)))
        elif m.group(5) is not None:
            runs.append(("italic", m.group(5), ""))
        else:
            runs.append(("url", m.group(6), m.group(6)))
        pos = m.end()
    if pos < len(text):
        runs.append(("text", text[pos:], ""))
    return runs


def _latex_inline(text: str) -> str:
    parts = []
    for kind, value, url in inline_runs(text):
        if kind == "code":
            parts.append(rf"\texttt{{{latex_breakable(value)}}}")
        elif kind == "bold":
            parts.append(rf"\textbf{{{latex_escape(value)}}}")
        elif kind == "italic":
            parts.append(rf"\emph{{{latex_escape(value)}}}")
        elif kind == "url":
            parts.append(rf"\texttt{{{latex_breakable(value)}}}")
        elif kind == "link":
            parts.append(latex_escape(value) if value == url else
                         rf"{latex_escape(value)} (\texttt{{{latex_breakable(url)}}})")
        else:
            parts.append(latex_escape(value))
    return "".join(parts)


def _visible_text(cell: str) -> str:
    """Text of a table cell as `_latex_inline` typesets it (a link shows its target)."""
    return "".join(f"{value} ({url})" if kind == "link" and value != url else value
                   for kind, value, url in inline_runs(cell))


def _unbreakable_segments(cell: str) -> list[str]:
    """Pieces of a cell's visible text between the line-break opportunities of its
    LaTeX (`_latex_inline`): spaces; after '-' in plain text (an explicit hyphen); in
    code, URLs and link targets also after '_', '.', '/' and inside long runs of
    letters and digits (`latex_breakable`). Hyphenation is ignored (conservative)."""
    segments: list[str] = []
    for kind, value, url in inline_runs(cell):
        texts = [(value, kind in ("code", "url"))]
        if kind == "link" and value != url:
            texts = [(value, False), (f"({url})", True)]
        for text, breakable in texts:
            for word in text.split():
                if breakable:
                    segments += [chunk for piece in re.split(r"(?<=[_\-./])", word)
                                 for chunk in _chunk_long_runs(piece)]
                else:
                    segments += re.split(r"(?<=-)", word)
    return [seg for seg in segments if seg]


_X_ALIGN: Mapping[str, str] = MappingProxyType(
    {"l": r"\raggedright", "r": r"\raggedleft", "c": r"\centering"})
"""Paragraph alignment of a wrapping column, after the column's markdown alignment."""


@dataclass(frozen=True)
class TableLayout:
    """LaTeX column layout of a markdown table: `spec` (column specification without
    the outer @{}), `columns` (number of columns) and `wrap` (some column wraps: the
    table is a tabularx at line width; otherwise a tabular at natural width)."""
    spec: str
    columns: int
    wrap: bool


def _share_width(width: float, wrapping: Sequence[int], minimum: Mapping[int, float],
                 claim: Mapping[int, float]) -> dict[int, float]:
    """Divide `width` among the wrapping columns in proportion to their `claim`, but give
    no column less than its `minimum` (water-filling: a column whose share would fall
    below its minimum is fixed at it and the rest is divided again)."""
    fixed: dict[int, float] = {}
    while True:
        free = [j for j in wrapping if j not in fixed]
        rest = width - sum(fixed.values())
        if not free:
            return fixed
        total = sum(claim[j] for j in free)
        share = {j: max(rest, 0.0) * claim[j] / total for j in free}
        low = [j for j in free if share[j] < minimum[j]]
        if not low:
            return {**fixed, **share}
        fixed.update({j: minimum[j] for j in low})


def table_layout(rows: Sequence[Sequence[str]], align: Sequence[str] = ()) -> TableLayout:
    """Column layout of a model-card table in the PDF.

    Method -- a heuristic in characters of the visible cell text, header cells counted
    TABLE_HEADER_SCALE times (bold): every column starts at its natural width (its
    longest cell), aligned as the separator row says (unspecified: left). A column may
    wrap only if that makes it narrower, i.e. if its longest cell is longer than its
    minimum wrapped width: its longest unbreakable segment (`_unbreakable_segments`)
    plus one character, but at least TABLE_WRAP_MIN_CHARS. While a row would be wider
    than TABLE_LINE_CHARS -- natural columns at their natural width, wrapping columns
    at their minimum, TABLE_COLSEP_CHARS between two columns -- one more column wraps:
    left-aligned columns first (numbers only if nothing else is left), the one that
    saves the most characters first. Wrapping columns are tabularx X columns that keep
    the column's alignment (ragged right, ragged left or centred); the width the natural columns leave is divided among them in proportion
    to their longest cell, capped at TABLE_WRAP_CAP_CHARS, but never below their
    minimum (`_share_width`), and passed to tabularx as \\hsize factors that sum to
    the number of wrapping columns. A long description thus gets more room than a
    licence text, numbers and short headers keep their natural width, and no column
    is narrower than its longest word."""
    n = max((len(row) for row in rows), default=0)
    if n == 0:
        return TableLayout("", 0, False)
    natural = [0.0] * n
    word = [0.0] * n
    for i, row in enumerate(rows):
        scale = TABLE_HEADER_SCALE if i == 0 else 1.0
        for j, cell in enumerate(row):
            natural[j] = max(natural[j], scale * len(_visible_text(cell)))
            word[j] = max([word[j]] + [scale * len(seg) for seg in _unbreakable_segments(cell)])
    minimum = {j: max(float(TABLE_WRAP_MIN_CHARS), word[j] + 1.0) for j in range(n)}
    kinds = [(align[j] if j < len(align) else "") or "l" for j in range(n)]
    seps = TABLE_COLSEP_CHARS * (n - 1)
    wrapping: list[int] = []

    def row_chars() -> float:
        return seps + sum(minimum[j] if j in wrapping else natural[j] for j in range(n))

    while row_chars() > TABLE_LINE_CHARS:
        candidates = [j for j in range(n) if j not in wrapping and natural[j] > minimum[j]]
        if not candidates:
            break
        wrapping.append(max(candidates, key=lambda j: (kinds[j] == "l", natural[j] - minimum[j])))
    wrapping.sort()
    spec = list(kinds)
    if len(wrapping) == 1:
        j = wrapping[0]
        spec[j] = rf">{{{_X_ALIGN[kinds[j]]}\arraybackslash}}X"
    elif wrapping:
        room = TABLE_LINE_CHARS - seps - sum(natural[j] for j in range(n) if j not in wrapping)
        claim = {j: min(natural[j], float(TABLE_WRAP_CAP_CHARS)) for j in wrapping}
        widths = _share_width(room, wrapping, minimum, claim)
        total = sum(widths.values())
        factors = {j: round(len(wrapping) * widths[j] / total, 3) for j in wrapping[:-1]}
        factors[wrapping[-1]] = round(len(wrapping) - sum(factors.values()), 3)   # exact sum
        for j, factor in factors.items():
            spec[j] = rf">{{{_X_ALIGN[kinds[j]]}\arraybackslash\hsize={factor:.3f}\hsize}}X"
    return TableLayout("".join(spec), n, bool(wrapping))


def markdown_to_latex(text: str) -> Latex:
    """LaTeX for the markdown subset (`parse_markdown`): headings become run-in bold
    headings of decreasing size, lists itemize/enumerate, tables booktabs tables laid
    out by `table_layout` (natural width, or tabularx at line width when columns must
    wrap), code a typewriter block; every text run is escaped."""
    out: list[str] = []
    for block in parse_markdown(text):
        if block.kind == "heading":
            size = {1: r"\large", 2: r"\normalsize"}.get(block.level, "")
            out.append(rf"\par\medskip{{{size}\bfseries {_latex_inline(block.text)}\par}}\nopagebreak[4]")
        elif block.kind == "paragraph":
            out.append(_latex_inline(block.text) + r"\par")
        elif block.kind == "list":
            env = "enumerate" if block.ordered else "itemize"
            body = "\n".join(rf"\item {_latex_inline(item)}" for item in block.items)
            out.append(rf"\begin{{{env}}}[nosep,leftmargin=*]" + "\n" + body + "\n" + rf"\end{{{env}}}")
        elif block.kind == "table":
            layout = table_layout(block.rows, block.align)
            rows = [list(row) + [""] * (layout.columns - len(row)) for row in block.rows]
            begin = (rf"\begin{{tabularx}}{{\linewidth}}{{@{{}}{layout.spec}@{{}}}}" if layout.wrap
                     else rf"\begin{{tabular}}{{@{{}}{layout.spec}@{{}}}}")
            lines = [r"\par\noindent" + begin, r"\toprule",
                     " & ".join(rf"\textbf{{{_latex_inline(c)}}}" for c in rows[0]) + r" \\",
                     r"\midrule"]
            lines += [" & ".join(_latex_inline(c) for c in row) + r" \\" for row in rows[1:]]
            lines += [r"\bottomrule", r"\end{%s}\par" % ("tabularx" if layout.wrap else "tabular")]
            out.append("\n".join(lines))
        elif block.kind == "code":
            body = r"\\".join(latex_escape(line).replace(" ", "~") or r"\mbox{}"
                              for line in block.text.splitlines() or [""])
            out.append(r"{\par\noindent\ttfamily\footnotesize " + body + r"\par}")
    return Latex("\n\n".join(out))


def _html_inline(text: str) -> str:
    parts = []
    for kind, value, url in inline_runs(text):
        esc = str(markupsafe.escape(value))
        if kind == "code":
            parts.append(f"<code>{esc}</code>")
        elif kind == "bold":
            parts.append(f"<strong>{esc}</strong>")
        elif kind == "italic":
            parts.append(f"<em>{esc}</em>")
        elif kind in ("url", "link") and re.match(r"https?://", url):
            parts.append(f'<a href="{markupsafe.escape(url)}">{esc}</a>')
        else:
            parts.append(esc)
    return "".join(parts)


def markdown_to_html(text: str) -> markupsafe.Markup:
    """HTML for the markdown subset (`parse_markdown`); all text is escaped and only
    http(s) links become anchors."""
    out: list[str] = []
    for block in parse_markdown(text):
        if block.kind == "heading":
            level = min(6, block.level + 2)          # the card sits below the page's h2
            out.append(f"<h{level}>{_html_inline(block.text)}</h{level}>")
        elif block.kind == "paragraph":
            out.append(f"<p>{_html_inline(block.text)}</p>")
        elif block.kind == "list":
            tag = "ol" if block.ordered else "ul"
            items = "".join(f"<li>{_html_inline(i)}</li>" for i in block.items)
            out.append(f"<{tag}>{items}</{tag}>")
        elif block.kind == "table":
            head = "".join(f"<th>{_html_inline(c)}</th>" for c in block.rows[0])
            body = "".join("<tr>" + "".join(f"<td>{_html_inline(c)}</td>" for c in row) + "</tr>"
                           for row in block.rows[1:])
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif block.kind == "code":
            out.append(f"<pre>{markupsafe.escape(block.text)}</pre>")
    return markupsafe.Markup("\n".join(out))


# ========================================================== report data (normalised)

@dataclass(frozen=True)
class DataRow:
    """Images of one source in this cycle: received (labelled images loaded by ingest;
    None if unknown), rejected by quality control, added to the pool (train/val) and
    the distinct photographs among the added images."""
    source: str
    received: int | None
    rejected: int
    added: int
    train: int
    val: int
    photographs: int
    license: str


@dataclass(frozen=True)
class ReasonRow:
    """Rejections of one reason category with up to three example image names."""
    reason: str
    count: int
    examples: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateRow:
    """One image of the cycle that shows an already known photograph."""
    image: str
    matched: str
    split: str
    method: str
    action: str


@dataclass(frozen=True)
class DriftRow:
    """KS test of one image feature: statistic D, p-value, Bonferroni-adjusted p."""
    feature: str
    statistic: float | None
    p_value: float | None
    p_adjusted: float | None
    shifted: bool


@dataclass(frozen=True)
class DriftSummary:
    """Drift test of the cycle (alpha = family-wise level of the Bonferroni test)."""
    rows: tuple[DriftRow, ...]
    alpha: float | None
    n_new: int | None
    n_reference: int | None

    @property
    def shifted(self) -> list[str]:
        """Features flagged as shifted."""
        return [row.feature for row in self.rows if row.shifted]


@dataclass(frozen=True)
class MetricRow:
    """One row of the metrics table: label and formatted cells per shown version."""
    label: str
    candidate: str
    baseline: str


@dataclass(frozen=True)
class VersionPoint:
    """One registered version in the version-history chart."""
    version: int
    dataset_version: int
    architecture: str
    train_images: int
    train_photographs: int
    dice: float | None
    ci: tuple[float, float] | None
    promoted: bool
    current: bool


@dataclass(frozen=True)
class RateBar:
    """A proportion (false-positive rate) of one category with its Wilson 95 % CI."""
    category: str
    n: int | None
    rate: float | None
    ci: tuple[float, float] | None


@dataclass(frozen=True)
class BarSeries:
    """One series of a grouped bar chart (values with optional CIs)."""
    label: str
    color: str
    values: tuple[float | None, ...]
    cis: tuple[tuple[float, float] | None, ...] = ()


@dataclass(frozen=True)
class Example:
    """A frozen-test image chosen for the overlay figure (values from the evaluation)."""
    role: str
    name: str
    dice: float
    gt_positive: bool
    pred_positive: bool
    pred_area: float


@dataclass(frozen=True)
class ExampleSource:
    """What the overlay figure needs to recompute predictions of one version."""
    version: int
    architecture: str
    weights: Path
    threshold: float
    data_root: Path
    rows: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class FigureFile:
    """A saved figure: vector PDF (LaTeX), PNG (HTML), caption and layout width."""
    key: str
    caption: str
    width: str
    pdf: Path
    png: Path

    @property
    def latex_name(self) -> Latex:
        """File name of the figure inside the LaTeX build directory."""
        return Latex(f"fig-{self.key.replace('_', '-')}.pdf")

    @property
    def label(self) -> Latex:
        """LaTeX label of the figure."""
        return Latex(f"fig:{self.key.replace('_', '-')}")


@dataclass(frozen=True)
class ReportData:
    """Everything the templates, figures and the fallback PDF show (see the module
    docstring for the meaning of candidate, baseline and in-service version; `subject`
    is the version whose evaluation is shown: the candidate, or the version in service
    when no model was trained in the cycle)."""
    cycle: int
    date: str
    demo: bool
    candidate: int | None
    baseline: int | None
    in_service: int | None
    subject: int | None
    dataset_version: int | None
    architecture: str | None
    version_line: str
    outcome: str
    gate_sentence: str
    gate_details: str
    key_numbers: tuple[tuple[str, str], ...]
    steps: tuple[tuple[str, str, str], ...]
    batches: tuple[str, ...]
    data_rows: tuple[DataRow, ...]
    data_total: DataRow | None
    pool_line: str
    rejections: tuple[ReasonRow, ...]
    n_rejected: int
    duplicates: tuple[DuplicateRow, ...]
    drift: DriftSummary | None
    subject_label: str
    baseline_label: str
    metric_rows: tuple[MetricRow, ...]
    versions: tuple[VersionPoint, ...]
    stage_series: tuple[BarSeries, ...]
    fpr_categories: tuple[str, ...]
    fpr_series: tuple[BarSeries, ...]
    examples: ExampleSource | None
    examples_note: str
    model_card: str | None
    notes: tuple[str, ...] = ()


# ------------------------------------------------------------------ normalisation

def normalise_metrics(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge metric dictionaries under the names of the evaluation's test block
    (dice_mean, ...; METRIC_ALIASES maps the names stored by the cycle and the
    registry, test_dice_mean, ...); later sources override earlier ones, but a
    missing value (None/NaN) never overrides a present one."""
    out: dict[str, Any] = {}
    for source in sources:
        for key, value in (source or {}).items():
            name = METRIC_ALIASES.get(str(key), str(key))
            missing = value is None or (isinstance(value, float) and not math.isfinite(value))
            if not missing or name not in out:
                out[name] = value
    return out


def _evaluation_metrics(evaluation: Mapping[str, Any] | None) -> dict[str, Any]:
    """Metrics of a stored evaluation (models/v{N}/evaluation.json): the headline numbers
    under the names the registry stores (`registry.metrics_from_evaluation`), plus the
    values of the frozen-test block that the registry does not keep but the report shows
    -- per-stage Dice (pooled pixels), image-level sensitivity with its Wilson CI and the
    IoU confidence interval."""
    if not evaluation or not isinstance(evaluation.get("test"), Mapping):
        return {}
    from .registry import metrics_from_evaluation
    out = metrics_from_evaluation(evaluation)
    if "inference_ms_per_image" in evaluation:
        out["inference_ms_per_image"] = finite(evaluation["inference_ms_per_image"])
    test = evaluation["test"]
    if isinstance(test.get("stage_dice"), Mapping):
        out["stage_dice"] = {str(k): finite(v) for k, v in test["stage_dice"].items()}
    for key in ("sensitivity", "sensitivity_ci95", "iou_ci95"):
        if key in test:
            out[key] = test[key]
    return out


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalise_drift(raw: Any) -> DriftSummary | None:
    """Normalise the drift step's result. Accepted: a mapping with ``features`` (or
    ``tests``) given as a list of {feature, statistic, p_value, ...} or as a mapping
    feature -> dict (or -> p-value); synonyms: statistic/ks/D, p_value/p/pvalue,
    p_adjusted/p_bonferroni, shifted/drift/flagged; ``alpha``/``drift_alpha``,
    ``n_new``, ``n_reference``/``n_pool``, and optionally a list ``shifted``. Missing
    adjusted p-values are min(1, k p) over the k features; a feature without explicit
    flag is shifted iff its adjusted p-value is below alpha. None if there is no test."""
    if not isinstance(raw, Mapping):
        return None
    features = raw.get("features", raw.get("tests"))
    if isinstance(features, Mapping):
        items = [(str(k), v) for k, v in features.items()]
    elif isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
        items = [(str(_first(f, "feature", "name") or f"feature {i + 1}"), f)
                 for i, f in enumerate(features) if isinstance(f, Mapping)]
    else:
        return None
    if not items:
        return None
    alpha = finite(_first(raw, "alpha", "drift_alpha"))
    listed = raw.get("shifted", raw.get("drifted"))
    flagged_names = {str(x) for x in listed} if isinstance(listed, (list, tuple, set)) else set()
    k = len(items)
    rows = []
    for name, value in items:
        entry = value if isinstance(value, Mapping) else {"p_value": value}
        p = finite(_first(entry, "p_value", "p", "pvalue"))
        p_adj = finite(_first(entry, "p_adjusted", "p_bonferroni", "p_corrected", "p_adj"))
        if p_adj is None and p is not None:
            p_adj = min(1.0, k * p)
        flag = _first(entry, "shifted", "drift", "flagged", "flag", "significant")
        if flag is None:
            shifted = name in flagged_names or (p_adj is not None and alpha is not None
                                                and p_adj < alpha)
        else:
            shifted = bool(flag)
        rows.append(DriftRow(name, finite(_first(entry, "statistic", "ks", "ks_statistic", "D", "d")),
                             p, p_adj, shifted))
    return DriftSummary(tuple(rows), alpha, _int_or_none(raw.get("n_new")),
                        _int_or_none(_first(raw, "n_reference", "n_pool")))


def reason_category(reason: Any) -> str:
    """Category of a rejection reason: its text up to the first colon or opening
    parenthesis (quality reasons are written as 'category: details')."""
    head = re.split(r":|\s\(", str(reason), maxsplit=1)[0].strip()
    return head or "unspecified"


def _record_name(record: Any) -> str:
    path = getattr(record, "path", None)
    return Path(str(path)).name if path else str(getattr(record, "name", record))


def rejection_rows(items: Iterable[Any]) -> tuple[list[ReasonRow], Counter[str]]:
    """Group rejections by `reason_category`. Items are mappings with ``reason`` and
    optional ``name``/``source`` or (record, reason) pairs. Returns the rows (most
    frequent first, ties by name) and the number of rejections per source."""
    by_reason: dict[str, list[str]] = {}
    by_source: Counter[str] = Counter()
    for item in items:
        if isinstance(item, Mapping):
            reason = item.get("reason", "")
            name = str(item.get("name") or item.get("image") or Path(str(item.get("path") or "")).name)
            source = item.get("source")
        else:
            record, reason = item
            name, source = _record_name(record), getattr(record, "source", None)
        by_reason.setdefault(reason_category(reason), []).append(name)
        if source:
            by_source[str(source)] += 1
    rows = [ReasonRow(cat, len(names), tuple(sorted(n for n in names if n)[:3]))
            for cat, names in by_reason.items()]
    rows.sort(key=lambda r: (-r.count, r.reason))
    return rows, by_source


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def duplicates_from_dedupe(result: Any, records: Sequence[Any] | None = None) -> list[dict[str, str]]:
    """Duplicate rows from a dedupe result (`dedupe.DedupeResult` or its to_dict form):
    images inheriting the split of a pool photograph, images not added because they
    copy a frozen evaluation photograph, split conflicts, and further images of a new
    photograph that occurs more than once in the cycle. `records` (the list given to
    dedupe) names the images when the result does not carry file names."""
    if result is None:
        return []
    new_groups = [list(g) for g in (_field(result, "new_groups") or [])]
    inherited = list(_field(result, "inherited") or [])
    dropped = list(_field(result, "dropped_test_matches") or [])
    conflicts = list(_field(result, "conflicts") or [])
    n = sum(len(g) for g in new_groups) + len(inherited) + len(dropped) + len(conflicts)
    recs = list(records) if records is not None and len(records) == n else None

    def image(index: Any, holder: Any = None) -> str:
        file = _field(holder, "file") if holder is not None else None
        if file:
            return Path(str(file)).name
        if recs is not None and isinstance(index, int) and 0 <= index < n:
            return _record_name(recs[index])
        return f"record {index}"

    rows: list[dict[str, str]] = []
    for m in inherited:
        split = str(_field(m, "split", ""))
        rows.append({"image": image(_field(m, "record"), m), "matched": str(_field(m, "pool_name", "")),
                     "split": split, "method": str(_field(m, "method", "")),
                     "action": f"added to {split} (same photograph)"})
    for m in dropped:
        split = str(_field(m, "split", ""))
        rows.append({"image": image(_field(m, "record"), m), "matched": str(_field(m, "pool_name", "")),
                     "split": split, "method": str(_field(m, "method", "")),
                     "action": f"not added: copy of a frozen {split} photograph"})
    for c in conflicts:
        matches = list(_field(c, "matches") or [])
        rows.append({"image": image(_field(c, "record"), c),
                     "matched": ", ".join(sorted({str(_field(x, "pool_name", "")) for x in matches})),
                     "split": "/".join(sorted({str(_field(x, "split", "")) for x in matches})),
                     "method": ", ".join(sorted({str(_field(x, "method", "")) for x in matches})),
                     "action": "not added: matches photographs of several splits"})
    for group in new_groups:
        for index in group[1:]:
            rows.append({"image": image(index), "matched": image(group[0]), "split": "new",
                         "method": "within the cycle",
                         "action": "added with the photograph of the matched image"})
    return rows


def _duplicate_row(item: Mapping[str, Any]) -> DuplicateRow:
    return DuplicateRow(*(str(item.get(key) or "") for key in
                          ("image", "matched", "split", "method", "action")))


# ------------------------------------------------------------------- decisions

def gate_sentence(candidate: int | None, gate: Mapping[str, Any] | None,
                  in_service: int | None, no_model_reason: str | None = None) -> str:
    """The gate decision of the cycle in one plain-text sentence (no characters that
    HTML or LaTeX would have to escape except the minus sign). It names the candidate,
    the version it was compared with, the paired-bootstrap mean Dice difference with
    its 95 % CI and the rule that decided (mean difference above 0 and lower bound
    above minus the margin)."""
    serving = f"version {in_service} stays in service" if in_service is not None \
        else "no model is in service"
    if candidate is None:
        why = f" ({no_model_reason.rstrip('.')})" if no_model_reason else ""
        return f"No new model was trained in this cycle{why}; {serving}."
    if not gate:
        return f"Version {candidate} was trained but the gate made no decision; {serving}."
    promoted = bool(gate.get("promoted"))
    diff, ci = finite(gate.get("mean_diff")), ci_pair(gate.get("ci95"))
    base = _int_or_none(gate.get("compared_to"))
    if diff is None or ci is None:
        if promoted:
            return (f"Version {candidate} was promoted and is now in service because no "
                    f"model was in service before it.")
        reason = str(gate.get("reason") or "the gate rule could not be evaluated").rstrip(".")
        return f"Version {candidate} was not promoted ({reason}); {serving}."
    margin = finite(gate.get("margin"))
    margin_text = f"{MINUS}{margin:g}" if margin is not None else "minus the margin"
    n = _int_or_none(gate.get("n"))
    other = f"version {base}" if base is not None else "the baseline"
    size = abs(diff)
    direction = "higher" if diff > 0 else "lower" if diff < 0 else "unchanged"
    change = (f"its mean test Dice is {size:.3f} {direction} than that of {other}"
              if diff != 0 else f"its mean test Dice equals that of {other}")
    evidence = (f"95 % CI {fmt_signed(ci[0])} to {fmt_signed(ci[1])}"
                + (f", n = {n} paired test images" if n is not None else ""))
    if promoted:
        return (f"Version {candidate} was promoted and replaces {other}: {change} "
                f"({evidence}), which meets the gate rule (mean difference above 0 and "
                f"lower CI bound above {margin_text}).")
    if diff <= 0:
        return (f"Version {candidate} was not promoted and {serving}: {change} "
                f"({evidence}), so it is not better on average.")
    return (f"Version {candidate} was not promoted and {serving}: {change} ({evidence}), "
            f"but the lower CI bound is not above {margin_text}, so it may be worse by "
            f"more than the margin.")


def gate_details(gate: Mapping[str, Any] | None) -> str:
    """The gate's statistics in one line (paired bootstrap of per-image test Dice)."""
    if not gate:
        return "The gate did not run in this cycle."
    diff = finite(gate.get("mean_diff"))
    if diff is None:
        return f"Gate: {gate.get('reason') or 'no comparison was made'}."
    parts = [f"mean Dice difference (candidate {MINUS} baseline) {fmt_signed(diff)}",
             f"95 % CI {fmt_interval(gate.get('ci95'), signed=True)}"]
    if finite(gate.get("p_value")) is not None:
        parts.append(f"p = {fmt_num(gate.get('p_value'))}")
    if _int_or_none(gate.get("n")) is not None:
        parts.append(f"n = {gate['n']} paired test images")
    if finite(gate.get("margin")) is not None:
        parts.append(f"non-inferiority margin {finite(gate.get('margin')):g}")
    decision = "promoted" if gate.get("promoted") else "not promoted"
    return f"Paired bootstrap of per-image test Dice: {', '.join(parts)}; decision: {decision}."


def metric_rows(subject: Mapping[str, Any] | None,
                baseline: Mapping[str, Any] | None) -> list[MetricRow]:
    """Rows of the metrics table (METRIC_TABLE plus test image counts) for the shown
    version and its baseline; a row is omitted if neither version has the value."""
    rows = []
    sources = [m for m in (subject, baseline)]
    for label, kind, key, ci_key in METRIC_TABLE:
        cells = [metric_text(m.get(key), m.get(ci_key) if ci_key else None, kind) if m else DASH
                 for m in sources]
        if any(c != DASH for c in cells):
            rows.append(MetricRow(label, cells[0], cells[1]))
    counts = [f"{fmt_count(m.get('n_test_positive'))} / {fmt_count(m.get('n_test_negative'))}"
              if m and (m.get("n_test_positive") is not None or m.get("n_test_negative") is not None)
              else DASH for m in sources]
    if any(c != DASH for c in counts):
        rows.append(MetricRow("Test images with ulcer / ulcer-free", counts[0], counts[1]))
    return rows


# ------------------------------------------------------------------ examples

def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def select_examples(rows: Iterable[Mapping[str, Any]]) -> list[Example]:
    """Pick up to four images of a per-image results table (selftrain.evaluate):
    among ulcer-containing images the best and the worst Dice, then the image whose
    Dice is closest to the median of all ulcer-containing images (each image at most
    once; ties broken by name), and the ulcer-free image with the largest predicted
    area -- a false positive if there is one, otherwise a correct rejection."""
    def example(role: str, r: Mapping[str, Any]) -> Example:
        return Example(role, str(r["name"]), float(r["dice"]), _truthy(r["gt_positive"]),
                       _truthy(r.get("pred_positive")), finite(r.get("pred_area")) or 0.0)

    rows = [r for r in rows if finite(r.get("dice")) is not None]
    positives = [r for r in rows if _truthy(r.get("gt_positive"))]
    negatives = [r for r in rows if not _truthy(r.get("gt_positive"))]
    chosen: list[Example] = []
    if positives:
        by_dice = sorted(positives, key=lambda r: (float(r["dice"]), str(r["name"])))
        best = min(positives, key=lambda r: (-float(r["dice"]), str(r["name"])))
        chosen.append(example("best", best))
        rest = [r for r in by_dice if r is not best]
        if rest:
            chosen.append(example("worst", rest[0]))
            median = float(np.median([float(r["dice"]) for r in positives]))
            left = rest[1:]
            if left:
                mid = min(left, key=lambda r: (abs(float(r["dice"]) - median), str(r["name"])))
                chosen.insert(1, example("median", mid))
    if negatives:
        neg = min(negatives, key=lambda r: (-(finite(r.get("pred_area")) or 0.0), str(r["name"])))
        chosen.append(example("ulcer-free", neg))
    return chosen


def _read_per_image(path: Path) -> tuple[dict[str, str], ...]:
    with open(path, newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def load_evaluation(ws: Workspace, version: int | None) -> dict[str, Any] | None:
    """models/v{N}/evaluation.json of a version, or None."""
    if version is None:
        return None
    path = ws.model_dir(version) / "evaluation.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _resolve(ws: Workspace, path: Any) -> Path:
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else Path(ws.root) / p


def example_source(ws: Workspace, record: VersionRecord | None,
                   evaluation: Mapping[str, Any] | None) -> tuple[ExampleSource | None, str]:
    """Inputs of the overlay figure for one version -- per-image test results, weights,
    operating threshold and dataset root -- or None with the reason."""
    if record is None:
        return None, "no registered model version to show"
    label = f"v{record.version}"
    if evaluation is None:
        return None, f"{label} has no evaluation.json"
    model_dir = ws.model_dir(record.version)
    rel = (evaluation.get("per_image_csv") or {}).get("test")
    table = model_dir / rel if rel else None
    if table is None or not table.is_file():
        found = sorted(model_dir.glob("**/*_test_per_image.csv"))
        table = found[0] if found else None
    if table is None:
        return None, f"{label} has no per-image results of the frozen test split"
    weights = record.weights_path(ws)
    if not weights.is_file():
        return None, f"the weights of {label} ({weights}) do not exist"
    from .steps import operating_threshold
    root = evaluation.get("data_root")
    data_root = _resolve(ws, root) if root else ws.dataset_dir(record.dataset_version)
    threshold = finite(evaluation.get("threshold"))
    if threshold is None:
        threshold = operating_threshold(ws, record.version, record.architecture)
    return ExampleSource(record.version, record.architecture, weights, threshold, data_root,
                         _read_per_image(table)), ""


# ------------------------------------------------------------------ collection

def version_points(ws: Workspace, reg: Registry) -> list[VersionPoint]:
    """Every registered version with its test Dice (registry metrics, else its
    evaluation.json), gate decision and whether it is in service."""
    points = []
    for rec in reg["versions"]:
        metrics = normalise_metrics(rec.metrics)
        if finite(metrics.get("dice_mean")) is None:
            metrics = normalise_metrics(metrics, _evaluation_metrics(load_evaluation(ws, rec.version)))
        points.append(VersionPoint(rec.version, rec.dataset_version, rec.architecture,
                                   rec.train_images, rec.train_photographs,
                                   finite(metrics.get("dice_mean")), ci_pair(metrics.get("dice_ci95")),
                                   bool(rec.gate.get("promoted")), rec.version == reg["current"]))
    return points


def _stage_values(metrics: Mapping[str, Any]) -> tuple[float | None, ...] | None:
    stage = metrics.get("stage_dice")
    if not isinstance(stage, Mapping):
        return None
    values = tuple(finite(stage.get(name)) for name in STAGE_NAMES)
    return values if any(v is not None for v in values) else None


def _wound_label(code: str) -> str:
    return WOUND_TYPES.get(code, code)


def _fpr_bars(evaluation: Mapping[str, Any] | None,
              metrics: Mapping[str, Any]) -> dict[str, RateBar]:
    """False-positive rates of one version by category: every wound type of the extra
    ulcer-free photographs, all of them together, and the ulcer-free test photographs."""
    bars: dict[str, RateBar] = {}
    extra = (evaluation or {}).get("extra_negatives")
    if isinstance(extra, Mapping):
        for code in sorted(k for k in extra if k != "all"):
            entry = extra[code]
            if isinstance(entry, Mapping):
                label = _wound_label(str(code))
                bars[label] = RateBar(label, _int_or_none(entry.get("n")),
                                      finite(entry.get("false_positive_rate")), ci_pair(entry.get("ci95")))
        if isinstance(extra.get("all"), Mapping):
            entry = extra["all"]
            bars["all wound types"] = RateBar("all wound types", _int_or_none(entry.get("n")),
                                              finite(entry.get("false_positive_rate")),
                                              ci_pair(entry.get("ci95")))
    if finite(metrics.get("false_positive_rate")) is not None:
        bars["frozen test, ulcer-free"] = RateBar(
            "frozen test, ulcer-free", _int_or_none(metrics.get("n_test_negative")),
            finite(metrics.get("false_positive_rate")), ci_pair(metrics.get("false_positive_rate_ci95")))
    return bars


def _date_text(value: Any) -> str:
    if not value:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def _data_rows(ws: Workspace, cycle: int, received: Mapping[str, Any] | None,
               rejected_by_source: Mapping[str, int]) -> tuple[list[DataRow], DataRow | None, str]:
    manifest = ws.read_manifest()
    added = manifest[manifest["added_cycle"] == cycle]
    received = {str(k): int(v) for k, v in (received or {}).items()}
    sources = sorted(set(received) | set(rejected_by_source) | set(added["source"]))
    rows = []
    for source in sources:
        part = added[added["source"] == source]
        licences = sorted({str(x) for x in part["license"] if str(x)})
        rows.append(DataRow(source, received.get(source) if received else None,
                            int(rejected_by_source.get(source, 0)), len(part),
                            int((part["split"] == "train").sum()), int((part["split"] == "val").sum()),
                            int(part["photograph"].nunique()), ", ".join(licences) or DASH))
    total = None
    if rows:
        total = DataRow("Total", sum(received.values()) if received else None,
                        sum(r.rejected for r in rows), len(added),
                        int((added["split"] == "train").sum()), int((added["split"] == "val").sum()),
                        int(added["photograph"].nunique()), "")
    if len(manifest):
        counts = manifest["split"].value_counts()
        splits = ", ".join(f"{s} {int(counts[s])}" for s in ("train", "val", "test", "external")
                           if s in counts)
        pool = (f"Pool after cycle {cycle}: {len(manifest)} images of "
                f"{manifest['photograph'].nunique()} photographs ({splits}).")
    else:
        pool = "The pool is empty."
    return rows, total, pool


def _steps(raw: Any) -> list[tuple[str, str, str]]:
    if not isinstance(raw, Mapping):
        return []
    out = []
    for name in [*STEP_NAMES, *sorted(k for k in raw if k not in STEP_NAMES)]:
        entry = raw.get(name)
        if entry is None:
            continue
        status = _field(entry, "status", "")
        if callable(status):
            status = status()
        out.append((name, str(status or ""), str(_field(entry, "summary", "") or "")))
    return out


def _key_numbers(subject_label: str, metrics: Mapping[str, Any], total: DataRow | None,
                 n_rejected: int, n_duplicates: int, drift: DriftSummary | None
                 ) -> list[tuple[str, str]]:
    numbers = []
    if finite(metrics.get("dice_mean")) is not None:
        numbers.append((f"Test Dice of {subject_label} (95 % CI)",
                        metric_text(metrics.get("dice_mean"), metrics.get("dice_ci95"), "ratio")))
    if finite(metrics.get("false_positive_rate")) is not None:
        numbers.append(("False-positive rate, ulcer-free test photographs",
                        metric_text(metrics.get("false_positive_rate"),
                                    metrics.get("false_positive_rate_ci95"), "pct")))
    added = f"{total.added} ({total.train} train, {total.val} val)" if total else "0"
    numbers.append(("Images added to the pool", added))
    numbers.append(("Rejected by quality control / duplicates", f"{n_rejected} / {n_duplicates}"))
    if drift is None:
        numbers.append(("Drift", "not tested in this cycle"))
    else:
        shifted = drift.shifted
        numbers.append(("Drift", f"{len(shifted)} of {len(drift.rows)} features shifted"
                        + (f" ({', '.join(shifted)})" if shifted else "")))
    return numbers


def collect(ws: Workspace, cycle: int, context_data: Mapping[str, Any] | None) -> ReportData:
    """Assemble the report content from the workspace and the cycle's context data
    (schema: `build_report`)."""
    from . import registry
    ctx = dict(context_data or {})
    reg = registry.load(ws)
    records = {rec.version: rec for rec in reg["versions"]}
    in_service = reg["current"]
    candidate = _int_or_none(ctx.get("model_version"))
    gate = ctx.get("gate") if isinstance(ctx.get("gate"), Mapping) else None
    if gate is None and candidate in records:
        gate = records[candidate].gate
    baseline = _int_or_none(gate.get("compared_to")) if gate else None
    if baseline is None and candidate in records:
        baseline = records[candidate].parent
    subject = candidate if candidate is not None else in_service
    subject_rec, baseline_rec = records.get(subject), records.get(baseline)
    subject_eval, baseline_eval = load_evaluation(ws, subject), load_evaluation(ws, baseline)
    subject_m = normalise_metrics(subject_rec.metrics if subject_rec else None,
                                  _evaluation_metrics(subject_eval),
                                  ctx.get("metrics") if subject == candidate else None)
    baseline_m = normalise_metrics(baseline_rec.metrics if baseline_rec else None,
                                   _evaluation_metrics(baseline_eval)) if baseline is not None else None

    steps = _steps(ctx.get("steps"))
    train_step = next((s for s in steps if s[0] == "train"), None)
    sentence = gate_sentence(candidate, gate, in_service,
                             train_step[2] if candidate is None and train_step else None)
    outcome = ("promoted" if candidate is not None and gate and gate.get("promoted")
               else "kept" if candidate is not None else "none")

    rejections, rejected_by_source = rejection_rows(ctx.get("rejected") or [])
    data_rows, total, pool_line = _data_rows(ws, cycle, ctx.get("received"), rejected_by_source)
    duplicates = tuple(_duplicate_row(d) for d in (ctx.get("duplicates") or []) if isinstance(d, Mapping))
    drift = normalise_drift(ctx.get("drift"))

    subject_label = (f"v{subject}" + (" (candidate)" if subject == candidate else " (in service)")
                     if subject is not None else "model")
    baseline_label = f"v{baseline} (baseline)" if baseline is not None else ""
    architecture = subject_rec.architecture if subject_rec else ctx.get("architecture")
    dataset_version = _int_or_none(ctx.get("dataset_version"))
    if dataset_version is None and subject_rec is not None and subject == candidate:
        dataset_version = subject_rec.dataset_version
    parts = []
    if candidate is not None:
        parts.append(f"candidate v{candidate}" + (f" ({architecture}" if architecture else "")
                     + (f", dataset v{dataset_version})" if dataset_version is not None else
                        (")" if architecture else "")))
    else:
        parts.append("no new model")
    if baseline is not None:
        parts.append(f"baseline v{baseline}")
    parts.append(f"in service: v{in_service}" if in_service is not None else "no model in service")

    stage_series = []
    for label, color, metrics, evaluation in (
            (subject_label, CANDIDATE_COLOR, subject_m, subject_eval),
            (baseline_label, BASELINE_COLOR, baseline_m, baseline_eval)):
        values = _stage_values(metrics) if metrics is not None else None
        if values is not None:
            stage_series.append(BarSeries(label, color, values))

    fpr_by_version = [(subject_label, CANDIDATE_COLOR, _fpr_bars(subject_eval, subject_m))]
    if baseline_m is not None:
        fpr_by_version.append((baseline_label, BASELINE_COLOR, _fpr_bars(baseline_eval, baseline_m)))
    categories = list(fpr_by_version[0][2])
    categories += [c for _, _, bars in fpr_by_version[1:] for c in bars if c not in categories]
    fpr_series = [BarSeries(label, color, tuple(bars[c].rate if c in bars else None for c in categories),
                            tuple(bars[c].ci if c in bars else None for c in categories))
                  for label, color, bars in fpr_by_version if bars]
    fpr_categories = []
    for c in categories:
        n = next((bars[c].n for _, _, bars in fpr_by_version if c in bars), None)
        fpr_categories.append(f"{c} (n = {n})" if n is not None else c)

    examples, examples_note = example_source(ws, subject_rec, subject_eval)
    card = ctx.get("model_card")
    if card is None and subject_rec is not None:
        path = _resolve(ws, subject_rec.model_card) if subject_rec.model_card else \
            ws.model_dir(subject_rec.version) / "model_card.md"
        card = path.read_text(encoding="utf-8") if path.is_file() else None

    return ReportData(
        cycle=cycle, date=_date_text(ctx.get("date")), demo=bool(ctx.get("demo", False)),
        candidate=candidate, baseline=baseline, in_service=in_service, subject=subject,
        dataset_version=dataset_version, architecture=architecture,
        version_line=" | ".join(parts), outcome=outcome, gate_sentence=sentence,
        gate_details=gate_details(gate) if candidate is not None else "",
        key_numbers=tuple(_key_numbers(subject_label, subject_m, total, sum(r.count for r in rejections),
                                       len(duplicates), drift)),
        steps=tuple(steps), batches=tuple(str(b) for b in (ctx.get("batches") or [])),
        data_rows=tuple(data_rows), data_total=total, pool_line=pool_line,
        rejections=tuple(rejections), n_rejected=sum(r.count for r in rejections),
        duplicates=duplicates, drift=drift, subject_label=subject_label,
        baseline_label=baseline_label,
        metric_rows=tuple(metric_rows(subject_m if subject is not None else None, baseline_m)),
        versions=tuple(version_points(ws, reg)), stage_series=tuple(stage_series),
        fpr_categories=tuple(fpr_categories), fpr_series=tuple(fpr_series),
        examples=examples, examples_note=examples_note,
        model_card=str(card) if card else None)


# ======================================================================== figures

@contextmanager
def figure_style() -> Iterator[None]:
    """Apply STYLE while figures are built and saved (restored afterwards)."""
    with matplotlib.rc_context(dict(STYLE)):
        yield


def _no_data(ax: Any, text: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha="center", va="center", color=GREY, transform=ax.transAxes)


def draw_version_history(points: Sequence[VersionPoint]) -> Figure | None:
    """Mean test Dice with 95 % CI per version (filled marker: promoted by the gate,
    hollow: not promoted, ring: in service) and training photographs per version as
    grey bars on a secondary axis. None without versions."""
    if not points:
        return None
    fig = Figure(figsize=(HALF_WIDTH_IN, 2.5))
    ax = fig.add_subplot()
    ax2 = ax.twinx()
    xs = [p.version for p in points]
    ax2.bar(xs, [p.train_photographs for p in points], width=0.6, color=LIGHT_GREY,
            label="training photographs", zorder=0)
    ax2.set_ylabel("Training photographs (count)")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax2.set_ylim(0, max(1, max(p.train_photographs for p in points)) * 1.25)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for promoted, label in ((True, "promoted"), (False, "not promoted")):
        sel = [p for p in points if p.promoted == promoted and p.dice is not None]
        if not sel:
            continue
        low = [p.dice - p.ci[0] if p.ci else 0.0 for p in sel]
        high = [p.ci[1] - p.dice if p.ci else 0.0 for p in sel]
        ax.errorbar([p.version for p in sel], [p.dice for p in sel],
                    yerr=[np.clip(low, 0, None), np.clip(high, 0, None)], fmt="o",
                    color=CANDIDATE_COLOR, ecolor=CANDIDATE_COLOR, elinewidth=0.9,
                    mfc=CANDIDATE_COLOR if promoted else "white", mec=CANDIDATE_COLOR,
                    mew=1.1, label=label, zorder=3)
    serving = [p for p in points if p.current and p.dice is not None]
    if serving:
        ax.scatter([p.version for p in serving], [p.dice for p in serving], s=110,
                   facecolors="none", edgecolors=IN_SERVICE_COLOR, linewidths=1.2,
                   label="in service", zorder=4)
    bounds = [b for p in points if p.dice is not None
              for b in ((p.ci[0], p.ci[1]) if p.ci else (p.dice, p.dice))]
    if bounds:
        ax.set_ylim(max(0.0, min(bounds) - 0.05), min(1.0, max(bounds) + 0.05))
    else:
        ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"v{v}" for v in xs], rotation=0 if len(xs) <= 12 else 90)
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_xlabel("Model version")
    ax.set_ylabel("Mean test Dice (0–1)")
    handles, labels = ax.get_legend_handles_labels()
    bar_handles, bar_labels = ax2.get_legend_handles_labels()
    ax.legend(handles + bar_handles, labels + bar_labels, loc="lower left",
              bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0.0, handlelength=1.4,
              columnspacing=1.0)
    return fig


def _grouped_bars(ax: Any, categories: Sequence[str], series: Sequence[BarSeries], *,
                  horizontal: bool, scale: float = 1.0, value_fmt: str | None = None) -> None:
    """Grouped bars (one group per category, one bar per series) with optional CI
    error bars; a missing value is marked 'n/a' instead of a bar."""
    k = max(1, len(series))
    width = 0.8 / k
    base = np.arange(len(categories), dtype=float)
    for j, s in enumerate(series):
        pos = base + (j - (k - 1) / 2) * width
        cis = s.cis or (None,) * len(s.values)
        heights = [0.0 if v is None else v * scale for v in s.values]
        if horizontal:
            ax.barh(pos, heights, height=width * 0.9, color=s.color, label=s.label, zorder=2)
        else:
            ax.bar(pos, heights, width=width * 0.9, color=s.color, label=s.label, zorder=2)
        for p, v, ci in zip(pos, s.values, cis):
            if v is None:
                if horizontal:
                    ax.text(0, p, " n/a", ha="left", va="center", fontsize=6, color=GREY)
                else:
                    ax.text(p, 0, "n/a", ha="center", va="bottom", fontsize=6, color=GREY)
                continue
            if ci is not None:
                err = [[max(0.0, v - ci[0]) * scale], [max(0.0, ci[1] - v) * scale]]
                kwargs = dict(fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
                if horizontal:
                    ax.errorbar(v * scale, p, xerr=err, **kwargs)
                else:
                    ax.errorbar(p, v * scale, yerr=err, **kwargs)
            if value_fmt:
                if horizontal:
                    ax.text(v * scale, p, " " + value_fmt.format(v), ha="left", va="center", fontsize=6)
                else:
                    ax.text(p, v * scale, value_fmt.format(v), ha="center", va="bottom", fontsize=6)
    if horizontal:
        ax.set_yticks(base)
        ax.set_yticklabels(list(categories))
        ax.invert_yaxis()
        ax.yaxis.grid(False)
        ax.xaxis.grid(True)
    else:
        ax.set_xticks(base)
        ax.set_xticklabels(list(categories))


def draw_stage_dice(series: Sequence[BarSeries]) -> Figure | None:
    """Pooled per-stage Dice (stages 1-4) on the frozen test split, one bar series per
    version. None if no version has a per-stage Dice."""
    series = [s for s in series if any(v is not None for v in s.values)]
    if not series:
        return None
    fig = Figure(figsize=(HALF_WIDTH_IN, 2.5))
    ax = fig.add_subplot()
    _grouped_bars(ax, [f"Stage {i + 1}" for i in range(len(STAGE_NAMES))], series,
                  horizontal=False, value_fmt="{:.2f}")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Pooled test Dice (0–1)")
    ax.set_xlabel("Stage (ground truth and prediction)")
    if len(series) > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=len(series), borderaxespad=0.0)
    return fig


def draw_fpr(categories: Sequence[str], series: Sequence[BarSeries]) -> Figure | None:
    """False-positive rate (%) per category of ulcer-free photographs with Wilson 95 %
    CIs, horizontal bars, one series per version. None without data."""
    series = [s for s in series if any(v is not None for v in s.values)]
    if not series or not categories:
        return None
    fig = Figure(figsize=(HALF_WIDTH_IN, 0.9 + 0.32 * len(categories) * max(1, len(series)) ** 0.5))
    ax = fig.add_subplot()
    _grouped_bars(ax, categories, series, horizontal=True, scale=100.0)
    upper = max((ci[1] if ci else (v or 0.0)) for s in series
                for v, ci in zip(s.values, s.cis or (None,) * len(s.values)))
    ax.set_xlim(0.0, min(100.0, max(5.0, 100.0 * upper * 1.15)))
    ax.set_xlabel("False-positive rate (% of photographs)")
    if len(series) > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=len(series), borderaxespad=0.0)
    return fig


def _wrap_label(text: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(text, width)[:2]) or text


def draw_data_added(rows: Sequence[DataRow], rejections: Sequence[ReasonRow]) -> Figure | None:
    """Two stacked panels: images added to the pool in this cycle per source (train
    and val stacked) and images rejected by quality control per reason category.
    None when nothing was added or rejected."""
    added = [r for r in rows if r.added > 0]
    if not added and not rejections:
        return None
    n_top, n_bottom = max(1, len(added)), max(1, len(rejections))
    fig = Figure(figsize=(HALF_WIDTH_IN, 1.3 + 0.3 * (n_top + n_bottom)))
    top, bottom = fig.subplots(2, 1, gridspec_kw={"height_ratios": [n_top + 0.8, n_bottom + 0.8]})
    if added:
        ys = np.arange(len(added))
        train = np.array([r.train for r in added])
        val = np.array([r.val for r in added])
        other = np.array([r.added - r.train - r.val for r in added])
        top.barh(ys, train, color=TRAIN_COLOR, label="train", height=0.6, zorder=2)
        top.barh(ys, val, left=train, color=VAL_COLOR, label="val", height=0.6, zorder=2)
        if other.any():
            top.barh(ys, other, left=train + val, color=GREY, label="other", height=0.6, zorder=2)
        top.set_yticks(ys)
        top.set_yticklabels([_wrap_label(r.source) for r in added])
        top.invert_yaxis()
        top.yaxis.grid(False)
        top.xaxis.grid(True)
        top.xaxis.set_major_locator(MaxNLocator(integer=True))
        top.set_xlabel("Images added to the pool (count)")
        top.legend(loc="lower right", ncol=3)
    else:
        _no_data(top, "no images added in this cycle")
    if rejections:
        ys = np.arange(len(rejections))
        bottom.barh(ys, [r.count for r in rejections], color=REJECT_COLOR, height=0.6, zorder=2)
        bottom.set_yticks(ys)
        bottom.set_yticklabels([_wrap_label(r.reason) for r in rejections])
        bottom.invert_yaxis()
        bottom.yaxis.grid(False)
        bottom.xaxis.grid(True)
        bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
        bottom.set_xlabel("Images rejected by quality control (count)")
    else:
        _no_data(bottom, "no images rejected in this cycle")
    fig.subplots_adjust(hspace=0.75)
    return fig


def overlay_layers(img_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                   max_side: int = DISPLAY_MAX_SIDE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downscale a photograph (area interpolation) and its ground-truth and predicted
    masks (nearest neighbour) to at most `max_side` px; returns RGB, gt, pred (bool)."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    rgb = cv2.cvtColor(cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    small = [cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0 for m in (gt, pred)]
    return rgb, small[0], small[1]


def predict_foreground(predictor: Any, img_bgr: np.ndarray, threshold: float) -> np.ndarray:
    """Binary prediction exactly as the evaluation scores it: semantic map at the
    operating threshold, regions smaller than MIN_COMPONENT_FRAC removed."""
    from ..metrics import remove_specks
    semantic = predictor.to_semantic(predictor.infer(img_bgr), threshold)
    return remove_specks(semantic > 0)


def _example_label(index: int, example: Example) -> str:
    letter = "abcdefgh"[index]
    if example.gt_positive:
        return f"({letter}) {example.role}, Dice {example.dice:.2f}"
    verdict = "false positive" if example.pred_positive else "no false positive"
    return f"({letter}) ulcer-free,\n{verdict}"        # two lines: fits a quarter-width panel


def draw_examples(source: ExampleSource, examples: Sequence[Example], predictor: Any) -> Figure | None:
    """Overlay panels: photograph, prediction as blue fill (recomputed with
    `predict_foreground`), ground truth as yellow contour over a black halo; the
    panel label gives the role and the evaluation's per-image Dice."""
    if not examples:
        return None
    images = source.data_root / "images" / "test"
    masks = source.data_root / "masks" / "test"
    fig = Figure(figsize=(TEXT_WIDTH_IN, 2.15))
    axes = fig.subplots(1, len(examples), squeeze=False)[0]
    fill = (*to_rgb(PRED_COLOR), PRED_ALPHA)
    for i, (ax, example) in enumerate(zip(axes, examples)):
        img = cv2.imread(str(images / f"{example.name}.jpg"))
        gt = cv2.imread(str(masks / f"{example.name}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            raise FileNotFoundError(f"test image or mask of {example.name} not found under {source.data_root}")
        rgb, gt_small, pred_small = overlay_layers(img, gt > 0, predict_foreground(predictor, img, source.threshold))
        ax.imshow(rgb)
        if pred_small.any():
            layer = np.zeros(pred_small.shape + (4,), dtype=float)
            layer[pred_small] = fill
            ax.imshow(layer)
        if gt_small.any() and not gt_small.all():
            ax.contour(gt_small.astype(float), levels=[0.5], colors=["black"], linewidths=2.2)
            ax.contour(gt_small.astype(float), levels=[0.5], colors=[GT_COLOR], linewidths=1.1)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(_example_label(i, example), fontsize=7)
    fig.legend([Patch(facecolor=fill, edgecolor="none"), Line2D([], [], color=GT_COLOR, lw=1.6,
                                                                path_effects=None)],
               ["prediction", "ground truth (contour)"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06))
    return fig


def save_figure(fig: Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Save a figure as vector PDF (reproducible: no creation date) and 200 dpi PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(pdf, format="pdf", metadata={"Creator": "selftrain.system.report", "CreationDate": None})
    fig.savefig(png, format="png", dpi=PNG_DPI)
    return pdf, png


def _captions(data: ReportData) -> dict[str, str]:
    subject = data.subject_label
    min_area = f"{100 * MIN_COMPONENT_FRAC:g} %"
    thr = fmt_num(data.examples.threshold, 2) if data.examples else DASH
    return {
        "version_history": (
            "Mean per-image Dice on the ulcer-containing photographs of the frozen test "
            "split for every registered model version, with 95 % bootstrap confidence "
            "intervals. Filled markers: promoted by the gate; hollow: not promoted; ring: "
            f"in service after cycle {data.cycle}. Grey bars (right axis): distinct "
            "training photographs per version."),
        "stage_dice": (
            f"Per-stage Dice of {subject}"
            + (f" and {data.baseline_label}" if len(data.stage_series) > 1 else "")
            + " on the frozen test split, pooled over all pixels of the split "
              "(2TP / (2TP + FP + FN) per stage); n/a: stage absent in ground truth and "
              "prediction."),
        "fpr_wound_type": (
            f"False-positive rate of {subject}"
            + (f" and {data.baseline_label}" if len(data.fpr_series) > 1 else "")
            + " on ulcer-free photographs: extra photographs per wound type and the "
              "ulcer-free photographs of the frozen test split, with Wilson 95 % "
              f"confidence intervals. A photograph counts as false positive if a predicted "
              f"region larger than {min_area} of the image remains."),
        "data_added": (
            f"Top: images added to the pool in cycle {data.cycle} per source and split. "
            "Bottom: images rejected by quality control, by reason."),
        "examples": (
            f"Frozen-test examples predicted by {subject} at its operating threshold "
            f"{thr}: best, median and worst per-image Dice among ulcer-containing "
            "photographs, and the ulcer-free photograph with the largest predicted area. "
            "Yellow contour: ground truth; blue fill: prediction."),
    }


def make_figures(data: ReportData, out_dir: Path, prefix: str,
                 progress: Callable[[str], None] = log.info) -> tuple[list[FigureFile], list[str]]:
    """Draw and save every figure the data allows. A figure that cannot be drawn is
    left out and explained in the returned notes; the report is still written."""
    captions = _captions(data)

    def examples() -> Figure | None:
        if data.examples is None:
            raise LookupError(data.examples_note)
        chosen = select_examples(data.examples.rows)
        if not chosen:
            raise LookupError("the per-image results list no frozen-test image")
        from .steps import load_predictor
        predictor = load_predictor(data.examples.architecture, data.examples.weights)
        return draw_examples(data.examples, chosen, predictor)

    specs: list[tuple[str, str, Callable[[], Figure | None]]] = [
        ("version_history", "half", lambda: draw_version_history(data.versions)),
        ("stage_dice", "half", lambda: draw_stage_dice(data.stage_series)),
        ("fpr_wound_type", "half", lambda: draw_fpr(data.fpr_categories, data.fpr_series)),
        ("data_added", "half", lambda: draw_data_added(data.data_rows, data.rejections)),
        ("examples", "full", examples),
    ]
    figures: list[FigureFile] = []
    notes: list[str] = []
    for key, width, draw in specs:
        try:
            with figure_style():
                fig = draw()
                if fig is None:
                    continue
                pdf, png = save_figure(fig, out_dir, f"{prefix}_{key}")
        except Exception as exc:                    # noqa: BLE001 -- reported in the document
            log.warning("report figure %s failed", key, exc_info=True)
            notes.append(f"Figure '{key.replace('_', ' ')}' is missing: {exc}")
            continue
        figures.append(FigureFile(key, captions[key], width, pdf, png))
        progress(f"figure {key.replace('_', ' ')} written")
    return figures, notes


def figure_rows(figures: Sequence[FigureFile]) -> list[tuple[FigureFile, ...]]:
    """Layout rows: consecutive half-width figures are paired side by side."""
    rows: list[tuple[FigureFile, ...]] = []
    pending: list[FigureFile] = []
    for fig in figures:
        if fig.width == "half":
            pending.append(fig)
            if len(pending) == 2:
                rows.append(tuple(pending))
                pending = []
            continue
        if pending:
            rows.append(tuple(pending))
            pending = []
        rows.append((fig,))
    if pending:
        rows.append(tuple(pending))
    return rows


# ====================================================================== rendering

def _environment(**options: Any) -> jinja2.Environment:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
                             undefined=jinja2.StrictUndefined, trim_blocks=True,
                             lstrip_blocks=True, keep_trailing_newline=True, **options)
    env.filters.update(count=fmt_count, num=fmt_num, pct=fmt_pct, shorten=shorten)
    return env


def _card_for_pdf(card: str | None) -> tuple[str | None, bool]:
    """The model card typeset in the PDF appendix, and whether it was shortened. A card
    of more than MODEL_CARD_MAX_LINES lines (its lineage tables grow with every version)
    is cut before the last heading that starts within the limit (outside fenced code)
    or, without one, at the last blank line within it, so the appendix never ends
    inside a table, list or paragraph; the HTML report and models/v{N}/model_card.md
    keep the whole card."""
    if not card:
        return None, False
    lines = card.splitlines()
    if len(lines) <= MODEL_CARD_MAX_LINES:
        return card, False
    headings: list[int] = []
    blanks: list[int] = []
    fenced = False
    for i, line in enumerate(lines[:MODEL_CARD_MAX_LINES + 1]):
        if _MD_FENCE.match(line):
            fenced = not fenced
        elif fenced:
            continue
        elif i > 0 and _MD_HEADING.match(line):
            headings.append(i)
        elif not line.strip():
            blanks.append(i)
    cut = (headings or blanks or [MODEL_CARD_MAX_LINES])[-1]
    return "\n".join(lines[:cut]).rstrip(), True


def render_latex(data: ReportData, figures: Sequence[FigureFile], pdf_note: str = "") -> str:
    """LaTeX source of the report (templates/report.tex.j2, custom delimiters, every
    value escaped by the finalize hook unless it is a Latex string)."""
    env = _environment(block_start_string="((*", block_end_string="*))",
                       variable_start_string="(((", variable_end_string=")))",
                       comment_start_string="((=", comment_end_string="=))",
                       autoescape=False, finalize=_latex_finalize)
    env.filters["breakable"] = latex_breakable
    card, truncated = _card_for_pdf(data.model_card)
    return env.get_template(LATEX_TEMPLATE).render(
        r=data, rows=figure_rows(figures), max_rows=MAX_LATEX_ROWS, pdf_note=pdf_note,
        card=markdown_to_latex(card) if card else None, card_truncated=truncated)


def render_html(data: ReportData, figures: Sequence[FigureFile], pdf_note: str = "",
                pdf_name: str = "") -> str:
    """Standalone HTML of the report (templates/report.html.j2, autoescaped, figures
    embedded as base64 PNG data URIs)."""
    env = _environment(autoescape=True)
    embedded = [{"key": f.key, "caption": f.caption, "width": f.width,
                 "uri": "data:image/png;base64," + base64.b64encode(f.png.read_bytes()).decode("ascii")}
                for f in figures]
    return env.get_template(HTML_TEMPLATE).render(
        r=data, figures=embedded, max_rows=MAX_HTML_ROWS, pdf_note=pdf_note, pdf_name=pdf_name,
        card=markdown_to_html(data.model_card) if data.model_card else None)


# ============================================================================ PDF

def latex_error_excerpt(log_text: str, limit: int = 300) -> str:
    """The first LaTeX error of a log ('! ...' or 'file:line: ...' and its context
    line), else the last log lines."""
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("!") or re.match(r"^\S+:\d+: ", line):
            return " ".join(x.strip() for x in lines[i:i + 3] if x.strip())[:limit]
    return " ".join(x.strip() for x in lines[-3:] if x.strip())[:limit] or "no message"


def _kill(proc: subprocess.Popen[bytes]) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)       # latexmk and its pdflatex children
        else:                                          # pragma: no cover (Windows)
            proc.kill()
    except ProcessLookupError:
        pass
    proc.communicate()


def compile_latex(tex_source: str, figures: Sequence[FigureFile], out_pdf: Path, *,
                  latexmk: str = "latexmk", timeout: float = LATEX_TIMEOUT_S) -> tuple[bool, str, str]:
    """Compile with ``latexmk -norc -pdf -interaction=nonstopmode -halt-on-error
    -no-shell-escape`` in a temporary directory that holds the source and the figure
    PDFs; the whole process group is killed after `timeout` seconds. On success the
    PDF replaces `out_pdf` atomically. Returns (ok, failure reason, LaTeX log)."""
    exe = shutil.which(latexmk)
    if exe is None:
        return False, f"{latexmk} not found on PATH", ""
    with tempfile.TemporaryDirectory(prefix="selftrain_report_") as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "report.tex").write_text(tex_source, encoding="utf-8")
        for fig in figures:
            shutil.copyfile(fig.pdf, tmp / fig.latex_name)
        cmd = [exe, "-norc", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
               "-file-line-error", "-no-shell-escape", "report.tex"]
        env = {**os.environ, "max_print_line": "1000", "error_line": "254", "half_error_line": "238"}
        proc = subprocess.Popen(cmd, cwd=tmp, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            log_file = tmp / "report.log"
            text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
            return False, f"latexmk timed out after {timeout:g} s", text
        log_file = tmp / "report.log"
        text = (log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists()
                else output.decode("utf-8", errors="replace"))
        built = tmp / "report.pdf"
        if proc.returncode != 0 or not built.is_file():
            return False, f"latexmk exit status {proc.returncode}: {latex_error_excerpt(text)}", text
        staged = out_pdf.with_name(f".{out_pdf.name}.tmp")
        shutil.copyfile(built, staged)
        os.replace(staged, out_pdf)
    return True, "", text


def _text_table(header: Sequence[str], rows: Iterable[Sequence[str]], widths: Sequence[int]) -> list[str]:
    def line(cells: Sequence[str]) -> str:
        return "  ".join(shorten(str(c), w).ljust(w) for c, w in zip(cells, widths)).rstrip()
    return [line(header), line(["-" * w for w in widths])] + [line(r) for r in rows]


def _wrapped(text: str, indent: str = "") -> list[str]:
    return textwrap.wrap(text, FALLBACK_LINE_CHARS, initial_indent=indent,
                         subsequent_indent=indent) or [""]


def fallback_lines(data: ReportData, reason: str) -> list[str]:
    """The text content of the fallback PDF: the same sections as the LaTeX report as
    monospaced plain text."""
    out = [f"SELF-RETRAINING CYCLE {data.cycle}: REPORT", *_wrapped(f"{data.date} | {data.version_line}"),
           "", *_wrapped(f"Note: typeset by the matplotlib fallback because LaTeX failed: {reason}"), ""]
    out += ["GATE DECISION", *_wrapped(data.gate_sentence)]
    if data.demo:
        out += _wrapped("Demo cycle: shortened training; the numbers are not representative.")
    out += [f"  {label}: {value}" for label, value in data.key_numbers] + [""]
    if data.steps:
        out += ["CYCLE STEPS"] + [ln for name, status, summary in data.steps
                                  for ln in _wrapped(f"{name:<9} {status:<8} {summary}")] + [""]
    out += ["DATA", *_wrapped(data.pool_line)]
    if data.data_rows:
        rows = [(r.source, fmt_count(r.received), r.rejected, r.added, r.train, r.val, r.photographs, r.license)
                for r in (*data.data_rows, *((data.data_total,) if data.data_total else ()))]
        out += _text_table(("source", "received", "rejected", "added", "train", "val", "photos", "licence"),
                           [[str(c) for c in row] for row in rows], (24, 8, 8, 6, 6, 4, 6, 20))
    out += ["", "QUALITY CONTROL"]
    out += (_text_table(("reason", "images", "examples"),
                        [(r.reason, str(r.count), ", ".join(r.examples)) for r in data.rejections],
                        (36, 6, 54)) if data.rejections else ["No image was rejected."])
    out += ["", "DUPLICATES"]
    out += (_text_table(("image", "matched photograph", "split", "action"),
                        [(d.image, d.matched, d.split, d.action) for d in data.duplicates],
                        (30, 30, 8, 28)) if data.duplicates else ["No duplicate was found."])
    out += ["", "DRIFT"]
    if data.drift is None:
        out.append("No drift test in this cycle.")
    else:
        out += _text_table(("feature", "KS D", "p", "p Bonferroni", "shifted"),
                           [(r.feature, fmt_num(r.statistic), fmt_num(r.p_value, 4),
                             fmt_num(r.p_adjusted, 4), "yes" if r.shifted else "no") for r in data.drift.rows],
                           (24, 8, 10, 12, 8))
    out += ["", "METRICS ON THE FROZEN TEST SPLIT"]
    out += _text_table(("metric", data.subject_label, data.baseline_label or ""),
                       [(m.label, m.candidate, m.baseline) for m in data.metric_rows], (50, 24, 24))
    if data.gate_details:
        out += ["", *_wrapped(data.gate_details)]
    if data.notes:
        out += ["", "NOTES"] + [ln for note in data.notes for ln in _wrapped(f"- {note}")]
    return out


def _text_page(lines: Sequence[str]) -> Figure:
    fig = Figure(figsize=A4_IN)
    step = 0.9 / FALLBACK_LINES_PER_PAGE
    for i, line in enumerate(lines):
        fig.text(0.06, 0.95 - i * step, line, family="monospace", fontsize=7.6, va="top",
                 parse_math=False)
    return fig


def _figure_page(figures: Sequence[FigureFile]) -> Figure:
    fig = Figure(figsize=A4_IN)
    slot = 0.9 / 2
    for i, item in enumerate(figures):
        top = 0.95 - i * slot
        ax = fig.add_axes((0.06, top - slot * 0.72, 0.88, slot * 0.7))
        ax.imshow(matplotlib.image.imread(str(item.png)))
        ax.set_axis_off()
        caption = "\n".join(textwrap.wrap(f"Figure {item.key.replace('_', ' ')}: {item.caption}", 110))
        fig.text(0.06, top - slot * 0.75, caption, fontsize=7, va="top", parse_math=False)
    return fig


def write_fallback_pdf(data: ReportData, figures: Sequence[FigureFile], out_pdf: Path,
                       reason: str) -> None:
    """Plain-layout A4 PDF with matplotlib's PdfPages: text pages with every section,
    pages with two figures each, then the model card as text. Written to a temporary
    file and moved over `out_pdf`."""
    lines = fallback_lines(data, reason)
    if data.model_card:
        lines += ["", f"APPENDIX: MODEL CARD OF {data.subject_label.upper()}"]
        lines += [ln for raw in data.model_card.splitlines() for ln in _wrapped(raw)]
    staged = out_pdf.with_name(f".{out_pdf.name}.tmp")
    with figure_style(), PdfPages(staged, metadata={"Creator": "selftrain.system.report",
                                                    "CreationDate": None}) as pdf:
        for start in range(0, len(lines), FALLBACK_LINES_PER_PAGE):
            pdf.savefig(_text_page(lines[start:start + FALLBACK_LINES_PER_PAGE]))
        for start in range(0, len(figures), 2):
            pdf.savefig(_figure_page(figures[start:start + 2]))
    os.replace(staged, out_pdf)


_PDF_PAGE = re.compile(rb"/Type\s*/Page(?![A-Za-z])")
_PDF_STREAM = re.compile(rb"(?<!end)stream\r?\n")


def pdf_page_count(path: Path) -> int:
    """Number of pages of a PDF: pypdf if installed, else the number of page objects
    (/Type /Page) in the file and in its Flate-compressed streams (object streams of
    PDF 1.5 files hold the page objects compressed)."""
    data = Path(path).read_bytes()
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"{path} is not a PDF file")
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None
    if PdfReader is not None:
        return len(PdfReader(str(path)).pages)
    chunks = [data]
    for match in _PDF_STREAM.finditer(data):
        end = data.find(b"endstream", match.end())
        if end < 0:
            break
        try:
            chunks.append(zlib.decompressobj().decompress(data[match.end():end]))
        except zlib.error:
            continue
    return sum(len(_PDF_PAGE.findall(chunk)) for chunk in chunks)


# ================================================================== orchestration

def _progress(emit: Emit | None) -> Callable[[str], None]:
    def report(message: str) -> None:
        log.info("[report] %s", message)
        if emit is not None:
            emit(STEP, "started", message)
    return report


def _check_cycle(cycle: Any) -> int:
    if isinstance(cycle, (bool, np.bool_)) or _int_or_none(cycle) is None or int(cycle) < 0:
        raise ValueError(f"cycle must be a non-negative integer, got {cycle!r}")
    return int(cycle)


def build_report(ws: Workspace, cycle: int, context_data: Mapping[str, Any] | None, *,
                 emit: Emit | None = None, latexmk: str = "latexmk",
                 timeout: float = LATEX_TIMEOUT_S) -> dict[str, Any]:
    """Write reports/cycle_{N}.html, reports/cycle_{N}.pdf (and the LaTeX source
    cycle_{N}.tex) and the figures under reports/figures/ (SPEC §3 step 9).

    `context_data` holds the transient results of the cycle; every key is optional:
      demo (bool), date (ISO-8601), batches [str], dataset_version, model_version
      (the candidate trained in the cycle), gate (gate.GateDecision.to_dict()),
      metrics (steps.summary_metrics), steps {name: {"status", "summary"}},
      received {source: labelled images loaded}, rejected [{"name", "source",
      "batch", "reason"}], duplicates [{"image", "matched", "split", "method",
      "action"}], drift (see normalise_drift), model_card (markdown; default: the
      candidate's model_card.md).
    Everything else is read from the workspace. `emit(step, status, message)` receives
    progress events (status "started"). Returns {"pdf": Path | None, "html": Path,
    "figures": [Path] (PDF and PNG of every figure), "tex": Path, "engine": "latex" |
    "matplotlib" | None, "pdf_error": reason LaTeX failed or "", "pages": int | None,
    "gate_sentence": str, "notes": [str], "summary": one line for the dashboard}."""
    cycle = _check_cycle(cycle)
    progress = _progress(emit)
    ws.reports.mkdir(parents=True, exist_ok=True)
    ws.figures.mkdir(parents=True, exist_ok=True)
    prefix = f"cycle_{cycle}"
    for stale in ws.figures.glob(f"{prefix}_*"):
        stale.unlink()
    progress("collecting results")
    data = collect(ws, cycle, context_data)
    progress("drawing figures")
    figures, notes = make_figures(data, ws.figures, prefix, progress)
    data = replace(data, notes=data.notes + tuple(notes))

    pdf: Path | None = ws.reports / f"{prefix}.pdf"
    tex = ws.reports / f"{prefix}.tex"
    tex.write_text(render_latex(data, figures), encoding="utf-8")
    progress("typesetting the PDF with LaTeX")
    ok, error, latex_log = compile_latex(tex.read_text(encoding="utf-8"), figures, pdf,
                                         latexmk=latexmk, timeout=timeout)
    failed_log = ws.reports / f"{prefix}_latex.log"
    engine: str | None = "latex"
    if not ok:
        progress(f"LaTeX failed ({error}); writing the matplotlib fallback PDF")
        if latex_log:
            failed_log.write_text(latex_log, encoding="utf-8")
        try:
            write_fallback_pdf(data, figures, pdf, error)
            engine = "matplotlib"
        except Exception as exc:                    # noqa: BLE001 -- recorded, HTML still written
            log.exception("fallback PDF failed")
            error = f"{error}; the fallback PDF failed too: {exc}"
            pdf, engine = None, None
    elif failed_log.exists():
        failed_log.unlink()
    pages = None
    if pdf is not None:
        try:
            pages = pdf_page_count(pdf)
        except (OSError, ValueError):
            pages = None
    pdf_note = "" if engine == "latex" else (
        f"The PDF was written by the matplotlib fallback because LaTeX failed: {error}"
        if engine == "matplotlib" else f"No PDF was written: {error}")
    html = ws.reports / f"{prefix}.html"
    html.write_text(render_html(data, figures, pdf_note, pdf.name if pdf else ""), encoding="utf-8")
    files = [p for f in figures for p in (f.pdf, f.png)]
    summary = (f"{prefix}.pdf ({pages if pages is not None else '?'} pages, {engine}) and "
               f"{prefix}.html, {len(figures)} figures" if pdf is not None
               else f"{prefix}.html, {len(figures)} figures; no PDF")
    progress(summary)
    return {"pdf": pdf, "html": html, "figures": files, "tex": tex, "engine": engine,
            "pdf_error": error if not ok else "", "pages": pages,
            "gate_sentence": data.gate_sentence, "notes": list(data.notes), "summary": summary}


def context_from_cycle(ctx: CycleContext) -> dict[str, Any]:
    """`context_data` of `build_report` from a running cycle: step summaries
    (ctx.results), labelled images per source (ctx.records), rejections
    (ctx.rejected), duplicates (the dedupe step's DedupeResult -- its step data if it
    has the DedupeResult.to_dict form with file names, else ctx.groups with
    ctx.accepted as the record list), drift (the drift step's data), candidate
    version, gate decision and metrics."""
    results = dict(getattr(ctx, "results", {}) or {})
    steps = {name: {"status": res.status, "summary": res.summary} for name, res in results.items()}
    received = Counter(str(getattr(r, "source", "unknown")) for r in ctx.records)
    rejected = [{"name": _record_name(rec), "source": str(getattr(rec, "source", "")),
                 "batch": str(getattr(rec, "batch", "")), "reason": str(reason)}
                for rec, reason in ctx.rejected]
    dedupe_data = results["dedupe"].data if "dedupe" in results else None
    if isinstance(dedupe_data, Mapping) and "inherited" in dedupe_data:
        duplicates = duplicates_from_dedupe(dedupe_data, ctx.accepted)
    else:
        duplicates = duplicates_from_dedupe(ctx.groups, ctx.accepted)
    drift = results.get("drift")
    batches = [str(getattr(b, "name", None) or Path(str(getattr(b, "path", b))).name)
               for b in ctx.batches]
    return {"demo": bool(ctx.demo), "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "batches": batches, "dataset_version": ctx.dataset_version,
            "model_version": ctx.model_version, "gate": ctx.artifacts.get("gate"),
            "metrics": dict(ctx.metrics), "steps": steps, "received": dict(received),
            "rejected": rejected, "duplicates": duplicates,
            "drift": dict(drift.data) if drift is not None and not drift.skipped else None}


def write_report(ws: Workspace, summary: Mapping[str, Any], *, emit: Emit | None = None,
                 latexmk: str = "latexmk", timeout: float = LATEX_TIMEOUT_S) -> dict[str, Any]:
    """Render the report of the cycle a stored summary describes (reports/cycle_{N}.json,
    written by `steps.cycle_summary`): `build_report` with the summary's cycle number
    and its ``report_context`` -- the `context_data` that `context_from_cycle` took
    from the running cycle -- so a re-rendered report shows that cycle's results,
    while the persistent state (registry, evaluations, manifest, model card) is read
    from the workspace as it is now. Returns the result of `build_report`; ValueError
    if `summary` is not such a summary."""
    if not isinstance(summary, Mapping):
        raise ValueError(f"a cycle summary must be a mapping, got {type(summary).__name__}")
    context = summary.get("report_context")
    if not isinstance(context, Mapping):
        raise ValueError("the cycle summary has no 'report_context' (written by "
                         "steps.cycle_summary); it cannot be rendered")
    return build_report(ws, _check_cycle(summary.get("cycle")), context, emit=emit,
                        latexmk=latexmk, timeout=timeout)
