"""Deterministic checks of the thesis sources in docs/thesis (run before every review).

  python scripts/check_thesis.py [--thesis docs/thesis] [--json out.json]

Hard failures (exit code 1): strings listed in docs/private/thesis_forbidden.txt,
secret values from .env, e-mail addresses, citation keys missing from references.bib,
references to undefined labels, figures/tables without caption or label, missing image
files, LaTeX errors in main.log.
Warnings (listed for manual review): figure/table labels never referenced in the text,
literal numbers that no result file contains, literal numbers that a macro already
provides, acronyms missing from the glossary, first-person or marketing wording,
units without a space, undefined references and overfull boxes in main.log.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULT_FILES = ["results", "system_ws/registry.json", "system_ws/reports", "logs/e2e_run.log",
                "system_ws/models", "docs/thesis/references_notes.md",
                "docs/thesis/generated/factsheet.md"]  # notes: numbers reported by the cited literature
REF_CMDS = r"\\(?:ref|cref|Cref|autoref|eqref|pageref|vref|crefrange|Crefrange)\*?\{([^}]*)\}"
MARKETING = (r"\b(revolutionary|cutting[- ]edge|groundbreaking|game[- ]chang\w*|seamless(?:ly)?|"
             r"unprecedented|incredibl[ey]|amazing|best[- ]in[- ]class|world[- ]class|"
             r"powerful|effortless(?:ly)?|perfect(?:ly)?)\b")


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def strip_verbatim(text: str) -> str:
    return re.sub(r"\\begin\{(lstlisting|verbatim)\}.*?\\end\{\1\}", " ", text, flags=re.S)


def load_sources(thesis: Path) -> dict[str, str]:
    files = [thesis / "main.tex", *sorted((thesis / "chapters").rglob("*.tex"))]
    return {str(f.relative_to(thesis)): strip_comments(f.read_text(encoding="utf-8"))
            for f in files if f.exists()}


def macro_values(thesis: Path) -> dict[str, str]:
    p = thesis / "generated" / "macros.tex"
    if not p.exists():
        return {}
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", p.read_text()))


def result_numbers() -> tuple[list[float], str]:
    """All numbers found in the result files, and their concatenated text."""
    nums, texts = [], []
    for rel in RESULT_FILES:
        base = REPO / rel
        paths = [base] if base.is_file() else sorted(base.rglob("*")) if base.exists() else []
        for p in paths:
            if p.suffix.lower() not in {".json", ".txt", ".csv", ".log", ".md", ".yaml"}:
                continue
            if p.stat().st_size > 5_000_000:
                continue
            t = p.read_text(encoding="utf-8", errors="replace")
            texts.append(t)
            nums.extend(float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", t)[:200_000])
    return nums, "\n".join(texts)


def matches_result(literal: str, nums: list[float], numset: set) -> bool:
    clean = literal.replace(",", "").replace("{,}", "")
    try:
        v = float(clean)
    except ValueError:
        return True
    dec = len(clean.split(".")[1]) if "." in clean else 0
    for cand in (v, v / 100.0):
        d = dec if cand == v else dec + 2
        key = round(cand, d)
        if key in numset.get(d, set()):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thesis", default=str(REPO / "docs" / "thesis"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    thesis = Path(args.thesis)
    src = load_sources(thesis)
    chapters = {k: v for k, v in src.items() if k.startswith("chapters/")}
    body = {k: strip_verbatim(v) for k, v in chapters.items()}
    fail: list[str] = []
    warn: list[str] = []

    # 1. forbidden strings and secrets
    forb = REPO / "docs" / "private" / "thesis_forbidden.txt"
    patterns = [l.strip() for l in forb.read_text().splitlines()
                if l.strip() and not l.startswith("#")] if forb.exists() else []
    secrets = []
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                val = line.split("=", 1)[1].strip()
                if len(val) >= 6:
                    secrets.append(val)
    for name, text in src.items():
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.I):
                fail.append(f"forbidden string /{pat}/ in {name}: ...{text[max(0, m.start()-40):m.end()+40]!r}")
        for s in secrets:
            if s in text:
                fail.append(f"secret value from .env appears in {name}")
        for m in re.finditer(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}", text):
            fail.append(f"e-mail address in {name}: {m.group(0)}")

    # 2. citations
    bib = (thesis / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
    cited: dict[str, int] = {}
    for name, text in src.items():
        for m in re.finditer(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", text):
            for k in (x.strip() for x in m.group(1).split(",")):
                if k:
                    cited[k] = cited.get(k, 0) + 1
                    if k not in bib_keys:
                        fail.append(f"citation key not in references.bib: {k} ({name})")

    # 3. labels and references
    defined: dict[str, str] = {}
    for name, text in src.items():
        for m in re.finditer(r"\\label\{([^}]*)\}|\blabel=\{?([\w:.-]+)\}?", text):
            lab = m.group(1) or m.group(2)
            if lab in defined:
                fail.append(f"label defined twice: {lab} ({defined[lab]}, {name})")
            defined[lab] = name
    referenced: set[str] = set()
    for name, text in src.items():
        for m in re.finditer(REF_CMDS, text):
            for k in (x.strip() for x in m.group(1).split(",")):
                if k:
                    referenced.add(k)
                    if k not in defined:
                        fail.append(f"reference to undefined label: {k} ({name})")
    for k, where in sorted(defined.items()):
        if k.startswith(("fig:", "tab:", "lst:")) and k not in referenced:
            warn.append(f"{k} ({where}) is never referenced in the text")

    # 4. floats and images
    for name, text in body.items():
        for m in re.finditer(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}", text, flags=re.S):
            blk = m.group(2)
            line = text[:m.start()].count("\n") + 1
            if "\\caption" not in blk:
                fail.append(f"{m.group(1)} without caption: {name}:{line}")
            if "\\label" not in blk:
                fail.append(f"{m.group(1)} without label: {name}:{line}")
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text):
            f = m.group(1)
            cands = [thesis / f] + [thesis / (f + ext) for ext in (".pdf", ".png", ".jpg")]
            if not any(c.is_file() for c in cands):
                fail.append(f"missing image file: {f} ({name})")

    # 5. literal numbers vs macros and result files
    macros = macro_values(thesis)
    by_value: dict[str, list[str]] = {}
    for k, v in macros.items():
        by_value.setdefault(v, []).append(k)
    nums, _ = result_numbers()
    numset: dict[int, set] = {}
    for d in range(0, 6):
        numset[d] = {round(x, d) for x in nums}
    unverified, macro_dupes = [], []
    for name, text in body.items():
        if name.endswith(("00_titlepage.tex", "00_declaration.tex")):
            continue
        t = re.sub(r"\\(cite[pt]?|label|ref|cref|Cref|eqref|includegraphics|url|href|input)"
                   r"(\[[^\]]*\])?\{[^}]*\}", " ", text)
        for m in re.finditer(r"(?<![\w.\\{-])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{2,})(?![\w.]*\d)(?!\.\d)", t):
            lit = m.group(1)
            if re.fullmatch(r"(19|20)\d\d", lit):
                continue
            ctx = t[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")
            if lit in by_value:
                macro_dupes.append(f"{name}: {lit} is \\{by_value[lit][0]} -- use the macro | {ctx!r}")
            elif not matches_result(lit, nums, numset):
                unverified.append(f"{name}: {lit} | {ctx!r}")
    warn += [f"number not found in results: {u}" for u in unverified]
    warn += [f"literal duplicates a macro: {u}" for u in macro_dupes]

    # 6. acronyms vs glossary
    gl = src.get("chapters/00_glossary.tex", "")
    gloss = set()
    for row in re.findall(r"^(.*?)&", gl, flags=re.M):
        g = re.sub(r"\\ul\{([^}]*)\}|\\underline\{([^}]*)\}", lambda m: m.group(1) or m.group(2) or "", row)
        g = re.sub(r"\\[a-zA-Z]+|[{}\s]", "", g)
        if g:
            gloss.add(g)
    used: dict[str, str] = {}
    for name, text in body.items():
        if name.endswith(("00_glossary.tex", "00_titlepage.tex", "10_appendix.tex")):
            continue
        t = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?(\{[^}]*\})?", lambda m: m.group(0) if m.group(0).startswith(("\\textbf", "\\emph", "\\textit")) else " ", text)
        t = re.sub(r"\$[^$]*\$", " ", t)
        for m in re.finditer(r"(?<![\w-])([A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*(?:-[A-Z0-9][A-Za-z0-9]*)?)(?![\w])", t):
            a = m.group(1)
            if len(a) >= 2 and sum(c.isupper() for c in a) >= 2:
                used.setdefault(a, name)
    for a, where in sorted(used.items()):
        base = re.sub(r"s$", "", a)
        if a not in gloss and base not in gloss:
            warn.append(f"acronym not in glossary: {a} (first in {where})")

    # 7. style
    for name, text in body.items():
        if name.endswith(("00_titlepage.tex", "00_declaration.tex")):
            continue
        plain = re.sub(r"\$[^$]*\$", " ", text)
        for m in re.finditer(r"(?<![\\\w])(I|we|We|our|Our|us|my|My)(?![\w])", plain):
            ctx = plain[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
            warn.append(f"first person '{m.group(1)}' in {name}: {ctx!r}")
        for m in re.finditer(MARKETING, plain, flags=re.I):
            warn.append(f"marketing word '{m.group(1)}' in {name}")
        for m in re.finditer(r"\b\w+(n't|'re|'ve|'ll)\b|\bit's\b", plain, flags=re.I):
            warn.append(f"contraction '{m.group(0)}' in {name}")
        for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(px|ms|GB|MB|KB|s|h|min|dpi)\b", plain):
            warn.append(f"unit without space '{m.group(0)}' in {name}")
        for m in re.finditer(r"\d\\%", plain):
            ctx = plain[max(0, m.start() - 20):m.end() + 5].replace("\n", " ")
            warn.append(f"percent without space in {name}: {ctx!r}")

    # 8. LaTeX log
    log = thesis / "main.log"
    pages = None
    if log.exists():
        lt = log.read_text(encoding="latin-1")
        for m in re.finditer(r"^! (.*)$", lt, flags=re.M):
            fail.append(f"LaTeX error: {m.group(1)}")
        for m in re.finditer(r"LaTeX Warning: (Reference|Citation) `([^']*)' on page (\d+) undefined", lt):
            warn.append(f"LaTeX: undefined {m.group(1).lower()} {m.group(2)} (page {m.group(3)})")
        over = re.findall(r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)", lt)
        big = [float(x) for x in over if float(x) > 5]
        if big:
            warn.append(f"LaTeX: {len(big)} overfull boxes wider than 5 pt (max {max(big):.0f} pt)")
        m = re.search(r"Output written on main\.pdf \((\d+) pages", lt)
        pages = int(m.group(1)) if m else None

    unused_bib = sorted(bib_keys - set(cited))
    summary = {"fail": fail, "warn": warn, "cited_keys": len(cited), "bib_keys": len(bib_keys),
               "uncited_bib_keys": unused_bib, "labels": len(defined), "pages": pages}
    print(f"thesis check: {len(fail)} failures, {len(warn)} warnings; {len(cited)} of "
          f"{len(bib_keys)} bib entries cited; {len(defined)} labels; pages: {pages}")
    for f in fail:
        print("FAIL", f)
    for w in warn:
        print("WARN", w)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
