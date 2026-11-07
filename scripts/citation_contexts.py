"""Pair every in-text citation of the thesis with the verified note of the cited source.

  python scripts/citation_contexts.py [--thesis docs/thesis] [--out <json>]

Writes a list of {id, file, line, keys, sentence, notes{key: note}} for a faithfulness audit:
the sentence that carries the citation and, for each cited key, what references_notes.md says
the source supports.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def notes_by_key(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"^- \*\*([^*]+)\*\* — (.*)$", path.read_text(encoding="utf-8"), flags=re.M):
        out.setdefault(m.group(1).strip(), m.group(2).strip())
    return out


def sentence_around(text: str, pos: int) -> str:
    start = max(text.rfind(". ", 0, pos), text.rfind("\n\n", 0, pos), text.rfind("\\item", 0, pos))
    end_candidates = [i for i in (text.find(". ", pos), text.find("\n\n", pos)) if i != -1]
    end = min(end_candidates) if end_candidates else len(text)
    s = text[start + 1 if start >= 0 else 0:end + 1]
    s = re.sub(r"%.*", "", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thesis", default=str(REPO / "docs" / "thesis"))
    ap.add_argument("--out", default=str(REPO / "docs" / "thesis" / "generated" / "citation_contexts.json"))
    args = ap.parse_args()
    thesis = Path(args.thesis)
    notes = notes_by_key(thesis / "references_notes.md")
    items = []
    for f in sorted((thesis / "chapters").rglob("*.tex")):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", text):
            keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
            items.append({"id": len(items) + 1, "file": str(f.relative_to(thesis)),
                          "line": text.count("\n", 0, m.start()) + 1, "keys": keys,
                          "sentence": sentence_around(text, m.start()),
                          "notes": {k: notes.get(k, "(no note)") for k in keys}})
    Path(args.out).write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} citation occurrences, {len({k for i in items for k in i['keys']})} distinct keys -> {args.out}")


if __name__ == "__main__":
    main()
