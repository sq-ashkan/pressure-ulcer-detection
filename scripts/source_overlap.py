"""Cross-source overlap of the segmentation dataset (reproducible from its manifest).

  python scripts/source_overlap.py [--manifest data/seg/manifest.csv] [--out results/seg_source_overlap.json]

The dataset builder (scripts/build_seg_dataset.py) assigns every image a photograph
identity (union-find over the stem rule, pHash/dHash, dihedral correlation and ORB/RANSAC).
Entry [a][b] counts the photographs of source a that also occur in source b, i.e. the
photograph identities shared by both sources; the diagonal is the number of distinct
photographs of the source. Images dropped by the builder (DTI/Unstageable) are not counted.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "data/seg/manifest.csv"))
    ap.add_argument("--out", default=str(REPO / "results/seg_source_overlap.json"))
    args = ap.parse_args()
    by_src: dict[str, set[str]] = collections.defaultdict(set)
    with open(args.manifest, newline="") as f:
        for row in csv.DictReader(f):
            by_src[row["source"]].add(row["photograph"])
    srcs = sorted(by_src)
    matrix = {a: {b: len(by_src[a] & by_src[b]) for b in srcs} for a in srcs}
    multi = collections.Counter()
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for s, ids in by_src.items():
        for i in ids:
            owners[i].add(s)
    for i, o in owners.items():
        multi[len(o)] += 1
    out = {"manifest": str(Path(args.manifest).relative_to(REPO)),
           "photographs_per_source": {s: len(by_src[s]) for s in srcs},
           "distinct_photographs": len(owners),
           "photographs_by_number_of_sources": dict(sorted(multi.items())),
           "shared_photographs": matrix}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "shared_photographs"}, indent=1))


if __name__ == "__main__":
    main()
