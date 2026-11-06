"""Photograph identity of new records (SPEC §3 step 3, feature F3).

Purpose
    Before a new image may enter the pool it is assigned to a *photograph*, the unit
    of every split. Two files show the same photograph when one is a re-encoding, a
    rotation or reflection, a crop or a re-export of the other. Assigning photographs
    before anything is written guarantees that (a) no photograph is ever divided
    between training and evaluation, and (b) the frozen evaluation sets never receive
    a copy of one of their own photographs.

Identity tests (the tests and thresholds of scripts/unify_dataset_v3.build_groups,
which built the thesis dataset, so the system never applies a weaker notion of
identity than the thesis did). Every image is decoded once with OpenCV (EXIF
orientation applied, i.e. the pixels the pool stores) and described by

    * pHash, 64 bit (imagehash, hash_size 8) ...... same photograph if Hamming <= 5
    * dHash, 64 bit (imagehash, hash_size 8) ...... same photograph if Hamming <= 8
    * a 32 x 32 grey-level grid, mean-centred and scaled to unit length: same
      photograph if the Pearson correlation, maximised over the eight rotations and
      reflections of the square (dihedral group D4), is >= 0.90 -- Roboflow
      augments by flipping and rotating, which no hash survives;
    * ORB, 1200 keypoints on a 480 x 480 grey image: same photograph if >= 25
      cross-checked Hamming matches are inliers of a RANSAC homography (5 px). This
      local-feature test catches crops; it is expensive, so it runs only against
      the 40 candidates with the highest dihedral correlation (shortlist).

Any single test suffices (union of the evidence), exactly as in build_groups.

Procedure
    1. New photographs within the cycle: union-find over the new records with the
       four tests plus the filename rule of build_groups (equal Roboflow stem after
       removing the ".rf.<hash>" export suffix, unify_dataset_v3.source_stem). The
       filename rule joins records of the SAME ingest batch only, because unrelated
       collections reuse names such as "100.jpg".
    2. Matches against the reference set: the four tests (no filename rule; pool
       names are namespaced) between every new record and every reference image --
       all pool images of every split plus the extra ulcer-free photographs on which
       false-positive rates are measured (pool/negatives_extra.csv).
    3. Decision per new photograph (group of step 1), from the matches of all of its
       records:
         * a match to a photograph of a frozen evaluation set (test, external, extra
           negatives) drops every image of the group and reports the evidence
           (frozen-test protection: the test split of every dataset version stays
           identical to version 0 and never gains a twin in training);
         * matches to pool photographs of more than one split are a split conflict:
           the group is dropped, because adding it to either split would give one of
           its images a twin on the other side;
         * matches to pool photographs of exactly one split (train or val): every
           image of the group inherits that split and the matched photograph id
           (the photograph with the most matching image pairs; ties: smallest id);
         * no match: the group is a new photograph, split later by the dataset step.

Cost: one decode per reference image per call (about 1 ms per small image, 5 ms per
640 px JPEG) plus ORB for the shortlisted pairs; no state is kept between calls.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

import cv2
import imagehash
import numpy as np
from PIL import Image

from ..config import ROOT
from . import dataset

_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:      # scripts/ is not a package (same idiom as build_seg_dataset)
    sys.path.insert(0, str(_SCRIPTS))
import unify_dataset_v3 as u  # noqa: E402  (identity tests and thresholds of the thesis dataset)

if TYPE_CHECKING:                      # type hints only
    from .workspace import Workspace

_LOG = logging.getLogger(__name__)
Log = Callable[[str], None]

EXTRA_NEGATIVES = "extra_negatives"     # pseudo-split of the extra ulcer-free photographs
FROZEN_SPLITS = frozenset(dataset.FROZEN_SPLITS) | {EXTRA_NEGATIVES}
METHODS = ("phash", "dhash", "pixel", "orb")   # order = priority when several tests fire


class RecordLike(Protocol):
    """What dedupe and the dataset step read from an ingested record (SPEC §2:
    `selftrain.system.ingest.Record`)."""
    path: Path
    polygons: list[tuple[int, np.ndarray]]
    source: str
    license: str
    batch: str


# ----------------------------------------------------------------- descriptors

@dataclass(frozen=True)
class Signature:
    """Global descriptors of one decoded image: pHash and dHash bits and the
    normalised grey-level grid of the dihedral correlation test."""
    phash: np.ndarray          # (PHASH_SIZE**2,) bool
    dhash: np.ndarray          # (DHASH_SIZE**2,) bool
    grid: np.ndarray           # (GRID_SIDE**2,) float32, zero mean, unit norm (zeros if flat)


def unit_grid(grey: np.ndarray) -> np.ndarray:
    """GRID_SIDE x GRID_SIDE bilinear thumbnail of a grey image, flattened,
    mean-centred and divided by its L2 norm (build_groups' norm_vec); a flat image
    gives the zero vector, which correlates with nothing."""
    v = cv2.resize(grey, (u.GRID_SIDE, u.GRID_SIDE)).astype(np.float32).ravel()
    v -= v.mean()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-6 else v


def describe(img_bgr: np.ndarray) -> Signature:
    """Signature of a decoded BGR image (pHash/dHash via imagehash with the hash
    sizes of unify_dataset_v3, grid from the grey image)."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return Signature(
        phash=imagehash.phash(pil, hash_size=u.PHASH_SIZE).hash.ravel().astype(bool),
        dhash=imagehash.dhash(pil, hash_size=u.DHASH_SIZE).hash.ravel().astype(bool),
        grid=unit_grid(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))


def read_signature(path: Path) -> Signature | None:
    """Signature of an image file, None if OpenCV cannot decode it."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if img is None else describe(img)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between the rows of two bit matrices
    (n, k) and (m, k) -> (n, m) int, computed as a.(1-b)^T + (1-a).b^T."""
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    return np.rint(fa @ (1.0 - fb).T + (1.0 - fa) @ fb.T).astype(np.int32)


def dihedral_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, m) matrix: correlation of grid a[i] with grid b[j], maximised over the
    eight dihedral transforms (unify_dataset_v3._dihedral_scores); negative
    correlations are reported as 0."""
    out = np.zeros((len(a), len(b)), np.float32)
    for i, vec in enumerate(a):
        out[i] = u._dihedral_scores(vec, b)
    return out


class OrbMatcher:
    """The ORB/RANSAC test of build_groups (unify_dataset_v3._describe and
    _feature_inliers: ORB_FEATURES keypoints on the FEATURE_SIDE x FEATURE_SIDE grey
    image, cross-checked Hamming matches, RANSAC homography with 5 px tolerance).
    The keypoints of a file are computed once and kept in a least-recently-used
    memo of at most `max_cached` files, which bounds memory on a large pool."""

    def __init__(self, max_cached: int = 512) -> None:
        if max_cached < 2:
            raise ValueError("max_cached must be >= 2 (both images of a pair are needed)")
        self._orb = cv2.ORB_create(nfeatures=u.ORB_FEATURES)
        self._features: OrderedDict[Path, tuple[Any, Any]] = OrderedDict()
        self._max_cached = max_cached

    def features(self, path: Path) -> tuple[Any, Any]:
        """ORB keypoints and descriptors of a file ((None, None) if unreadable)."""
        cached = self._features.get(path)
        if cached is not None:
            self._features.move_to_end(path)
            return cached
        feats = u._describe(path, self._orb)
        self._features[path] = feats
        if len(self._features) > self._max_cached:
            self._features.popitem(last=False)
        return feats

    def inliers(self, a: Path, b: Path) -> int:
        """Number of cross-checked ORB matches that are RANSAC homography inliers."""
        fa = self.features(a)
        return u._feature_inliers(*fa, *self.features(b), cv2.NORM_HAMMING)


@dataclass(frozen=True)
class ImageSet:
    """Signatures of a list of images stacked into matrices."""
    paths: list[Path]
    phash: np.ndarray          # (n, 64) bool
    dhash: np.ndarray          # (n, 64) bool
    grid: np.ndarray           # (n, GRID_SIDE**2) float32

    @classmethod
    def of(cls, paths: Sequence[Path], sigs: Sequence[Signature]) -> ImageSet:
        """Stack the signatures of `paths` (same order)."""
        if len(paths) != len(sigs):
            raise ValueError("one signature per path required")
        nbits_p, nbits_d, ngrid = u.PHASH_SIZE ** 2, u.DHASH_SIZE ** 2, u.GRID_SIDE ** 2
        return cls(list(paths),
                   np.array([s.phash for s in sigs], bool).reshape(len(sigs), nbits_p),
                   np.array([s.dhash for s in sigs], bool).reshape(len(sigs), nbits_d),
                   np.array([s.grid for s in sigs], np.float32).reshape(len(sigs), ngrid))

    def __len__(self) -> int:
        return len(self.paths)


Evidence = dict[int, tuple[str, float]]     # other image index -> (test, score)


def global_evidence(q: ImageSet, r: ImageSet) -> tuple[list[Evidence], np.ndarray]:
    """Tests 1-3 (pHash, dHash, dihedral pixel correlation) between every image of
    `q` and every image of `r`. Returns, per query image, the reference images that
    passed with the first test (in METHODS order) that fired and its score
    (Hamming distance or correlation), and the correlation matrix (for the ORB
    shortlist)."""
    ph, dh = hamming(q.phash, r.phash), hamming(q.dhash, r.dhash)
    corr = dihedral_correlation(q.grid, r.grid)
    out: list[Evidence] = []
    for i in range(len(q)):
        found: Evidence = {}
        for j in np.flatnonzero(ph[i] <= u.PHASH_THRESHOLD):
            found[int(j)] = ("phash", float(ph[i, j]))
        for j in np.flatnonzero(dh[i] <= u.DHASH_THRESHOLD):
            found.setdefault(int(j), ("dhash", float(dh[i, j])))
        for j in np.flatnonzero(corr[i] >= u.GRID_CORR):
            found.setdefault(int(j), ("pixel", float(corr[i, j])))
        out.append(found)
    return out, corr


def shortlist(corr_row: np.ndarray, exclude: int | None = None) -> list[int]:
    """The ORB_SHORTLIST candidates with the highest dihedral correlation
    (stable order; `exclude` removes the query itself when q and r coincide)."""
    order = [int(j) for j in np.argsort(-corr_row, kind="stable") if j != exclude]
    return order[:u.ORB_SHORTLIST]


# ----------------------------------------------------------------------- step 1

def stem_key(record: RecordLike) -> tuple[str, str]:
    """Key of the filename rule: (ingest batch, Roboflow stem without .rf.<hash>)."""
    return str(record.batch), u.source_stem(Path(record.path).name)


def group_records(keys: Sequence[tuple[str, str]], images: ImageSet,
                  orb: OrbMatcher) -> list[list[int]]:
    """Step 1: union-find over the new records with the filename rule (equal key,
    i.e. same batch and same Roboflow stem) and identity tests 1-4. ORB runs on the
    shortlist of each record and skips pairs that are already joined (as in
    build_groups). Returns the groups as sorted index lists, ordered by first index."""
    n = len(images)
    if len(keys) != n:
        raise ValueError("one filename key per record required")
    uf = u.UnionFind(n)
    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        by_key[key].append(i)
    for idxs in by_key.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    evidence, corr = global_evidence(images, images)
    for i, found in enumerate(evidence):
        for j in found:
            if j != i:
                uf.union(i, j)
    for i in range(n):
        for j in shortlist(corr[i], exclude=i):
            if uf.find(i) != uf.find(j) and \
                    orb.inliers(images.paths[i], images.paths[j]) >= u.ORB_MIN_INLIERS:
                uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


# ----------------------------------------------------------------------- step 2

@dataclass(frozen=True)
class Reference:
    """An image the new records are compared with: a pool image (name = pool name,
    photograph = its integer photograph id as text, split = its pool split) or an
    extra negative photograph (name = photograph = its file path, split =
    EXTRA_NEGATIVES)."""
    name: str
    photograph: str
    split: str
    path: Path


def match_references(new: ImageSet, refs: ImageSet, orb: OrbMatcher) -> list[Evidence]:
    """Step 2: identity tests 1-4 of every new record against every reference
    image; ORB runs on the shortlist of each record for references that no global
    test matched. Returns, per record, {reference index: (test, score)} with score
    = Hamming distance, correlation or RANSAC inlier count."""
    evidence, corr = global_evidence(new, refs)
    for i, found in enumerate(evidence):
        for j in shortlist(corr[i]):
            if j in found:
                continue
            n_in = orb.inliers(new.paths[i], refs.paths[j])
            if n_in >= u.ORB_MIN_INLIERS:
                found[j] = ("orb", float(n_in))
    return evidence


# ----------------------------------------------------------------------- result

@dataclass(frozen=True)
class PoolMatch:
    """Evidence that new record `record` shows the existing photograph
    `photograph` (pool split `split`): reference image `pool_name` passed identity
    test `method` with `score`. `via` is set when the evidence belongs to another
    record of the same new photograph (index of that record)."""
    record: int
    photograph: str
    split: str
    pool_name: str
    method: str
    score: float
    via: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


@dataclass(frozen=True)
class SplitConflict:
    """New record `record` belongs to a photograph that matches pool photographs of
    several splits; `matches` lists all the evidence of its group."""
    record: int
    matches: tuple[PoolMatch, ...]

    @property
    def splits(self) -> list[str]:
        """The pool splits the photograph matched."""
        return sorted({m.split for m in self.matches})

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"record": self.record, "splits": self.splits,
                "matches": [m.to_dict() for m in self.matches]}


@dataclass(frozen=True)
class DedupeResult:
    """Photograph assignment of the records of one cycle (indices into the record
    list given to dedupe). Every record appears exactly once: in one of
    `new_groups` (new photographs, one list of record indices each), in `inherited`
    (joins an existing train or val photograph), in `dropped_test_matches` (copy of
    a photograph of a frozen evaluation set -- test, external or extra negatives --
    never added) or in `conflicts` (matches several splits, never added).
    `n_duplicates` = records that are not the first image of a new photograph,
    i.e. copies of a pool photograph or of another record of the cycle
    (= number of records - number of new photographs)."""
    new_groups: list[list[int]]
    inherited: list[PoolMatch]
    dropped_test_matches: list[PoolMatch]
    n_duplicates: int
    conflicts: list[SplitConflict] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen = [i for g in self.new_groups for i in g]
        seen += [m.record for m in self.inherited]
        seen += [m.record for m in self.dropped_test_matches]
        seen += [c.record for c in self.conflicts]
        if sorted(seen) != list(range(len(seen))):
            raise ValueError("DedupeResult must assign every record index 0..n-1 exactly once")
        if any(not g for g in self.new_groups):
            raise ValueError("empty photograph group")
        if self.n_duplicates != len(seen) - len(self.new_groups):
            raise ValueError("n_duplicates must equal records - new photographs")
        if any(m.split in FROZEN_SPLITS for m in self.inherited):
            raise ValueError("a record cannot inherit a frozen evaluation split")
        if any(m.split not in FROZEN_SPLITS for m in self.dropped_test_matches):
            raise ValueError("dropped_test_matches must reference frozen splits only")

    @property
    def n_records(self) -> int:
        """Number of records the result covers."""
        return (sum(len(g) for g in self.new_groups) + len(self.inherited)
                + len(self.dropped_test_matches) + len(self.conflicts))

    def kept(self) -> list[int]:
        """Records that may enter the pool (new photographs and inherited ones)."""
        return sorted([i for g in self.new_groups for i in g] + [m.record for m in self.inherited])

    def summary(self) -> str:
        """One line for the dashboard; the four duplicate counts add up to
        n_duplicates."""
        within = sum(len(g) - 1 for g in self.new_groups)
        return (f"{self.n_records} images -> {len(self.new_groups)} new photographs; "
                f"{self.n_duplicates} duplicates: {within} within the cycle, "
                f"{len(self.inherited)} of train/val photographs, "
                f"{len(self.dropped_test_matches)} of frozen evaluation photographs (dropped), "
                f"{len(self.conflicts)} split conflicts (dropped)")

    def to_dict(self, records: Sequence[RecordLike] | None = None) -> dict[str, Any]:
        """JSON-ready form; with `records`, every match also names the record's file."""
        def with_file(d: dict[str, Any]) -> dict[str, Any]:
            if records is not None:
                d["file"] = str(records[d["record"]].path)
            return d
        return {
            "n_records": self.n_records, "n_new_photographs": len(self.new_groups),
            "n_duplicates": self.n_duplicates, "new_groups": [list(g) for g in self.new_groups],
            "inherited": [with_file(m.to_dict()) for m in self.inherited],
            "dropped_test_matches": [with_file(m.to_dict()) for m in self.dropped_test_matches],
            "conflicts": [with_file(c.to_dict()) for c in self.conflicts],
        }


# ----------------------------------------------------------------------- step 3

Hit = tuple[int, int, str, float]           # (record, reference index, test, score)


def _group_hits(group: Sequence[int], evidence: Sequence[Evidence]) -> list[Hit]:
    """All matches of the records of one group, ordered by record, test priority
    and reference index (deterministic choice of the reported evidence)."""
    hits = [(i, j, method, score) for i in group for j, (method, score) in evidence[i].items()]
    return sorted(hits, key=lambda h: (h[0], METHODS.index(h[2]), h[1]))


def _match_for(record: int, hits: Sequence[Hit], refs: Sequence[Reference]) -> PoolMatch:
    """The evidence reported for `record`: its own first hit among `hits`, else the
    first hit of a twin record of the same group (then `via` names that twin)."""
    own = [h for h in hits if h[0] == record]
    i, j, method, score = own[0] if own else hits[0]
    ref = refs[j]
    return PoolMatch(record, ref.photograph, ref.split, ref.name, method, score,
                     None if own else i)


def photograph_order(photograph: str) -> tuple[int, int, str]:
    """Sort key of photograph ids: integer ids in numeric order, then other ids
    (paths of extra negatives) in string order."""
    try:
        return 0, int(photograph), ""
    except ValueError:
        return 1, 0, photograph


def choose_photograph(hits: Sequence[Hit], refs: Sequence[Reference]) -> str:
    """Photograph a group joins: the one with the most matching (record, image)
    pairs; ties broken by the smallest photograph id (photograph_order)."""
    counts = Counter(refs[j].photograph for _, j, _, _ in hits)
    return min(counts, key=lambda p: (-counts[p], photograph_order(p)))


def decide(groups: Sequence[Sequence[int]], evidence: Sequence[Evidence],
           refs: Sequence[Reference]) -> DedupeResult:
    """Step 3: frozen match -> drop; several splits -> conflict; one split ->
    inherit; no match -> new photograph (see the module docstring)."""
    new_groups: list[list[int]] = []
    inherited: list[PoolMatch] = []
    dropped: list[PoolMatch] = []
    conflicts: list[SplitConflict] = []
    for group in groups:
        hits = _group_hits(group, evidence)
        if not hits:
            new_groups.append(list(group))
            continue
        frozen = [h for h in hits if refs[h[1]].split in FROZEN_SPLITS]
        splits = {refs[h[1]].split for h in hits}
        if frozen:
            dropped += [_match_for(i, frozen, refs) for i in group]
        elif len(splits) > 1:
            for i in group:
                matches = tuple(PoolMatch(i, refs[j].photograph, refs[j].split, refs[j].name,
                                          method, score, None if r == i else r)
                                for r, j, method, score in hits)
                conflicts.append(SplitConflict(i, matches))
        else:
            photo = choose_photograph(hits, refs)
            chosen = [h for h in hits if refs[h[1]].photograph == photo]
            inherited += [_match_for(i, chosen, refs) for i in group]
    n = sum(len(g) for g in groups)
    return DedupeResult(new_groups=new_groups,
                        inherited=sorted(inherited, key=lambda m: m.record),
                        dropped_test_matches=sorted(dropped, key=lambda m: m.record),
                        n_duplicates=n - len(new_groups),
                        conflicts=sorted(conflicts, key=lambda c: c.record))


# ------------------------------------------------------------------ entry points

def record_signatures(records: Sequence[RecordLike]) -> list[Signature]:
    """Signatures of the new records; an undecodable image is a contract violation
    (the quality step rejects unreadable images) and raises ValueError."""
    sigs = []
    for rec in records:
        sig = read_signature(Path(rec.path))
        if sig is None:
            raise ValueError(f"cannot decode {rec.path}: quality control must reject it first")
        sigs.append(sig)
    return sigs


def reference_set(references: Sequence[Reference], log: Log) -> tuple[list[Reference], ImageSet]:
    """Signatures of the reference images. A pool image that cannot be decoded
    raises FileNotFoundError: without it the frozen-test protection would be
    incomplete. An unreadable extra negative (a file outside the workspace) is
    skipped with a log line, as selftrain.evaluate.evaluate_negatives skips it."""
    kept: list[Reference] = []
    sigs: list[Signature] = []
    skipped = 0
    for ref in references:
        sig = read_signature(ref.path)
        if sig is None:
            if ref.split != EXTRA_NEGATIVES:
                raise FileNotFoundError(f"pool image {ref.name} ({ref.path}) cannot be read")
            skipped += 1
            continue
        kept.append(ref)
        sigs.append(sig)
    if skipped:
        log(f"[dedupe] {skipped} unreadable extra-negative photographs skipped")
    return kept, ImageSet.of([r.path for r in kept], sigs)


def find_duplicates(records: Sequence[RecordLike], references: Sequence[Reference],
                    log: Log | None = None) -> DedupeResult:
    """Assign every record to a photograph (steps 1-3 of the module docstring),
    comparing with `references` (pool images and extra negatives)."""
    log = log or _LOG.info
    if not records:
        return DedupeResult([], [], [], 0)
    paths = [Path(r.path) for r in records]
    new = ImageSet.of(paths, record_signatures(records))
    refs, ref_set = reference_set(references, log)
    log(f"[dedupe] {len(records)} new images vs {len(refs)} reference images")
    orb = OrbMatcher()
    groups = group_records([stem_key(r) for r in records], new, orb)
    result = decide(groups, match_references(new, ref_set, orb), refs)
    log(f"[dedupe] {result.summary()}")
    return result


def reference_images(ws: Workspace) -> list[Reference]:
    """Reference set of a workspace: every pool image (manifest order) with its
    photograph and split, then the extra negative photographs of
    pool/negatives_extra.csv (each its own photograph, split EXTRA_NEGATIVES)."""
    pool = ws.read_manifest()
    refs = [Reference(name, str(int(photo)), split, dataset.pool_file(ws, "images", name))
            for name, photo, split in zip(pool["name"], pool["photograph"], pool["split"])]
    refs += [Reference(str(p), str(p), EXTRA_NEGATIVES, p) for p in dataset.extra_negative_paths(ws)]
    return refs


def dedupe_records(ws: Workspace, records: Sequence[RecordLike],
                   log: Log | None = None) -> DedupeResult:
    """SPEC §3 step 3: photograph identity of `records` against the workspace pool
    (all splits) and the extra negatives, and within the cycle."""
    return find_duplicates(records, reference_images(ws), log=log)
