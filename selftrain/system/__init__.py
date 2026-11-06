"""selftrain.system -- the self-retraining system around the experiment library.

Every cycle runs the steps of STEP_NAMES in this order (SPEC section 3)::

    ingest -> quality -> dedupe -> drift -> dataset -> train -> evaluate -> gate -> report -> notify

and stops at the first step that fails. Each step reports its progress as events whose
status is one of STEP_STATUSES (SPEC section 4: one JSON object per line of events.jsonl,
which the dashboard tails). All state lives in one Workspace directory; the file
selftrain/system/SPEC.md is the contract between the modules:

  workspace   directory layout, pool manifest, SystemConfig (config.yaml), atomic writes
  ingest      batch formats accepted in inbox/
  dedupe      photograph identity of new images against the pool (frozen-test protection)
  dataset     pool and dataset versions datasets/v{N}
  steps       the ten cycle steps; cycle runs them and writes events and status
  gate        promotion rule (paired bootstrap of per-photograph test Dice)
  registry    model versions, current model, promote / rollback
  modelcard   generated model card per version
  active      active-learning ranking of unlabeled photographs
"""
from .workspace import SystemConfig, Workspace

STEP_NAMES: list[str] = [
    "ingest", "quality", "dedupe", "drift", "dataset",
    "train", "evaluate", "gate", "report", "notify",
]
"""The ten cycle steps in execution order, exactly as in SPEC section 4. A list like the
SPEC's, so it compares equal to step-name lists built by other modules; treat it as a
constant."""

STEP_STATUSES: tuple[str, ...] = ("started", "done", "failed", "skipped")
"""Allowed values of an event's "status" field (SPEC section 4)."""

__all__ = ["STEP_NAMES", "STEP_STATUSES", "SystemConfig", "Workspace"]
