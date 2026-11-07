#!/bin/bash
# Data-growth demonstration of the self-retraining system (WP4), exactly as run for the
# thesis (logs/run_piid_growth.sh, logs/run_growth.sh), from the repository root:
#
#   bash scripts/data_growth.sh
#
# 1. scripts/make_growth_batches.py -> data/growth/batch{1..4}.txt, data/growth/seed_v25/,
#    data/growth/inbox_batch{2..4}/ (train photographs of data/seg in four seeded batches)
# 2. initial U-Net on batch 1        -> runs/growth/v25/best.pt     (logs/train_growth_v25.log)
# 3. selftrain init                  -> system_ws/ with model v0     (logs/growth_init.log)
# 4. three cycles, one inbox batch each (batches 2, 3, 4) -> models v1-v3, reports/
#                                                                     (logs/growth_cycle_{2,3,4}.log)
# 5. selftrain status                                                (logs/growth_status.log)
#
# WARNING: deletes and rebuilds system_ws/. The thesis run additionally switched on the
# Telegram notifications in system_ws/config.yaml; that does not change any result and is
# left out here. Deterministic up to GPU non-determinism of training (seeds are fixed).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs
$PY scripts/make_growth_batches.py
$PY scripts/train_model.py --model unet --data data/growth/seed_v25 --out runs/growth/v25 \
  > logs/train_growth_v25.log 2>&1
rm -rf system_ws
$PY -m selftrain init --workspace system_ws --from data/growth/seed_v25 --architecture unet \
  --weights runs/growth/v25/best.pt > logs/growth_init.log 2>&1
for b in 2 3 4; do
  cp -R "data/growth/inbox_batch$b" system_ws/inbox/
  $PY -m selftrain run --workspace system_ws > "logs/growth_cycle_$b.log" 2>&1
  echo "cycle with inbox_batch$b done"
done
$PY -m selftrain status --workspace system_ws > logs/growth_status.log 2>&1
