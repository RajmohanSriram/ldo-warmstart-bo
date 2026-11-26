#!/usr/bin/env bash
# Overnight chain (started 2026-11-03): GF180 comparison batch, then the SKY130 dataset.
# Both resume if restarted: just run this script again.
cd "$(dirname "$0")"
PY=./venv/bin/python
echo "=== $(date '+%F %T') GF180 batch start"
$PY -u overnight_batch.py > overnight_batch.log 2>&1
echo "=== $(date '+%F %T') GF180 batch exit code $?"
echo "=== $(date '+%F %T') SKY130 dataset start"
$PY -u gen_dataset.py --pdk sky130 --vout 3.3 > gen_dataset_sky130.log 2>&1
echo "=== $(date '+%F %T') SKY130 dataset exit code $?"
echo "=== $(date '+%F %T') ALL DONE"
