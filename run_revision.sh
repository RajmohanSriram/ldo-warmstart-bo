#!/usr/bin/env bash
cd ~/ldo_project/optimizer
PY=./venv/bin/python
echo "=== $(date '+%F %T') queue start"
$PY -u revision_batch.py queue --jobs 4 --workers 3
echo "=== $(date '+%F %T') ensemble decisions"
$PY -u revision_batch.py ens
echo "=== $(date '+%F %T') screening baselines"
$PY -u revision_batch.py screen
$PY -u revision_batch.py report | tee runs_batch/report_rev.txt
tar czf ~/rev_results.tgz --exclude='*/scratch' runs_batch
cp ~/rev_results.tgz /mnt/c/Users/Admin/Downloads/
echo "=== $(date '+%F %T') ALL DONE -> Downloads/rev_results.tgz"
