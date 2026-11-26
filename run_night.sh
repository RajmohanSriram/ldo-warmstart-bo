#!/usr/bin/env bash
# Overnight dataset run: GF180 5 V first, then 3.3 V. Safe to re-run: finished samples are skipped.
cd ~/ldo_project/optimizer || exit 1
source venv/bin/activate
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
mkdir -p dataset
python gen_dataset.py --vout 5.0 --sobol 20000 --local 8000 --test 1000 2>&1 | tee -a dataset/night_5V.log
python gen_dataset.py --vout 3.3 --sobol 20000 --local 8000 --test 1000 2>&1 | tee -a dataset/night_3v3.log
echo "ALL DONE $(date)" | tee -a dataset/night_5V.log
