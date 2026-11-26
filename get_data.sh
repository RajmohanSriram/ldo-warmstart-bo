#!/usr/bin/env bash
# Unpack the GF180MCU data sets into ./dataset and print the line to point the code at them.
# If the .xz files are not in ./dataset (too large for the repository), download them from the
# GitHub Release "data-v1" of this repository into ./dataset first.
set -e
cd "$(dirname "$0")/dataset"
shopt -s nullglob
files=(*.jsonl.xz)
if [ ${#files[@]} -eq 0 ]; then
  echo "No .jsonl.xz files in dataset/. Download them from this repository's Releases page (tag data-v1) into dataset/ and rerun."
  exit 1
fi
for f in "${files[@]}"; do
  [ -f "${f%.xz}" ] || { echo "unpacking $f"; xz -dk "$f"; }
done
echo
echo "Done. Before training surrogates, run:"
echo "  export LDO_DATASET=$(pwd)"
