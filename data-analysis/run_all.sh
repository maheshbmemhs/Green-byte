#!/bin/sh
# Rebuilds the clean data and all results from data/raw
set -e
cd "$(dirname "$0")"
python3 cleaning/prepare_data.py
python3 cleaning/export_clean_data.py
Rscript analysis/run_all.R
