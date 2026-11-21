#!/usr/bin/env bash
# Reproduces every number, table and figure from scratch; the same steps as `make all`.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
python -m demand.pipeline data
python -m demand.pipeline panel
python -m demand.pipeline run
python -m demand.pipeline report
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_analysis.ipynb --ExecutePreprocessor.timeout=600
echo "done: see README.md, docs/index.html and results/"
