#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
case "${1:---analysis}" in
  --check)
    "$PY" scripts/verify_artifact.py
    "$PY" src/test_ground_truth.py
    "$PY" src/test_stats.py
    "$PY" src/test_cf_scoring.py
    ;;
  --analysis)
    "$PY" scripts/run_oracle.py
    "$PY" scripts/aggregate.py
    "$PY" scripts/rev2611_analysis.py
    "$PY" scripts/rev2611_code_behaviour.py
    MPLBACKEND=Agg "$PY" scripts/make_figures.py
    ;;
  *) echo "Usage: bash scripts/run_all.sh [--check|--analysis]" >&2; exit 2 ;;
esac
