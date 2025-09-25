#!/usr/bin/env bash
# Simple orchestration script to run the default experiments with structured output.
# Usage:
#   bash run_experiments.sh [output_dir]
# Default output_dir: experiments_glow_refined

set -euo pipefail

OUT_DIR=${1:-experiments_glow_refined}
mkdir -p "$OUT_DIR"

# Pre-flight checks
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found in PATH. Please install Python 3 or activate your environment." >&2
    exit 1
fi

if [ ! -f "glow_experiments.py" ]; then
    echo "Error: glow_experiments.py not found in current directory $(pwd). Run this script from the repository root." >&2
    exit 1
fi

# Activate conda environment if available
if command -v conda >/dev/null 2>&1 && [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
elif command -v conda >/dev/null 2>&1 && conda info --envs | grep -q "3MASB"; then
    echo "Activating conda environment 3MASB..."
    eval "$(conda shell.bash hook)"
    conda activate 3MASB
else
    echo "Warning: No conda environment detected. Using system python3."
fi

# Ensure current directory is importable by Python
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Run the experiments
echo "Running experiments (output -> $OUT_DIR)"
python3 glow_experiments.py --both --output "$OUT_DIR" --save_metrics --enable_covariance_analysis
EX_CODE=$?
if [ $EX_CODE -ne 0 ]; then
    echo "Experiment runner exited with code $EX_CODE" >&2
    exit $EX_CODE
fi

# After run, aggregate metrics into a CSV
echo "Aggregating metrics into CSV"
OUT_DIR="$OUT_DIR" python3 - <<PY
import os
from glow_experiments import GLOWExperimentRunner
base = os.environ['OUT_DIR']
runner = GLOWExperimentRunner(base)
runner.aggregate_all_metrics()
print('Aggregation complete.')
PY
