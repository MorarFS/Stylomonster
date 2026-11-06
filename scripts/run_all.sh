#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$ROOT_DIR/.venv/bin/python"
PYTEST="$ROOT_DIR/.venv/bin/pytest"

echo "=================================================================="
echo "STYLOMETRY TOOLKIT — END-TO-END PIPELINE & BENCHMARK SUITE"
echo "=================================================================="

echo "[Step 0] Running pytest unit & integration test suite..."
"$PYTEST" "$ROOT_DIR/tests" -v

echo "[Step 1] Verifying & parsing benchmark corpora..."
"$PYTHON" "$SCRIPT_DIR/01_download_corpora.py"

echo "[Step 2] Extracting tabular stylometric feature matrices (CSV)..."
"$PYTHON" "$SCRIPT_DIR/02_extract_features.py"

echo "[Step 3] Running 12-method cross-validation & disputed authorship benchmark..."
"$PYTHON" "$SCRIPT_DIR/03_test_authorship.py"

echo "[Step 4] Running Zeta vocabulary, Rolling Stylometry, Unmasking & BCT stability..."
"$PYTHON" "$SCRIPT_DIR/04_rolling_zeta_unmasking.py"

echo "[Step 5] Running sample forensic attribution report on Federalist No. 51..."
"$PYTHON" "$SCRIPT_DIR/05_attribute_custom_text.py" --corpus federalist --paper 51

echo "=================================================================="
echo "ALL STYLOMETRY SCRIPTS AND BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "=================================================================="
