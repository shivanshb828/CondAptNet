#!/usr/bin/env bash
# scripts/infra/run_all_smoke_tests.sh
#
# Orchestrates all GCP smoke tests in order. Run this after gcp_setup.sh
# has provisioned the VM and you have SSH'd into it.
#
# Usage (on the GCP VM):
#   cd ~/condaptnet
#   source condaptnet_env/bin/activate
#   bash scripts/infra/run_all_smoke_tests.sh
#
# Options:
#   --skip-dnabert2   Skip DNABERT-2 steps (faster; use if transformers not installed)
#   --n-steps N       Training steps per encoder for step 5 (default: 50)
#   --batch-size N    Batch size for step 5 (default: 16)
#   --require-cuda    Fail step 2 if CUDA is not available (always pass on GCP)
#
# Output: logs to smoke_test_results/YYYY-MM-DD_HHMMSS/
# Exit code: 0 = all steps passed, 1 = one or more failures

set -uo pipefail

SKIP_DNABERT2="false"
N_STEPS=50
BATCH_SIZE=16
REQUIRE_CUDA=""
TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")
LOG_DIR="smoke_test_results/${TIMESTAMP}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-dnabert2) SKIP_DNABERT2="true"; shift;;
        --n-steps)       N_STEPS="$2"; shift 2;;
        --batch-size)    BATCH_SIZE="$2"; shift 2;;
        --require-cuda)  REQUIRE_CUDA="--require-cuda"; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p "$LOG_DIR"
SUMMARY_FILE="$LOG_DIR/summary.txt"

PASS="[PASS]"
FAIL="[FAIL]"
WARN="[WARN]"

declare -A STEP_RESULTS

header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo ""
}

run_step() {
    local step_name="$1"
    local log_file="$LOG_DIR/${step_name}.log"
    shift
    local cmd="$@"

    header "Running: $step_name"
    echo "Command: $cmd"
    echo "Log: $log_file"
    echo ""

    local start_time
    start_time=$(date +%s)

    if eval "$cmd" 2>&1 | tee "$log_file"; then
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo ""
        echo "$PASS $step_name  (${elapsed}s)"
        STEP_RESULTS["$step_name"]="PASS (${elapsed}s)"
    else
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo ""
        echo "$FAIL $step_name  (${elapsed}s)"
        STEP_RESULTS["$step_name"]="FAIL (${elapsed}s)"
    fi
}

header "CondAptNet GCP Smoke Tests — $TIMESTAMP"
echo "Log directory: $LOG_DIR"
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'UNKNOWN')"
echo ""

# ── Step 1: Environment sanity ────────────────────────────────────────────────
# Quick sanity check that all imports work before running the heavier tests.
header "Pre-flight: import sanity"
python -c "
import torch, numpy, pandas, scipy, sklearn
import sys; sys.path.insert(0, '.')
import config
from models.condaptnet import CondAptNet
from scripts.model.tokenizer import DNATokenizer
from scripts.training.losses import CondAptNetLoss
print('[PASS] All core imports succeeded')
print(f'  torch   : {torch.__version__}')
print(f'  numpy   : {numpy.__version__}')
print(f'  pandas  : {pandas.__version__}')
print(f'  device  : {config.DEVICE}')
" 2>&1 | tee "$LOG_DIR/preflight.log"

# ESM-2 import check
python -c "import esm; print('[PASS] fair-esm import OK')" 2>&1 \
    | tee -a "$LOG_DIR/preflight.log" \
    || echo "[WARN] fair-esm not importable — protein encoder will fail" \
    | tee -a "$LOG_DIR/preflight.log"

# ViennaRNA check
python -c "import RNA; print(f'[PASS] ViennaRNA import OK: {RNA.__version__}')" 2>&1 \
    | tee -a "$LOG_DIR/preflight.log" \
    || echo "[WARN] ViennaRNA not importable — structure features will zero-out" \
    | tee -a "$LOG_DIR/preflight.log"

# ── Step 2: CUDA routing check ────────────────────────────────────────────────
run_step "step2_cuda_check" \
    python scripts/infra/smoke_cuda_check.py $REQUIRE_CUDA

# ── Step 3: Data pipeline ─────────────────────────────────────────────────────
SKIP_DB2_FLAG=""
[ "$SKIP_DNABERT2" = "true" ] && SKIP_DB2_FLAG="--skip-dnabert2"

run_step "step3_data_loading" \
    python scripts/infra/smoke_data_loading.py --n-rows 64 $SKIP_DB2_FLAG

# ── Step 4: Checkpoint save/load ──────────────────────────────────────────────
run_step "step4_checkpoint" \
    python scripts/infra/smoke_checkpoint.py --n-steps 5

# ── Step 5: Short training run ────────────────────────────────────────────────
if [ "$SKIP_DNABERT2" = "true" ]; then
    run_step "step5_training_scratch" \
        python scripts/infra/smoke_training.py \
            --encoder scratch \
            --n-steps "$N_STEPS" \
            --batch-size "$BATCH_SIZE" \
            --n-rows 256
else
    run_step "step5_training_both" \
        python scripts/infra/smoke_training.py \
            --encoder both \
            --n-steps "$N_STEPS" \
            --batch-size "$BATCH_SIZE" \
            --n-rows 256
fi

# ── Summary ───────────────────────────────────────────────────────────────────
header "Smoke Test Summary"
{
echo "CondAptNet GCP Smoke Test — $TIMESTAMP"
echo ""
echo "System:"
echo "  Python  : $(python --version 2>&1)"
echo "  PyTorch : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "  GPU     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")' 2>/dev/null)"
echo ""
echo "Results:"
} | tee "$SUMMARY_FILE"

ALL_PASS=true
for step_name in "${!STEP_RESULTS[@]}"; do
    result="${STEP_RESULTS[$step_name]}"
    status_icon="$PASS"
    if [[ "$result" == FAIL* ]]; then
        status_icon="$FAIL"
        ALL_PASS=false
    fi
    echo "  $status_icon  $step_name: $result" | tee -a "$SUMMARY_FILE"
done

echo "" | tee -a "$SUMMARY_FILE"
if [ "$ALL_PASS" = true ]; then
    echo "$PASS All smoke tests PASSED" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Ready to launch full A/B training run:" | tee -a "$SUMMARY_FILE"
    echo "  python scripts/training/train.py --use-amp --batch-size 32" | tee -a "$SUMMARY_FILE"
    exit 0
else
    echo "$FAIL One or more smoke tests FAILED — check logs in $LOG_DIR" | tee -a "$SUMMARY_FILE"
    echo "Fix the failures above before launching a full training run." | tee -a "$SUMMARY_FILE"
    exit 1
fi
