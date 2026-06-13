#!/bin/bash
set -euo pipefail

echo "=========================================================="
echo "  Stage 7 -- Parallel 10ns Direct Campaign (4x Spot H100)"
echo "  Target: Azure centralus"
echo "  $(date -u)"
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORCHESTRATOR="${SCRIPT_DIR}/run_full_eval_10ns_direct_azure.sh"
STANDALONE_EVAL="${SCRIPT_DIR}/run_eval_standalone.sh"

if [ ! -f "$ORCHESTRATOR" ]; then
  echo "ERROR: Orchestrator ${ORCHESTRATOR} not found."
  exit 1
fi

# Load infra env to perform central upload
INFRA_ENV="${SCRIPT_DIR}/../../../scripts/azure/.infra_env"
source "$INFRA_ENV"

# Perform central upload of run_eval.sh
echo "Performing central upload of run_eval.sh..."
az storage file upload \
  --share-name "$BIOMNI_SHARE_NAME" \
  --account-name "$BIOMNI_STORAGE_ACCT" \
  --account-key "$BIOMNI_STORAGE_KEY" \
  --path "stage7_scripts_pipeline_10ns/run_eval.sh" \
  --source "${SCRIPT_DIR}/run_eval.sh" -o none

# Cache golden image ID
echo "Resolving golden image ID..."
GOLDEN_IMAGE_ID=$(az image show --resource-group "$BIOMNI_RG_INFRA" --name "biomni-gpu-golden" --query id -o tsv 2>/dev/null || echo "")
export GOLDEN_IMAGE_ID
if [ -z "$GOLDEN_IMAGE_ID" ]; then
  echo "WARNING: Golden image not found. Using base Ubuntu."
fi

RUN_ID="h$(date +%m%d%H%M)"
CASES=("A_ERDRP_WT" "B_ERDRP_MUT" "C_BMS_WT" "D_BMS_MUT")

echo "Launching 4 parallel analysis campaigns..."
for CASE in "${CASES[@]}"; do
  EVAL_ID="${CASE}_${RUN_ID}"
  SHARE_DIR="stage7_evals_10ns_direct_${CASE}_${RUN_ID}"
  
  echo "Launching worker for CASE: ${CASE} (Log: worker_${CASE}.log)..."
  bash "$ORCHESTRATOR" "$EVAL_ID" "$SHARE_DIR" "true" "$CASE" > "worker_${CASE}.log" 2>&1 &
done

echo "Waiting for all 4 campaigns to initialize and start monitoring..."
echo "Monitor completion by checking Azure Files shares for eval_complete.txt"
wait

echo "Campaign launcher finished (all sub-orchestrators returned)."
