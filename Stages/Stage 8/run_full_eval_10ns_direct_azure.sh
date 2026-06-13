#!/bin/bash
# run_full_eval_10ns_direct_azure.sh — Stage 7 10ns Direct eval launcher
# Launches 4 spot H100 VMs (one per case) in parallel, polls sentinels, verifies results.
# Usage: ./run_full_eval_10ns_direct_azure.sh [EVAL_ID] [SHARE_DIR_BASELINE] [SKIP_UPLOAD] [CASE_FILTER]
# CASE_FILTER: optional, restricts to a single case (e.g. A_ERDRP_WT) for smoke testing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFRA_ENV="${SCRIPT_DIR}/../../../scripts/azure/.infra_env"

if [ ! -f "$INFRA_ENV" ]; then
  echo "ERROR: ${INFRA_ENV} not found."
  exit 1
fi
source "$INFRA_ENV"

export AZURE_CORE_LOGGING_ENABLE_LOG_FILE=false

EVAL_ID="${1:-direct}"
SHARE_DIR_BASELINE="${2:-stage7_evals_10ns_direct}"
SKIP_UPLOAD="${3:-false}"
CASE_FILTER="${4:-}"
MOUNT_POINT="/mnt/biomni"
VM_SIZE="Standard_NC40ads_H100_v5"
SCRIPTS_SHARE="stage7_scripts_pipeline_10ns"
POLL_INTERVAL=60
MAX_WAIT_SECS=$((12 * 3600))

# Parse 4 cases from azure_results_locations.txt
CASES_FILE="${SCRIPT_DIR}/../azure_results_locations.txt"
if [ ! -f "$CASES_FILE" ]; then
  echo "ERROR: ${CASES_FILE} not found."
  exit 1
fi
CASES=()
while IFS= read -r case_name; do
  CASES+=("$case_name")
done < <(grep -E '^\s*- [A-Z]_' "$CASES_FILE" | awk '{print $2}' | cut -d/ -f1 | sort -u)
if [ "${#CASES[@]}" -eq 0 ]; then
  echo "ERROR: No cases found in ${CASES_FILE}."
  exit 1
fi
if [ -n "$CASE_FILTER" ]; then
  CASES=("$CASE_FILTER")
  echo "CASE_FILTER active — running single case: ${CASE_FILTER}"
fi
echo "Cases to evaluate: ${CASES[*]}"

pick_golden_image() {
  local image_id
  for name in biomni-gpu-golden-stage7eval biomni-gpu-golden; do
    image_id=$(az image show \
      --resource-group "$BIOMNI_RG_INFRA" \
      --name "$name" \
      --query id -o tsv 2>/dev/null || true)
    if [ -n "$image_id" ]; then
      echo "$image_id"
      return 0
    fi
  done
  return 1
}

azure_file_exists() {
  az storage file exists \
    --share-name "$BIOMNI_SHARE_NAME" \
    --account-name "$BIOMNI_STORAGE_ACCT" \
    --account-key "$BIOMNI_STORAGE_KEY" \
    --path "$1" \
    --query exists -o tsv 2>/dev/null || echo "false"
}

# Upload analysis scripts + run_eval.sh to share (SKIP_UPLOAD=true skips this block only)
if [ "$SKIP_UPLOAD" != "true" ]; then
  echo "Uploading analysis scripts and run_eval.sh to ${SCRIPTS_SHARE}..."
  az storage directory create \
    --share-name "$BIOMNI_SHARE_NAME" \
    --account-name "$BIOMNI_STORAGE_ACCT" \
    --account-key "$BIOMNI_STORAGE_KEY" \
    --name "${SCRIPTS_SHARE}" -o none 2>/dev/null || true

  az storage file upload \
    --share-name "$BIOMNI_SHARE_NAME" \
    --account-name "$BIOMNI_STORAGE_ACCT" \
    --account-key "$BIOMNI_STORAGE_KEY" \
    --path "${SCRIPTS_SHARE}/run_eval.sh" \
    --source "${SCRIPT_DIR}/run_eval.sh" -o none

  for script_name in \
    stage7_eval_common.py \
    stage7_success_criteria.py \
    verify_eval_ready_trajectories.py \
    analyze_backbone_rmsd.py \
    analyze_ligand_rmsd.py \
    analyze_rmsf.py \
    analyze_hbond.py \
    analyze_contacts.py \
    analyze_mmgbsa.py \
    analyze_pca.py \
    analyze_pocket_volume.py \
    analyze_decomp.py \
    analyze_interaction_entropy.py \
    analyze_plif.py; do
    az storage file upload \
      --share-name "$BIOMNI_SHARE_NAME" \
      --account-name "$BIOMNI_STORAGE_ACCT" \
      --account-key "$BIOMNI_STORAGE_KEY" \
      --path "${SCRIPTS_SHARE}/${script_name}" \
      --source "${SCRIPT_DIR}/${script_name}" -o none
  done

  az storage file upload \
    --share-name "$BIOMNI_SHARE_NAME" \
    --account-name "$BIOMNI_STORAGE_ACCT" \
    --account-key "$BIOMNI_STORAGE_KEY" \
    --path "${SCRIPTS_SHARE}/stage7_production_validation.py" \
    --source "${SCRIPT_DIR}/../stage7_production_validation.py" -o none

  echo "Upload complete."
fi

VM_IMAGE=""
if GOLDEN=$(pick_golden_image); then
  VM_IMAGE="$GOLDEN"
else
  VM_IMAGE="Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"
  echo "WARNING: No golden image found — VM will install conda env from scratch (~30 min extra)."
fi

# launch_and_poll: creates VM for one case, bootstraps, polls sentinel. Runs as a background job.
launch_and_poll() {
  local CASE=$1
  local EVAL_SHARE_DIR="${SHARE_DIR_BASELINE}/${CASE}"
  local RG="biomni-eval-10ns-${EVAL_ID}-${CASE}"
  local VM_NAME="eval-10ns-h100-${EVAL_ID}-${CASE}"

  echo "[${CASE}] Creating resource group ${RG}..."
  az group create --name "$RG" --location "$BIOMNI_LOCATION" -o none
  az network vnet create \
    --resource-group "$RG" \
    --name "eval-vnet" \
    --subnet-name "eval-subnet" -o none

  local CI_FILE
  CI_FILE=$(mktemp /tmp/cloud-init-eval-XXXXXX)
  printf '#cloud-config\n# Eval bootstrap runs post-boot via az vm run-command invoke.\n' > "$CI_FILE"

  echo "[${CASE}] Launching spot VM ${VM_NAME}..."
  local RETRY=0
  local ERR_FILE="/tmp/vm_create_err_${EVAL_ID}_${CASE}.txt"
  while [ "$RETRY" -lt 5 ]; do
    if az vm create \
      --resource-group "$RG" \
      --name "$VM_NAME" \
      --location "$BIOMNI_LOCATION" \
      --size "$VM_SIZE" \
      --vnet-name "eval-vnet" \
      --subnet "eval-subnet" \
      --priority Spot --max-price -1 \
      --eviction-policy Deallocate \
      --image "$VM_IMAGE" \
      --admin-username azureuser \
      --ssh-key-values ~/.ssh/id_rsa.pub \
      --os-disk-size-gb 256 \
      --assign-identity \
      --custom-data "@${CI_FILE}" \
      -o none 2>"$ERR_FILE"; then
      break
    else
      local ERR
      ERR=$(cat "$ERR_FILE")
      if echo "$ERR" | grep -qE "OverconstrainedAllocationRequest|SkuNotAvailable"; then
        echo "[${CASE}] Spot capacity unavailable. Retry $((RETRY+1))/5 in 60s..."
        sleep 60
        RETRY=$((RETRY + 1))
      else
        echo "[${CASE}] ERROR: VM creation failed: $ERR"
        rm -f "$CI_FILE" "$ERR_FILE"
        return 1
      fi
    fi
  done
  rm -f "$CI_FILE" "$ERR_FILE"

  if [ "$RETRY" -eq 5 ]; then
    echo "[${CASE}] Spot capacity exhausted — skipping case."
    return 1
  fi

  # Grant VM identity permission to self-deallocate
  local PRINCIPAL_ID
  PRINCIPAL_ID=$(az vm show --resource-group "$RG" --name "$VM_NAME" \
    --query identity.principalId -o tsv 2>/dev/null || echo "")
  if [ -n "$PRINCIPAL_ID" ]; then
    local VM_RES
    VM_RES=$(az vm show --resource-group "$RG" --name "$VM_NAME" --query id -o tsv)
    az role assignment create \
      --assignee "$PRINCIPAL_ID" \
      --role "Virtual Machine Contributor" \
      --scope "$VM_RES" -o none
  fi

  # Bootstrap script: mounts share, copies run_eval.sh, injects env vars, launches as azureuser.
  # Variables expanded at launcher time; runtime shell sees hardcoded values.
  # SHARE_DIR_BASELINE is NOT passed — only the pre-composed STAGE7_EVAL_SHARE_DIR is injected.
  local BOOTSTRAP_FILE
  BOOTSTRAP_FILE=$(mktemp /tmp/bootstrap-eval-XXXXXX)
  cat > "$BOOTSTRAP_FILE" << ENDBOOTSTRAP
#!/bin/sh
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y cifs-utils util-linux sudo >/dev/null 2>&1 || true

export MOUNT_POINT="${MOUNT_POINT}"
export STORAGE_ACCT="${BIOMNI_STORAGE_ACCT}"
export STORAGE_KEY="${BIOMNI_STORAGE_KEY}"
export SHARE_NAME="${BIOMNI_SHARE_NAME}"
export SCRIPTS_SHARE="${SCRIPTS_SHARE}"
export STAGE7_TARGET_CASE="${CASE}"
export STAGE7_EVAL_SHARE_DIR="${EVAL_SHARE_DIR}"

mkdir -p "\${MOUNT_POINT}"
if ! mountpoint -q "\${MOUNT_POINT}"; then
  mount -t cifs "//\${STORAGE_ACCT}.file.core.windows.net/\${SHARE_NAME}" "\${MOUNT_POINT}" \
    -o "vers=3.0,username=\${STORAGE_ACCT},password=\${STORAGE_KEY},dir_mode=0777,file_mode=0777,serverino,cache=none"
fi

id azureuser >/dev/null 2>&1 || useradd -m -s /bin/bash azureuser
mkdir -p /home/azureuser
chown -R azureuser:azureuser /home/azureuser

cp "\${MOUNT_POINT}/\${SCRIPTS_SHARE}/run_eval.sh" /home/azureuser/run_eval.sh
chmod +x /home/azureuser/run_eval.sh
chown azureuser:azureuser /home/azureuser/run_eval.sh

nohup sudo -H -u azureuser -E bash /home/azureuser/run_eval.sh >/dev/null 2>&1 &
ENDBOOTSTRAP

  echo "[${CASE}] Bootstrapping VM..."
  local BOOT_RETRY=0
  while [ "$BOOT_RETRY" -lt 3 ]; do
    if az vm run-command invoke \
      --resource-group "$RG" \
      --name "$VM_NAME" \
      --command-id RunShellScript \
      --scripts "@${BOOTSTRAP_FILE}" \
      -o none; then
      break
    else
      BOOT_RETRY=$((BOOT_RETRY + 1))
      if [ "$BOOT_RETRY" -lt 3 ]; then
        echo "[${CASE}] Bootstrap attempt failed, retrying in 30s ($((BOOT_RETRY+1))/3)..."
        sleep 30
      else
        echo "[${CASE}] ERROR: Bootstrap failed after 3 attempts."
        rm -f "$BOOTSTRAP_FILE"
        return 1
      fi
    fi
  done
  rm -f "$BOOTSTRAP_FILE"

  echo "[${CASE}] Polling for completion sentinel..."
  local ELAPSED=0
  while true; do
    if [ "$(azure_file_exists "${EVAL_SHARE_DIR}/eval_complete.txt")" = "true" ]; then
      echo "[${CASE}] DONE."
      return 0
    fi
    if [ "$(azure_file_exists "${EVAL_SHARE_DIR}/eval_failed.txt")" = "true" ]; then
      echo "[${CASE}] FAILED. Check Azure Files: ${EVAL_SHARE_DIR}/eval_vm.log"
      return 1
    fi
    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
    printf "[%s] %ds elapsed...\n" "$CASE" "$ELAPSED"
    if [ "$ELAPSED" -ge "$MAX_WAIT_SECS" ]; then
      echo "[${CASE}] ERROR: Timed out after $((MAX_WAIT_SECS / 3600))h."
      return 1
    fi
  done
}

# Launch all cases in parallel
echo "Launching ${#CASES[@]} VMs in parallel..."
PIDS=()
for i in "${!CASES[@]}"; do
  CASE="${CASES[$i]}"
  launch_and_poll "$CASE" &
  PIDS[$i]=$!
done

# Collect per-case results
CASE_STATUS=()
FAILED_CASES=()
for i in "${!CASES[@]}"; do
  CASE="${CASES[$i]}"
  if wait "${PIDS[$i]}"; then
    CASE_STATUS[$i]="PASS"
  else
    CASE_STATUS[$i]="FAIL"
    FAILED_CASES+=("$CASE")
  fi
done

# Post-completion verification: confirm metric index exists for each passing case
echo ""
echo "=== POST-COMPLETION VERIFICATION ==="
VERIFY_FAILED=()
for i in "${!CASES[@]}"; do
  CASE="${CASES[$i]}"
  if [ "${CASE_STATUS[$i]}" != "PASS" ]; then
    continue
  fi
  INDEX_PATH="${SHARE_DIR_BASELINE}/${CASE}/rmsd_10ns_direct/backbone_plateau_metric_index.json"
  if [ "$(azure_file_exists "$INDEX_PATH")" = "true" ]; then
    echo "[${CASE}] backbone_plateau_metric_index.json: PRESENT"
  else
    echo "[${CASE}] backbone_plateau_metric_index.json: MISSING"
    VERIFY_FAILED+=("$CASE")
  fi
done

# Cleanup resource groups
echo ""
echo "Cleaning up resource groups..."
for CASE in "${CASES[@]}"; do
  RG="biomni-eval-10ns-${EVAL_ID}-${CASE}"
  az group delete --name "$RG" --yes --no-wait 2>/dev/null || true
done

# Final summary
echo ""
echo "=== SUMMARY ==="
for i in "${!CASES[@]}"; do
  CASE="${CASES[$i]}"
  echo "  ${CASE}: ${CASE_STATUS[$i]}"
done

if [ "${#FAILED_CASES[@]}" -gt 0 ]; then
  echo "FAILED cases: ${FAILED_CASES[*]}"
  exit 1
fi
if [ "${#VERIFY_FAILED[@]}" -gt 0 ]; then
  echo "WARNING: Metric index missing for: ${VERIFY_FAILED[*]}"
  exit 1
fi
echo "All ${#CASES[@]} cases completed and verified."
