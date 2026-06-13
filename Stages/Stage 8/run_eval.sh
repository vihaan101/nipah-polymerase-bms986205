#!/bin/bash
# run_eval.sh — Stage 7 10ns Direct eval orchestrator
# Runs on the Azure spot H100 VM as azureuser.
# Injected env vars (set by bootstrap): MOUNT_POINT STORAGE_ACCT STORAGE_KEY SHARE_NAME
#   SCRIPTS_SHARE SHARE_DIR_BASELINE STAGE7_TARGET_CASE STAGE7_EVAL_SHARE_DIR
set -euxo pipefail

cd /home/azureuser
export HOME=/home/azureuser
export XDG_CONFIG_HOME="/home/azureuser/.config"
mkdir -p "/home/azureuser/.config"

touch /home/azureuser/eval.log
exec > >(tee -a /home/azureuser/eval.log) 2>&1
echo "=== EVALUATION START ==="
echo "Case: ${STAGE7_TARGET_CASE}"

# Launcher injects a share-relative directory (e.g. stage7_evals_10ns_smoke/CASE).
# Normalize it once so every write lands on the mounted Azure Files share.
if [[ "${STAGE7_EVAL_SHARE_DIR}" != /* ]]; then
  STAGE7_EVAL_SHARE_DIR="${MOUNT_POINT%/}/${STAGE7_EVAL_SHARE_DIR}"
fi
export STAGE7_EVAL_SHARE_DIR
echo "Eval share dir: ${STAGE7_EVAL_SHARE_DIR}"

publish_failure() {
  local exit_code=$1
  echo "Publishing failure (exit code: ${exit_code})..."
  if ! mountpoint -q "${MOUNT_POINT}"; then
    sudo mount -t cifs "//${STORAGE_ACCT}.file.core.windows.net/${SHARE_NAME}" "${MOUNT_POINT}" \
      -o "vers=3.0,username=${STORAGE_ACCT},password=${STORAGE_KEY},dir_mode=0777,file_mode=0777,serverino,cache=none" || true
  fi
  if mountpoint -q "${MOUNT_POINT}"; then
    mkdir -p "${STAGE7_EVAL_SHARE_DIR}"
    cp /home/azureuser/eval.log "${STAGE7_EVAL_SHARE_DIR}/eval_vm.log" 2>/dev/null || true
    printf "failed\n" > "${STAGE7_EVAL_SHARE_DIR}/eval_failed.txt" 2>/dev/null || true
    printf "%s\n" "${exit_code}" > "${STAGE7_EVAL_SHARE_DIR}/eval_exit_code.txt" 2>/dev/null || true
    rm -f "${STAGE7_EVAL_SHARE_DIR}/eval_running.txt" 2>/dev/null || true
    sync
  fi
}

on_exit() {
  local exit_code=$?
  if [ "${exit_code}" -ne 0 ]; then
    publish_failure "${exit_code}"
  fi
  exit "${exit_code}"
}
trap on_exit EXIT

# Mount share
sudo mkdir -p "${MOUNT_POINT}"
if ! mountpoint -q "${MOUNT_POINT}"; then
  sudo mount -t cifs "//${STORAGE_ACCT}.file.core.windows.net/${SHARE_NAME}" "${MOUNT_POINT}" \
    -o "vers=3.0,username=${STORAGE_ACCT},password=${STORAGE_KEY},dir_mode=0777,file_mode=0777,serverino,cache=none"
fi
mkdir -p "${STAGE7_EVAL_SHARE_DIR}"
echo "running" > "${STAGE7_EVAL_SHARE_DIR}/eval_running.txt"
sync

# Start NVMe staging in the background immediately — overlaps with conda env setup below.
# NVME_COPY_PID is waited on before analysis starts.
NVME_MOUNT="/mnt/nvme"
NVME_COPY_PID=""
NVME_DEV=$(lsblk -dno NAME,SIZE | awk '$2 ~ /1\.[6-9]T|1\.7T|1\.8T|1\.9T/ {print "/dev/"$1; exit}')
if [ -n "$NVME_DEV" ]; then
  echo "NVMe found at $NVME_DEV — formatting and staging ${STAGE7_TARGET_CASE} in background..."
  sudo mkdir -p "$NVME_MOUNT"
  if ! mountpoint -q "$NVME_MOUNT"; then
    sudo mkfs.xfs -f "$NVME_DEV" >/dev/null 2>&1 || true
    sudo mount -o discard,defaults,noatime "$NVME_DEV" "$NVME_MOUNT"
    sudo chown azureuser:azureuser "$NVME_MOUNT"
  fi
  sudo ln -s "${MOUNT_POINT}/stage6" "$NVME_MOUNT/stage6"
  mkdir -p "$NVME_MOUNT/stage7_10ns_direct/results"
  cp -r "${MOUNT_POINT}/stage7_10ns_direct/results/${STAGE7_TARGET_CASE}" \
        "$NVME_MOUNT/stage7_10ns_direct/results/" &
  NVME_COPY_PID=$!
fi

# Wait for cloud-init / dpkg to settle
echo "Waiting for dpkg lock..."
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
  echo "Lock held, waiting..."
  sleep 5
done

if ! nvidia-smi &>/dev/null; then
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-550-server cifs-utils --no-install-recommends
  sudo depmod -a || true
  sudo modprobe nvidia || true
else
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cifs-utils --no-install-recommends
fi

# Conda setup — DO NOT use conda activate or conda shell.bash hook:
# both trigger CONDA_MKL_INTERFACE_LAYER_BACKUP under set -u. Use env python directly.
if [ -d "/opt/miniforge3" ]; then
  export PATH="/opt/miniforge3/bin:$PATH"
fi
if [ -d "/home/azureuser/miniforge3" ]; then
  export PATH="/home/azureuser/miniforge3/bin:$PATH"
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "Installing Miniforge..."
  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "/home/azureuser/miniforge3"
  rm -f /tmp/miniforge.sh
  export PATH="/home/azureuser/miniforge3/bin:$PATH"
fi

if ! conda env list | awk '{print $1}' | grep -qx "stage7eval"; then
  conda create -y -n stage7eval python=3.11
  conda run -n stage7eval conda install -y -c conda-forge \
    openmm cudatoolkit pdbfixer rdkit openmmforcefields openff-toolkit \
    numpy ambertools mdanalysis matplotlib pandas scipy
fi
conda run -n stage7eval pip install --quiet prolif==2.1.0

export STAGE7EVAL_PREFIX
STAGE7EVAL_PREFIX=$(conda env list | awk '$1=="stage7eval"{print $NF; exit}')
export STAGE7EVAL_PYTHON="${STAGE7EVAL_PREFIX}/bin/python"
export STAGE7EVAL_PIP="${STAGE7EVAL_PREFIX}/bin/pip"

# Workspace
rm -rf "/home/azureuser/workspace"
mkdir -p "/home/azureuser/workspace/Stages/Stage 7"
cd /home/azureuser/workspace
export STAGE7_AZURE_ONLY=1

# Wait for background NVMe staging to finish (started before conda setup above)
if [ -n "$NVME_COPY_PID" ]; then
  echo "Waiting for NVMe staging to complete..."
  wait "$NVME_COPY_PID"
  echo "NVMe staging done."
  export STAGE7_AZURE_MOUNT="$NVME_MOUNT"
else
  echo "No NVMe staging in progress — using CIFS mount directly."
  export STAGE7_AZURE_MOUNT="${MOUNT_POINT}"
fi

export STAGE7_STAGE6_RESULTS_DIR="${MOUNT_POINT}/stage6"
export STAGE7_STAGE6_MANIFEST="${MOUNT_POINT}/stage6/stage6_manifest.json"
export STAGE7_EVAL_10NS_DIR="${STAGE7_EVAL_SHARE_DIR}"
export STAGE7_AZURE_SYNC_DIR="${STAGE7_EVAL_SHARE_DIR}"

# Copy named analysis scripts from share (explicit list avoids stale uploads)
for sf in \
  stage7_eval_common.py \
  stage7_success_criteria.py \
  stage7_production_validation.py \
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
  cp "${MOUNT_POINT}/${SCRIPTS_SHARE}/${sf}" "Stages/Stage 7/"
done

cd "Stages/Stage 7"

# Pre-flight: confirm trajectory is eval-ready before running analysis
echo "Running trajectory verifier..."
"${STAGE7EVAL_PYTHON}" verify_eval_ready_trajectories.py \
  --results-root "${STAGE7_AZURE_MOUNT}/stage7_10ns_direct/results" \
  --output-json  "${STAGE7_EVAL_SHARE_DIR}/trajectory_eval_ready_manifest.json" \
  --output-md    "${STAGE7_EVAL_SHARE_DIR}/trajectory_eval_ready_summary.md" \
  --fail-on-not-ready \
  --case "${STAGE7_TARGET_CASE}"
echo "Trajectory verifier passed."

run_task() {
  local script=$1
  local metric=$2
  local out_subdir=$3

  # Check Azure Files for a completed result from a previous run (handles VM restart)
  if [ -f "${STAGE7_EVAL_10NS_DIR}/${out_subdir}/${metric}_metric_index.json" ]; then
    echo "[CHECKPOINT] Skipping ${script}: ${metric}_metric_index.json found on share."
    mkdir -p "${out_subdir}"
    cp -r "${STAGE7_EVAL_10NS_DIR}/${out_subdir}/." "${out_subdir}/" 2>/dev/null || true
    return 0
  fi

  echo "Running ${script}..."
  export PATH="${STAGE7EVAL_PREFIX}/bin:$PATH"
  "${STAGE7EVAL_PYTHON}" "${script}" > "${script}.log" 2>&1

  # Flush this task's output directory to Azure Files immediately.
  # If the VM is restarted before the final batch upload, the checkpoint above
  # will skip this task on the next run.
  if [ -d "${out_subdir}" ]; then
    mkdir -p "${STAGE7_EVAL_10NS_DIR}/${out_subdir}"
    cp -r "${out_subdir}/." "${STAGE7_EVAL_10NS_DIR}/${out_subdir}/" 2>/dev/null || true
  fi
}
# export -f makes run_task available to bash -c subprocesses spawned by taskset
export -f run_task

echo "Starting light CPU tasks (cores 4-10)..."
ANALYSIS_PIDS=()
taskset -c 4  bash -c 'run_task analyze_rmsf.py rmsf rmsf_10ns_direct' &
ANALYSIS_PIDS+=($!)
taskset -c 5  bash -c 'run_task analyze_backbone_rmsd.py backbone_plateau rmsd_10ns_direct' &
ANALYSIS_PIDS+=($!)
taskset -c 6  bash -c 'run_task analyze_ligand_rmsd.py ligand_rmsd rmsd_10ns_direct' &
ANALYSIS_PIDS+=($!)
taskset -c 7  bash -c 'run_task analyze_pocket_volume.py pocket_volume pocket_volume_10ns_direct' &
ANALYSIS_PIDS+=($!)
taskset -c 8  bash -c 'run_task analyze_pca.py pca pca_10ns_direct' &
ANALYSIS_PIDS+=($!)
taskset -c 9  bash -c 'run_task analyze_hbond.py hbonds hbonds_10ns_direct' &
ANALYSIS_PIDS+=($!)
taskset -c 10 bash -c 'run_task analyze_contacts.py contacts contacts_10ns_direct' &
ANALYSIS_PIDS+=($!)

echo "Starting PLIF (cores 11-39, 29 joblib workers)..."
export STAGE7_PLIF_N_JOBS=29
taskset -c 11-39 bash -c 'run_task analyze_plif.py plif plif_10ns_direct' &
ANALYSIS_PIDS+=($!)

echo "Starting GPU track (cores 0-3): mmgbsa → IE → decomp..."
(
  taskset -c 0-3 bash -c 'run_task analyze_mmgbsa.py mmgbsa mmgbsa_10ns_direct'
  taskset -c 0-3 bash -c 'run_task analyze_interaction_entropy.py interaction_entropy mmgbsa_10ns_direct'
  taskset -c 0-3 bash -c 'run_task analyze_decomp.py decomp decomp_10ns_direct'
) &
ANALYSIS_PIDS+=($!)

echo "Waiting for all analysis tracks..."
for pid in "${ANALYSIS_PIDS[@]}"; do
  wait "$pid"
done
echo "All tracks done."

# Write-back block — sentinel ordering: results → log → complete → sync → sleep → deallocate
echo "Copying results to Azure Files..."
shopt -s dotglob nullglob
cp -r ./* "${STAGE7_EVAL_SHARE_DIR}/"
shopt -u dotglob nullglob

cp /home/azureuser/eval.log "${STAGE7_EVAL_SHARE_DIR}/eval_vm.log"
echo "eval_complete" > "${STAGE7_EVAL_SHARE_DIR}/eval_complete.txt"
rm -f "${STAGE7_EVAL_SHARE_DIR}/eval_running.txt"
sync
sleep 5
trap - EXIT

# Self-deallocate via instance metadata
echo "Deallocating VM..."
VMID=$(curl -sf -H "Metadata:true" \
  "http://169.254.169.254/metadata/instance?api-version=2021-02-01" | \
  python3 -c "import sys,json; d=json.load(sys.stdin)['compute']; print('/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Compute/virtualMachines/{}'.format(d['subscriptionId'],d['resourceGroupName'],d['name']))")
TOKEN=$(curl -sf -H "Metadata:true" \
  "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/" | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -s -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Length: 0" \
  "https://management.azure.com${VMID}/deallocate?api-version=2023-07-01"
echo "Deallocate request sent."
