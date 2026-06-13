#!/bin/bash
set -euo pipefail

cd /home/azureuser
export HOME=/home/azureuser
export XDG_CONFIG_HOME="/home/azureuser/.config"
mkdir -p "/home/azureuser/.config"

# These variables are injected by the bootstrap process
# MOUNT_POINT, STORAGE_ACCT, STORAGE_KEY, SHARE_NAME, SCRIPTS_SHARE, SHARE_DIR_BASELINE, STAGE7_TARGET_CASE

touch /home/azureuser/eval.log
exec > >(tee -a /home/azureuser/eval.log) 2>&1
echo "=== EVALUATION START ==="
if [ -n "${STAGE7_TARGET_CASE:-}" ]; then
  echo "Target Case: ${STAGE7_TARGET_CASE}"
fi

publish_failure() {
  local exit_code=$1
  echo "Publishing failure (exit code: ${exit_code})..."
  if ! mountpoint -q "${MOUNT_POINT}"; then
    sudo mount -t cifs "//${STORAGE_ACCT}.file.core.windows.net/${SHARE_NAME}" "${MOUNT_POINT}" \
      -o "vers=3.0,username=${STORAGE_ACCT},password=${STORAGE_KEY},dir_mode=0777,file_mode=0777,serverino,cache=none" || true
  fi

  if mountpoint -q "${MOUNT_POINT}" && [ -d "${MOUNT_POINT}/${SHARE_DIR_BASELINE}" ]; then
    cp /home/azureuser/eval.log "${MOUNT_POINT}/${SHARE_DIR_BASELINE}/eval_vm.log" 2>/dev/null || true
    printf "failed\n" > "${MOUNT_POINT}/${SHARE_DIR_BASELINE}/eval_failed.txt" 2>/dev/null || true
    printf "%s\n" "${exit_code}" > "${MOUNT_POINT}/${SHARE_DIR_BASELINE}/eval_exit_code.txt" 2>/dev/null || true
    rm -f "${MOUNT_POINT}/${SHARE_DIR_BASELINE}/eval_running.txt" 2>/dev/null || true
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

sudo mkdir -p "${MOUNT_POINT}"
if ! mountpoint -q "${MOUNT_POINT}"; then
  sudo mount -t cifs "//${STORAGE_ACCT}.file.core.windows.net/${SHARE_NAME}" "${MOUNT_POINT}" \
    -o "vers=3.0,username=${STORAGE_ACCT},password=${STORAGE_KEY},dir_mode=0777,file_mode=0777,serverino,cache=none"
fi
mkdir -p "${MOUNT_POINT}/${SHARE_DIR_BASELINE}"
echo "running" > "${MOUNT_POINT}/${SHARE_DIR_BASELINE}/eval_running.txt" 2>/dev/null || true
sync

echo "Waiting for dpkg lock..."
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
  echo "Lock held by another process, waiting..."
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

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "stage7eval"; then
  conda create -y -n stage7eval python=3.11
  conda run -n stage7eval conda install -y -c conda-forge openmm cudatoolkit pdbfixer rdkit openmmforcefields openff-toolkit numpy ambertools mdanalysis matplotlib pandas scipy
fi
STAGE7EVAL_PREFIX=$(conda env list | awk '$1=="stage7eval"{print $NF; exit}')
STAGE7EVAL_PYTHON="${STAGE7EVAL_PREFIX}/bin/python"
STAGE7EVAL_PIP="${STAGE7EVAL_PREFIX}/bin/pip"
"${STAGE7EVAL_PIP}" install --quiet prolif==2.1.0

rm -rf "/home/azureuser/workspace"
mkdir -p "/home/azureuser/workspace/Stages/Stage 7"
cd /home/azureuser/workspace

export STAGE7_AZURE_ONLY=1

# --- NVMe Local Storage Setup ---
NVME_MOUNT="/mnt/nvme"
NVME_DEV=$(lsblk -dno NAME,SIZE | awk '$2 ~ /2.9T|3T|3.2T/ {print "/dev/"$1; exit}')

if [ -n "$NVME_DEV" ]; then
  echo "Found NVMe local storage at $NVME_DEV. Mounting to $NVME_MOUNT..."
  sudo mkdir -p "$NVME_MOUNT"
  if ! mountpoint -q "$NVME_MOUNT"; then
    sudo mkfs.xfs -f "$NVME_DEV" >/dev/null 2>&1 || true
    sudo mount -o discard,defaults,noatime "$NVME_DEV" "$NVME_MOUNT"
    sudo chown azureuser:azureuser "$NVME_MOUNT"
  fi
  
  echo "Priming local NVMe cache..."
  mkdir -p "$NVME_MOUNT/stage7_10ns_direct/results"
  if [ -n "${STAGE7_TARGET_CASE:-}" ]; then
    echo "Copying DCDs for case ${STAGE7_TARGET_CASE} only..."
    mkdir -p "$NVME_MOUNT/stage7_10ns_direct/results/${STAGE7_TARGET_CASE}"
    cp -r "${MOUNT_POINT}/stage7_10ns_direct/results/${STAGE7_TARGET_CASE}" "$NVME_MOUNT/stage7_10ns_direct/results/"
  else
    echo "WARNING: No target case specified. Copying ALL results..."
    cp -r "${MOUNT_POINT}/stage7_10ns_direct/results" "$NVME_MOUNT/stage7_10ns_direct/"
  fi
  sudo ln -s "${MOUNT_POINT}/stage6" "$NVME_MOUNT/stage6"
  echo "Local cache ready."
  export STAGE7_AZURE_MOUNT="$NVME_MOUNT"
else
  echo "WARNING: No NVMe local storage found."
  export STAGE7_AZURE_MOUNT="${MOUNT_POINT}"
fi

export STAGE7_STAGE6_RESULTS_DIR="${MOUNT_POINT}/stage6"
export STAGE7_STAGE6_MANIFEST="${MOUNT_POINT}/stage6/stage6_manifest.json"
export STAGE7_EVAL_10NS_DIR="${MOUNT_POINT}/${SHARE_DIR_BASELINE}"
export STAGE7_AZURE_SYNC_DIR="${MOUNT_POINT}/${SHARE_DIR_BASELINE}"

for sf in stage7_eval_common.py stage7_success_criteria.py analyze_backbone_rmsd.py analyze_ligand_rmsd.py analyze_rmsf.py analyze_hbond.py analyze_contacts.py analyze_mmgbsa.py analyze_pca.py analyze_pocket_volume.py analyze_decomp.py analyze_interaction_entropy.py analyze_plif.py; do
  cp "${MOUNT_POINT}/${SCRIPTS_SHARE}/${sf}" "Stages/Stage 7/"
done

cd "Stages/Stage 7"

run_task() {
  local script=$1
  local metric=$2
  local out_subdir=$3

  if [ -f "${STAGE7_EVAL_10NS_DIR}/${out_subdir}/${metric}_metric_index.json" ]; then
    echo "[CHECKPOINT] Skipping ${script}: result found in ${out_subdir}."
    mkdir -p "${out_subdir}"
    cp -r "${STAGE7_EVAL_10NS_DIR}/${out_subdir}/." "${out_subdir}/" 2>/dev/null || true
    return 0
  fi

  echo "Running ${script}..."
  export PATH="${STAGE7EVAL_PREFIX}/bin:$PATH"
  "${STAGE7EVAL_PYTHON}" "${script}" --case "${STAGE7_TARGET_CASE}" > "${script}.log" 2>&1
}

echo "Starting Light CPU tasks (Cores 4-10)..."
taskset -c 4 run_task analyze_rmsf.py rmsf rmsf_10ns_direct &
taskset -c 5 run_task analyze_backbone_rmsd.py backbone_plateau rmsd_10ns_direct &
taskset -c 6 run_task analyze_ligand_rmsd.py ligand_rmsd rmsd_10ns_direct &
taskset -c 7 run_task analyze_pocket_volume.py pocket_volume pocket_volume_10ns_direct &
taskset -c 8 run_task analyze_pca.py pca pca_10ns_direct &
taskset -c 9 run_task analyze_hbond.py hbonds hbonds_10ns_direct &
taskset -c 10 run_task analyze_contacts.py contacts contacts_10ns_direct &

echo "Starting Heavy CPU task (PLIF, Cores 11-39)..."
export STAGE7_PLIF_N_JOBS=29
taskset -c 11-39 run_task analyze_plif.py plif plif_10ns_direct &

echo "Starting GPU Track (MMGBSA, Interaction Entropy, Decomposition, Cores 0-3)..."
(
  taskset -c 0-3 run_task analyze_mmgbsa.py mmgbsa mmgbsa_10ns_direct
  taskset -c 0-3 run_task analyze_interaction_entropy.py interaction_entropy mmgbsa_10ns_direct
  taskset -c 0-3 run_task analyze_decomp.py decomp decomp_10ns_direct
) &

echo "Waiting for all analysis tracks to complete..."
wait

echo "All scripts finished. Copying results to mounted share..."
shopt -s dotglob nullglob
cp -r ./* "${STAGE7_EVAL_10NS_DIR}/"
shopt -u dotglob nullglob

cp /home/azureuser/eval.log "${STAGE7_EVAL_10NS_DIR}/eval_vm.log"
echo "eval_complete" > "${STAGE7_EVAL_10NS_DIR}/eval_complete.txt"
rm -f "${STAGE7_EVAL_10NS_DIR}/eval_failed.txt" "${STAGE7_EVAL_10NS_DIR}/eval_exit_code.txt" "${STAGE7_EVAL_10NS_DIR}/eval_running.txt"
sync
sleep 10
trap - EXIT

VMID=$(curl -s -H "Metadata:true" "http://169.254.169.254/metadata/instance?api-version=2021-02-01" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'/subscriptions/{d[\"compute\"][\"subscriptionId\"]}/resourceGroups/{d[\"compute\"][\"resourceGroupName\"]}/providers/Microsoft.Compute/virtualMachines/{d[\"compute\"][\"name\"]}')")
TOKEN=$(curl -s -H "Metadata:true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Length: 0" "https://management.azure.com${VMID}/deallocate?api-version=2023-07-01"
