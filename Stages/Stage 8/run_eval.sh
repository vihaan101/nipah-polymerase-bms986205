#!/bin/bash
# run_eval.sh — Stage 7 50ns Direct eval orchestrator (SunLab / local)
#
# Environment (optional overrides — see Stages/README.md):
#   NIPAH_PROJECT_ROOT, STAGE7_STAGE6_RESULTS_DIR, STAGE7_DIRECT_RESULTS_ROOT,
#   STAGE7_EVAL_50NS_DIR, STAGE7_EVAL_WORK_DIR, STAGE7_TARGET_CASE,
#   STAGE7_EVAL_SYNC_DIR, STAGE7_CONDA_ENV
#
# Usage:
#   ./run_eval.sh                    # all cases, default paths
#   STAGE7_TARGET_CASE=A_ERDRP_WT ./run_eval.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export NIPAH_PROJECT_ROOT="${NIPAH_PROJECT_ROOT:-$REPO_ROOT}"
export PYTHONPATH="${NIPAH_PROJECT_ROOT}/Stages/common:${PYTHONPATH:-}"

LOG_DIR="${STAGE7_EVAL_LOG_DIR:-${REPO_ROOT}/Stages/Stage 7/logs}"
mkdir -p "$LOG_DIR"
EVAL_LOG="${LOG_DIR}/eval_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$EVAL_LOG") 2>&1

echo "=== Stage 7 50ns direct evaluation (SunLab) ==="
echo "Repo: ${NIPAH_PROJECT_ROOT}"
echo "Case filter: ${STAGE7_TARGET_CASE:-all}"
echo "Log: ${EVAL_LOG}"

CONDA_ENV="${STAGE7_CONDA_ENV:-stage7eval}"
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "$CONDA_BASE" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
      conda activate "$CONDA_ENV"
    else
      echo "WARNING: conda env '${CONDA_ENV}' not found; using current python: $(command -v python3)"
    fi
  fi
fi

PYTHON="${STAGE7_PYTHON:-$(command -v python3)}"
echo "Python: ${PYTHON}"

WORK_BASE="${STAGE7_EVAL_WORK_DIR:-${STAGE7_EVAL_50NS_DIR:-${REPO_ROOT}/Stages/Stage 7/analysis_50ns_direct}}"
export STAGE7_EVAL_WORK_DIR="$WORK_BASE"
export STAGE7_EVAL_50NS_DIR="${STAGE7_EVAL_50NS_DIR:-$WORK_BASE}"
export STAGE7_EVAL_SYNC_DIR="${STAGE7_EVAL_SYNC_DIR:-}"

RESULTS_ROOT="${STAGE7_DIRECT_RESULTS_ROOT:-${REPO_ROOT}/stage7_50ns_direct/results}"
export STAGE7_DIRECT_RESULTS_ROOT="$RESULTS_ROOT"

echo "Trajectories: ${STAGE7_DIRECT_RESULTS_ROOT}"
echo "Eval work dir: ${STAGE7_EVAL_WORK_DIR}"

run_task() {
  local module=$1
  local label=$2
  echo "--- ${label} (${module}) ---"
  "${PYTHON}" -m "${module}"
}

echo "Pre-flight trajectory check..."
"${PYTHON}" -m md_eval.verify_eval_ready_trajectories \
  --results-root "${STAGE7_DIRECT_RESULTS_ROOT}" \
  --output-json "${WORK_BASE}/trajectory_eval_ready_manifest.json" \
  --output-md "${WORK_BASE}/trajectory_eval_ready_summary.md" \
  --fail-on-not-ready \
  ${STAGE7_TARGET_CASE:+--case "${STAGE7_TARGET_CASE}"}

run_task md_eval.analyze_rmsf "RMSF"
run_task md_eval.analyze_backbone_rmsd "Backbone RMSD"
run_task md_eval.analyze_ligand_rmsd "Ligand RMSD"
run_task md_eval.analyze_pocket_volume "Pocket volume"
run_task md_eval.analyze_pca "PCA"
run_task md_eval.analyze_hbond "H-bonds"
run_task md_eval.analyze_contacts "Contacts"
run_task md_eval.analyze_plif "PLIF"
run_task md_eval.analyze_mmgbsa "MM-GBSA"
run_task md_eval.analyze_interaction_entropy "Interaction entropy"
run_task md_eval.analyze_decomp "Decomposition"

echo "=== Evaluation complete ==="
echo "Outputs under: ${STAGE7_EVAL_WORK_DIR}"
