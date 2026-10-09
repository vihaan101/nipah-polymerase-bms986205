#!/usr/bin/env bash
# EC2 user-data / launch bootstrap: clone repo, sync Stage 6 from S3, verify GPU env.
set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nipah-aws-env.sh
source "${AWS_DIR}/nipah-aws-env.sh"

export NIPAH_PROJECT_ROOT="${NIPAH_PROJECT_ROOT:-/opt/nipah}"
export NIPAH_REPO_URL="${NIPAH_REPO_URL:-https://github.com/vihaan101/nipah-polymerase-bms986205.git}"
export CONDA_ENV="${CONDA_ENV:-nipah-md}"
export AWS_DEFAULT_REGION="${NIPAH_EC2_REGION}"
STAGE6_S3_URI="${STAGE6_S3_URI:-s3://${NIPAH_S3_BUCKET}/${NIPAH_S3_PREFIX}/staging/stage6/results/}"

MINICONDA_DIR="${MINICONDA_DIR:-/home/ubuntu/miniconda3}"
if [[ -d /home/ec2-user/miniconda3 ]]; then
  MINICONDA_DIR="/home/ec2-user/miniconda3"
fi

log() { echo "[instance_bootstrap] $*"; }

log "NIPAH_PROJECT_ROOT=${NIPAH_PROJECT_ROOT}"

if [[ -f "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  log "WARNING: Miniconda not at ${MINICONDA_DIR}; expect custom AMI with conda"
fi

sudo mkdir -p "${NIPAH_PROJECT_ROOT}"
sudo chown "$(whoami):$(whoami)" "${NIPAH_PROJECT_ROOT}" || true

if [[ -d "${NIPAH_PROJECT_ROOT}/.git" ]]; then
  log "Updating existing clone..."
  git -C "${NIPAH_PROJECT_ROOT}" pull --ff-only
else
  log "Cloning ${NIPAH_REPO_URL}..."
  git clone --depth 1 "${NIPAH_REPO_URL}" "${NIPAH_PROJECT_ROOT}"
fi

STAGE6_LOCAL="${NIPAH_PROJECT_ROOT}/Stages/Stage 6/results"
mkdir -p "${STAGE6_LOCAL}"
log "Syncing Stage 6 from ${STAGE6_S3_URI}..."
aws s3 sync "${STAGE6_S3_URI}" "${STAGE6_LOCAL}/" --region "${NIPAH_S3_REGION}"

export PYTHONPATH="${NIPAH_PROJECT_ROOT}/Stages/common:${PYTHONPATH:-}"

AWS_DIR="${NIPAH_PROJECT_ROOT}/Stages/aws"
if [[ -x "${AWS_DIR}/verify_gpu_env.sh" ]]; then
  export CONDA_ENV
  "${AWS_DIR}/verify_gpu_env.sh"
else
  log "WARNING: verify_gpu_env.sh not found under repo; skipping"
fi

log "Bootstrap complete."
