#!/usr/bin/env bash
# Wrapper for Stage 7 50 ns direct production on EC2 (archive to nipah-archive).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nipah-aws-env.sh
source "${SCRIPT_DIR}/nipah-aws-env.sh"

NIPAH_PROJECT_ROOT="${NIPAH_PROJECT_ROOT:-/opt/nipah}"
CONDA_ENV="${CONDA_ENV:-nipah-md}"

CASE=""
REPLICATE=""
MAX_STEPS_ARGS=()

usage() {
  echo "Usage: $0 --case C_BMS_WT|D_BMS_MUT|... --replicate N [--max-steps N]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      CASE="$2"
      shift 2
      ;;
    --replicate)
      REPLICATE="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS_ARGS=(--max-steps "$2")
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

[[ -n "${CASE}" && -n "${REPLICATE}" ]] || usage

MINICONDA_DIR="${MINICONDA_DIR:-/home/ubuntu/miniconda3}"
if [[ -d /home/ec2-user/miniconda3 ]]; then
  MINICONDA_DIR="/home/ec2-user/miniconda3"
fi
if [[ -f "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${NIPAH_PROJECT_ROOT}/Stages/common:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUNNER="${NIPAH_PROJECT_ROOT}/Stages/Stage 7/run_production_50ns_direct.py"
if [[ ! -f "${RUNNER}" ]]; then
  echo "FATAL: ${RUNNER} not found — run instance_bootstrap.sh first"
  exit 1
fi

exec python "${RUNNER}" \
  --case "${CASE}" \
  --replicate "${REPLICATE}" \
  --resume \
  --archive-s3 \
  --s3-bucket "${NIPAH_S3_BUCKET}" \
  --s3-prefix "${NIPAH_S3_PREFIX}" \
  --s3-region "${NIPAH_S3_REGION}" \
  "${MAX_STEPS_ARGS[@]}"
