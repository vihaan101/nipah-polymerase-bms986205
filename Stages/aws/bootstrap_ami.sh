#!/usr/bin/env bash
# Idempotent AMI builder: Miniconda, nipah-md env, AWS CLI v2, verify_gpu_env.sh.
# Run on a g5.xlarge Deep Learning AMI (Ubuntu 22.04) before creating an EC2 image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-nipah-md}"
MINICONDA_DIR="${MINICONDA_DIR:-${HOME}/miniconda3}"

echo "=== bootstrap_ami.sh (CONDA_ENV=${CONDA_ENV}) ==="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: run on a GPU instance (nvidia-smi missing)"
  exit 1
fi

if [[ ! -d "${MINICONDA_DIR}" ]]; then
  echo "Installing Miniconda to ${MINICONDA_DIR}..."
  tmp="$(mktemp)"
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "${tmp}"
  bash "${tmp}" -b -p "${MINICONDA_DIR}"
  rm -f "${tmp}"
fi

# shellcheck source=/dev/null
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda config --set auto_activate_base false

if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "Updating conda env ${CONDA_ENV}..."
  conda env update -n "${CONDA_ENV}" -f "${SCRIPT_DIR}/environment.yml" --prune
else
  echo "Creating conda env ${CONDA_ENV}..."
  conda env create -f "${SCRIPT_DIR}/environment.yml"
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "Installing AWS CLI v2..."
  tmpdir="$(mktemp -d)"
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "${tmpdir}/awscliv2.zip"
  unzip -q "${tmpdir}/awscliv2.zip" -d "${tmpdir}"
  sudo "${tmpdir}/aws/install" --update
  rm -rf "${tmpdir}"
fi

export CONDA_ENV
"${SCRIPT_DIR}/verify_gpu_env.sh"

echo "=== bootstrap_ami.sh: done — create AMI from this instance ==="
