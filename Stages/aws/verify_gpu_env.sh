#!/usr/bin/env bash
# Exit non-zero unless GPU stack + MD deps + AWS CLI are usable on the instance.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-nipah-md}"

echo "=== verify_gpu_env.sh ==="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found"
  exit 1
fi
nvidia-smi -L

# Activate conda env when available (AMI bake or post-bootstrap).
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/opt/conda/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV}" 2>/dev/null || true
fi

python - <<'PY'
import sys

import MDAnalysis as mda  # noqa: F401

import openmm
import openmm.unit as unit

platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print("OpenMM platforms:", platforms)

gpu_name = None
for name in ("CUDA", "OpenCL"):
    if name in platforms:
        gpu_name = name
        break
if gpu_name is None:
    print("FATAL: no CUDA or OpenCL OpenMM platform")
    sys.exit(1)

platform = openmm.Platform.getPlatformByName(gpu_name)
system = openmm.System()
system.addParticle(1.0 * unit.amu)
integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
context = openmm.Context(system, integrator, platform)
del context
print(f"OpenMM {gpu_name} context: OK")
PY

if ! command -v aws >/dev/null 2>&1; then
  echo "FATAL: aws CLI not found"
  exit 1
fi
aws --version

echo "=== verify_gpu_env.sh: PASS ==="
