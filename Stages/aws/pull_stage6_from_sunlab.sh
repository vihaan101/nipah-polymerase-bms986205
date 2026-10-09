#!/usr/bin/env bash
# Pull Stage 6 results from sunlab into local Stages/Stage 6/results/ (~500 MB).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGE6_RESULTS="${STAGE6_RESULTS:-${STAGES_ROOT}/Stage 6/results}"
SUNLAB_HOST="${SUNLAB_HOST:-vihaan5@sunlab-serv-03.cs.illinois.edu}"
# Backslash-escaped space (macOS openrsync does not support --protect-args).
REMOTE_PATH="${SUNLAB_REMOTE_PATH:-/home/vihaan5/Stages/Stage\\ 6/results/}"

mkdir -p "${STAGE6_RESULTS}"
rsync -avz -e ssh --progress "${SUNLAB_HOST}:${REMOTE_PATH}" "${STAGE6_RESULTS}/"
test -f "${STAGE6_RESULTS}/stage6_manifest.json"
echo "Stage 6 results ready at ${STAGE6_RESULTS}"
