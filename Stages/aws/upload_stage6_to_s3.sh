#!/usr/bin/env bash
# One-time sync of local Stage 6 results to S3 staging.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nipah-aws-env.sh
source "${SCRIPT_DIR}/nipah-aws-env.sh"
STAGES_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGE6_RESULTS="${STAGE6_RESULTS:-${STAGES_ROOT}/Stage 6/results}"
S3_URI="${S3_URI:-s3://${NIPAH_S3_BUCKET}/${NIPAH_S3_PREFIX}/staging/stage6/results/}"

MANIFEST="${STAGE6_RESULTS}/stage6_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "FATAL: missing ${MANIFEST}"
  echo "Pull Stage 6 results from sunlab (C_BMS_WT, D_BMS_MUT) before upload."
  exit 1
fi

for case_id in C_BMS_WT D_BMS_MUT; do
  if ! python3 -c "import json; m=json.load(open('${MANIFEST}')); assert '${case_id}' in m.get('cases',{})" 2>/dev/null; then
    echo "FATAL: stage6_manifest.json missing case ${case_id}"
    exit 1
  fi
done

echo "Syncing ${STAGE6_RESULTS}/ -> ${S3_URI} (S3 region ${NIPAH_S3_REGION})"
aws s3 sync "${STAGE6_RESULTS}/" "${S3_URI}" --region "${NIPAH_S3_REGION}"
echo "Upload complete."
