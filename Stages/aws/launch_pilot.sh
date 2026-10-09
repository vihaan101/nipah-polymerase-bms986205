#!/usr/bin/env bash
# Launch two Spot g5.xlarge instances (C_BMS_WT and D_BMS_MUT) for Stage 7 pilot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/ami-id.env" ]]; then
  # shellcheck source=ami-id.env
  source "${SCRIPT_DIR}/ami-id.env"
fi
# shellcheck source=nipah-aws-env.sh
source "${SCRIPT_DIR}/nipah-aws-env.sh"
AWS_REGION="${AWS_REGION:-${NIPAH_EC2_REGION}}"
TEMPLATE_NAME="${TEMPLATE_NAME:-nipah-stage7-g5-spot}"
TEMPLATE_VERSION="${TEMPLATE_VERSION:-\$Latest}"

launch_one() {
  local case_tag="$1"
  echo "Launching Spot instance Case=${case_tag}..."
  aws ec2 run-instances \
    --region "${AWS_REGION}" \
    --launch-template "LaunchTemplateName=${TEMPLATE_NAME},Version=${TEMPLATE_VERSION}" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=nipah-stage7},{Key=Case,Value=${case_tag}}]" \
    --count 1 \
    --output table
}

if ! aws ec2 describe-launch-templates \
  --region "${AWS_REGION}" \
  --launch-template-names "${TEMPLATE_NAME}" >/dev/null 2>&1; then
  echo "FATAL: launch template ${TEMPLATE_NAME} not found in ${AWS_REGION}"
  echo "Run create_launch_template.sh with CUSTOM_AMI_ID set first."
  exit 1
fi

launch_one "C_BMS_WT"
launch_one "D_BMS_MUT"
echo "Pilot instances requested. Use SSM Session Manager or SSH to monitor cloud-init."
