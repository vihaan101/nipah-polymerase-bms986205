#!/usr/bin/env bash
# Register EC2 launch template nipah-stage7-g5-spot (custom AMI + Spot + user-data bootstrap).
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
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
IAM_PROFILE="${IAM_PROFILE:-NipahStage7EC2Role}"
CUSTOM_AMI_ID="${CUSTOM_AMI_ID:-}"
KEY_NAME="${KEY_NAME:-}"
SECURITY_GROUP_IDS="${SECURITY_GROUP_IDS:-}"

if [[ -z "${CUSTOM_AMI_ID}" ]]; then
  echo "CUSTOM_AMI_ID is not set."
  echo "After baking AMI, run:"
  echo "  export CUSTOM_AMI_ID=ami-xxxxxxxx"
  echo "  export SECURITY_GROUP_IDS=sg-xxxxxxxx   # egress HTTPS; SSM recommended"
  echo "  ${SCRIPT_DIR}/create_launch_template.sh"
  echo ""
  echo "Reference template JSON (placeholder ImageId): ${SCRIPT_DIR}/launch-template-nipah-stage7-g5-spot.json"
  exit 0
fi

if [[ -z "${SECURITY_GROUP_IDS}" ]]; then
  echo "FATAL: set SECURITY_GROUP_IDS (comma-separated) for the template"
  exit 1
fi

USER_DATA_B64="$(base64 < "${SCRIPT_DIR}/instance_bootstrap.sh" | tr -d '\n')"
export USER_DATA_B64 CUSTOM_AMI_ID INSTANCE_TYPE IAM_PROFILE SECURITY_GROUP_IDS KEY_NAME

LAUNCH_DATA="$(mktemp)"
python3 - <<PY
import json
import os

sg_ids = [s.strip() for s in os.environ["SECURITY_GROUP_IDS"].split(",") if s.strip()]
data = {
    "ImageId": os.environ["CUSTOM_AMI_ID"],
    "InstanceType": os.environ["INSTANCE_TYPE"],
    "IamInstanceProfile": {"Name": os.environ["IAM_PROFILE"]},
    "BlockDeviceMappings": [{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 200,
            "VolumeType": "gp3",
            "DeleteOnTermination": True,
        },
    }],
    "InstanceMarketOptions": {
        "MarketType": "spot",
        "SpotOptions": {
            "SpotInstanceType": "one-time",
            "InstanceInterruptionBehavior": "terminate",
        },
    },
    "UserData": os.environ["USER_DATA_B64"],
    "TagSpecifications": [{
        "ResourceType": "instance",
        "Tags": [
            {"Key": "Project", "Value": "nipah-stage7"},
        ],
    }],
    "NetworkInterfaces": [{
        "DeviceIndex": 0,
        "AssociatePublicIpAddress": True,
        "Groups": sg_ids,
    }],
}
key = os.environ.get("KEY_NAME", "")
if key:
    data["KeyName"] = key
print(json.dumps(data))
PY

if aws ec2 describe-launch-templates \
  --region "${AWS_REGION}" \
  --launch-template-names "${TEMPLATE_NAME}" >/dev/null 2>&1; then
  aws ec2 create-launch-template-version \
    --region "${AWS_REGION}" \
    --launch-template-name "${TEMPLATE_NAME}" \
    --launch-template-data "file://${LAUNCH_DATA}" \
    --source-version "\$Latest" \
    --version-description "Update AMI/SG/user-data"
  aws ec2 modify-launch-template \
    --region "${AWS_REGION}" \
    --launch-template-name "${TEMPLATE_NAME}" \
    --default-version "\$Latest"
else
  aws ec2 create-launch-template \
    --region "${AWS_REGION}" \
    --launch-template-name "${TEMPLATE_NAME}" \
    --launch-template-data "file://${LAUNCH_DATA}"
fi

rm -f "${LAUNCH_DATA}"
echo "Launch template ${TEMPLATE_NAME} registered in ${AWS_REGION}."
