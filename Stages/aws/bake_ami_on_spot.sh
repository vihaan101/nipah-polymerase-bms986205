#!/usr/bin/env bash
# Launch cheapest GPU Spot builder → bootstrap OpenMM AMI → register launch template.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nipah-aws-env.sh
source "${SCRIPT_DIR}/nipah-aws-env.sh"

AWS_REGION="${NIPAH_EC2_REGION}"
BUILDER_TYPE="${BUILDER_TYPE:-g4dn.xlarge}"
IAM_PROFILE="${IAM_PROFILE:-NipahStage7EC2Role}"
AMI_NAME="${AMI_NAME:-nipah-md-openmm-$(date +%Y%m%d)}"
SPOT_MAX_PRICE="${SPOT_MAX_PRICE:-0.60}"
NIPAH_REPO_URL="${NIPAH_REPO_URL:-https://github.com/vihaan101/nipah-polymerase-bms986205.git}"
NIPAH_REPO_BRANCH="${NIPAH_REPO_BRANCH:-main}"
SG_NAME="${SG_NAME:-nipah-stage7-ec2}"
AMI_ENV_FILE="${SCRIPT_DIR}/ami-id.env"

log() { echo "[bake_ami_on_spot] $*"; }

ensure_security_group() {
  local vpc_id sg_id
  vpc_id="$(aws ec2 describe-vpcs --region "${AWS_REGION}" \
    --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' --output text)"
  sg_id="$(aws ec2 describe-security-groups --region "${AWS_REGION}" \
    --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${vpc_id}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
  if [[ -n "${sg_id}" && "${sg_id}" != "None" ]]; then
    log "Using security group ${sg_id} (${SG_NAME})"
    echo "${sg_id}"
    return
  fi
  log "Creating security group ${SG_NAME} in ${vpc_id}..."
  sg_id="$(aws ec2 create-security-group \
    --region "${AWS_REGION}" \
    --group-name "${SG_NAME}" \
    --description "Nipah Stage 7 EC2 (SSM; egress HTTPS)" \
    --vpc-id "${vpc_id}" \
    --query GroupId --output text)"
  aws ec2 authorize-security-group-egress --region "${AWS_REGION}" \
    --group-id "${sg_id}" \
    --ip-permissions 'IpProtocol=-1,IpRanges=[{CidrIp=0.0.0.0/0,Description="egress"}]' \
    2>/dev/null || true
  echo "${sg_id}"
}

DL_AMI="$(aws ec2 describe-images \
  --region "${AWS_REGION}" \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
  --output text)"

USER_DATA="$(mktemp)"
cat > "${USER_DATA}" <<EOF
#!/bin/bash
set -euo pipefail
exec > /var/log/nipah-ami-bake.log 2>&1
echo "=== nipah ami bake user-data ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git
rm -rf /tmp/nipah-src
git clone --depth 1 --branch ${NIPAH_REPO_BRANCH} ${NIPAH_REPO_URL} /tmp/nipah-src
cd /tmp/nipah-src/Stages/aws
chmod +x bootstrap_ami.sh verify_gpu_env.sh
./bootstrap_ami.sh
touch /var/lib/nipah-ami-bake-done
echo "=== bake user-data complete ==="
EOF

SG_ID="$(ensure_security_group)"
log "Launching Spot ${BUILDER_TYPE} (max \$${SPOT_MAX_PRICE}/hr) from ${DL_AMI}..."

INSTANCE_ID="$(aws ec2 run-instances \
  --region "${AWS_REGION}" \
  --image-id "${DL_AMI}" \
  --instance-type "${BUILDER_TYPE}" \
  --iam-instance-profile "Name=${IAM_PROFILE}" \
  --security-group-ids "${SG_ID}" \
  --instance-market-options "$(printf '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate","MaxPrice":"%s"}}' "${SPOT_MAX_PRICE}")" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":80,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=nipah-stage7},{Key=Role,Value=ami-bake}]' \
  --user-data "file://${USER_DATA}" \
  --query 'Instances[0].InstanceId' \
  --output text)"
rm -f "${USER_DATA}"
log "Builder instance: ${INSTANCE_ID}"

aws ec2 wait instance-running --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"
log "Instance running; waiting for SSM Online (up to ~15 min)..."
for _ in $(seq 1 90); do
  status="$(aws ssm describe-instance-information --region "${AWS_REGION}" \
    --filters "Key=InstanceIds,Values=${INSTANCE_ID}" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)"
  if [[ "${status}" == "Online" ]]; then
    log "SSM Online"
    break
  fi
  sleep 10
done

log "Waiting for bootstrap completion on instance (conda/OpenMM; may take 30–60 min)..."
for _ in $(seq 1 180); do
  done_flag="$(aws ssm send-command --region "${AWS_REGION}" \
    --instance-ids "${INSTANCE_ID}" \
    --document-name AWS-RunShellScript \
    --parameters 'commands=["test -f /var/lib/nipah-ami-bake-done && echo DONE || (tail -5 /var/log/nipah-ami-bake.log 2>/dev/null || echo WAIT)"]' \
    --query 'Command.CommandId' --output text)"
  sleep 8
  out="$(aws ssm get-command-invocation --region "${AWS_REGION}" \
    --command-id "${done_flag}" --instance-id "${INSTANCE_ID}" \
    --query 'StandardOutputContent' --output text 2>/dev/null || true)"
  if [[ "${out}" == *DONE* ]]; then
    log "Bootstrap finished on builder"
    break
  fi
  if [[ $((_ % 6)) -eq 0 ]]; then
    log "Still baking... last log tail: ${out//$'\n'/ }"
  fi
  sleep 30
done

if [[ "${out:-}" != *DONE* ]]; then
  log "FATAL: bootstrap did not complete in time. Check SSM on ${INSTANCE_ID} and /var/log/nipah-ami-bake.log"
  exit 1
fi

log "Creating AMI ${AMI_NAME}..."
IMAGE_ID="$(aws ec2 create-image \
  --region "${AWS_REGION}" \
  --instance-id "${INSTANCE_ID}" \
  --name "${AMI_NAME}" \
  --description "OpenMM conda nipah-md for Stage 7" \
  --no-reboot \
  --query ImageId --output text)"
log "AMI ${IMAGE_ID} pending..."

aws ec2 wait image-available --region "${AWS_REGION}" --image-ids "${IMAGE_ID}"
log "AMI available: ${IMAGE_ID}"

log "Terminating builder ${INSTANCE_ID}..."
aws ec2 terminate-instances --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}" >/dev/null

cat > "${AMI_ENV_FILE}" <<ENV
# Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by bake_ami_on_spot.sh
CUSTOM_AMI_ID=${IMAGE_ID}
SECURITY_GROUP_IDS=${SG_ID}
NIPAH_EC2_REGION=${AWS_REGION}
ENV

log "Wrote ${AMI_ENV_FILE}"
# shellcheck source=/dev/null
source "${AMI_ENV_FILE}"
"${SCRIPT_DIR}/create_launch_template.sh"

LT_VER="$(aws ec2 describe-launch-templates --region "${AWS_REGION}" \
  --launch-template-names nipah-stage7-g5-spot \
  --query 'LaunchTemplates[0].LatestVersionNumber' --output text)"
log "Launch template nipah-stage7-g5-spot version ${LT_VER}"
log "Done. Pilot: ${SCRIPT_DIR}/launch_pilot.sh"
