#!/usr/bin/env bash
# Report recent Spot prices and dry-run feasibility for Stage 7 GPU types in us-east-2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nipah-aws-env.sh
source "${SCRIPT_DIR}/nipah-aws-env.sh"

AWS_REGION="${NIPAH_EC2_REGION}"
INSTANCE_TYPES="${INSTANCE_TYPES:-g4dn.xlarge g5.xlarge g4dn.2xlarge}"
PILOT_COUNT="${PILOT_COUNT:-2}"
PILOT_TYPE="${PILOT_TYPE:-g5.xlarge}"
BUILDER_TYPE="${BUILDER_TYPE:-g4dn.xlarge}"
DL_AMI_FILTER="${DL_AMI_FILTER:-Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*}"
IAM_PROFILE="${IAM_PROFILE:-NipahStage7EC2Role}"

log() { echo "[check_spot_gpu_capacity] $*"; }

log "Region: ${AWS_REGION}"
log "Recent Spot prices (Linux/UNIX):"
aws ec2 describe-spot-price-history \
  --region "${AWS_REGION}" \
  --instance-types ${INSTANCE_TYPES} \
  --product-descriptions "Linux/UNIX" \
  --max-items 30 \
  --query 'SpotPriceHistory[*].[InstanceType,AvailabilityZone,SpotPrice,Timestamp]' \
  --output table

log "Instance type offerings (AZ):"
aws ec2 describe-instance-type-offerings \
  --region "${AWS_REGION}" \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=${INSTANCE_TYPES// /,}" \
  --query 'InstanceTypeOfferings[*].[InstanceType,Location]' \
  --output table

DL_AMI="$(aws ec2 describe-images \
  --region "${AWS_REGION}" \
  --owners amazon \
  --filters "Name=name,Values=${DL_AMI_FILTER}" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
  --output text)"
if [[ -z "${DL_AMI}" || "${DL_AMI}" == "None" ]]; then
  log "FATAL: could not resolve Deep Learning GPU AMI"
  exit 1
fi
log "Probe AMI: ${DL_AMI}"

dry_run() {
  local itype="$1"
  local count="$2"
  local max_price="$3"
  if aws ec2 run-instances --region "${AWS_REGION}" --dry-run \
    --image-id "${DL_AMI}" \
    --instance-type "${itype}" \
    --count "${count}" \
    --iam-instance-profile "Name=${IAM_PROFILE}" \
    --instance-market-options "$(printf '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate","MaxPrice":"%s"}}' "${max_price}")" \
    >/dev/null 2>&1; then
    echo "  OK: dry-run ${count}x ${itype} Spot (MaxPrice=${max_price})"
  else
    local err
    err="$(aws ec2 run-instances --region "${AWS_REGION}" --dry-run \
      --image-id "${DL_AMI}" \
      --instance-type "${itype}" \
      --count "${count}" \
      --iam-instance-profile "Name=${IAM_PROFILE}" \
      --instance-market-options "$(printf '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate","MaxPrice":"%s"}}' "${max_price}")" \
      2>&1 || true)"
    if [[ "${err}" == *DryRunOperation* ]]; then
      echo "  OK: dry-run ${count}x ${itype} Spot (MaxPrice=${max_price})"
    else
      echo "  FAIL: ${count}x ${itype}: ${err}"
    fi
  fi
}

log "Dry-run Spot launch (capacity / permission probe):"
dry_run "${BUILDER_TYPE}" 1 "0.60"
dry_run "${PILOT_TYPE}" "${PILOT_COUNT}" "1.50"

log "Cheapest recent g4dn.xlarge Spot is typically ~\$0.27/hr; g5.xlarge ~\$0.52/hr (see table above)."
log "Dry-run success implies ${PILOT_COUNT}x ${PILOT_TYPE} is likely launchable; actual Spot still subject to real-time capacity."
