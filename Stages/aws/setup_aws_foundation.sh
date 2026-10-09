#!/usr/bin/env bash
# Create S3 bucket (us-east-2), IAM role + instance profile for Stage 7 EC2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_REGION="${AWS_REGION:-us-east-2}"
BUCKET="${BUCKET:-nipah-archive}"
ROLE_NAME="${ROLE_NAME:-NipahStage7EC2Role}"
PROFILE_NAME="${PROFILE_NAME:-NipahStage7EC2Role}"
POLICY_NAME="${POLICY_NAME:-NipahStage7EC2S3Policy}"

log() { echo "[setup_aws_foundation] $*"; }

if ! aws sts get-caller-identity --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "FATAL: AWS credentials not configured"
  exit 1
fi

log "Ensuring S3 bucket s3://${BUCKET} in ${AWS_REGION}..."
if aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
  existing_region="$(aws s3api get-bucket-location --bucket "${BUCKET}" --output text)"
  if [[ "${existing_region}" == "None" ]]; then
    existing_region="us-east-1"
  fi
  if [[ "${existing_region}" != "${AWS_REGION}" ]]; then
    log "WARNING: bucket ${BUCKET} is in ${existing_region}, script target is ${AWS_REGION}"
    log "         Use --s3-region ${existing_region} in production or create a regional bucket."
  else
    log "Bucket already exists in ${AWS_REGION}"
  fi
else
  if [[ "${AWS_REGION}" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "${BUCKET}" --region "${AWS_REGION}"
  else
    aws s3api create-bucket \
      --bucket "${BUCKET}" \
      --region "${AWS_REGION}" \
      --create-bucket-configuration "LocationConstraint=${AWS_REGION}"
  fi
  log "Created bucket"
fi

aws s3api put-public-access-block \
  --bucket "${BUCKET}" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true \
  2>/dev/null || true

TRUST_DOC='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  log "IAM role ${ROLE_NAME} already exists"
else
  log "Creating IAM role ${ROLE_NAME}..."
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "${TRUST_DOC}" \
    --description "Stage 7 production EC2 (S3 nipah-archive + SSM)"
fi

POLICY_ARN="$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${POLICY_NAME}'].Arn" --output text)"
if [[ -z "${POLICY_ARN}" || "${POLICY_ARN}" == "None" ]]; then
  log "Creating IAM policy ${POLICY_NAME}..."
  POLICY_ARN="$(aws iam create-policy \
    --policy-name "${POLICY_NAME}" \
    --policy-document "file://${SCRIPT_DIR}/iam-policy-stage7-ec2.json" \
    --query Policy.Arn --output text)"
else
  log "Updating IAM policy document ${POLICY_NAME}..."
  aws iam create-policy-version \
    --policy-arn "${POLICY_ARN}" \
    --policy-document "file://${SCRIPT_DIR}/iam-policy-stage7-ec2.json" \
    --set-as-default >/dev/null || true
fi

aws iam attach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${POLICY_ARN}" 2>/dev/null || true
aws iam attach-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore" 2>/dev/null || true

if aws iam get-instance-profile --instance-profile-name "${PROFILE_NAME}" >/dev/null 2>&1; then
  log "Instance profile ${PROFILE_NAME} already exists"
else
  log "Creating instance profile ${PROFILE_NAME}..."
  aws iam create-instance-profile --instance-profile-name "${PROFILE_NAME}"
  aws iam add-role-to-instance-profile \
    --instance-profile-name "${PROFILE_NAME}" \
    --role-name "${ROLE_NAME}"
fi

log "Foundation ready:"
log "  Bucket: s3://${BUCKET} (${AWS_REGION})"
log "  Instance profile: ${PROFILE_NAME}"
log "  Attach ${PROFILE_NAME} to launch template / run-instances"
