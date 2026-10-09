#!/usr/bin/env bash
# Helper notes for manual AMI bake (cannot complete without an interactive builder instance).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_REGION="${AWS_REGION:-us-east-2}"
AMI_NAME="${AMI_NAME:-nipah-md-openmm-$(date +%Y%m%d)}"

cat <<EOF
=== Bake custom AMI (manual steps) ===

1. Launch g5.xlarge ON-DEMAND in ${AWS_REGION} from:
   Deep Learning AMI GPU (Ubuntu 22.04) — pick latest in EC2 console.

2. Attach instance profile NipahStage7EC2Role (optional for bake; needed for S3 smoke later).

3. On the builder:
   git clone https://github.com/vihaan101/nipah-polymerase-bms986205.git /tmp/nipah-src
   cd /tmp/nipah-src/Stages/aws
   ./bootstrap_ami.sh

4. Gate: ./verify_gpu_env.sh must exit 0.

5. From laptop (replace INSTANCE_ID):
   aws ec2 create-image --region ${AWS_REGION} \\
     --instance-id INSTANCE_ID \\
     --name "${AMI_NAME}" \\
     --description "OpenMM conda nipah-md for Stage 7"

6. Wait until AMI state is available, then terminate builder.

7. Register launch template:
   export CUSTOM_AMI_ID=ami-xxxxxxxx
   export SECURITY_GROUP_IDS=sg-xxxxxxxx
   ${SCRIPT_DIR}/create_launch_template.sh

EOF
