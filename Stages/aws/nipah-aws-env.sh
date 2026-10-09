# Source from Stages/aws scripts: separates EC2 region from S3 bucket region.
# nipah-archive currently lives in us-east-1; GPU instances use us-east-2.
export NIPAH_EC2_REGION="${NIPAH_EC2_REGION:-us-east-2}"
export NIPAH_S3_REGION="${NIPAH_S3_REGION:-us-east-1}"
export NIPAH_S3_BUCKET="${NIPAH_S3_BUCKET:-nipah-archive}"
export NIPAH_S3_PREFIX="${NIPAH_S3_PREFIX:-nipah}"
