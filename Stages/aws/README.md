# AWS Stage 7 (custom AMI, us-east-2)

Run OpenMM Stage 7 production on **g5.xlarge Spot** with a **custom AMI** (conda `nipah-md` only). Application code and Stage 6 inputs are refreshed on each launch via `git clone` and S3 sync.

**GPU quotas (us-east-2):** EC2 must allow **Running On-Demand G and VT** (`L-DB2E81BA`) and **All G and VT Spot** (`L-3819A6DF`) vCPUs ≥ 4 for one `g4dn.xlarge`/`g5.xlarge`. New accounts often start at **0**; request increases in Service Quotas before baking or launching pilots.

**Regions (defaults in `nipah-aws-env.sh`):**

| Resource | Default region |
|----------|----------------|
| EC2 / AMI / Spot (`NIPAH_EC2_REGION`) | `us-east-2` |
| S3 bucket `nipah-archive` (`NIPAH_S3_REGION`) | `us-east-1` (existing bucket) |

Cross-region S3 from us-east-2 instances is normal. Production uses `--s3-region "${NIPAH_S3_REGION}"`. Override env vars if you migrate the bucket.

| Purpose | S3 path |
|---------|---------|
| Stage 6 staging | `s3://nipah-archive/nipah/staging/stage6/results/` |
| Stage 7 DCD archive | `s3://nipah-archive/nipah/stage7_50ns_direct/results/{case}/replicate_{N}/production.dcd` |

## 1. Foundation (S3 + IAM)

```bash
cd Stages/aws
./setup_aws_foundation.sh
```

Creates (if missing):

- S3 bucket `nipah-archive` in **us-east-2** (public access blocked)
- IAM role **`NipahStage7EC2Role`** with inline policy from `iam-policy-stage7-ec2.json`
- Attached **`AmazonSSMManagedInstanceCore`** (SSM shell without opening SSH)
- Instance profile **`NipahStage7EC2Role`** (same name as role)

Policy scope: `ListBucket` on `nipah-archive` with `s3:prefix` `nipah/*`; `GetObject` / `PutObject` on `arn:aws:s3:::nipah-archive/nipah/*`.

Create a security group with egress HTTPS (and optional SSH). Prefer **SSM Session Manager** for access.

## 2. Stage 6 upload (one-time, from laptop or sunlab)

Requires local `Stages/Stage 6/results/stage6_manifest.json` with **C_BMS_WT** and **D_BMS_MUT**.

```bash
./upload_stage6_to_s3.sh
```

If `Stage 6/results/` is not local:

```bash
./pull_stage6_from_sunlab.sh
./upload_stage6_to_s3.sh
```

## 3. Spot capacity check

```bash
./check_spot_gpu_capacity.sh
```

Reports recent Spot prices in `NIPAH_EC2_REGION` and dry-runs `2× g5.xlarge` (pilot) plus `1× g4dn.xlarge` (AMI bake).

## 4. Bake custom AMI (automated Spot builder)

Push this repo to GitHub first (`instance_bootstrap` / bake user-data `git clone`).

```bash
./bake_ami_on_spot.sh
```

Launches **Spot `g4dn.xlarge`** (cheapest GPU with CUDA), runs `bootstrap_ami.sh` via user-data, creates `nipah-md-openmm-YYYYMMDD`, writes **`ami-id.env`**, and registers launch template **`nipah-stage7-g5-spot`**.

Manual checklist (fallback): `./bake_ami.sh`.

## 5. Launch template

After bake, `ami-id.env` holds `CUSTOM_AMI_ID` and `SECURITY_GROUP_IDS`. Re-register manually if needed:

```bash
source ./ami-id.env
# optional: export KEY_NAME=my-key
./create_launch_template.sh
```

Without `CUSTOM_AMI_ID`, the script prints instructions and references `launch-template-nipah-stage7-g5-spot.json` (placeholder `ImageId`).

Template name: **`nipah-stage7-g5-spot`** — Spot `g5.xlarge`, 200 GB gp3, user-data = `instance_bootstrap.sh`.

## 6. Instance bootstrap (automatic via user-data)

`instance_bootstrap.sh`:

- Activates conda env `nipah-md`
- `NIPAH_PROJECT_ROOT=/opt/nipah`
- `git clone` / pull `https://github.com/vihaan101/nipah-polymerase-bms986205.git`
- `aws s3 sync` Stage 6 → `$NIPAH_PROJECT_ROOT/Stages/Stage 6/results/`
- `PYTHONPATH` includes `Stages/common`
- Runs `verify_gpu_env.sh`

Check `/var/log/cloud-init-output.log` after launch.

## 7. Smoke verification (manual — do not run 50 ns from CI/agent)

SSM or SSH into the instance:

```bash
export NIPAH_PROJECT_ROOT=/opt/nipah
cd "$NIPAH_PROJECT_ROOT/Stages/aws"
./run_stage7_production.sh --case C_BMS_WT --replicate 1 --max-steps 1000
```

Expect output under `Stages/Stage 7/results/C_BMS_WT/replicate_1/`. OpenMM must use GPU (not CPU-only fatal).

Optional archive smoke (validates IAM + S3): same command **without** `--max-steps` is production length — for archive path only, use a short run and inspect `s3_archive.json` if you enable archive on a completed short trajectory per your policy.

## 8. Pilot production (two Spot instances — user-triggered)

```bash
./launch_pilot.sh
```

Tags: `Case=C_BMS_WT` and `Case=D_BMS_MUT`. On each instance after bootstrap:

```bash
./run_stage7_production.sh --case C_BMS_WT --replicate 1
# or
./run_stage7_production.sh --case D_BMS_MUT --replicate 1
```

Uses `--archive-s3`, `--s3-bucket nipah-archive`, `--s3-prefix nipah`, `--s3-region` from `nipah-aws-env.sh` (default **us-east-1** for the bucket), `--resume`, `CUDA_VISIBLE_DEVICES=0`. Target **12_500_000** steps (`md_protocol_50ns_direct.py`). Do not combine `--archive-s3` and `--eval` on the same run.

Spot eviction: relaunch with the same case and **`--resume`**; keep `Stage 7/results/` on EBS or sync checkpoints to S3 between attempts.

## Files

| File | Role |
|------|------|
| `iam-policy-stage7-ec2.json` | S3 policy for instance role |
| `environment.yml` | Conda env for AMI |
| `bootstrap_ami.sh` | AMI builder (idempotent) |
| `verify_gpu_env.sh` | Pre-flight gate |
| `upload_stage6_to_s3.sh` | Laptop → S3 staging |
| `instance_bootstrap.sh` | EC2 user-data |
| `run_stage7_production.sh` | Stage 7 wrapper |
| `setup_aws_foundation.sh` | Bucket + IAM |
| `create_launch_template.sh` | Register launch template |
| `launch_pilot.sh` | Two Spot launches |
| `bake_ami.sh` | AMI bake instructions (manual) |
| `bake_ami_on_spot.sh` | Spot g4dn bake → `ami-id.env` + launch template |
| `check_spot_gpu_capacity.sh` | Spot price + dry-run probe |
| `ami-id.env` | Generated `CUSTOM_AMI_ID` / `SECURITY_GROUP_IDS` |
