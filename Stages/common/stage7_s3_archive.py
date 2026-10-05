"""Stage 7 production.dcd archive to S3 (verify, upload, sidecar, local DCD removal)."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

TIER_LABEL = "50ns_direct"
DEFAULT_BUCKET = "nipah-archive"
DEFAULT_PREFIX = "nipah"
DEFAULT_PROFILE = "default"
DEFAULT_REGION = "us-east-1"
DCD_INTERVAL = 2000
SIDECAR_NAME = "s3_archive.json"


@dataclass(frozen=True)
class S3ArchiveConfig:
    bucket: str
    prefix: str
    profile: str
    region: str
    delete_chk_above_mb: float | None = None


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_s3_config(
    *,
    archive_enabled: bool | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
    profile: str | None = None,
    region: str | None = None,
    delete_chk_above_mb: float | None = None,
) -> tuple[S3ArchiveConfig | None, bool]:
    """Resolve S3 settings from CLI overrides and STAGE7_S3_* env vars."""
    enabled = archive_enabled
    if enabled is None:
        enabled = _truthy(os.environ.get("STAGE7_S3_ARCHIVE"))

    bucket = bucket or os.environ.get("STAGE7_S3_BUCKET") or DEFAULT_BUCKET
    prefix = prefix or os.environ.get("STAGE7_S3_PREFIX") or DEFAULT_PREFIX
    profile = profile or os.environ.get("STAGE7_S3_PROFILE") or DEFAULT_PROFILE
    region = region or os.environ.get("STAGE7_S3_REGION") or DEFAULT_REGION

    if delete_chk_above_mb is None:
        raw = os.environ.get("STAGE7_S3_DELETE_CHK_ABOVE_MB")
        if raw:
            delete_chk_above_mb = float(raw)

    if not enabled:
        return None, False
    return S3ArchiveConfig(
        bucket=bucket,
        prefix=prefix.strip("/"),
        profile=profile,
        region=region,
        delete_chk_above_mb=delete_chk_above_mb,
    ), True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_dcd_precheck(dcd: Path, topology: Path, total_steps: int, dcd_interval: int = DCD_INTERVAL) -> None:
    """Run H3 validation with effective step count (supports --max-steps smoke)."""
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from md_eval.stage7_production_validation import validate_trajectory

    validate_trajectory(str(dcd), str(topology), TIER_LABEL, total_steps, dcd_interval)


def _aws_base(profile: str, region: str) -> list[str]:
    return ["aws", "--profile", profile, "--region", region]


def upload_dcd_with_verify(
    local_path: Path,
    s3_uri: str,
    profile: str,
    region: str = DEFAULT_REGION,
) -> dict:
    """Upload with checksum; verify size and SHA256 via head-object."""
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"Local DCD not found: {local_path}")

    local_size = local_path.stat().st_size
    local_sha = sha256_file(local_path)

    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    cp_cmd = [
        *_aws_base(profile, region),
        "s3",
        "cp",
        str(local_path),
        s3_uri,
        "--metadata",
        f"sha256={local_sha}",
    ]
    cp = subprocess.run(cp_cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"aws s3 cp failed ({cp.returncode}): {cp.stderr.strip() or cp.stdout}")

    head_cmd = [
        *_aws_base(profile, region),
        "s3api",
        "head-object",
        "--bucket",
        bucket,
        "--key",
        key,
    ]
    head = subprocess.run(head_cmd, capture_output=True, text=True)
    if head.returncode != 0:
        raise RuntimeError(f"aws s3api head-object failed ({head.returncode}): {head.stderr.strip()}")

    meta = json.loads(head.stdout)
    remote_size = int(meta.get("ContentLength", -1))
    remote_meta_sha = (meta.get("Metadata") or {}).get("sha256")
    if remote_meta_sha and remote_meta_sha != local_sha:
        raise RuntimeError(
            f"S3 metadata sha256 mismatch for {s3_uri}: local={local_sha[:16]}… "
            f"remote={remote_meta_sha[:16]}…"
        )
    remote_sha_b64 = meta.get("ChecksumSHA256")
    if remote_sha_b64 and not remote_meta_sha:
        remote_hex = base64.b64decode(remote_sha_b64).hex()
        if remote_hex != local_sha:
            raise RuntimeError(
                f"S3 checksum mismatch for {s3_uri}: local sha256={local_sha[:16]}… "
                f"remote={remote_hex[:16]}…"
            )
    if remote_size != local_size:
        raise RuntimeError(
            f"S3 size mismatch for {s3_uri}: local={local_size} remote={remote_size}"
        )

    return {
        "s3_uri": s3_uri,
        "size_bytes": local_size,
        "sha256": local_sha,
        "uploaded_at": _ts(),
    }


def replicate_s3_key(
    prefix: str,
    results_root: Path,
    case_id: str,
    replicate: int,
    filename: str = "production.dcd",
) -> str:
    rel = Path(case_id) / f"replicate_{replicate}" / filename
    return f"{prefix.strip('/')}/stage7_50ns_direct/results/{rel.as_posix()}"


def archive_replicate_dir(
    replicate_dir: Path,
    case_id: str,
    replicate: int,
    topology_path: Path,
    config: S3ArchiveConfig,
    total_steps: int,
    results_root: Path | None = None,
) -> Path:
    """Upload production.dcd, write sidecar, delete local DCD on success."""
    replicate_dir = Path(replicate_dir)
    dcd = replicate_dir / "production.dcd"
    local_dcd_precheck(dcd, topology_path, total_steps)

    bucket = config.bucket
    key = replicate_s3_key(
        config.prefix,
        results_root or replicate_dir.parent.parent,
        case_id,
        replicate,
    )
    s3_uri = f"s3://{bucket}/{key}"
    upload_meta = upload_dcd_with_verify(dcd, s3_uri, config.profile, config.region)

    sidecar = {
        "case_id": case_id,
        "replicate": replicate,
        "tier": TIER_LABEL,
        "total_steps": total_steps,
        **upload_meta,
    }
    sidecar_path = replicate_dir / SIDECAR_NAME
    tmp = sidecar_path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(sidecar, fh, indent=2)
    tmp.replace(sidecar_path)

    dcd.unlink()
    print(f"  [S3] Archived {dcd.name} -> {s3_uri}; local DCD removed.")
    print(f"  [S3] Sidecar: {sidecar_path}")

    chk = replicate_dir / "production.chk"
    if config.delete_chk_above_mb is not None and chk.is_file():
        mb = chk.stat().st_size / (1024 * 1024)
        if mb > config.delete_chk_above_mb:
            chk.unlink()
            print(f"  [S3] Removed checkpoint ({mb:.1f} MB > {config.delete_chk_above_mb} MB)")

    return sidecar_path
