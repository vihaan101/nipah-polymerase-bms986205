#!/usr/bin/env python3
"""CLI: archive one replicate's production.dcd to S3."""
import argparse
import sys
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from stage7_s3_archive import (  # noqa: E402
    DEFAULT_REGION,
    archive_replicate_dir,
    resolve_s3_config,
)

TOTAL_STEPS_DEFAULT = 25_000_000


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive Stage 7 production.dcd to S3")
    parser.add_argument("--replicate-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS_DEFAULT)
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default=None)
    parser.add_argument("--s3-profile", default=None)
    parser.add_argument("--s3-region", default=None)
    parser.add_argument("--s3-delete-chk-above-mb", type=float, default=None)
    args = parser.parse_args()

    config, _enabled = resolve_s3_config(
        archive_enabled=True,
        bucket=args.s3_bucket,
        prefix=args.s3_prefix,
        profile=args.s3_profile,
        region=args.s3_region or DEFAULT_REGION,
        delete_chk_above_mb=args.s3_delete_chk_above_mb,
    )
    if config is None:
        print("FATAL: could not resolve S3 config", file=sys.stderr)
        return 1

    try:
        archive_replicate_dir(
            args.replicate_dir,
            args.case,
            args.replicate,
            args.topology,
            config,
            args.total_steps,
            results_root=args.results_root,
        )
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
