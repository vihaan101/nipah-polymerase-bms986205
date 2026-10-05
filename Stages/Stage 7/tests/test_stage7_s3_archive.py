"""Unit tests for stage7_s3_archive (mocked AWS CLI)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_COMMON = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(_COMMON))

from stage7_s3_archive import (  # noqa: E402
    DEFAULT_BUCKET,
    S3ArchiveConfig,
    archive_replicate_dir,
    replicate_s3_key,
    resolve_s3_config,
    sha256_file,
    upload_dcd_with_verify,
)


class ResolveS3ConfigTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            config, enabled = resolve_s3_config()
        self.assertFalse(enabled)
        self.assertIsNone(config)

    def test_env_enables(self) -> None:
        env = {
            "STAGE7_S3_ARCHIVE": "1",
            "STAGE7_S3_BUCKET": "my-bucket",
            "STAGE7_S3_PREFIX": "campaign",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            config, enabled = resolve_s3_config()
        self.assertTrue(enabled)
        self.assertEqual(config.bucket, "my-bucket")
        self.assertEqual(config.prefix, "campaign")
        self.assertEqual(config.profile, "default")

    def test_default_bucket_when_archive_on(self) -> None:
        env = {"STAGE7_S3_ARCHIVE": "1"}
        with mock.patch.dict("os.environ", env, clear=True):
            config, enabled = resolve_s3_config()
        self.assertTrue(enabled)
        self.assertEqual(config.bucket, DEFAULT_BUCKET)


class UploadVerifyTests(unittest.TestCase):
    def test_upload_and_head_verify(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            dcd = Path(td) / "production.dcd"
            dcd.write_bytes(b"fake-dcd-bytes")
            local_sha = sha256_file(dcd)

            def fake_run(cmd, capture_output=True, text=True):
                if "s3" in cmd and "cp" in cmd:
                    return mock.Mock(returncode=0, stdout="", stderr="")
                if "head-object" in cmd:
                    payload = {
                        "ContentLength": dcd.stat().st_size,
                        "Metadata": {"sha256": local_sha},
                    }
                    return mock.Mock(returncode=0, stdout=json.dumps(payload), stderr="")
                return mock.Mock(returncode=1, stdout="", stderr="fail")

            with mock.patch("stage7_s3_archive.subprocess.run", side_effect=fake_run):
                meta = upload_dcd_with_verify(
                    dcd, "s3://bucket/key/production.dcd", "default", "us-east-1"
                )
            self.assertEqual(meta["size_bytes"], dcd.stat().st_size)
            self.assertEqual(meta["sha256"], local_sha)


class ArchiveReplicateTests(unittest.TestCase):
    def test_archive_writes_sidecar_and_deletes_dcd(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rep = Path(td) / "A_ERDRP_WT" / "replicate_1"
            rep.mkdir(parents=True)
            dcd = rep / "production.dcd"
            dcd.write_bytes(b"x" * 128)
            topo = Path(td) / "topo.pdb"
            topo.write_text("END\n")

            config = S3ArchiveConfig(
                bucket="b", prefix="pfx", profile="default", region="us-east-1"
            )

            with mock.patch("stage7_s3_archive.local_dcd_precheck"):
                with mock.patch(
                    "stage7_s3_archive.upload_dcd_with_verify",
                    return_value={
                        "s3_uri": "s3://b/k",
                        "size_bytes": 128,
                        "sha256": "abc",
                        "uploaded_at": "t",
                    },
                ):
                    sidecar = archive_replicate_dir(
                        rep,
                        "A_ERDRP_WT",
                        1,
                        topo,
                        config,
                        2000,
                        results_root=Path(td),
                    )
            self.assertFalse(dcd.exists())
            self.assertTrue(sidecar.exists())
            data = json.loads(sidecar.read_text())
            self.assertEqual(data["case_id"], "A_ERDRP_WT")

    def test_replicate_s3_key_layout(self) -> None:
        key = replicate_s3_key("smoke", Path("/tmp"), "A_ERDRP_WT", 99)
        self.assertEqual(
            key,
            "smoke/stage7_50ns_direct/results/A_ERDRP_WT/replicate_99/production.dcd",
        )


if __name__ == "__main__":
    unittest.main()
