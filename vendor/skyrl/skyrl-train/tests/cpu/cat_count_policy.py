"""Pinned policy artifacts for CatCount canaries."""

import hashlib
import json
from pathlib import Path

from marinskyrl.remote_io import create_s3_filesystem

POLICY_PREFIX = "s3://marin-us-east-02a/marin/rl-canaries/cat-count/cpu/pretrained/llama-530k/v1"
POLICY_MANIFEST_SHA256 = "7a5f1047648a90262514a663168580610b9f6bbe1522b2c38b7578bbc5eb82ae"


def download_policy(directory: Path) -> None:
    filesystem = create_s3_filesystem(
        config_kwargs={"connect_timeout": 5, "read_timeout": 10, "retries": {"max_attempts": 1}}
    )
    manifest_path = directory / "MANIFEST.json"
    manifest_bytes = (
        manifest_path.read_bytes() if manifest_path.exists() else filesystem.cat(f"{POLICY_PREFIX}/MANIFEST.json")
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != POLICY_MANIFEST_SHA256:
        raise ValueError("CatCount policy manifest SHA256 differs from the calibrated artifact")
    manifest = json.loads(manifest_bytes)
    for name, receipt in manifest["files"].items():
        path = directory / name
        data = path.read_bytes() if path.exists() else filesystem.cat(f"{POLICY_PREFIX}/{name}")
        if len(data) != receipt["bytes"] or hashlib.sha256(data).hexdigest() != receipt["sha256"]:
            raise ValueError(f"CatCount policy SHA256 mismatch: {name}")
        path.write_bytes(data)
    (directory / "MANIFEST.json").write_bytes(manifest_bytes)

