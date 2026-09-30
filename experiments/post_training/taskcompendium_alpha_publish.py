# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload an exact reviewed regional alpha artifact, then make it public."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from rigging.filesystem.storage_path import StoragePath

REPO_ID = "open-athena/taskcompendium-alpha-1"
EXPECTED_ROWS = {
    "data/workplace/train.jsonl": 1255,
    "data/workplace/validation.jsonl": 545,
    "data/tasktrove_clean/mcqa.jsonl": 23711,
    "data/tasktrove_clean/prism_math.jsonl": 2219,
}


def _copy_and_verify(uri: str, destination: Path, expected_sha256: str, expected_rows: int | None = None) -> None:
    digest = hashlib.sha256()
    rows = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with StoragePath(uri).open("rb") as source, destination.open("wb") as output:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            rows += chunk.count(b"\n")
            output.write(chunk)
    if digest.hexdigest() != expected_sha256 or (expected_rows is not None and rows != expected_rows):
        raise ValueError(f"Regional release file differs from its reviewed digest/count: {uri}")


def _verify_hub_file(repo_id: str, revision: str, path: str, expected_sha256: str) -> None:
    downloaded = Path(hf_hub_download(repo_id, path, repo_type="dataset", revision=revision, force_download=True))
    with downloaded.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Hub readback differs from reviewed file: {path}")


def _stage_release(
    source_prefix: str, local: Path, manifest_sha256: str, readme_sha256: str
) -> dict[str, dict[str, Any]]:
    _copy_and_verify(f"{source_prefix}/manifest.json", local / "manifest.json", manifest_sha256)
    manifest = json.loads((local / "manifest.json").read_text())
    if manifest["repo_id"] != REPO_ID or manifest["publication_ready"] is not True:
        raise ValueError("Reviewed manifest does not authorize the target public dataset")
    data_files = {entry["path"]: entry for entry in manifest["data_files"]}
    if set(data_files) != set(EXPECTED_ROWS):
        raise ValueError("Reviewed release has an unexpected config, cohort, or file")
    if any(
        entry["exported_rows"] != EXPECTED_ROWS[path] or entry["accepted_rows"] != EXPECTED_ROWS[path]
        for path, entry in data_files.items()
    ):
        raise ValueError("Reviewed release row counts changed")
    _copy_and_verify(f"{source_prefix}/README.md", local / "README.md", readme_sha256)
    for path, entry in data_files.items():
        _copy_and_verify(f"{source_prefix}/{path}", local / path, entry["sha256"], EXPECTED_ROWS[path])
    return data_files


def _upload_private(local: Path) -> tuple[HfApi, str]:
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=False)
    commit = api.upload_folder(
        folder_path=str(local),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Publish TaskCompendium alpha 1 public task data",
    )
    return api, commit.oid


def _verify_and_publish(
    api: HfApi, revision: str, data_files: dict[str, dict[str, Any]], manifest_sha256: str, readme_sha256: str
) -> None:
    expected_files = {
        "manifest.json": manifest_sha256,
        "README.md": readme_sha256,
        **{path: entry["sha256"] for path, entry in data_files.items()},
    }
    for path, digest in expected_files.items():
        _verify_hub_file(REPO_ID, revision, path, digest)
    api.update_repo_settings(REPO_ID, private=False, repo_type="dataset")
    public = HfApi(token=False).repo_info(REPO_ID, repo_type="dataset", revision=revision)
    if public.private or public.sha != revision:
        raise ValueError("Hub public readback did not match the reviewed commit")


def _write_publication_summary(
    revision: str, data_files: dict[str, dict[str, Any]], manifest_sha256: str, readme_sha256: str
) -> None:
    summary = {
        "repo_id": REPO_ID,
        "revision": revision,
        "public": True,
        "manifest_sha256": manifest_sha256,
        "readme_sha256": readme_sha256,
        "files": {path: data_files[path]["sha256"] for path in sorted(data_files)},
        "rows": EXPECTED_ROWS,
    }
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "publication-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the reviewed TaskCompendium alpha-1 dataset")
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--readme-sha256", required=True)
    args = parser.parse_args()
    if not args.source_prefix.startswith("s3://marin-us-east-02a/marin/taskcompendium/releases/"):
        raise ValueError("Source must be the reviewed regional release prefix")
    with tempfile.TemporaryDirectory(prefix="taskcompendium-alpha-upload-") as directory:
        local = Path(directory)
        data_files = _stage_release(args.source_prefix.rstrip("/"), local, args.manifest_sha256, args.readme_sha256)
        api, revision = _upload_private(local)
        _verify_and_publish(api, revision, data_files, args.manifest_sha256, args.readme_sha256)
        _write_publication_summary(revision, data_files, args.manifest_sha256, args.readme_sha256)


if __name__ == "__main__":
    main()
