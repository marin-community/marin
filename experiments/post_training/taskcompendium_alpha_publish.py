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

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.mixed_release import PublishedRow
from taskcompendium.models import SCHEMA_VERSION
from taskcompendium.release_common import REPO_ID

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
    downloaded = Path(
        hf_hub_download(repo_id, path, repo_type="dataset", revision=revision, force_download=True, token=False)
    )
    with downloaded.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Hub readback differs from reviewed file: {path}")


def _stage_release(
    source_prefix: str, local: Path, manifest_sha256: str, readme_sha256: str
) -> dict[str, dict[str, Any]]:
    _copy_and_verify(str(StoragePath(source_prefix) / "manifest.json"), local / "manifest.json", manifest_sha256)
    manifest = json.loads((local / "manifest.json").read_text())
    if (
        manifest["repo_id"] != REPO_ID
        or manifest["publication_ready"] is not True
        or manifest["public_record_version"] != 3
        or manifest["task_spec_schema"] != SCHEMA_VERSION
    ):
        raise ValueError("Reviewed manifest does not authorize the target public dataset")
    audit = manifest["reconstruction_audit"]
    if (
        audit["workplace_expected_state_matches"] != {"train": 1255, "validation": 545}
        or audit["harbor_trials"] != 8
        or audit["harbor_rewards"]
        != {
            "workplace/train": [1.0, 0.0],
            "workplace/validation": [1.0, 0.0],
            "tasktrove_clean/mcqa": [1.0, 0.0],
            "tasktrove_clean/prism_math": [1.0, 0.0],
        }
    ):
        raise ValueError("Reviewed release lacks successful reconstructed source and Harbor gates")
    data_files = {entry["path"]: entry for entry in manifest["data_files"]}
    if set(data_files) != set(EXPECTED_ROWS):
        raise ValueError("Reviewed release has an unexpected config, cohort, or file")
    if any(
        entry["exported_rows"] != EXPECTED_ROWS[path] or entry["accepted_rows"] != EXPECTED_ROWS[path]
        for path, entry in data_files.items()
    ):
        raise ValueError("Reviewed release row counts changed")
    _copy_and_verify(str(StoragePath(source_prefix) / "README.md"), local / "README.md", readme_sha256)
    for path, entry in data_files.items():
        _copy_and_verify(str(StoragePath(source_prefix) / path), local / path, entry["sha256"], EXPECTED_ROWS[path])
        with (local / path).open(encoding="utf-8") as stream:
            for line in stream:
                row = PublishedRow.model_validate_json(line)
                if row.source.dataset.startswith(("s3://", "gs://")):
                    raise ValueError(f"Public row exposes a regional bucket: {path}")
    for path in local.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                if any(b"s3://" in line or b"marin-us-east-02a" in line for line in stream):
                    raise ValueError(f"Public release exposes a regional bucket: {path.relative_to(local)}")
    return data_files


def _upload_update(local: Path, base_revision: str) -> tuple[HfApi, str]:
    api = HfApi()
    current = api.repo_info(REPO_ID, repo_type="dataset")
    if current.private or current.sha != base_revision:
        raise ValueError("Public dataset head differs from the reviewed base revision")
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=path.relative_to(local).as_posix(), path_or_fileobj=path)
            for path in sorted(local.rglob("*"))
            if path.is_file()
        ],
        parent_commit=base_revision,
        commit_message="Publish complete TaskCompendium alpha 1 demonstration tasks",
    )
    return api, commit.oid


def _verify_public_update(
    revision: str, data_files: dict[str, dict[str, Any]], manifest_sha256: str, readme_sha256: str
) -> None:
    expected_files = {
        "manifest.json": manifest_sha256,
        "README.md": readme_sha256,
        **{path: entry["sha256"] for path, entry in data_files.items()},
    }
    for path, digest in expected_files.items():
        _verify_hub_file(REPO_ID, revision, path, digest)
    public = HfApi(token=False).repo_info(REPO_ID, repo_type="dataset")
    if public.private or public.sha != revision:
        raise ValueError("Hub public readback did not match the reviewed commit")


def _write_publication_summary(
    revision: str, data_files: dict[str, dict[str, Any]], manifest_sha256: str, readme_sha256: str, output: Path
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
    output.mkdir(parents=True, exist_ok=True)
    (output / "publication-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the reviewed TaskCompendium alpha-1 dataset")
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--readme-sha256", required=True)
    parser.add_argument("--base-revision", required=True)
    args = parser.parse_args()
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    if not args.source_prefix.startswith("s3://marin-us-east-02a/marin/taskcompendium/releases/"):
        raise ValueError("Source must be the reviewed regional release prefix")
    if len(args.base_revision) != 40 or any(char not in "0123456789abcdef" for char in args.base_revision):
        raise ValueError("Base revision must be a full Git commit")
    with tempfile.TemporaryDirectory(prefix="taskcompendium-alpha-upload-") as directory:
        local = Path(directory)
        data_files = _stage_release(args.source_prefix, local, args.manifest_sha256, args.readme_sha256)
        _, revision = _upload_update(local, args.base_revision)
        _verify_public_update(revision, data_files, args.manifest_sha256, args.readme_sha256)
        _write_publication_summary(revision, data_files, args.manifest_sha256, args.readme_sha256, output)


if __name__ == "__main__":
    main()
