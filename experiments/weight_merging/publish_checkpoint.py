# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish one verified merge from object storage without staging full weights."""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from marin.merging.checkpoint import MANIFEST_NAME
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

logger = logging.getLogger(__name__)
WARNING = (
    "Research artifact. This model has not been properly tested or evaluated and is not necessarily secure. "
    "Use at your own risk in production settings."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--license-sha256", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    card = args.model_card.read_bytes()
    license_text = args.license.read_bytes()
    assert WARNING in card.decode()
    assert yaml.safe_load(card.decode().split("---", 2)[1])["license"] == "openmdw-1.1"
    assert hashlib.sha256(license_text).hexdigest() == args.license_sha256
    fs, path = filesystem_for(args.checkpoint)
    manifest_bytes = fs.cat_file(prefix_join(path, MANIFEST_NAME))
    manifest = json.loads(manifest_bytes)
    objects = manifest["objects"]
    assert len({item["name"] for item in objects}) == len(objects)
    assert len([item for item in objects if item["name"].endswith(".safetensors")]) == manifest["tensor_count"]
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
    existing = set(api.list_repo_files(args.repo_id, repo_type="model"))
    if existing - {".gitattributes"}:
        raise FileExistsError(f"Refuse to overwrite populated repository: {args.repo_id}")

    def preupload(item: dict) -> CommitOperationAdd:
        payload = fs.cat_file(prefix_join(path, item["name"]))
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"Checkpoint object differs from completion manifest: {item['name']}")
        operation = CommitOperationAdd(path_in_repo=item["name"], path_or_fileobj=payload)
        api.preupload_lfs_files(args.repo_id, additions=[operation], repo_type="model", num_threads=1, free_memory=True)
        logger.info("Preuploaded and hash-checked %s", item["name"])
        return operation

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        operations = list(executor.map(preupload, objects))
    operations.extend(
        [
            CommitOperationAdd(path_in_repo=MANIFEST_NAME, path_or_fileobj=manifest_bytes),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card),
            CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=license_text),
        ]
    )
    commit = api.create_commit(
        args.repo_id, repo_type="model", operations=operations, commit_message="Publish evaluated Grug merge checkpoint"
    )
    public = HfApi(token=False)
    info = public.model_info(args.repo_id, revision=commit.oid, files_metadata=True)
    assert not info.private
    siblings = {item.rfilename: item for item in info.siblings}
    for item in objects:
        remote = siblings[item["name"]]
        assert remote.size == item["bytes"]
        if remote.lfs is not None:
            assert remote.lfs.sha256 == item["sha256"]
        else:
            local = hf_hub_download(args.repo_id, item["name"], revision=commit.oid, token=False)
            assert hashlib.sha256(Path(local).read_bytes()).hexdigest() == item["sha256"]
    for filename, expected in (("README.md", card), ("LICENSE", license_text), (MANIFEST_NAME, manifest_bytes)):
        local = hf_hub_download(args.repo_id, filename, revision=commit.oid, token=False)
        assert Path(local).read_bytes() == expected
    report = {
        "repo_id": args.repo_id,
        "commit": commit.oid,
        "public": True,
        "checkpoint": args.checkpoint,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "objects_verified": len(objects),
        "tensor_count": manifest["tensor_count"],
        "license_sha256": args.license_sha256,
        "model_card_sha256": hashlib.sha256(card).hexdigest(),
        "verification": "Public metadata size/LFS SHA256; regular-file contents; exact model card and license",
    }
    output_fs, output_path = filesystem_for(args.report)
    output_fs.pipe_file(output_path, json.dumps(report, indent=2).encode())
    logger.info("Published and verified %s at %s", args.repo_id, commit.oid)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
