# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-time, cluster-side clean-history publication; never changes the private source repo."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi, hf_hub_download
from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import require_secret

from experiments.grug.moe_hero_ep.ops.upload_hf import validate_model_card
from experiments.post_training.long_loop.archive import write_once
from experiments.post_training.long_loop.hf_archive import (
    PUBLIC_SIDECARS,
    UPLOAD_REQUEST,
    audit_public_bytes,
    audit_public_fields,
    audit_public_file,
    location_commitment,
    public_weight_name,
    verify_hf_checkpoint,
)

SOURCE_REPO = "open-athena/Snowball-LongLoop-Plumbing-Step3-2026.11.09"
SOURCE_REVISION = "8a5c545ee1090776a00316108b1261f0250c6653"
TARGET_REPO = "open-athena/Snowball-LongLoop-Plumbing-Step3-2026.11.09-r1"
SOURCE_RECEIPT = "s3://marin-us-east-08a/marin/long-loop/archive/2026.11.09.10-hf1/hf-checkpoint.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--receipt-output", required=True)
    args = parser.parse_args()
    api = HfApi(token=require_secret("HF_TOKEN"))
    fs, key = filesystem_for(SOURCE_RECEIPT)
    receipt = json.loads(fs.cat_file(key))
    assert receipt["repo_id"] == SOURCE_REPO and receipt["revision"] == SOURCE_REVISION
    info = api.model_info(SOURCE_REPO, revision=SOURCE_REVISION, files_metadata=True)
    assert info.private and info.sha == SOURCE_REVISION
    actual = {item.rfilename: item for item in info.siblings if item.rfilename != ".gitattributes"}
    assert set(actual) == {item["path"] for item in receipt["files"]}
    card = args.card.read_bytes()
    audit_public_bytes("README.md", card)
    records, operations, small = [], [], {}
    for item in receipt["files"]:
        name = item["path"]
        remote = actual[name]
        assert remote.size == item["size"]
        if public_weight_name(name):
            assert remote.lfs is not None and remote.lfs.sha256 == item["sha256"]
            header = api.parse_safetensors_file_metadata(SOURCE_REPO, name, revision=SOURCE_REVISION)
            audit_public_fields(name, header.metadata)
            audit_public_bytes(name, json.dumps({"metadata": header.metadata, "tensors": list(header.tensors)}).encode())
            operations.append(
                CommitOperationCopy(
                    src_path_in_repo=name,
                    path_in_repo=name,
                    src_revision=SOURCE_REVISION,
                    src_repo_id=SOURCE_REPO,
                    src_repo_type="model",
                )
            )
            records.append(item)
            continue
        source = Path(hf_hub_download(SOURCE_REPO, name, revision=SOURCE_REVISION, token=api.token))
        content = source.read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
        if name in {"README.md", UPLOAD_REQUEST}:
            continue
        if name not in PUBLIC_SIDECARS | {"LICENSE"}:
            raise ValueError(f"Unexpected source sidecar: {name}")
        audit_public_file(source, name)
        small[name] = content
        records.append(item)
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=content))
    validate_model_card(card, small["LICENSE"])
    records.append({"path": "README.md", "size": len(card), "sha256": hashlib.sha256(card).hexdigest()})
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))
    request = {
        "schema": 2,
        "location_commitment_sha256": location_commitment(receipt["source"], receipt["native_checkpoint"]),
        "files": sorted(records, key=lambda item: item["path"]),
    }
    raw = json.dumps(request, sort_keys=True, indent=2).encode()
    audit_public_bytes(UPLOAD_REQUEST, raw)
    operations.append(CommitOperationAdd(path_in_repo=UPLOAD_REQUEST, path_or_fileobj=raw))
    records.append({"path": UPLOAD_REQUEST, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    # A fresh repository is mandatory: never reuse a history containing private paths.
    api.create_repo(TARGET_REPO, repo_type="model", private=False, exist_ok=False)
    commit = api.create_commit(
        repo_id=TARGET_REPO,
        repo_type="model",
        operations=operations,
        commit_message="Publish terminal long-loop plumbing checkpoint with reviewed provenance",
    )
    public_receipt = dict(
        receipt,
        repo_id=TARGET_REPO,
        revision=commit.oid,
        private=False,
        files=sorted(records, key=lambda item: item["path"]),
    )
    verify_hf_checkpoint(public_receipt)
    write_once(args.receipt_output, json.dumps(public_receipt, sort_keys=True, indent=2).encode())
    print(json.dumps({"repo_id": TARGET_REPO, "revision": commit.oid, "verified": True}))


if __name__ == "__main__":
    main()
