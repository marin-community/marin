# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish exact staged raw source snapshots as new private provenance archives."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from finestore.rl.licenses import LicenseReceipt
from finestore.rl.status_taxonomy import DataRestriction
from huggingface_hub import HfApi, hf_hub_download
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX, read_license_receipts
from experiments.post_training.task_curation.license_publication import (
    check_publication,
    licensed_dataset_card,
    prepare_license_publication,
    require_publication_license,
)
from experiments.post_training.task_curation.native_quick import source_slug
from experiments.post_training.task_curation.publication_gate import require_public_data
from experiments.post_training.task_curation.remediation_publish import RawSourceArchive


def publish_raw_source(
    stage: dict, receipt: LicenseReceipt, *, version: str, evidence_root: str, use: str
) -> RawSourceArchive:
    """Archive one raw snapshot; refused outright for a PRIVATE_ALWAYS source.

    The staged snapshot carries its source's typed tier (``data_restriction``):
    an exact re-publish of its content is an export, and the private store never
    exports. A staged snapshot without the tier is refused.
    """
    raw_tier = stage.get("data_restriction")
    if not isinstance(raw_tier, str):
        raise ValueError(f"Staged raw snapshot declares no restriction tier: {stage['source_id']}")
    require_public_data(DataRestriction(raw_tier), stage["source_id"], "raw source archive")
    require_publication_license(receipt, stage["source_id"], stage["dataset_id"], use)
    identity = hashlib.sha256(stage["source_id"].encode()).hexdigest()[:8]
    receipt_path = StoragePath(evidence_root) / identity / "raw-source-archive.json"
    if receipt_path.exists():
        raise FileExistsError("Raw archive receipt requires a new evidence root")
    api = HfApi()
    upstream = api.dataset_info(stage["dataset_id"], revision=stage["revision"])
    if upstream.sha != stage["revision"] or upstream.siblings is None:
        raise ValueError("Raw snapshot repository metadata does not identify its complete pinned source")
    expected = sorted(file.rfilename for file in upstream.siblings)
    if upstream.sha != stage["revision"] or sorted(stage["source_files"]) != expected:
        raise ValueError("Raw snapshot inventory differs from the complete pinned upstream repository")
    names = [file["path"] for file in stage["files"]]
    if len(names) != len(set(names)) or sorted(names) != expected:
        raise ValueError("Staged raw files do not cover the complete pinned snapshot exactly once")
    slug = source_slug(stage["source_id"])[:40].rstrip("-")
    repo = f"open-athena/atlas-raw-{slug}-{identity}-{version}"
    with TemporaryDirectory(prefix="atlas-raw-source-") as scratch:
        root = Path(scratch)
        publication = prepare_license_publication(receipt, root, use=use)
        hashes = {}
        for file in stage["files"]:
            name = PurePosixPath(file["path"])
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe raw source snapshot path")
            destination = root / "raw" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy(file["uri"], str(destination))
            digest = file_sha256(StoragePath(str(destination)))
            if digest != file["sha256"]:
                raise ValueError(f"Staged raw source bytes changed: {name}")
            hashes["raw/" + file["path"]] = digest
        (root / "raw-snapshot.json").write_text(json.dumps(stage, indent=2) + "\n")
        body = (
            f"# Original raw source archive\n\nPinned input: `{stage['dataset_id']}@{stage['revision']}`. "
            "Every original repository file is preserved under `raw/`; `raw-snapshot.json` records its hash. "
            "This archive supplies provenance and no execution qualification or admission.\n"
        )
        (root / "README.md").write_text(licensed_dataset_card(body, publication, pretty_name="Atlas raw source archive"))
        for path in (root / "README.md", root / "raw-snapshot.json", *publication.notice_files):
            hashes[path.name] = file_sha256(StoragePath(str(path)))
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=False)
        if not api.repo_info(repo, repo_type="dataset").private:
            raise ValueError("Raw source archive destination is not private")
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=root,
            allow_patterns=list(hashes),
            commit_message="Archive exact pinned raw source files privately",
        )
        if not api.repo_info(repo, repo_type="dataset", revision=commit.oid).private:
            raise ValueError("Published raw source archive is not private")
        for name, digest in hashes.items():
            downloaded = hf_hub_download(repo, name, repo_type="dataset", revision=commit.oid)
            if file_sha256(StoragePath(downloaded)) != digest:
                raise ValueError(f"Raw source archive readback differs: {name}")
        result = RawSourceArchive(
            source_id=stage["source_id"],
            dataset_id=stage["dataset_id"],
            source_revision=stage["revision"],
            repo=repo,
            revision=commit.oid,
            url=f"https://huggingface.co/datasets/{repo}/tree/{commit.oid}",
            private=True,
            scope="exact_pinned_raw_source_archive",
            source_artifact=stage["artifact"],
            verified_raw_files=len(names),
            file_hashes=hashes,
            license_check=check_publication(stage["source_id"], repo, commit.oid, use),
        )
    receipt_path.write_text(result.model_dump_json(indent=2) + "\n", auto_mkdir=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Frozen staged snapshot list with per-file URI and SHA-256")
    parser.add_argument("--version", required=True, help="New private repository version suffix")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--use", choices=["train", "eval"], required=True)
    parser.add_argument("--license-receipts", type=Path, default=LICENSE_RECEIPT_INDEX)
    args = parser.parse_args()
    configure_coreweave_s3()
    stages = json.loads(StoragePath(args.snapshots).read_text())
    output = StoragePath(args.evidence_root) / "raw-source-archives.json"
    if output.exists():
        raise FileExistsError("Raw archive collection requires a new evidence root")
    if len({stage["source_id"] for stage in stages}) != len(stages):
        raise ValueError("Duplicate raw snapshot source identity")
    receipts = read_license_receipts(args.license_receipts)
    results = [
        publish_raw_source(
            stage,
            receipts[stage["source_id"]],
            version=args.version,
            evidence_root=args.evidence_root,
            use=args.use,
        )
        for stage in stages
    ]
    output.write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2) + "\n", auto_mkdir=True
    )


if __name__ == "__main__":
    main()
