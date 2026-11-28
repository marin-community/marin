# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage and preserve Endless's pristine SIF images as independently sealed artifacts."""

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from finestore.rl.licenses import LicenseReceipt
from huggingface_hub import hf_hub_download
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.storage_path import StoragePath
from shellbox.sif import OciTools, SifTools, file_sha256, oci_from_filesystem, sif_runtime, unpack_sif

from experiments.post_training.task_curation.datasets.facet.artifacts import PreservedImage
from experiments.post_training.task_curation.datasets.facet.images import (
    ImageConfig,
    image_main,
    pinned_json,
    upload_verified,
)
from experiments.post_training.task_curation.license_publication import require_publication_license

DATASET = "FWeindel/validated-tasks"
TASK_REVISION = "317659ac88c8b706d318020c5fc3813a931fb6ec"
IMAGE_REVISION = "0e3d341e05545d2e997602cfa95f4f7d46d87898"
REGIONAL_PREFIX = "s3://marin-us-east-02a/marin"
EXPECTED_IMAGES = 1777
EXPECTED_TASKS = {"endless_v2": 1958, "endless_v3": 121}


def image_files(index: dict) -> dict[str, dict]:
    """Return image bindings only after both independent task maps are accounted for."""
    if (index["repo"], index["revision"]) != (DATASET, IMAGE_REVISION):
        raise ValueError("Endless image index has a different upstream pin")
    if {folder: len(rows) for folder, rows in index["tasks"].items()} != EXPECTED_TASKS:
        raise ValueError("Endless image index does not cover both kept populations")
    files = {item["bundle_id"]: {**item, "path": path} for path, item in index["files"].items()}
    if len(files) != len(index["files"]) or len(files) != EXPECTED_IMAGES:
        raise ValueError("Endless image file bindings are repeated or incomplete")
    requested = {
        env["bundle"] for rows in index["tasks"].values() for row in rows.values() for env in row["environments"]
    }
    if requested != files.keys() or requested != index["bundles"].keys():
        raise ValueError("Endless task and image bundle maps differ")
    for bundle, item in files.items():
        source = index["bundles"][bundle]
        if item["path"] != source["prefix"] + "/" + item["filename"]:
            raise ValueError("Endless SIF directory/file binding differs")
        if source["files"][item["filename"]] != {"bytes": item["bytes"], "sha256": item["sha256"]}:
            raise ValueError("Endless SIF checksum binding differs")
    return files


def preserve_image(config: ImageConfig) -> PreservedImage:
    if get_job_info() is None:
        raise RuntimeError("Preserve Endless images inside the approved regional Iris CPU job")
    StoragePath(config.output_path).relative_to(StoragePath(REGIONAL_PREFIX))
    licenses = pinned_json(config.licenses_uri, config.licenses_sha256)
    for folder in EXPECTED_TASKS:
        source_id = DATASET.replace("/", ":", 1) + "/" + folder
        receipt = LicenseReceipt.model_validate(licenses[source_id])
        require_publication_license(receipt, source_id, DATASET, "train")
    item = image_files(pinned_json(config.index_uri, config.index_sha256))[config.bundle_id]
    if (item["path"], item["sha256"], item["bytes"]) != (config.filename, config.sif_sha256, config.sif_bytes):
        raise ValueError("Endless step differs from its frozen bundle binding")
    for name, path in config.tools.items():
        if file_sha256(Path(path)) != config.tool_sha256[name]:
            raise ValueError(f"Endless tool bytes changed: {name}")
    output = StoragePath(config.output_path)
    Path(config.workdir).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=config.bundle_id + "-", dir=config.workdir) as temporary:
        work = Path(temporary)
        cached = Path(
            hf_hub_download(
                DATASET, config.filename, repo_type="dataset", revision=IMAGE_REVISION, cache_dir=work / "hub"
            )
        )
        if cached.stat().st_size != config.sif_bytes or file_sha256(cached) != config.sif_sha256:
            raise ValueError("Downloaded SIF differs from the frozen upstream checksum")
        original = upload_verified(cached, output / "original.sif")
        unpacked = unpack_sif(
            cached,
            config.sif_sha256,
            work / "source",
            SifTools(Path(config.tools["siftool"]), Path(config.tools["unsquashfs"])),
            4,
        )
        runtime = sif_runtime(unpacked.root, Path(config.tools["bwrap"]), 60)
        converted = oci_from_filesystem(
            unpacked.root, runtime, work / "converted", OciTools(Path(config.tools["tar"]), Path(config.tools["umoci"]))
        )
        blobs = {
            str(path.relative_to(converted.layout)): upload_verified(
                path, output / "oci" / str(path.relative_to(converted.layout))
            )
            for path in sorted(converted.layout.rglob("*"))
            if path.is_file()
        }
        tree = work / "filesystem-entries.json"
        tree.write_text(
            json.dumps(
                [asdict(entry) for entry in converted.entries], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        )
        tree_binding = upload_verified(tree, output / tree.name)
        if tree_binding["sha256"] != converted.tree_sha256:
            raise ValueError("Filesystem proof serialization differs from the conversion")
        document = {
            "bundle_id": config.bundle_id,
            "source_repo": DATASET,
            "source_revision": IMAGE_REVISION,
            "source_filename": config.filename,
            "source_index": {"uri": config.index_uri, "sha256": config.index_sha256},
            "original_sif": original,
            "partition_sha256": unpacked.partition_sha256,
            "descriptor_listing": unpacked.descriptor_listing,
            "runtime": asdict(runtime),
            "manifest_digest": converted.manifest_digest,
            "config_digest": converted.config_digest,
            "diff_id": converted.diff_id,
            "filesystem_equality": True,
            "tree": tree_binding,
            "tree_entries": len(converted.entries),
            "oci_files": blobs,
            "tools": config.tool_sha256,
            "registry_push": False,
            "execution_qualification": False,
        }
        proof = work / "preservation.json"
        proof.write_text(json.dumps(document, indent=2) + "\n")
        binding = upload_verified(proof, output / proof.name)
    return PreservedImage(
        path=config.output_path,
        bundle_id=config.bundle_id,
        source_repo=DATASET,
        source_revision=IMAGE_REVISION,
        source_filename=config.filename,
        sif_sha256=config.sif_sha256,
        sif_bytes=config.sif_bytes,
        partition_sha256=unpacked.partition_sha256,
        manifest_digest=converted.manifest_digest,
        config_digest=converted.config_digest,
        diff_id=converted.diff_id,
        tree_sha256=converted.tree_sha256,
        tree_entries=len(converted.entries),
        provenance_uri=binding["uri"],
        provenance_sha256=binding["sha256"],
    )


def main() -> None:
    image_main(
        description="Preserve Endless pristine SIF images as independent artifacts",
        artifact_namespace="endless",
        files_for_index=image_files,
        producer=preserve_image,
    )


if __name__ == "__main__":
    main()
