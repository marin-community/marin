# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage and preserve FACET's pristine SIF images as independently sealed artifacts."""

import argparse
import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from finestore.rl.licenses import LicenseReceipt, require_license
from huggingface_hub import hf_hub_download
from iris.cluster.client.job_info import get_job_info
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from shellbox.sif import OciTools, SifTools, file_sha256, oci_from_filesystem, sif_runtime, unpack_sif

from experiments.post_training.task_curation.datasets.facet.artifacts import PreservedImage
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned

DATASET = "FWeindel/validated-tasks"
TASK_REVISION = "317659ac88c8b706d318020c5fc3813a931fb6ec"
IMAGE_REVISION = "815988416b9a982932b5531eee9853d4c0e3b55a"
REGIONAL_PREFIX = "s3://marin-us-east-02a/marin"
EXPECTED_IMAGES = 747
EXPECTED_TASKS = {"facet_v2": 780, "facet_v3": 728}


@dataclass(frozen=True)
class ImageConfig:
    bundle_id: str
    filename: str
    sif_sha256: str
    sif_bytes: int
    index_uri: str
    index_sha256: str
    licenses_uri: str
    licenses_sha256: str
    tools: dict[str, str]
    tool_sha256: dict[str, str]
    workdir: str
    output_path: str


def pinned_json(uri: str, sha256: str) -> dict:
    raw = StoragePath(uri).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"Pinned FACET manifest changed: {uri}")
    return json.loads(raw)


def image_files(index: dict) -> dict[str, dict]:
    """Return image bindings only after both independent task maps are accounted for."""
    if (index["repo"], index["revision"]) != (DATASET, IMAGE_REVISION):
        raise ValueError("FACET image index has a different upstream pin")
    if {folder: len(rows) for folder, rows in index["tasks"].items()} != EXPECTED_TASKS:
        raise ValueError("FACET image index does not cover both kept populations")
    files = {item["bundle_id"]: {**item, "path": path} for path, item in index["files"].items()}
    if len(files) != len(index["files"]) or len(files) != EXPECTED_IMAGES:
        raise ValueError("FACET image file bindings are repeated or incomplete")
    requested = {
        env["bundle"] for rows in index["tasks"].values() for row in rows.values() for env in row["environments"]
    }
    if requested != files.keys() or requested != index["bundles"].keys():
        raise ValueError("FACET task and image bundle maps differ")
    for bundle, item in files.items():
        source = index["bundles"][bundle]
        if item["path"] != source["prefix"] + "/" + item["filename"]:
            raise ValueError("FACET SIF directory/file binding differs")
        if source["files"][item["filename"]] != {"bytes": item["bytes"], "sha256": item["sha256"]}:
            raise ValueError("FACET SIF checksum binding differs")
    return files


def upload_verified(source: Path, destination: StoragePath) -> dict:
    if destination.exists():
        raise FileExistsError(str(destination))
    with source.open("rb") as reader, destination.open("wb", auto_mkdir=True) as writer:
        shutil.copyfileobj(reader, writer)
    digest = file_sha256(source)
    with destination.open("rb") as reader:
        if hashlib.file_digest(reader, "sha256").hexdigest() != digest:
            raise ValueError(f"FACET artifact readback differs: {destination}")
    return {"uri": str(destination), "sha256": digest, "bytes": source.stat().st_size}


def preserve_image(config: ImageConfig) -> PreservedImage:
    if get_job_info() is None:
        raise RuntimeError("Preserve FACET images inside the approved regional Iris CPU job")
    StoragePath(config.output_path).relative_to(StoragePath(REGIONAL_PREFIX))
    licenses = pinned_json(config.licenses_uri, config.licenses_sha256)
    for folder in EXPECTED_TASKS:
        source_id = DATASET.replace("/", ":", 1) + "/" + folder
        receipt = LicenseReceipt.model_validate(licenses[source_id])
        require_license(receipt, source_id, DATASET, TASK_REVISION, "train", ban=None)
        require_unbanned(source_id, DATASET, receipt, "train")
    item = image_files(pinned_json(config.index_uri, config.index_sha256))[config.bundle_id]
    if (item["path"], item["sha256"], item["bytes"]) != (config.filename, config.sif_sha256, config.sif_bytes):
        raise ValueError("FACET step differs from its frozen bundle binding")
    for name, path in config.tools.items():
        if file_sha256(Path(path)) != config.tool_sha256[name]:
            raise ValueError(f"FACET tool bytes changed: {name}")
    output = StoragePath(config.output_path)
    Path(config.workdir).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=config.bundle_id + "-", dir=config.workdir) as temporary:
        work = Path(temporary)
        cached = Path(hf_hub_download(DATASET, config.filename, repo_type="dataset", revision=IMAGE_REVISION))
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


def image_config(config: ImageConfig, context: StepContext) -> ImageConfig:
    return replace(config, output_path=context.output_path)


def image_main(
    *,
    description: str,
    artifact_namespace: str,
    files_for_index: Callable[[dict], dict[str, dict]],
    producer: Callable[[ImageConfig], PreservedImage],
) -> None:
    """Run the shared pristine-SIF artifact CLI with source-specific binding gates."""
    parser = argparse.ArgumentParser(description=description)
    for name in (
        "index",
        "index-sha256",
        "licenses",
        "licenses-sha256",
        "tools",
        "tools-sha256",
        "workdir",
        "artifact-root",
        "artifact-version",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-concurrent", type=int, required=True)
    parser.add_argument("--bundle", action="append")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    logging.basicConfig(level=logging.INFO)
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    index = pinned_json(args.index, args.index_sha256)
    files = files_for_index(index)
    tools = pinned_json(args.tools, args.tools_sha256)
    selected = sorted(args.bundle if args.bundle else files)
    if len(set(selected)) != len(selected):
        raise ValueError("Image bundle selection repeats an image")
    with artifact_prefix(args.artifact_root):
        steps = []
        for bundle in selected:
            item = files[bundle]
            config = ImageConfig(
                bundle,
                item["path"],
                item["sha256"],
                item["bytes"],
                args.index,
                args.index_sha256,
                args.licenses,
                args.licenses_sha256,
                {name: tool["path"] for name, tool in tools.items()},
                {name: tool["binary_sha256"] for name, tool in tools.items()},
                args.workdir,
                "",
            )
            steps.append(
                ArtifactStep(
                    name=f"rl-data/{artifact_namespace}/pristine-oci/" + bundle,
                    version=args.artifact_version,
                    artifact_type=PreservedImage,
                    run=producer,
                    build_config=partial(image_config, config),
                )
            )
        if not args.run:
            print(json.dumps([{"path": step.path(), "fingerprint": step.fingerprint()} for step in steps], indent=2))
            return
        with local_client():
            results = run(*steps, max_concurrent=args.max_concurrent)
        destination = StoragePath(args.output)
        if destination.exists():
            raise FileExistsError(str(destination))
        destination.write_text(json.dumps([result.model_dump(mode="json") for result in results], indent=2) + "\n")


def main() -> None:
    image_main(
        description="Preserve FACET pristine SIF images as independent artifacts",
        artifact_namespace="facet",
        files_for_index=image_files,
        producer=preserve_image,
    )


if __name__ == "__main__":
    main()
