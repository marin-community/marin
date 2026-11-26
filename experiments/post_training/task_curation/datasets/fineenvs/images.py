# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build checksum-bound FineEnvs input images without changing the base runtime."""

import argparse
import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from finestore.rl.licenses import LicenseReceipt, require_license
from finestore.rl.status_taxonomy import DataRestriction
from iris.cluster.client.job_info import get_job_info
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from shellbox.sif import OciTools, file_sha256

from experiments.post_training.task_curation.datasets.facet.images import pinned_json, upload_verified
from experiments.post_training.task_curation.datasets.fineenvs.artifacts import GroupedInputImage
from experiments.post_training.task_curation.datasets.fineenvs.image_inputs import (
    NativeInputFile,
    NativeInputSet,
    derive_group_image,
    image_documents,
)
from experiments.post_training.task_curation.datasets.fineenvs.metadata import (
    DATASET,
    REVISION,
    TASK_COUNTS,
    source_info,
)
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned
from infra.marina.apps.rl_data_catalog.server.restrictions import require_public_data

REGIONAL_PREFIX = "s3://marin-us-east-02a/marin"
BASE_IMAGE = (
    "docker.io/savatar101/env-data-agent-train@sha256:7b3f7e22780b431ce0d43717dcef6875cde84baeee824cc87e968374d62d3470"
)


@dataclass(frozen=True)
class ImageConfig:
    group: int
    data_restriction: DataRestriction
    plan_uri: str
    plan_sha256: str
    packages_uri: str
    packages_sha256: str
    base_uri: str
    base_sha256: str
    licenses_uri: str
    licenses_sha256: str
    qualifications_uri: str
    qualifications_sha256: str
    tools: dict[str, str]
    tool_sha256: dict[str, str]
    workdir: str
    output_path: str


def download_file(pin: dict, destination: Path) -> None:
    StoragePath(pin["uri"]).relative_to(StoragePath(REGIONAL_PREFIX))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with StoragePath(pin["uri"]).open("rb") as source, destination.open("xb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    if destination.stat().st_size != pin["bytes"] or file_sha256(destination) != pin["sha256"]:
        raise ValueError(f"FineEnvs staged file differs from its frozen binding: {pin['uri']}")


def build_group(config: ImageConfig) -> GroupedInputImage:
    """Seal a verified OCI image, its complete filesystem inventory and input provenance."""
    if get_job_info() is None:
        raise RuntimeError("Build FineEnvs images inside the approved regional Iris CPU job")
    StoragePath(config.output_path).relative_to(StoragePath(REGIONAL_PREFIX))
    infos = [source_info(folder) for folder in TASK_COUNTS]
    for info in infos:
        require_public_data(info.data_restriction, info.id, "grouped public-input image")
    (restriction,) = {info.data_restriction for info in infos}
    if config.data_restriction is not restriction:
        raise ValueError("FineEnvs image configuration declares another source restriction tier")
    licenses = pinned_json(config.licenses_uri, config.licenses_sha256)
    for info in infos:
        source = info.id
        receipt = LicenseReceipt.model_validate(licenses[source]["receipt"])
        require_license(receipt, source, DATASET, REVISION, "train", ban=None)
        require_unbanned(source, DATASET, receipt, "train")
    qualification_raw = StoragePath(config.qualifications_uri).read_bytes()
    if hashlib.sha256(qualification_raw).hexdigest() != config.qualifications_sha256:
        raise ValueError("FineEnvs qualification table differs from its frozen binding")
    qualifications = json.loads(qualification_raw)
    expected_profiles = {(1, 1024, 5120, "allow"), (2, 4096, 10240, "allow")}
    if {
        tuple(row["profile"][key] for key in ("cpus", "memory_mb", "storage_mb", "network")) for row in qualifications
    } != expected_profiles:
        raise ValueError("FineEnvs qualification omits an exact native profile")
    for row in qualifications:
        result = pinned_json(row["result"]["uri"], row["result"]["sha256"])
        if result["profile"] != row["profile"] or result["image"] != BASE_IMAGE:
            raise ValueError("FineEnvs qualification measured another native profile or base image")
        if set(result["resolved_cells"]) != {
            "provider_profile",
            "memory",
            "storage",
            "cpu",
            "network_allow",
            "metadata",
            "setup",
        } or set(result["resolved_cells"].values()) != {"SUPPORTED"}:
            raise ValueError("FineEnvs image build requires every native-profile cell to be supported")
    plan = pinned_json(config.plan_uri, config.plan_sha256)
    packages = pinned_json(config.packages_uri, config.packages_sha256)
    base = pinned_json(config.base_uri, config.base_sha256)
    if plan["base_image"] != BASE_IMAGE or base["image"] != BASE_IMAGE:
        raise ValueError("FineEnvs base image differs from its pinned native image")
    if len(plan["groups"]) > 10 or plan["input_set_count"] != 419 or plan["task_count"] != 4516:
        raise ValueError("FineEnvs grouped-image plan has another population or exceeds the snapshot budget")
    if packages["plan_sha256"] != config.plan_sha256 or packages["package_count"] != 419:
        raise ValueError("FineEnvs staged packages do not bind the grouped-image plan")
    (group,) = [row for row in plan["groups"] if row["group"] == config.group]
    for name, path in config.tools.items():
        if file_sha256(Path(path)) != config.tool_sha256[name]:
            raise ValueError(f"FineEnvs image tool changed: {name}")
    Path(config.workdir).mkdir(parents=True, exist_ok=True)
    output = StoragePath(config.output_path)
    with TemporaryDirectory(prefix=f"fineenvs-group-{config.group}-", dir=config.workdir) as temporary:
        work = Path(temporary)
        base_layout = work / "base"
        for pin in base["oci_files"]:
            relative = Path(pin["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("FineEnvs base OCI file has an invalid path")
            download_file(pin, base_layout / relative)
        if image_documents(base_layout, "source").manifest_digest != base["selected_platform_manifest"]:
            raise ValueError("FineEnvs base OCI manifest does not match the inspected platform")
        inputs = []
        bindings = []
        for row in group["input_sets"]:
            pin = packages["packages"][row["signature"]]
            if pin["group"] != config.group:
                raise ValueError("FineEnvs input set is staged for another image group")
            package = work / "packages" / (row["signature"] + ".tar.gz")
            download_file(pin, package)
            item = NativeInputSet(package, tuple(NativeInputFile.model_validate(file) for file in row["files"]))
            if item.signature != row["signature"]:
                raise ValueError("FineEnvs input-set signature differs from its complete file inventory")
            inputs.append(item)
            bindings.append(
                {
                    "signature": item.signature,
                    "image_prefix": str(item.image_prefix),
                    "files": row["files"],
                    "package": pin,
                }
            )
        derived = derive_group_image(
            base_layout,
            tuple(inputs),
            work / "derived",
            OciTools(Path(config.tools["tar"]), Path(config.tools["umoci"])),
        )
        oci_files = {
            str(path.relative_to(derived.layout)): upload_verified(
                path, output / "oci" / str(path.relative_to(derived.layout))
            )
            for path in sorted(derived.layout.rglob("*"))
            if path.is_file()
        }
        inventory = work / "filesystem-entries.json"
        inventory.write_text(
            json.dumps(
                [asdict(entry) for entry in derived.entries], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        )
        tree_pin = upload_verified(inventory, output / inventory.name)
        provenance = {
            "data_restriction": restriction.value,
            "group": config.group,
            "base_image": BASE_IMAGE,
            "plan": {"uri": config.plan_uri, "sha256": config.plan_sha256},
            "packages": {"uri": config.packages_uri, "sha256": config.packages_sha256},
            "base_inspection": {"uri": config.base_uri, "sha256": config.base_sha256},
            "license_receipts": {"uri": config.licenses_uri, "sha256": config.licenses_sha256},
            "qualifications": {"uri": config.qualifications_uri, "sha256": config.qualifications_sha256},
            "input_sets": bindings,
            "manifest_digest": derived.manifest_digest,
            "tree": tree_pin,
            "tree_entries": len(derived.entries),
            "oci_files": oci_files,
            "base_filesystem_preserved": True,
            "runtime_configuration_preserved": True,
            "derived_oci_roundtrip_equal": True,
            "tools": config.tool_sha256,
            "deviation": plan["deviation"],
            "task_binding": plan["task_binding"],
            "registry_push": False,
            "execution_qualification": False,
        }
        path = work / "provenance.json"
        path.write_text(json.dumps(provenance, indent=2) + "\n")
        pin = upload_verified(path, output / path.name)
    return GroupedInputImage(
        data_restriction=restriction,
        path=config.output_path,
        group=config.group,
        base_image=BASE_IMAGE,
        input_signatures=derived.input_signatures,
        manifest_digest=derived.manifest_digest,
        tree_sha256=tree_pin["sha256"],
        provenance_uri=pin["uri"],
        provenance_sha256=pin["sha256"],
    )


def image_config(config: ImageConfig, context: StepContext) -> ImageConfig:
    return replace(config, output_path=context.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "plan-sha256",
        "packages",
        "packages-sha256",
        "base",
        "base-sha256",
        "licenses",
        "licenses-sha256",
        "qualifications",
        "qualifications-sha256",
        "tools",
        "tools-sha256",
        "workdir",
        "artifact-root",
        "artifact-version",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--group", type=int, action="append")
    parser.add_argument("--max-concurrent", type=int, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    logging.basicConfig(level=logging.INFO)
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    # Source policy applies before the artifact runner can select a cached image.
    infos = [source_info(folder) for folder in TASK_COUNTS]
    for info in infos:
        require_public_data(info.data_restriction, info.id, "grouped image artifact selection")
    (restriction,) = {info.data_restriction for info in infos}
    plan = pinned_json(args.plan, args.plan_sha256)
    tools = pinned_json(args.tools, args.tools_sha256)
    selected = sorted(args.group if args.group is not None else [row["group"] for row in plan["groups"]])
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("FineEnvs image selection must contain distinct groups")
    steps = []
    with artifact_prefix(args.artifact_root):
        for group in selected:
            config = ImageConfig(
                group,
                restriction,
                args.plan,
                args.plan_sha256,
                args.packages,
                args.packages_sha256,
                args.base,
                args.base_sha256,
                args.licenses,
                args.licenses_sha256,
                args.qualifications,
                args.qualifications_sha256,
                {name: tool["path"] for name, tool in tools.items()},
                {name: tool["binary_sha256"] for name, tool in tools.items()},
                args.workdir,
                "",
            )
            steps.append(
                ArtifactStep(
                    name=f"rl-data/fineenvs/grouped-input-oci/group-{group:02d}",
                    version=args.artifact_version,
                    artifact_type=GroupedInputImage,
                    run=build_group,
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
    destination.write_text(
        json.dumps([row.model_dump(mode="json") for row in results], indent=2) + "\n", auto_mkdir=True
    )


if __name__ == "__main__":
    main()
