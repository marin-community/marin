# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACET terminal packages bound to their preserved pristine image digests."""

import gzip
import hashlib
import io
import json
import tomllib
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
from finestore.rl.curation import ContentEvidenceReference
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.harbor.root_delivery import REWARD_PATH, home_restore_commands, shell_quiesce_command
from taskcompendium.integrity import DeliveredVerifierFile, RootDeliveryContract, SharedDependencyState
from taskcompendium.models import (
    ArtifactKind,
    FileReward,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskSpec,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.staging import resource_files, write_archive

from experiments.post_training.task_curation.datasets.facet.artifacts import PreservedImage
from experiments.post_training.task_curation.datasets.facet.images import (
    DATASET,
    EXPECTED_TASKS,
    IMAGE_REVISION,
    TASK_REVISION,
    image_files,
    pinned_json,
)
from experiments.post_training.task_curation.datasets.native_harbor import native_task, unpack_native
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat
from finestore.rl.status_taxonomy import DataRestriction

VERSION = "2026.11.17.1"
PREFIX = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/fweindel-facet/2026.11.17.1"
IMAGE_INDEX = ContentEvidenceReference(
    uri=PREFIX + "/inputs/verified-sif-image-index-v1.json",
    sha256="91a1825aa7f94b0e56f590aaa1f8e293d1fed736966cbbd051547e6efdd9e6d1",
)
PUBLISHED_IMAGES = ContentEvidenceReference(
    uri=PREFIX + "/registry-full-v1/result.json",
    sha256="22c638f532b22ae81ee0f8a278a08345b976fe148db6195beea622b950533eab",
)
PRIVATE_NAMES = frozenset({"setup.sh", "test.sh", "test_state.py"})
WORKDIR = "/task_file"
TEST_OUTPUT = "/logs/verifier/test-output.txt"
TEST_EXIT = "/logs/verifier/exit-code.txt"
COPIED_REWARD = "/source-reward.txt"
RUBRIC = """Compare the public terminal instruction with the original native verifier and solution.
The pinned image preserves the source SIF filesystem and runtime metadata. Public setup_files
are uploaded before the actor; private tests are restored only after the actor stops.
The original root actor and mutable dependencies remain part of the shared grading contract.
"""


def read_facet_rows(
    path: StoragePath,
    _context: ConversionContext,
    *,
    folder: str,
    index_reference: ContentEvidenceReference,
    images_reference: ContentEvidenceReference,
) -> Iterator[dict]:
    """Keep each native row's image binding from the pinned Parquet metadata."""
    index = pinned_json(index_reference.uri, index_reference.sha256)
    files = image_files(index)
    published = pinned_json(images_reference.uri, images_reference.sha256)
    if published["index_sha256"] != index_reference.sha256 or published["task_counts"] != EXPECTED_TASKS:
        raise ValueError("FACET published images refer to another task index")
    images = {}
    for record in published["images"]:
        preserved = PreservedImage.model_validate(record["preserved_image"])
        original = files[preserved.bundle_id]
        if preserved.bundle_id in images or (preserved.source_repo, preserved.source_revision) != (
            DATASET,
            IMAGE_REVISION,
        ):
            raise ValueError("FACET preserved image identity is repeated or changed")
        if (preserved.source_filename, preserved.sif_sha256, preserved.sif_bytes) != (
            original["path"],
            original["sha256"],
            original["bytes"],
        ):
            raise ValueError("FACET image differs from its pinned SIF")
        expected = published["registry"] + "@" + preserved.manifest_digest
        if record["image"] != expected or not record["registry_manifest_verified"]:
            raise ValueError("FACET registry image lacks its preserved digest readback")
        if record["visibility"] not in {"private", "internal"}:
            raise ValueError("FACET images must not be public")
        images[preserved.bundle_id] = record
    if images.keys() != files.keys():
        raise ValueError("FACET publication does not cover every pristine image")
    with path.open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        metadata = json.loads(parquet.schema_arrow.metadata[b"ot.images.v1"])
        if metadata["missing"] or metadata["tasks"] != index["tasks"][folder]:
            raise ValueError("FACET native Parquet image bindings changed")
        if parquet.metadata.num_rows != EXPECTED_TASKS[folder]:
            raise ValueError("FACET native population count changed")
        seen = set()
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                name = row["path"]
                binding = metadata["tasks"][name]
                if name in seen or row["content_sha256"] != binding["content_sha256"]:
                    raise ValueError("FACET native task identity is repeated or changed")
                seen.add(name)
                (environment,) = binding["environments"]
                if environment["path"] != "environment":
                    raise ValueError("FACET native task has another environment layout")
                yield {**row, "facet_image": images[environment["bundle"]]}
        if seen != metadata["tasks"].keys():
            raise ValueError("FACET native task rows differ from the complete image map")


def decode_facet(row: dict, context: ConversionContext) -> dict:
    return {**unpack_native(row, context), "facet_image": row["facet_image"]}


def convert_facet(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    """Retain native task contents and run the source verifier on its pristine image."""
    task = native_task(row, context)
    if isinstance(task, ImportRejection):
        return task
    resources = {resource.path: resource for resource in task.resources.verifier}
    config = tomllib.loads(resource_bytes(resources["task.toml"]).decode())
    private = {
        name.removeprefix("tests/"): resource.model_copy(update={"path": name.removeprefix("tests/")})
        for name, resource in resources.items()
        if name.startswith("tests/")
    }
    if private.keys() != PRIVATE_NAMES or config.get("agent", {}).get("user") is not None:
        raise ValueError("FACET private verifier layout or original root actor changed")
    if config.get("metadata", {}).get("workdir") != WORKDIR:
        raise ValueError("FACET native working directory changed")
    for name, resource in private.items():
        if int(resource.mode, 8) != (0o644 if name == "test_state.py" else 0o755):
            raise ValueError("FACET native verifier mode changed")
    with TemporaryDirectory(prefix="facet-verifier-") as temporary:
        archive = Path(temporary) / "tests.tar"
        write_archive(archive, resource_files(tuple(private.values()), ""), [])
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as writer:
            writer.write(archive.read_bytes())
    image = row.data["facet_image"]
    environment = task.environment_requirements.model_copy(
        update={
            "docker_image": image["image"],
            "docker_build": None,
            "working_directory": WORKDIR,
            "capabilities": ("shell",),
        }
    )
    transport = environment.model_copy(
        update={
            "working_directory": "/",
            "setup_commands": (),
            "environment_variables": {},
        }
    )
    execute = VerifierCommand(
        argv=(
            "bash",
            "-c",
            "set -eu\nmkdir -p /logs/verifier\n"
            "status=0\nbash /tests/test.sh > " + TEST_OUTPUT + " 2>&1 || status=$?\n"
            "printf '%s\\n' \"$status\" > " + TEST_EXIT + "\nexit 0",
        ),
        cwd=WORKDIR,
        env=config.get("verifier", {}).get("env", {}),
    )
    archived = tuple(
        resource.model_copy(update={"path": "source_archive/" + resource.path}) for resource in task.resources.verifier
    )
    public = tuple(resource for name, resource in resources.items() if name.startswith("setup_files/"))
    return task.model_copy(
        update={
            "environment_requirements": environment,
            "verifier_execution_requirements": task.execution_requirements,
            "resources": ResourceGroups(
                all=task.resources.all, worker=public, oracle=task.resources.oracle, verifier=archived
            ),
            "grader": ScriptGrader(
                argv=("cp", COPIED_REWARD, REWARD_PATH),
                cwd="/",
                environment=transport,
                answer_path=None,
                collect=(
                    shell_quiesce_command(),
                    *home_restore_commands(compressed.getvalue(), restore_root="/tests"),
                    execute,
                ),
                artifacts=(
                    VerifierArtifact(source=TEST_OUTPUT, target="/source-test-output.txt", kind=ArtifactKind.FILE),
                    VerifierArtifact(source=TEST_EXIT, target="/source-exit-code.txt", kind=ArtifactKind.FILE),
                    VerifierArtifact(source=REWARD_PATH, target=COPIED_REWARD, kind=ArtifactKind.FILE),
                ),
                reward=FileReward(files=(RewardFile(path=REWARD_PATH, format=RewardFileFormat.NUMBER),)),
                timeout=config["verifier"]["timeout_sec"],
            ),
            "integrity_contract": RootDeliveryContract(
                dependency_state=SharedDependencyState.AGENT_MUTABLE,
                agent_user=None,
                undeclared_deadline=False,
                verifier_files=tuple(
                    DeliveredVerifierFile(
                        name=name,
                        mode=int(resource.mode, 8),
                        sha256=hashlib.sha256(resource_bytes(resource)).hexdigest(),
                    )
                    for name, resource in sorted(private.items())
                ),
            ),
            "tags": ("root-delivery", "facet", "pristine-sif"),
            "labels": {
                **task.labels,
                "facet_delivery_version": VERSION,
                "facet_sif_sha256": image["preserved_image"]["sif_sha256"],
                "facet_image_provenance": image["preserved_image"]["provenance_uri"],
                "facet_image_provenance_sha256": image["preserved_image"]["provenance_sha256"],
            },
        }
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.FWEINDEL,
                upstream_dataset="validated-tasks",
                upstream_name="FACET terminal tasks",
                upstream_url=f"https://huggingface.co/datasets/{DATASET}/tree/{TASK_REVISION}/{folder}",
                subset=folder,
                count=count,
                family="terminal",
                build_path="Pristine SIF filesystem preserved as pinned OCI image; original-root verifier delivery",
                task_format=TaskFormat.TASKCOMPENDIUM,
                tags=("license:apache-2.0", "facet"),
                notes="Original native task and verifier contents; pending current-pin controls and admission.",
            ),
            config=CurationRecipe(
                name="facet-" + folder.removeprefix("facet_"),
                version=VERSION,
                source=HfSource(
                    DATASET,
                    TASK_REVISION,
                    (folder + "/tasks.parquet",),
                    SourceFormat.PARQUET,
                    read=partial(
                        read_facet_rows, folder=folder, index_reference=IMAGE_INDEX, images_reference=PUBLISHED_IMAGES
                    ),
                    decode=decode_facet,
                ),
                convert=convert_facet,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                oracle_solutions=True,
            ),
            pipeline=process_rows,
        )
        for folder, count in EXPECTED_TASKS.items()
    ]
