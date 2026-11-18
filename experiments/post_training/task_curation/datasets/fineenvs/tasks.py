# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve FineEnvs native tasks while binding their public inputs to pinned images."""

import hashlib
import io
import shlex
import tarfile
import tomllib
from collections.abc import Iterator
from decimal import Decimal
from functools import partial
from pathlib import PurePosixPath

import pyarrow.parquet as pq
from finestore.rl.curation import ContentEvidenceReference
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.harbor.root_delivery import TEST_EXIT, TEST_OUTPUT, native_file_reward_grader
from taskcompendium.integrity import DeliveredVerifierFile, RootDeliveryContract, SharedDependencyState
from taskcompendium.models import (
    ResourceGroups,
    TaskSpec,
    VerifierCommand,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.facet.images import pinned_json
from experiments.post_training.task_curation.datasets.fineenvs.artifacts import GroupedInputImage
from experiments.post_training.task_curation.datasets.fineenvs.image_inputs import (
    INPUT_PREFIX,
    NativeInputFile,
    input_signature,
)
from experiments.post_training.task_curation.datasets.fineenvs.images import BASE_IMAGE
from experiments.post_training.task_curation.datasets.fineenvs.metadata import (
    DATASET,
    REVISION,
    TASK_COUNTS,
    source_info,
)
from experiments.post_training.task_curation.datasets.native_harbor import native_task, unpack_native
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource

VERSION = "2026.11.17.1"
PREFIX = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/fweindel-fineenvs/2026.11.17.1"
INPUT_SUMMARY = ContentEvidenceReference(
    uri=PREFIX + "/native-input-summary-v1/result.json",
    sha256="b9a03a4fe7a3728dbf9c76e04e9735e6f87d4e02b6b9b9e949a7340d9077346d",
)
SETUP_PROTOCOL_SHA = "14dd18868c4c0385a28808b990552ae586d0d252eb73534b0aac175cab5f5034"
CONTRACT_SHA = "e3179104ebff6933a2054c9ccdb692cfdc4d7626e186bbfaf06ee21881e0ad29"
NATIVE_SETUP_REVISION = "c313b7b18f5245619a13da12e6943d703ea5b0fb"
NATIVE_VERIFIER_SETUP_SHA = "97be873552fe710949ad3fe1825a9d083a78523bc69d1b1dd036604a7271baf9"
NATIVE_SETUP_REVIEW_SHA = "051a80e170b933230ad4fd0e1459f44bd9901e2b399e388a9959c2fc73358517"
SETUP_OUTPUT = "/logs/verifier/setup-output.txt"
WORKDIR = "/workdir"
RUBRIC = """Compare the public instruction with the unchanged native verifier and oracle.
Public setup_files/setup.sh runs before the actor. Private tests/setup.sh runs immediately
before tests/test.sh inside the verifier deadline, as the frozen upstream harness specifies.
The root actor and mutable dependencies remain shared. Grouped images carry other tasks'
public input sets outside the workdir; only this task's own set is linked at /setup_files/inputs.
Prompts, source scripts, answers and original resource allocations are unchanged.
"""


def read_fineenvs_rows(
    path: StoragePath, _context: ConversionContext, *, folder: str, images_reference: ContentEvidenceReference
) -> Iterator[dict]:
    """Bind every pinned native row to its complete file census and published input image."""
    summary = pinned_json(INPUT_SUMMARY.uri, INPUT_SUMMARY.sha256)
    pinned_json(PREFIX + "/inputs/native-setup-protocol-v1.json", SETUP_PROTOCOL_SHA)
    published = pinned_json(images_reference.uri, images_reference.sha256)
    if (
        not 1 <= published["image_count"] <= 10
        or published["input_set_count"] != 419
        or published["task_counts"] != TASK_COUNTS
    ):
        raise ValueError("FineEnvs image publication does not cover the frozen native population")
    images = {}
    groups = set()
    for row in published["images"]:
        record = GroupedInputImage.model_validate(row["grouped_input_image"])
        if record.group in groups or record.data_restriction is not source_info(folder).data_restriction:
            raise ValueError("FineEnvs publication repeats a group or changes its source restriction")
        groups.add(record.group)
        if record.base_image != BASE_IMAGE or row["image"] != published["registry"] + "@" + record.manifest_digest:
            raise ValueError("FineEnvs published image differs from its preserved native image")
        if not row["registry_manifest_verified"] or row["visibility"] not in {"private", "internal"}:
            raise ValueError("FineEnvs image has no nonpublic digest readback")
        for signature in record.input_signatures:
            if signature in images:
                raise ValueError("FineEnvs input set is assigned to multiple images")
            images[signature] = row
    if groups != set(range(published["image_count"])) or images.keys() != summary["input_groups"].keys():
        raise ValueError("FineEnvs published images differ from the complete input-set census")
    contracts = {row["task"]: row for row in summary["task_contracts"] if row["folder"] == folder}
    if len(contracts) != TASK_COUNTS[folder]:
        raise ValueError("FineEnvs native task census repeats or omits an identity")
    seen = set()
    with path.open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        if parquet.metadata.num_rows != TASK_COUNTS[folder]:
            raise ValueError("FineEnvs native Parquet population count changed")
        for batch in parquet.iter_batches(batch_size=1):
            for row in batch.to_pylist():
                name = row["path"]
                if name in seen or name not in contracts:
                    raise ValueError("FineEnvs native Parquet task identity changed")
                seen.add(name)
                contract = contracts[name]
                receipt = pinned_json(contract["receipt"]["uri"], contract["receipt"]["sha256"])
                if (receipt["folder"], receipt["path"], receipt["content_sha256"]) != (
                    folder,
                    name,
                    row["content_sha256"],
                ):
                    raise ValueError("FineEnvs native row differs from its frozen file census")
                yield {
                    **row,
                    "fineenvs_contract": contract,
                    "fineenvs_receipt": receipt,
                    "fineenvs_image": images[contract["input_signature"]],
                }
    if seen != contracts.keys():
        raise ValueError("FineEnvs conversion did not visit the complete native task census")


def decode_fineenvs(row: dict, context: ConversionContext) -> dict:
    """Verify every native file, retaining large public inputs only in the bound image."""
    original = row["task_binary"]
    receipt = row["fineenvs_receipt"]
    if hashlib.sha256(original).hexdigest() != receipt["archive_sha256"]:
        raise ValueError("FineEnvs native archive differs from its frozen checksum")
    files = {file["path"]: file for file in receipt["files"]}
    inputs = tuple(
        NativeInputFile.model_validate(file)
        for file in files.values()
        if PurePosixPath(file["path"]).is_relative_to(INPUT_PREFIX)
    )
    signature = input_signature(inputs)
    if signature != row["fineenvs_contract"]["input_signature"]:
        raise ValueError("FineEnvs native inputs differ from the selected image input set")
    image = GroupedInputImage.model_validate(row["fineenvs_image"]["grouped_input_image"])
    if signature not in image.input_signatures:
        raise ValueError("FineEnvs native input set is absent from its selected image")
    filtered = io.BytesIO()
    seen = set()
    with (
        tarfile.open(fileobj=io.BytesIO(original), mode="r|*") as source,
        tarfile.open(fileobj=filtered, mode="w", format=tarfile.PAX_FORMAT) as target,
    ):
        for member in source:
            name = member.name.removeprefix("./")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe FineEnvs native archive path: {name}")
            if member.isdir():
                target.addfile(member)
                continue
            if name in seen or name not in files or not member.isfile():
                raise ValueError(f"FineEnvs native archive member differs from its census: {name}")
            seen.add(name)
            expected = files[name]
            mtime = Decimal(str(member.pax_headers.get("mtime", member.mtime)))
            if (member.size, member.mode, mtime) != (expected["bytes"], expected["mode"], Decimal(expected["mtime"])):
                raise ValueError(f"FineEnvs native file metadata changed: {name}")
            data = source.extractfile(member)
            if path.is_relative_to(INPUT_PREFIX):
                digest = hashlib.file_digest(data, "sha256").hexdigest()
            else:
                content = data.read()
                digest = hashlib.sha256(content).hexdigest()
                target.addfile(member, io.BytesIO(content))
            if digest != expected["sha256"]:
                raise ValueError(f"FineEnvs native file checksum changed: {name}")
    if seen != files.keys():
        raise ValueError("FineEnvs native archive omits a censused file")
    data = unpack_native({**row, "task_binary": filtered.getvalue()}, context)
    return {
        **data,
        "archive_sha256": receipt["archive_sha256"],
        "fineenvs_filtered_archive_sha256": hashlib.sha256(filtered.getvalue()).hexdigest(),
        "fineenvs_input_signature": signature,
        "fineenvs_image": row["fineenvs_image"],
    }


def convert_fineenvs(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    """Keep the native root actor, setup order, oracle and private shared verifier."""
    task = native_task(row, context)
    if isinstance(task, ImportRejection):
        return task
    resources = {resource.path: resource for resource in task.resources.verifier}
    config = tomllib.loads(resource_bytes(resources["task.toml"]).decode())
    if task.environment_requirements.setup_commands or config.get("agent", {}).get("user") is not None:
        raise ValueError("FineEnvs native task declares another setup or actor identity")
    if task.environment_requirements.working_directory not in {None, WORKDIR}:
        raise ValueError("FineEnvs task overrides the native image working directory")
    if "setup_files/setup.sh" not in resources or "tests/test.sh" not in resources:
        raise ValueError("FineEnvs task has no original public setup or private verifier")
    private = tuple(
        resource.model_copy(update={"path": name.removeprefix("tests/")})
        for name, resource in resources.items()
        if name.startswith("tests/")
    )
    signature = row.data["fineenvs_input_signature"]
    inert = "/opt/fineenvs-inputs/" + signature
    setup = (
        "set -eu\ntest -d "
        + shlex.quote(inert)
        + "\ntest ! -e /setup_files/inputs\ntest ! -L /setup_files/inputs\nln -s "
        + shlex.quote(inert)
        + " /setup_files/inputs\nbash /setup_files/setup.sh"
    )
    environment = task.environment_requirements.model_copy(
        update={
            "docker_image": row.data["fineenvs_image"]["image"],
            "docker_build": None,
            "working_directory": WORKDIR,
            "capabilities": ("shell",),
            "setup_commands": (setup,),
        }
    )
    execute = VerifierCommand(
        argv=(
            "bash",
            "-c",
            "set -eu\nmkdir -p /logs/verifier\n: > "
            + TEST_OUTPUT
            + "\nrm -f /logs/verifier/reward.txt /logs/verifier/reward.json\n"
            "status=0\nif [ -f /tests/setup.sh ]; then bash /tests/setup.sh > "
            + SETUP_OUTPUT
            + " 2>&1 || status=$?; fi\n"
            'if [ "$status" -ne 0 ]; then\n'
            "  if test ! -e /logs/verifier/reward.json && test -f /logs/verifier/reward.txt && "
            'test "$(cat /logs/verifier/reward.txt)" = 0; then\n'
            "    printf '%s\\n' \"$status\" > " + TEST_EXIT + "; exit 0\n"
            '  fi\n  exit "$status"\nfi\n'
            "status=0\nbash /tests/test.sh > " + TEST_OUTPUT + " 2>&1 || status=$?\n"
            "printf '%s\\n' \"$status\" > " + TEST_EXIT + "\nexit 0",
        ),
        cwd=environment.working_directory,
        env=config.get("verifier", {}).get("env", {}),
    )
    archived = tuple(
        resource.model_copy(update={"path": "source_archive/" + resource.path}) for resource in task.resources.verifier
    )
    public = tuple(resource for name, resource in resources.items() if name.startswith("setup_files/"))
    image = row.data["fineenvs_image"]
    return task.model_copy(
        update={
            "environment_requirements": environment,
            "verifier_execution_requirements": task.execution_requirements,
            "resources": ResourceGroups(
                all=task.resources.all, worker=public, oracle=task.resources.oracle, verifier=archived
            ),
            "grader": native_file_reward_grader(environment, private, execute, config["verifier"]["timeout_sec"]),
            "integrity_contract": RootDeliveryContract(
                dependency_state=SharedDependencyState.AGENT_MUTABLE,
                agent_user=None,
                undeclared_deadline=False,
                verifier_files=tuple(
                    DeliveredVerifierFile(
                        name=resource.path,
                        mode=int(resource.mode, 8),
                        sha256=hashlib.sha256(resource_bytes(resource)).hexdigest(),
                    )
                    for resource in sorted(private, key=lambda item: item.path)
                ),
            ),
            "tags": ("root-delivery", "fineenvs", "grouped-public-inputs"),
            "labels": {
                **task.labels,
                "fineenvs_delivery_version": VERSION,
                "fineenvs_input_signature": signature,
                "fineenvs_image": image["image"],
                "fineenvs_image_provenance": image["provenance"]["uri"],
                "fineenvs_image_provenance_sha256": image["provenance"]["sha256"],
                "fineenvs_filtered_archive_sha256": row.data["fineenvs_filtered_archive_sha256"],
                "fineenvs_setup_protocol_uri": PREFIX + "/inputs/native-setup-protocol-v1.json",
                "fineenvs_setup_protocol_sha256": SETUP_PROTOCOL_SHA,
                "fineenvs_native_contract_sha256": CONTRACT_SHA,
                "fineenvs_native_setup_revision": NATIVE_SETUP_REVISION,
                "fineenvs_native_verifier_setup_sha256": NATIVE_VERIFIER_SETUP_SHA,
                "fineenvs_native_setup_review_sha256": NATIVE_SETUP_REVIEW_SHA,
            },
        }
    )


def sources(images_reference: ContentEvidenceReference) -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=source_info(folder),
            config=CurationRecipe(
                name="fineenvs-" + folder.removeprefix("fineenvs_"),
                version=VERSION,
                source=HfSource(
                    DATASET,
                    REVISION,
                    (folder + "/tasks.parquet",),
                    SourceFormat.PARQUET,
                    read=partial(read_fineenvs_rows, folder=folder, images_reference=images_reference),
                    decode=decode_fineenvs,
                ),
                convert=convert_fineenvs,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                oracle_solutions=True,
            ),
            pipeline=process_rows,
        )
        for folder in TASK_COUNTS
    ]
