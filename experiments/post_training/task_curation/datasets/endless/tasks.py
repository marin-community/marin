# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native Endless tasks bound to their pristine images and complete file census."""

import base64
import hashlib
import json
import tomllib
from collections.abc import Iterator
from decimal import Decimal
from functools import cache, partial

import pyarrow.parquet as pq
from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.status_taxonomy import DataRestriction
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.harbor.root_delivery import TEST_EXIT, TEST_OUTPUT, native_file_reward_grader
from taskcompendium.integrity import DeliveredVerifierFile, RootDeliveryContract, SharedDependencyState
from taskcompendium.models import ResourceGroups, TaskSpec, VerifierCommand
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.endless.images import (
    DATASET,
    EXPECTED_TASKS,
    IMAGE_REVISION,
    TASK_REVISION,
    image_files,
)
from experiments.post_training.task_curation.datasets.facet.artifacts import PreservedImage
from experiments.post_training.task_curation.datasets.facet.images import pinned_json
from experiments.post_training.task_curation.datasets.native_harbor import native_task, unpack_native
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

VERSION = "2026.11.20.1"
PREFIX = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/endless/2026.11.19.1"
COHORT_SOURCE_ID = "FWeindel:validated-tasks/endless"
COHORT_COUNT = 2069
COHORT_LOCATORS = ContentEvidenceReference(
    uri=PREFIX + "/inputs/endless-cohort-locators-v1.json",
    sha256="2ad3b5910a64af0b3d6fb62d0baebbf7810dc7b24616d76ce3542b85d65c6e47",
)
COHORT_LOCATOR_NOTE = (
    "The verified original-pair cohort's stable row locators, extracted from the sealed oracle-filter cohort "
    "(endless_v2: 1,948 + endless_v3: 121); the declared conversion selects exactly these rows."
)
COHORT_IMAGES = ContentEvidenceReference(
    uri=PREFIX + "/registry-v6/result.json",
    sha256="c694feec0d0f6acff758ebf5cfe55babd43be2a8e9ac52384c47cd8e760d4611",
)
COHORT_CENSUS = ContentEvidenceReference(
    uri=PREFIX + "/native-contract-census-v1/report.json",
    sha256="56a0e368e276697e5ed78a315f6239b508a7aa945c6c27ed8ddebb286a2f8347",
)


def cohort_source() -> list[RlDataSource[CurationRecipe]]:
    """The combined original-pair cohort declaration at its sealed image and census pins."""
    return cohort_sources(COHORT_IMAGES, COHORT_CENSUS)


IMAGE_INDEX = ContentEvidenceReference(
    uri=PREFIX + "/inputs/verified-sif-image-index-v1.json",
    sha256="81424898d51e7db35d16c93aaecef5e2a44ecb0287933fe034434f2fce5394f6",
)
PRIVATE_NAMES = frozenset({"test.sh", "test_final_state.py"})
RUBRIC = """Compare the public instruction with the unchanged native verifier and oracle.
Each pinned OCI image preserves its original SIF filesystem and runtime metadata.
Public setup_files are delivered unchanged; the original oracle invokes its own setup.
No additional setup command or source deadline is invented. Private verifier files are
restored after the original root actor stops, with agent-mutable dependencies declared.
"""


def read_endless_rows(
    path: StoragePath,
    _context: ConversionContext,
    *,
    folder: str,
    images_reference: ContentEvidenceReference,
    census_reference: ContentEvidenceReference,
) -> Iterator[dict]:
    """Join every native row to its sealed file census and exact published image."""
    index = pinned_json(IMAGE_INDEX.uri, IMAGE_INDEX.sha256)
    files = image_files(index)
    published = pinned_json(images_reference.uri, images_reference.sha256)
    census = pinned_json(census_reference.uri, census_reference.sha256)
    if published["index_sha256"] != IMAGE_INDEX.sha256 or published["task_counts"] != EXPECTED_TASKS:
        raise ValueError("Endless publication measures a different task/image index")
    if census["publication"] != images_reference.model_dump(mode="json"):
        raise ValueError("Endless census measures a different published image inventory")
    images = {}
    for row in published["images"]:
        image = PreservedImage.model_validate(row["preserved_image"])
        original = files[image.bundle_id]
        if image.bundle_id in images or (image.source_repo, image.source_revision) != (DATASET, IMAGE_REVISION):
            raise ValueError("Endless pristine image identity repeated or changed")
        if (image.source_filename, image.sif_sha256, image.sif_bytes) != (
            original["path"],
            original["sha256"],
            original["bytes"],
        ):
            raise ValueError("Endless image differs from its original SIF binding")
        if row["image"] != published["registry"] + "@" + image.manifest_digest:
            raise ValueError("Endless image differs from its preserved manifest digest")
        if row["visibility"] != "internal" or row["registry_manifest_verified"] is not True:
            raise ValueError("Endless image lacks its INTERNAL digest readback")
        images[image.bundle_id] = row
    if images.keys() != files.keys():
        raise ValueError("Endless publication omits pristine images")
    pin = census["folders"][folder]["census"]
    raw = StoragePath(pin["uri"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("Endless native census bytes changed")
    contracts = {}
    for line in raw.splitlines():
        contract = json.loads(line)
        if contract["folder"] != folder or contract["path"] in contracts:
            raise ValueError("Endless census repeats or changes a native task identity")
        contracts[contract["path"]] = contract
    if len(contracts) != EXPECTED_TASKS[folder]:
        raise ValueError("Endless native census is incomplete")
    seen = set()
    with path.open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        metadata = json.loads(parquet.schema_arrow.metadata[b"ot.images.v1"])
        if metadata["missing"] or metadata["tasks"] != index["tasks"][folder]:
            raise ValueError("Endless native Parquet image bindings changed")
        if parquet.metadata.num_rows != EXPECTED_TASKS[folder]:
            raise ValueError("Endless native population count changed")
        for batch in parquet.iter_batches(batch_size=1):
            for row in batch.to_pylist():
                name = row["path"]
                if name in seen or name not in contracts:
                    raise ValueError("Endless native task identity repeated or changed")
                contract = contracts[name]
                binding = metadata["tasks"][name]
                if row["content_sha256"] != binding["content_sha256"] or contract["image_binding"] != binding:
                    raise ValueError("Endless task differs from its original image binding")
                (environment,) = binding["environments"]
                if environment["path"] != "environment":
                    raise ValueError("Endless native environment layout changed")
                image = images[environment["bundle"]]
                if image["preserved_image"] != contract["preserved_image"] or image["image"] != contract["image"]:
                    raise ValueError("Endless task census and published image disagree")
                seen.add(name)
                yield {**row, "endless_image": image, "endless_contract": contract}
    if seen != contracts.keys():
        raise ValueError("Endless conversion did not visit the complete native census")


def decode_endless(row: dict, context: ConversionContext) -> dict:
    """Verify archive and regular-file bytes, modes and timestamps before conversion."""
    contract = row["endless_contract"]
    if hashlib.sha256(row["task_binary"]).hexdigest() != contract["archive_sha256"]:
        raise ValueError("Endless native task archive changed")
    data = unpack_native(row, context)
    if data["archive_links"] or data["files"].keys() != contract["files"].keys():
        raise ValueError("Endless native package file layout changed")
    for name, expected in contract["files"].items():
        raw = base64.b64decode(data["files"][name], validate=True)
        metadata = data["file_metadata"][name]
        if hashlib.sha256(raw).hexdigest() != expected["sha256"] or len(raw) != expected["bytes"]:
            raise ValueError(f"Endless native file changed: {name}")
        if metadata["mode"] != expected["mode"] or metadata["mtime_ns"] != int(
            Decimal(expected["mtime"]) * 1_000_000_000
        ):
            raise ValueError(f"Endless native file metadata changed: {name}")
    return {**data, "endless_image": row["endless_image"], "endless_contract": contract}


def convert_endless(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    """Keep the original root actor, oracle, verifier and pinned image runtime."""
    task = native_task(row, context)
    if isinstance(task, ImportRejection):
        return task
    resources = {item.path: item for item in task.resources.verifier}
    config = tomllib.loads(resource_bytes(resources["task.toml"]).decode())
    private = tuple(
        item.model_copy(update={"path": name.removeprefix("tests/")})
        for name, item in resources.items()
        if name.startswith("tests/")
    )
    runtime = row.data["endless_contract"]["image_runtime"]
    if {item.path for item in private} != PRIVATE_NAMES or config.get("agent", {}).get("user") is not None:
        raise ValueError("Endless native verifier layout or actor identity changed")
    if runtime["user"] not in {"", "root", "0"} or task.environment_requirements.setup_commands:
        raise ValueError("Endless native actor user or declared setup changed")
    workdir = task.environment_requirements.working_directory or runtime["working_directory"]
    if not workdir or not workdir.startswith("/"):
        raise ValueError("Endless image has no absolute native working directory")
    image = row.data["endless_image"]
    environment = task.environment_requirements.model_copy(
        update={
            "docker_image": image["image"],
            "docker_build": None,
            "working_directory": workdir,
            "capabilities": ("shell",),
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
        cwd=workdir,
        env=config.get("verifier", {}).get("env", {}),
    )
    return task.model_copy(
        update={
            "environment_requirements": environment,
            "verifier_execution_requirements": task.execution_requirements,
            "resources": ResourceGroups(
                all=task.resources.all,
                worker=tuple(item for name, item in resources.items() if name.startswith("setup_files/")),
                oracle=task.resources.oracle,
                verifier=tuple(
                    item.model_copy(update={"path": "source_archive/" + item.path}) for item in task.resources.verifier
                ),
            ),
            "grader": native_file_reward_grader(environment, private, execute, config["verifier"]["timeout_sec"]),
            "integrity_contract": RootDeliveryContract(
                dependency_state=SharedDependencyState.AGENT_MUTABLE,
                agent_user=None,
                undeclared_deadline=False,
                verifier_files=tuple(
                    DeliveredVerifierFile(
                        name=item.path, mode=int(item.mode, 8), sha256=hashlib.sha256(resource_bytes(item)).hexdigest()
                    )
                    for item in sorted(private, key=lambda item: item.path)
                ),
            ),
            "tags": ("root-delivery", "endless", "pristine-sif"),
            "labels": {
                **task.labels,
                "endless_delivery_version": VERSION,
                "endless_sif_sha256": image["preserved_image"]["sif_sha256"],
                "endless_image_provenance": image["preserved_image"]["provenance_uri"],
                "endless_image_provenance_sha256": image["preserved_image"]["provenance_sha256"],
            },
        }
    )


def read_cohort_rows(
    path: StoragePath,
    context: ConversionContext,
    *,
    images_reference: ContentEvidenceReference,
    census_reference: ContentEvidenceReference,
) -> Iterator[dict]:
    """Yield every native row of one folder in upstream order, annotated with its stable locator.

    The annotation carries the row's original ``file:index`` locator so the cohort select
    can drop the ten nop-only rows without renumbering the survivors: the declared
    conversion's task identities stay byte-identical to the sealed cohort's.
    """
    file = "/".join(str(path).rsplit("/", 2)[-2:])
    rows = read_endless_rows(
        path, context, folder=folder_of(file), images_reference=images_reference, census_reference=census_reference
    )
    for index, row in enumerate(rows):
        yield {**row, "endless_cohort_locator": f"{file}:{index}"}


def folder_of(file: str) -> str:
    folder = file.rsplit("/", 1)[0]
    if folder not in EXPECTED_TASKS:
        raise ValueError(f"Endless cohort file names another folder: {file}")
    return folder


def select_cohort_rows(row: dict, _context: ConversionContext, *, locators_reference: ContentEvidenceReference) -> bool:
    """Keep exactly the sealed original-pair cohort's rows, identified by their stable locators."""
    locator = row["endless_cohort_locator"]
    file, separator, index = locator.rpartition(":")
    if not separator or not index.isdigit():
        raise ValueError(f"Endless cohort row carries no stable locator: {locator}")
    locators = cohort_locators(locators_reference.uri, locators_reference.sha256)
    return int(index) in locators.get(file, frozenset())


@cache
def cohort_locators(uri: str, sha256: str) -> dict[str, frozenset[int]]:
    document = pinned_json(uri, sha256)
    locators = {folder: frozenset(indices) for folder, indices in document.items()}
    if {folder: len(indices) for folder, indices in locators.items()} != EXPECTED_COHORT_COUNTS:
        raise ValueError("Endless cohort locator set differs from the sealed original-pair accounting")
    return locators


EXPECTED_COHORT_COUNTS = {"endless_v2/tasks.parquet": 1948, "endless_v3/tasks.parquet": 121}


def cohort_sources(
    images: ContentEvidenceReference, census: ContentEvidenceReference
) -> list[RlDataSource[CurationRecipe]]:
    """The combined original-pair cohort at the upstream pin (ruling 2026-11-24, option B).

    One source over both kept folders whose population is exactly the sealed 2,069-task
    verified original-pair cohort: the read keeps every upstream row in order and the
    select drops the ten nop-only rows by their stable locators, so the declared
    conversion reproduces the cohort's task identities and bytes exactly. The per-folder
    license evidence stays bound per folder (the repackager ruling); the sealed release
    card carries the per-folder lineage table.
    """
    return [
        RlDataSource(
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.FWEINDEL,
                upstream_dataset="validated-tasks",
                upstream_name="EndlessTerminals native tasks",
                upstream_url=f"https://huggingface.co/datasets/{DATASET}/tree/{TASK_REVISION}",
                subset="endless",
                count=COHORT_COUNT,
                family="terminal",
                task_format=TaskFormat.TASKSPEC,
                build_path=(
                    "Combined original-pair cohort at the upstream pin; pristine per-folder image runtimes preserved"
                ),
                tags=("endless", "terminal", "pristine-sif", "cohort:original-pair"),
                notes=(
                    "The verified original-pair cohort (2,069 of the 2,079 census tasks; the ten nop-only v2 rows "
                    "never completed original oracle pairs). The declared conversion reproduces the sealed cohort's "
                    f"task identities and bytes exactly except one grader field: the shell quiesce script carries the "
                    "shellbox keeper-payload guard landed after the cohort was built (taskcompendium "
                    f"2026-11-24); {COHORT_LOCATOR_NOTE} "
                    "The admission serves the sealed release bytes, and the census, lexical and dedup receipts bind "
                    "the sealed cohort. Per-folder license evidence stays bound per folder (the repackager ruling). "
                    "Coordinator disposition 2026-11-24 15:35 EST: the sealed pre-guard quiesce is the "
                    "machine-keeper-killing variant on pre-PID-1 shellbox, so the sealed-bytes admission stands only "
                    "after the landed keeper PID 1 fix (atlas-env-audit dc44e1740b) is verified in-region on one "
                    "sealed Endless task grading on the LOCAL tier; until then the source does not mark Available. "
                    "If the fix had not been landed in time, the disposition was to re-release with the current "
                    "converter's keeper-safe quiesce (ids unchanged; SAMPLE re-run on the new task_specs_sha)."
                ),
            ),
            config=CurationRecipe(
                name="endless",
                version=VERSION,
                source=HfSource(
                    DATASET,
                    TASK_REVISION,
                    ("endless_v2/tasks.parquet", "endless_v3/tasks.parquet"),
                    SourceFormat.PARQUET,
                    read=partial(read_cohort_rows, images_reference=images, census_reference=census),
                    select=partial(select_cohort_rows, locators_reference=COHORT_LOCATORS),
                    decode=decode_endless,
                ),
                convert=convert_endless,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                oracle_solutions=True,
            ),
            pipeline=process_rows,
        )
    ]


def sources(images: ContentEvidenceReference, census: ContentEvidenceReference) -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.FWEINDEL,
                upstream_dataset="validated-tasks",
                upstream_name="EndlessTerminals native tasks",
                upstream_url=f"https://huggingface.co/datasets/{DATASET}/tree/{TASK_REVISION}/{folder}",
                subset=folder,
                count=count,
                family="terminal",
                task_format=TaskFormat.TASKSPEC,
                build_path="Original pristine SIF image, original root actor and private native verifier delivery",
            ),
            config=CurationRecipe(
                name="endless-" + folder.removeprefix("endless_"),
                version=VERSION,
                source=HfSource(
                    DATASET,
                    TASK_REVISION,
                    (folder + "/tasks.parquet",),
                    SourceFormat.PARQUET,
                    read=partial(read_endless_rows, folder=folder, images_reference=images, census_reference=census),
                    decode=decode_endless,
                ),
                convert=convert_endless,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                oracle_solutions=True,
            ),
            pipeline=process_rows,
        )
        for folder, count in EXPECTED_TASKS.items()
    ]
