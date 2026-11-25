# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native PyMethods2Test tasks bound to their sealed census pin, prepared images and rule-C graders.

The folder's 4,990 raw validated rows are complete native Harbor packages sharing
three pristine Apptainer environment bundles. Conversion keeps the original package
bytes, binds each task's environment to its class's digest-pinned prepared image (the
3 declared Dockerfile classes built once in region; coordinator GO 2026-11-25), and
binds the SHARED grader and ``RootDeliveryContract`` at conversion: the intake census
reports no contract-less rows, so later stages consume byte-exact task digests.
"""

import hashlib
import json
import tomllib
from collections.abc import Iterator

import pyarrow.parquet as pq
from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.status_taxonomy import DataRestriction
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.harbor.root_delivery import TEST_EXIT, TEST_OUTPUT, native_file_reward_grader
from taskcompendium.integrity import DeliveredVerifierFile, RootDeliveryContract, SharedDependencyState
from taskcompendium.models import ResourceGroups, TaskSpec, VerifierCommand
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.facet.images import pinned_json
from experiments.post_training.task_curation.datasets.native_harbor import native_task, unpack_native
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

VERSION = "2026.11.25.2"
PREFIX = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.24.1"
IMAGES_PREFIX = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.25.1"
DATASET = "FWeindel/validated-tasks"
TASK_REVISION = "317659ac88c8b706d318020c5fc3813a931fb6ec"
FOLDER = "pymethods2test_v1"
COUNT = 4990
CENSUS = ContentEvidenceReference(
    uri=PREFIX + "/pymethods2test-intake-census-v1/census.json",
    sha256="574937d678632ded04ebe284995771436f1682a0fc938b93bad43e5fc09ff781",
)
IMAGES = ContentEvidenceReference(
    uri=IMAGES_PREFIX + "/pymethods2test-prepared-images-v1/manifest-v2.json",
    sha256="729065e3c0ee32b3def3756ce35a72fb4bfe035991eb977c456b34cc3a8def8b",
)

NOTES = (
    "The complete raw validated-task population at the repackager pin (intake census "
    f"{CENSUS.sha256[:16]}: 4,990 rows, every row stages 1,2,3,4 passed, 0 duplicates). Every row is a "
    "complete native Harbor package; conversion keeps the original package bytes, binds the SHARED grader "
    "and RootDeliveryContract at conversion (rule C 2026-11-24), and binds each task's environment to its "
    "class's digest-pinned prepared image (coordinator GO 2026-11-25: the 3 declared Dockerfile classes "
    "built once in region through the taskforge image builder and pinned by manifest digest; the census "
    f"marked every task unbuilt_image until this pin; images manifest {IMAGES.sha256[:16]}). The license "
    "scan PASSED under the coordinator's 2026-11-25 ruling (open-thoughts/TaskTrove @94688470 declares "
    "apache-2.0 at the pin and all revisions; receipt bound). No publication or admission claim."
)

RUBRIC = """
For partitioning and scheduling problems, check whether contiguity, order, indivisibility, and coverage
restrictions are explicitly supplied. Construct a better valid solution under the public rules before
accepting a narrower hidden optimum.

Check tests against the stated input domain, including zero values and allowed worker counts. Reject a
contradiction in expected behavior; distinguish explicitly described edge cases from a merely abbreviated
constraints list.

Check method signatures, class context, return values, and exceptions against the hidden tests.

Check parameter meanings and essential transition rules against the tests and oracle; an algorithmic
constraint missing from the public request is a defect even when the oracle passes.

Do not equate a quantity, rate, time limit, and lower bound; identify concrete parameter mismatches.

Flag undefined behavior, missing fixtures, unavailable dependencies, and contradictory examples.

Hidden tests and oracle solutions are review evidence and must not be shown to the solving actor.

A passing oracle shows test compatibility, not specification coverage; cite a concrete defect when rejecting.
"""


def prepared_images() -> dict[str, dict]:
    """The sealed class→pinned-image manifest, keyed by the task's environment digest."""
    manifest = pinned_json(IMAGES.uri, IMAGES.sha256)
    classes = manifest["image_classes"]
    if manifest.get("task_coverage") != COUNT or sorted(classes) != sorted({e for e in classes}):
        raise ValueError("PyMethods2Test prepared-image manifest covers a different population")
    images = {}
    for environment_sha256, record in classes.items():
        image = record["image"]
        if not image.startswith("envreg.") or "@sha256:" not in image:
            raise ValueError(f"Prepared image is not a digest-pinned task-registry reference: {image}")
        if record["readback"]["digest_readback"] is not True or record["readback"]["architecture"] != "amd64":
            raise ValueError(f"Prepared image lacks its digest readback: {image}")
        images[environment_sha256] = {
            "image": image,
            "dockerfile_sha256": record["dockerfile_sha256"],
            "tasks": record["tasks"],
        }
    if sum(item["tasks"] for item in images.values()) != COUNT:
        raise ValueError("PyMethods2Test prepared-image manifest covers a different task count")
    return images


def read_pymethods_rows(path: StoragePath, _context: ConversionContext) -> Iterator[dict]:
    """Yield every raw validated task of the folder in upstream order, pin-verified.

    The staged Parquet must equal the sealed intake census's byte digest, and every
    row's content hash must agree with its ``ot.images.v1`` environment binding: one
    pristine bundle identity per task, resolved to its digest-pinned prepared image.
    """
    images = prepared_images()
    census = pinned_json(CENSUS.uri, CENSUS.sha256)
    if (census["source_id"], census["folder"], census["tasks"], census["distinct_paths"]) != (
        f"{DATASET.replace('/', ':', 1)}/{FOLDER}",
        FOLDER,
        COUNT,
        COUNT,
    ):
        raise ValueError("PyMethods2Test census measures a different population")
    if file_sha256(path) != census["tasks_parquet_sha256"]:
        raise ValueError("PyMethods2Test population bytes differ from the sealed census")
    seen = set()
    with path.open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        metadata = json.loads(parquet.schema_arrow.metadata[b"ot.images.v1"])
        if metadata["repo_id"] != DATASET or metadata["missing"]:
            raise ValueError("PyMethods2Test native metadata names another repository or missing rows")
        if len(metadata["tasks"]) != COUNT:
            raise ValueError("PyMethods2Test native metadata covers a different population")
        for batch in parquet.iter_batches(batch_size=1):
            for row in batch.to_pylist():
                name = row["path"]
                if name in seen:
                    raise ValueError("PyMethods2Test native task identity repeated")
                binding = metadata["tasks"].get(name)
                if binding is None or row["content_sha256"] != binding["content_sha256"]:
                    raise ValueError("PyMethods2Test task differs from its native metadata binding")
                (environment,) = binding["environments"]
                if environment["path"] != "environment":
                    raise ValueError("PyMethods2Test native environment layout changed")
                bundle = metadata["bundles"].get(environment["bundle"])
                if bundle is None or bundle["provenance"]["environment_sha256"] != environment["environment_sha256"]:
                    raise ValueError("PyMethods2Test environment binding lacks its pristine bundle")
                provenance = bundle["provenance"]
                if not provenance["pristine"] or provenance["format"] != "apptainer-sif":
                    raise ValueError("PyMethods2Test environment bundle is not a pristine SIF")
                seen.add(name)
                yield {
                    **row,
                    "pymethods_environment": {
                        "bundle": environment["bundle"],
                        "environment_sha256": environment["environment_sha256"],
                        "runtime": provenance["runtime"],
                        "bundle_prefix": bundle["prefix"],
                    },
                    "pymethods_image": images[environment["environment_sha256"]],
                }
    if len(seen) != COUNT or set(metadata["tasks"]) != seen:
        raise ValueError("PyMethods2Test conversion did not visit the complete native population")


def decode_pymethods(row: dict, context: ConversionContext) -> dict:
    """Verify the native archive unpacks to complete per-file metadata before conversion."""
    data = unpack_native(row, context)
    if data["archive_links"]:
        raise ValueError("PyMethods2Test native archive carries links")
    if set(data["file_metadata"]) != set(data["files"]):
        raise ValueError("PyMethods2Test native file metadata is incomplete")
    return {
        **data,
        "pymethods_environment": row["pymethods_environment"],
        "pymethods_image": row["pymethods_image"],
    }


def convert_pymethods(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    """Keep the native package bytes and bind the pinned image, SHARED grader and contract.

    The environment is the task's own declared Dockerfile build, executed once and pinned
    by manifest digest (coordinator GO 2026-11-25: the prepared-image route) — the task's
    docker_image is its class's digest-pinned build, docker_build is dropped, and every
    other byte stays unchanged. The grader restores the pinned private ``tests/`` files
    after the actor stops, runs the package's original ``test.sh``, and reads its own
    reward file; the contract records every delivered verifier file's mode and digest.
    """
    task = native_task(row, context)
    if isinstance(task, ImportRejection):
        return task
    resources = {item.path: item for item in task.resources.verifier}
    config = tomllib.loads(resource_bytes(resources["task.toml"]).decode())
    verifier = config.get("verifier", {})
    if verifier.get("environment_mode") != "shared" or "timeout_sec" not in verifier:
        raise ValueError("PyMethods2Test native package does not declare its shared verifier deadline")
    if config.get("agent", {}).get("user") is not None or config.get("agent", {}).get("timeout_sec") is None:
        raise ValueError("PyMethods2Test native actor identity or deadline declaration changed")
    image = row.data["pymethods_image"]
    environment = task.environment_requirements.model_copy(update={"docker_image": image["image"], "docker_build": None})
    private = tuple(
        item.model_copy(update={"path": name.removeprefix("tests/")})
        for name, item in resources.items()
        if name.startswith("tests/")
    )
    if not private or not any(item.path == "test.sh" for item in private):
        raise ValueError("PyMethods2Test native verifier layout changed")
    execute = VerifierCommand(
        argv=(
            "bash",
            "-c",
            "set -eu\nmkdir -p /logs/verifier\n"
            "status=0\nbash /tests/test.sh > " + TEST_OUTPUT + " 2>&1 || status=$?\n"
            "printf '%s\\n' \"$status\" > " + TEST_EXIT + "\nexit 0",
        ),
        env=verifier.get("env", {}),
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
            "grader": native_file_reward_grader(environment, private, execute, verifier["timeout_sec"]),
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
            "tags": ("root-delivery", "pymethods2test"),
            "labels": {
                **task.labels,
                "pymethods2test_delivery_version": VERSION,
                "pymethods2test_environment_sha256": row.data["pymethods_environment"]["environment_sha256"],
                "pymethods2test_environment_bundle": row.data["pymethods_environment"]["bundle"],
                "pymethods2test_environment_runtime": row.data["pymethods_environment"]["runtime"],
                "pymethods2test_prepared_image": image["image"],
                "pymethods2test_images_manifest_sha256": IMAGES.sha256,
            },
        }
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    """The raw validated PyMethods2Test population at its sealed census pin."""
    return [
        RlDataSource(
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.FWEINDEL,
                upstream_dataset="validated-tasks",
                upstream_name="PyMethods2Test native tasks",
                upstream_url=f"https://huggingface.co/datasets/{DATASET}/tree/{TASK_REVISION}/{FOLDER}",
                subset=FOLDER,
                count=COUNT,
                family="unit-test-gen",
                task_format=TaskFormat.TASKSPEC,
                build_path=(
                    "Original native Harbor package bytes with each environment bound to its class's "
                    "digest-pinned prepared image; the SHARED grader and RootDeliveryContract bind at "
                    "conversion (rule C)"
                ),
                tags=("pymethods2test", "language:python", "root-delivery"),
                notes=NOTES,
            ),
            config=CurationRecipe(
                name="pymethods2test",
                version=VERSION,
                source=HfSource(
                    DATASET,
                    TASK_REVISION,
                    (f"{FOLDER}/tasks.parquet",),
                    SourceFormat.PARQUET,
                    read=read_pymethods_rows,
                    decode=decode_pymethods,
                ),
                convert=convert_pymethods,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                oracle_solutions=True,
            ),
            pipeline=process_rows,
        )
    ]
