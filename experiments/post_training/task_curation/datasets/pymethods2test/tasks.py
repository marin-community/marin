# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native PyMethods2Test tasks bound to their sealed census pin and rule-C graders.

The folder's 4,990 raw validated rows are complete native Harbor packages sharing
three pristine Apptainer environment bundles. Conversion keeps the original
package bytes (the environment stays its declared Dockerfile build) and binds
the SHARED grader and ``RootDeliveryContract`` at conversion: the intake census
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

VERSION = "2026.11.25.1"
PREFIX = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.24.1"
DATASET = "FWeindel/validated-tasks"
TASK_REVISION = "317659ac88c8b706d318020c5fc3813a931fb6ec"
FOLDER = "pymethods2test_v1"
COUNT = 4990
CENSUS = ContentEvidenceReference(
    uri=PREFIX + "/pymethods2test-intake-census-v1/census.json",
    sha256="574937d678632ded04ebe284995771436f1682a0fc938b93bad43e5fc09ff781",
)

NOTES = (
    "The complete raw validated-task population at the repackager pin (intake census "
    f"{CENSUS.sha256[:16]}: 4,990 rows, every row stages 1,2,3,4 passed, 0 duplicates). Every row is a "
    "complete native Harbor package; conversion keeps the original package bytes and Dockerfile build "
    "and binds the SHARED grader and RootDeliveryContract at conversion (rule C 2026-11-24), so the "
    "population enters later stages with no contract-less row. The upstream license and lineage scan "
    "is HELD: the recorded CANDIDATES P1 basis (license PASS on the original upstream) does not resolve "
    "to a checkable receipt, and the revision-history scan awaits the user ruling on license-less early "
    "revisions. No publication or admission claim."
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


def read_pymethods_rows(path: StoragePath, _context: ConversionContext) -> Iterator[dict]:
    """Yield every raw validated task of the folder in upstream order, pin-verified.

    The staged Parquet must equal the sealed intake census's byte digest, and every
    row's content hash must agree with its ``ot.images.v1`` environment binding: one
    pristine bundle identity per task, carried through conversion as provenance.
    """
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
    return {**data, "pymethods_environment": row["pymethods_environment"]}


def convert_pymethods(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    """Keep the native package bytes and bind the SHARED grader and contract (rule C).

    The environment stays the package's declared Dockerfile build (no image is
    substituted). The grader restores the pinned private ``tests/`` files after the
    actor stops, runs the package's original ``test.sh``, and reads its own reward
    file; the contract records every delivered verifier file's mode and digest.
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
    if task.environment_requirements.docker_image is not None or task.environment_requirements.docker_build is None:
        raise ValueError("PyMethods2Test environment is its Dockerfile build, not an image pin")
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
    environment = task.environment_requirements
    return task.model_copy(
        update={
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
                    "Original native Harbor package bytes with the original Dockerfile build; "
                    "the SHARED grader and RootDeliveryContract bind at conversion (rule C)"
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
