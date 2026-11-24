# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build fixed checkpoint cohorts from the same pinned packages as curated evaluation."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from shellbox.machine import NetworkPolicy
from skyrlrecipe.declared import EvaluationProfile
from skyrlrecipe.sampling import EvalSamplingParams
from taskcompendium.importers.harbor.config import EnvironmentConfig, read_task_config
from taskcompendium.importers.harbor.package import JobSettings, lower_package
from taskcompendium.models import Source, grader_environment
from taskcompendium.pipeline.sources import row_locator
from zephyr.writers import write_parquet_file

from experiments.evaluation.benchmarks.agentic import BFCL_PARITY_PI, OT_TBLITE_3TRIAL, SWEBENCH_VERIFIED_MINI_SWE
from experiments.evaluation.benchmarks.harbor import HarborDataset, image_lock
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.harbor_packages import ARCHIVE_SUFFIX, extract_package
from experiments.post_training.task_curation.pipeline import HarborArchiveSource, download_step

CHECKPOINT_EVALUATION_REVISION = "2026.11.23.1"
CHECKPOINT_TASKS_FILENAME = "tasks.parquet"
CHECKPOINT_EVALUATION_TIMEOUT = 10_800
MINIMUM_COHORT_COVERAGE = 0.9
COHORT_SIZE = 30
CHECKPOINT_REPLY_TOKENS = 8192
CHECKPOINT_DAYTONA_TTL = 90
AGENT_SETUP_TIMEOUT = 360
AGENT_TIMEOUT = 600
CLEANUP_TIMEOUT = 30
CHECKPOINT_HARNESS = "terminus-2"
CHECKPOINT_HARNESS_OPTIONS = {"max_main_turns": 32, "setup_timeout": AGENT_SETUP_TIMEOUT}
CHECKPOINT_PROFILE = EvaluationProfile(
    n_samples_per_prompt=1,
    sampling_params=EvalSamplingParams(temperature=1.0, top_p=1.0, max_generate_length=CHECKPOINT_REPLY_TOKENS),
)
_COHORT_DATASETS = {
    "bfcl_parity": BFCL_PARITY_PI,
    "swebench_verified": SWEBENCH_VERIFIED_MINI_SWE,
    "openthoughts_tblite": OT_TBLITE_3TRIAL,
}


@dataclass(frozen=True)
class CheckpointCohort:
    name: str
    packages: tuple[str, ...]
    attempt_timeout: int
    verifier_timeout: int | None = None
    verifier_timeout_cap: int | None = None

    def __post_init__(self) -> None:
        if len(self.packages) != COHORT_SIZE or len(set(self.packages)) != COHORT_SIZE:
            raise ValueError(f"{self.name} must declare {COHORT_SIZE} distinct checkpoint tasks")

    @property
    def dataset(self) -> HarborDataset:
        return _COHORT_DATASETS[self.name]

    @property
    def source(self) -> HarborArchiveSource:
        source = self.dataset.source
        if not isinstance(source, HarborArchiveSource):
            raise ValueError(f"{self.name} needs a pinned archive source")
        return replace(source, packages=frozenset(self.packages))


def checkpoint_cohorts() -> tuple[CheckpointCohort, ...]:
    """Load the pinned members and the run policy for each cohort."""
    packages = yaml.safe_load(Path(__file__).with_name("rl_checkpoint_cohorts.yaml").read_text())
    return (
        CheckpointCohort("bfcl_parity", tuple(packages["bfcl_parity"]), 1800, verifier_timeout=300),
        CheckpointCohort(
            "swebench_verified",
            tuple(packages["swebench_verified"]),
            4500,
            verifier_timeout=3000,
        ),
        CheckpointCohort(
            "openthoughts_tblite",
            tuple(packages["openthoughts_tblite"]),
            5200,
            verifier_timeout_cap=3600,
        ),
    )


def _machine(environment: EnvironmentConfig, network: NetworkPolicy) -> MachineRuntimeSpec:
    return MachineRuntimeSpec(
        backend="daytona-eval",
        network=network,
        cpus=environment.cpus,
        memory_mb=environment.memory_mb,
        storage_mb=environment.storage_mb,
        gpus=0,
        user=None,
        startup_timeout=environment.build_timeout_sec,
        cleanup_timeout=CLEANUP_TIMEOUT,
    )


def _rollout_spec(
    cohort: CheckpointCohort,
    package: str,
    archive: StoragePath,
    provenance: Source,
    image_digests: dict[str, str],
) -> TaskRolloutSpec:
    with TemporaryDirectory(prefix="checkpoint-task-") as root:
        directory = Path(root)
        extract_package(archive.read_bytes(), directory)
        config = read_task_config(directory)
        lowered = lower_package(
            directory,
            task_id=f"{cohort.name}/{package}",
            source=provenance,
            rule=cohort.dataset.rule(directory),
            image_digests=image_digests,
            job=JobSettings(agent_timeout=AGENT_TIMEOUT, verifier_timeout=cohort.verifier_timeout),
        )
    task = lowered.task.model_copy(update={"tags": ("eval", "agentic")})
    timeout = lowered.episode.verifier_timeout
    assert timeout is not None
    if cohort.verifier_timeout_cap is not None:
        timeout = min(timeout, cohort.verifier_timeout_cap)
    verifier_machine = None
    if grader_environment(task.grader) is not None:
        verifier = config.verifier_environment
        verifier_machine = _machine(verifier, NetworkPolicy.ALLOW if verifier.allow_internet else NetworkPolicy.DENY)
    return TaskRolloutSpec(
        task=task,
        harness=CHECKPOINT_HARNESS,
        task_machine=_machine(config.environment, lowered.episode.network),
        verifier_machine=verifier_machine,
        limits=RolloutLimits(
            max_turns=None,
            max_total_tokens=None,
            model_turn_timeout=None,
            tool_turn_timeout=None,
            total_turn_timeout=AGENT_TIMEOUT,
            attempt_timeout=cohort.attempt_timeout,
            verifier_timeout=timeout,
            cleanup_timeout=CLEANUP_TIMEOUT,
        ),
    )


def write_checkpoint_tasks(
    *,
    cohort: CheckpointCohort,
    image_digests: dict[str, str],
    downloaded: str,
    output: str,
) -> None:
    """Convert every package selected for this cohort and publish SkyRL's rollout column."""
    source = cohort.source
    root = StoragePath(downloaded)

    def rows() -> Iterator[dict[str, str]]:
        for package in cohort.packages:
            path = f"tasks/{package}{ARCHIVE_SUFFIX}"
            provenance = Source(
                dataset=source.name,
                revision=source.revision,
                row=row_locator(path, 0),
                importer_revision=CHECKPOINT_EVALUATION_REVISION,
            )
            rollout = _rollout_spec(cohort, package, root / path, provenance, image_digests)
            yield {"task_rollout_spec": rollout.model_dump_json()}

    write_parquet_file(rows(), str(StoragePath(output) / CHECKPOINT_TASKS_FILENAME))


def checkpoint_tasks(cohort: CheckpointCohort) -> ArtifactStep[Artifact]:
    """Bind a cohort's package acquisition and native rollout publication."""
    return apply(
        f"data/rl/checkpoint-eval/{cohort.name}",
        write_checkpoint_tasks,
        version=CHECKPOINT_EVALUATION_REVISION,
        cohort=cohort,
        image_digests=image_lock(cohort.dataset.image_lock_name),
        downloaded=download_step(cohort.source, CampaignRuntime()),
        output=OUT,
    )
