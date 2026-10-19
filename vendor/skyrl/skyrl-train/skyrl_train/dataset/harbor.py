from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from itertools import batched
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from loguru import logger
from transformers import PreTrainedTokenizerBase
from rolloutengine.lowering import SHELLBOX_SESSION
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.machine import Backend, NetworkPolicy
from taskcompendium.models import NoGrader, ScriptGrader, Source, TaskSpec, VerifyitGrader

from marinskyrl.packed_tasks import PackedTaskMaterializer, PackedTaskReference, select_task_references
from marinskyrl.task_sources import DirectoryDataSource, TaskTroveParquetSource, data_source
from skyrl_train.dataset.tasks import LOWERED_TASK_COLUMN, TaskDataset, cache_tasks
from skyrl_train.dataset.harbor_import import (
    HARBOR_TRIAL_SESSION,
    EnvironmentConfig,
    TaskConfig,
    harbor_task,
    harbor_trial_task,
)

MATERIALIZATION_BATCH_SIZE = 64
HARBOR_ID_PREFIX = "harbor-id:"


class TerminalBenchTaskDataset:
    """Terminal-bench directory tasks and lazily materialized packed tasks."""

    def __init__(self, data_files: Sequence[str | Mapping[str, Any]]):
        self.data_files = data_files
        self._items = self._load_data_files()
        logger.info(f"TerminalBenchTaskDataset initialized with {len(self._items)} tasks")

    def _directory_tasks(self, source_path: Path) -> list[Path]:
        if not source_path.exists():
            logger.warning(f"Path does not exist: {source_path}")
            return []
        if not source_path.is_dir():
            logger.warning(f"File {source_path} cannot be a valid task directory (missing instruction.md)")
            return []

        all_dirs = [path for path in source_path.iterdir() if path.is_dir()]
        valid = [path for path in all_dirs if self._is_valid_task_directory(path)]
        if valid:
            logger.info(f"Found {len(valid)} valid task directories out of {len(all_dirs)} total directories")
            return valid
        if self._is_valid_task_directory(source_path):
            return [source_path]
        logger.warning(f"No valid task directories found in {source_path}")
        return []

    def _packed_tasks(self, source: TaskTroveParquetSource) -> list[PackedTaskReference]:
        summary = select_task_references(source, dataset_path=source.resolved_path())
        if source.snapshot is None:
            return list(summary.references)
        if source.snapshot.count != len(summary.references):
            raise ValueError(
                f"TaskTrove selection count changed: launch selected {source.snapshot.count}, "
                f"Ray selected {len(summary.references)}"
            )
        if source.snapshot.digest != summary.digest:
            raise ValueError("TaskTrove selection digest changed between launch and Ray")
        if source.snapshot.distinct_environment_count != summary.distinct_environment_count:
            raise ValueError("TaskTrove environment count changed between launch and Ray")
        return list(summary.references)

    def _load_data_files(self) -> list[Path | PackedTaskReference]:
        items: list[Path | PackedTaskReference] = []
        for value in self.data_files:
            if isinstance(value, str):
                items.extend(self._directory_tasks(Path(value)))
                continue
            source = data_source(value)
            if isinstance(source, DirectoryDataSource):
                items.extend(self._directory_tasks(Path(source.resolved_path())))
            else:
                items.extend(self._packed_tasks(source))
        if all(isinstance(item, Path) for item in items):
            return sorted(items)
        return items

    def _is_valid_task_directory(self, task_path: Path) -> bool:
        if not task_path.is_dir():
            return False
        return (task_path / "instruction.md").is_file() or (
            (task_path / "task.toml").is_file() and (task_path / "steps").is_dir()
        )

    def __getitem__(self, index: int) -> Path | PackedTaskReference:
        if index >= len(self._items):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._items)}")
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def harbor_task_ids(task_path: Path) -> set[str]:
    """Return Harbor's directory ID and any source-dataset ID in task metadata."""
    task_ids = {task_path.name}

    swe_gym_config = task_path / "tests" / "config.json"
    if swe_gym_config.is_file():
        config = json.loads(swe_gym_config.read_text())
        instance_id = config.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            task_ids.add(instance_id)

    r2e_info = task_path / "tests" / "test_info.json"
    if r2e_info.is_file():
        info = json.loads(r2e_info.read_text())
        github_repo = info.get("github_repo")
        base_commit = info.get("base_commit")
        if isinstance(github_repo, str) and github_repo and isinstance(base_commit, str) and base_commit:
            task_ids.add(f"{github_repo.replace('/', '__')}-{base_commit}")

    return {task_id.casefold() for task_id in task_ids}


def harbor_machine(environment: EnvironmentConfig, user: str | int | None) -> MachineRuntimeSpec:
    return MachineRuntimeSpec(
        backend=Backend.DOCKER,
        network=NetworkPolicy.ALLOW if environment.allow_internet else NetworkPolicy.DENY,
        cpus=environment.cpus,
        memory_mb=environment.memory_mb,
        storage_mb=environment.storage_mb,
        gpus=0 if environment.gpus is None else environment.gpus,
        user=None if user is None else str(user),
        startup_timeout=environment.build_timeout_sec,
        cleanup_timeout=None,
    )


def lower_harbor_task(directory: Path, task: TaskSpec, *, session: TaskSessionSpec) -> LoweredTaskSpec:
    """Use package machine settings and phase deadlines, with other limits from the caller."""
    config = TaskConfig.model_validate_toml((directory / "task.toml").read_text())

    if session.task_session == HARBOR_TRIAL_SESSION:
        return LoweredTaskSpec(
            task=task,
            runtime=TaskRuntimeSpec(task_machine=None, verifier_machine=None),
            session=session,
        )

    task_machine = harbor_machine(config.environment, config.agent.user)
    verifier_machine = (
        harbor_machine(config.verifier.environment or config.environment, config.verifier.user)
        if isinstance(task.grader, ScriptGrader)
        or (isinstance(task.grader, VerifyitGrader) and task.grader.environment is not None)
        else None
    )
    return LoweredTaskSpec(
        task=task,
        runtime=TaskRuntimeSpec(task_machine=task_machine, verifier_machine=verifier_machine),
        session=TaskSessionSpec.model_validate(
            {
                **session.model_dump(),
                "task_session": SHELLBOX_SESSION,
                "total_turn_timeout": config.agent.timeout_sec,
                "verifier_timeout": config.verifier.timeout_sec,
            }
        ),
    )


def materialize_harbor_tasks(
    data_files: Sequence[str | Mapping[str, Any]],
    *,
    cache_dir: Path,
    session: TaskSessionSpec,
    grader_override: NoGrader | None = None,
) -> Path:
    """Combine Harbor packages and pre-lowered task Parquet for one worker pool."""
    task_files = [source for source in data_files if isinstance(source, str) and source.endswith(".parquet")]
    sources = TerminalBenchTaskDataset([source for source in data_files if source not in task_files])
    if not len(sources) and not task_files:
        raise ValueError("No Harbor tasks matched the configured sources")

    def tasks() -> Iterator[LoweredTaskSpec]:
        with closing(PackedTaskMaterializer(cache_dir / "archives")) as materializer:
            for batch in batched(sources, MATERIALIZATION_BATCH_SIZE):
                references = [item for item in batch if isinstance(item, PackedTaskReference)]
                directories = materializer.materialize_batch(references)
                for item in batch:
                    if isinstance(item, Path):
                        directory = item
                        data_source = str(item)
                        uid = item.name
                    else:
                        directory = directories[item]
                        data_source = item.stable_uri()
                        uid = item.uid()
                    importer = harbor_trial_task if session.task_session == HARBOR_TRIAL_SESSION else harbor_task
                    task = importer(
                        directory,
                        source=Source(
                            dataset=data_source,
                            revision="unhashed",
                            row=uid,
                            importer_revision="skyrl-harbor-v1",
                        ),
                    )
                    if grader_override is not None:
                        if session.task_session == HARBOR_TRIAL_SESSION:
                            raise ValueError(
                                "Full Harbor trials configure verification through Harbor, not grader overrides"
                            )
                        task = task.model_copy(
                            update={
                                "grader": grader_override,
                                "resources": task.resources.model_copy(update={"verifier": ()}),
                            }
                        )
                    content = json.dumps(task.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
                    task = task.model_copy(
                        update={
                            "id": uid,
                            "tags": (
                                *task.tags,
                                *(
                                    f"{HARBOR_ID_PREFIX}{identifier}"
                                    for identifier in sorted(harbor_task_ids(directory))
                                ),
                            ),
                            "source": task.source.model_copy(
                                update={"revision": f"sha256:{hashlib.sha256(content).hexdigest()}"}
                            ),
                        }
                    )
                    yield lower_harbor_task(directory, task, session=session)
        for path in task_files:
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=MATERIALIZATION_BATCH_SIZE, columns=[LOWERED_TASK_COLUMN]):
                    for row in batch.to_pylist():
                        yield LoweredTaskSpec.model_validate_json(row[LOWERED_TASK_COLUMN])

    return cache_tasks(tasks(), cache_dir)


class HarborTaskDataset(TaskDataset):
    """Prepare Harbor source packages for the common task worker."""

    def __init__(
        self,
        data_files: Sequence[str | Mapping[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        *,
        cache_dir: Path,
        session: TaskSessionSpec,
        grader_override: NoGrader | None = None,
        num_workers: int = 8,
    ):
        self.task_path = materialize_harbor_tasks(
            data_files, cache_dir=cache_dir.expanduser(), session=session, grader_override=grader_override
        )
        super().__init__([str(self.task_path)], tokenizer, max_prompt_length, num_workers=num_workers)

    def uid(self, index: int) -> str:
        return LoweredTaskSpec.model_validate_json(self.dataframe[index][LOWERED_TASK_COLUMN]).task.id
