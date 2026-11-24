# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize unchanged sampled contracts for production Harbor candidate trials."""

import argparse
import json
import tomllib
from collections.abc import Sequence
from pathlib import Path

from finestore.rl.difficulty import DifficultyGap, difficulty_gap_reason, require_difficulty_gaps
from rigging.filesystem.s3_compat import configure_coreweave_s3
from taskcompendium.harbor.adapter import declared_execution_limits, unsupported_harbor_contract, write_harbor_task
from taskcompendium.harbor.native_package import write_native_task
from taskcompendium.harbor.runtime import write_source_runtime
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, NoGrader, TaskSpec, judge_spec
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.glm import GLM_MODEL
from experiments.post_training.task_curation.calibration_inventory import packages_lock_available
from experiments.post_training.task_curation.difficulty_artifacts import restore_original_lock
from experiments.post_training.task_curation.difficulty_config import (
    DECLARED_JUDGE_ENV_VAR,
    DECLARED_JUDGE_PINS,
    DECLARED_JUDGE_PROVIDER_PREFIX,
    SUBSTITUTE_JUDGE_MODEL,
    judge_route_repair,
)
from experiments.post_training.task_curation.difficulty_contracts import (
    difficulty_task_contract,
    resolve_receipt_vintage,
)
from experiments.post_training.task_curation.difficulty_prepared import (
    PreparedSource,
    PreparedTask,
    grader_requires_multi_uid,
    resolve_judge_template,
)
from experiments.post_training.task_curation.difficulty_samples import DifficultySource, load_queue_sources
from experiments.post_training.task_curation.judges import JUDGE_TIERS
from experiments.post_training.task_curation.prebuilt import validate_native_package
from experiments.post_training.task_curation.sample_qc import panel_task_runtimes

# Judges the Difficulty campaign can serve: the unjudged default, the GLM route
# and every pinned judge tier. The endpoint itself is validated at serving time
# (judge probe and identity receipt), never assumed here.
SERVED_JUDGE_MODELS = frozenset({"", GLM_MODEL, *(definition.spec.model for definition in JUDGE_TIERS.values())})

# Judges a frozen native bundle may declare for its own verifier on the lane's
# Together route: the qualified declared pins and the standing-ruling substitute.
SERVED_DECLARED_JUDGE_MODELS = frozenset(DECLARED_JUDGE_PINS) | {SUBSTITUTE_JUDGE_MODEL}


def bundle_declared_judge_model(task: TaskSpec) -> str | None:
    """The judge a frozen native bundle's verifier declares, with `${VAR:-default}` templates resolved.

    Reads the frozen task.toml resource bytes directly: the source gate runs
    before staging, so no written bundle exists yet. Returns None when the bundle
    declares no judge or its template carries no default to resolve.
    """
    if not isinstance(task.grader, NoGrader) or "harbor" not in task.tags:
        return None
    resource = next((file for file in task.resources.verifier if file.path == "task.toml"), None)
    if resource is None:
        return None
    document = tomllib.loads(resource_bytes(resource).decode())
    declared = ((document.get("verifier") or {}).get("env") or {}).get(DECLARED_JUDGE_ENV_VAR)
    if declared is None:
        return None
    return resolve_judge_template(declared)


def source_gap_reason(gap: DifficultyGap, detail: str | None = None) -> str:
    """Compose a typed not-measured reason; Difficulty never emits free text here."""
    reason = difficulty_gap_reason(gap, detail)
    require_difficulty_gaps(reason)
    return reason


LOCAL_ACTOR_SOURCES: frozenset[str] = frozenset({"OpenResearcher:OpenResearcher-Dataset"})
"""Sources whose actor image exceeds the Daytona snapshot limit and rerun on the maintained LOCAL backend."""


def grader_tier_backend(source_backend: str, *, native: bool, requires_multi_uid: bool) -> str:
    """The backend serving a task's trials, routing up the tier ladder when the grader needs it.

    A grader that drops privileges to a fixed uid cannot run on the LOCAL bubblewrap tier,
    whose user namespace maps a single uid; it routes up to Daytona, whose containers carry
    a full uid map. The grader bytes stay unchanged.
    """
    if source_backend == "local" and not native and requires_multi_uid:
        return "daytona"
    return source_backend


def restore_source_locks(tasks: Sequence[TaskSpec]) -> None:
    locks = {
        environment.packages_lock
        for task in tasks
        for environment in (
            task.environment_requirements,
            None if isinstance(task.grader, NoGrader) else task.grader.environment,
        )
        if environment is not None and environment.packages_lock is not None
    }
    restored = False
    for lock in sorted(locks):
        restored = restore_original_lock(lock) or restored
    if restored:
        packages_lock_available.cache_clear()


def prepare_source(source: DifficultySource, root: Path) -> PreparedSource:
    tasks, runtimes = panel_task_runtimes(source.panel)
    frozen = {selected.task_id: selected.task_sha256 for selected in source.sample.tasks}
    for task in tasks:
        if task.id not in frozen:
            raise ValueError("Prepared task is outside the frozen difficulty selection")
    # The receipt was frozen by one code state; reproduce exactly that vintage's
    # serialization instead of assuming the current one (the hosted-v13 break).
    vintage = resolve_receipt_vintage(frozen, tasks, runtimes)
    restore_source_locks(tasks)
    prepared = []
    for task, runtime in zip(tasks, runtimes, strict=True):
        contract = difficulty_task_contract(task, runtime, vintage)
        selected = next(item for item in source.sample.tasks if item.task_id == task.id)
        if selected.task_sha256 != contract["task_sha256"]:
            raise ValueError("Prepared task differs from the frozen sample receipt")
        native = isinstance(task.grader, NoGrader) and "harbor" in task.tags
        reason = None if native else unsupported_harbor_contract(task)
        if reason is not None:
            reason = source_gap_reason(DifficultyGap.EXECUTABILITY_ADMISSION_PENDING, reason)
        judge = judge_spec(task.grader)
        if reason is None and judge is not None and judge.api != "chat_completions":
            reason = source_gap_reason(DifficultyGap.JUDGE_API_UNQUALIFIED, judge.api)
        if reason is None and judge is not None and judge.model not in SERVED_JUDGE_MODELS:
            reason = source_gap_reason(DifficultyGap.JUDGE_ROUTE_UNQUALIFIED, judge.model)
        served_judge = None if judge is None else judge.model
        if reason is None and native:
            # A native bundle's own judge declaration is typed at this source gate,
            # never at runtime: the env template resolves to the model the verifier
            # would call, and an unserved one either rides its versioned repair or
            # fails closed here as a typed gap.
            declared = bundle_declared_judge_model(task)
            repair = None if declared is None else judge_route_repair(declared)
            if repair is not None:
                served_judge = repair.served_model
            elif (
                declared is not None
                and declared.removeprefix(DECLARED_JUDGE_PROVIDER_PREFIX) not in SERVED_DECLARED_JUDGE_MODELS
            ):
                reason = source_gap_reason(DifficultyGap.JUDGE_ROUTE_UNQUALIFIED, declared)
        if reason is None:
            environments = [task.environment_requirements]
            if not isinstance(task.grader, NoGrader):
                environments.append(task.grader.environment)
            missing = sorted(
                {
                    environment.packages_lock
                    for environment in environments
                    if environment is not None
                    and environment.packages_lock is not None
                    and not packages_lock_available(environment.packages_lock)
                }
            )
            if missing:
                reason = source_gap_reason(DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, ",".join(missing))
        directory = root / canonical_sha256(source.declaration.id)[:20] / canonical_sha256(task.id)[:20]
        if reason is None:
            if native:
                validate_native_package(task)
                write_native_task(task, directory)
            else:
                shape = ExecutionRequirements.model_validate(
                    {
                        **INITIAL_REQUIREMENTS.model_dump(),
                        **declared_execution_limits(task),
                    }
                )
                write_harbor_task(task, directory, shape)
            write_source_runtime(task, runtime, directory)
        prepared.append(
            PreparedTask(
                task_id=task.id,
                runtime_sha256=contract["runtime_sha256"],
                path=str(directory) if reason is None else None,
                workload="native" if native else "task_spec",
                # The frozen backend auto-chooses Daytona for static-safe sources; a source whose
                # actor image exceeds the Daytona snapshot limit reruns on the LOCAL backend, and
                # a grader that drops privileges to a fixed uid routes up to Daytona.
                backend=grader_tier_backend(
                    "local" if source.sample.source_id in LOCAL_ACTOR_SOURCES else source.backend,
                    native=native,
                    requires_multi_uid=grader_requires_multi_uid(task.model_dump(mode="json")),
                ),
                reason=reason,
                judge_model=(served_judge if served_judge is not None else None if judge is None else judge.model),
            )
        )
    return PreparedSource(sample=source.sample, sample_receipt=source.sample_receipt, tasks=prepared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-id", action="append")
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.root.exists():
        raise FileExistsError(args.root)
    sources = load_queue_sources(args.sources, args.sources_sha256, tuple(args.source_id or ()))
    args.root.mkdir(parents=True)
    prepared = [prepare_source(source, args.root / "tasks") for source in sources]
    (args.root / "sources.json").write_text(
        json.dumps([source.model_dump(mode="json") for source in prepared], indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
