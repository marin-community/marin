# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run selected Harbor tasks and publish typed ledger rows without source aggregates."""

import argparse
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from harbor.host_capabilities import HostCapabilities
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.request_budget import RequestBudget

from experiments.post_training.task_curation.difficulty.accounting import GpuAccounting
from experiments.post_training.task_curation.difficulty.importer import population_revision, receipt_row
from experiments.post_training.task_curation.difficulty.ledger import append_rows, digest
from experiments.post_training.task_curation.difficulty.legacy import legacy_gaps
from experiments.post_training.task_curation.difficulty.planner import AttemptBatch, action_id, load_action
from experiments.post_training.task_curation.difficulty.selection import selected_tasks
from experiments.post_training.task_curation.difficulty.state import Gap, GapReason, LedgerRow, Outcome
from experiments.post_training.task_curation.difficulty_config import CandidateRun
from experiments.post_training.task_curation.difficulty_evidence import write_json_evidence
from experiments.post_training.task_curation.difficulty_prepared import (
    PreparedSource,
    PreparedTask,
    attempt_judge_model,
)
from experiments.post_training.task_curation.difficulty_relay import budget_relay
from experiments.post_training.task_curation.difficulty_sampling import request_parameters
from experiments.post_training.task_curation.difficulty_trials import (
    JUDGE_TEMPERATURE,
    RunSealState,
    archive_model_traffic,
    preserved_model_traffic,
    resolve_provider_secrets,
    run_task,
    seal_model_traffic,
    substitute_judge_credentials,
    task_guest_environment,
    task_required_provider_secrets,
    worker_host_capabilities,
    write_replaced_json_evidence,
)

LOGGER = logging.getLogger(__name__)


def source_has_judge_traffic(tasks: Iterable[PreparedTask]) -> bool:
    """Whether this source's tasks route judge traffic: a prepared or a declared judge."""
    return any(task.reason is None and attempt_judge_model(task) is not None for task in tasks)


def record_attempt(
    path: StoragePath,
    *,
    source: PreparedSource,
    config: CandidateRun,
    output: StoragePath,
    ledger: str,
    meter: GpuAccounting,
    task_key: str,
) -> None:
    hours = meter.finish(task_key)
    meter.start(task_key)
    row = receipt_row(
        {"uri": str(path), "sha256": file_sha256(path)},
        source.sample.model_dump(mode="json"),
        config.solver,
        str(output.parent),
        datetime.now(UTC).isoformat(),
    )
    row = replace(
        row,
        gpu_hours=hours,
        outcome=Outcome.OBSERVATION if row.reason == GapReason.PROVIDER_CAPACITY_WAIT else row.outcome,
    )
    if row.outcome == Outcome.SCORED and (
        isinstance(row.reward, bool) or not isinstance(row.reward, (int, float)) or not 0 <= row.reward <= 1
    ):
        row = replace(
            row,
            outcome=Outcome.UNSCORED,
            reward=None,
            reason=GapReason.GRADER_INFRASTRUCTURE_FAILURE,
            error_identity="reward_outside_unit_interval",
        )
    append_rows(ledger, [row])
    LOGGER.info(
        "Recorded %s/%s: %s, %s, %.6f GPU-h",
        source.sample.source_id,
        row.task_id,
        row.outcome,
        row.reason or "-",
        hours,
    )
    LOGGER.info(
        "DIFFICULTY_OUTCOME %s",
        json.dumps(
            {
                "source_id": row.source_id,
                "task_id": row.task_id,
                "attempt_id": row.attempt_id,
                "outcome": row.outcome,
                "reason": row.reason,
                "evidence_uri": row.evidence_uri,
            },
            sort_keys=True,
        ),
    )


def execute_task(
    task: PreparedTask,
    source: PreparedSource,
    config: CandidateRun,
    relay: str,
    local: Path,
    output: StoragePath,
    judge_relay: str | None,
    ledger: str,
    meter: GpuAccounting,
    provider_secrets: Mapping[str, str],
    host_capabilities: HostCapabilities,
) -> None:
    source_key = hashlib.sha256(source.sample.source_id.encode()).hexdigest()
    task_key = hashlib.sha256(task.task_id.encode()).hexdigest()
    destination = output / source_key / "attempts"
    meter.start(task_key)
    recorder = partial(
        record_attempt, source=source, config=config, output=output, ledger=ledger, meter=meter, task_key=task_key
    )
    outcome = run_task(
        task,
        config,
        relay,
        local / source_key,
        str(destination),
        judge_relay,
        0,
        host_capabilities=host_capabilities,
        record_attempt=recorder,
        guest_environment=task_guest_environment(task, provider_secrets),
    )
    hours = meter.finish(task_key)
    references = (destination / task_key / "*" / "attempt.json").glob()
    if references:
        row = LedgerRow(
            population_revision(source.sample.model_dump(mode="json")),
            source.sample.source_id,
            None,
            config.solver,
            digest([str(destination / task_key), "postlude"]),
            Outcome.COST,
            None,
            str(output.parent),
            hours,
            datetime.now(UTC).isoformat(),
        )
    else:
        gaps = legacy_gaps(outcome.pending_reason or task.reason)
        gap = gaps[0] if gaps else Gap(GapReason.COVERAGE_SHORTFALL)
        row = LedgerRow(
            population_revision(source.sample.model_dump(mode="json")),
            source.sample.source_id,
            next(t.task_sha256 for t in source.sample.tasks if t.task_id == task.task_id),
            config.solver,
            digest(str(destination / task_key)),
            Outcome.UNSCORED,
            gap.reason,
            str(output.parent),
            hours,
            datetime.now(UTC).isoformat(),
            task_id=task.task_id,
            artifact_uri=gap.artifact_uri,
            artifact_kind=gap.artifact_kind,
            judge_model=gap.judge_model,
            capability=gap.capability,
            secondary_gaps=gaps[1:],
            evidence_uri=str(destination / task_key),
        )
    append_rows(ledger, [row])


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--action-manifest", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    config = CandidateRun.model_validate_json(Path(args.config).read_text())
    sources = [PreparedSource.model_validate(row) for row in json.loads(Path(args.sources).read_text())]
    action = load_action(args.action_manifest)
    if not isinstance(action, AttemptBatch) or action.solver != config.solver:
        raise ValueError("Worker does not match its action")
    if args.local_root.exists() or StoragePath(args.output_root).exists():
        raise FileExistsError("Batch outputs are immutable; select a new action")
    args.local_root.mkdir(parents=True)
    # Resolve the wave's provider credentials once, before any attempt runs: a
    # bundle-referenced credential that cannot resolve fails the wave closed at
    # dispatch instead of mid-verifier. A standing-ruling substitute judge covers
    # its bundle's own credential templates with the relay's guest dummy.
    provider_secrets = resolve_provider_secrets(
        task_required_provider_secrets(
            Path(task.path) for source in sources for task in source.tasks if task.path is not None
        )
        - substitute_judge_credentials(task for source in sources for task in source.tasks)
    )
    # Daytona's SDK temporarily clears os.environ while uploading build contexts.
    # Resolve authority once before any parallel trial can enter that SDK path.
    host_capabilities = worker_host_capabilities(
        {}, {**os.environ, "DAYTONA_DATA_API_KEY": require_secret("DAYTONA_DATA_API_KEY")}
    )
    output = StoragePath(args.output_root)
    run_seal = RunSealState(
        config.solver.value, tuple(source.sample.source_id for source in sources), datetime.now(UTC).isoformat()
    )
    key = "EMPTY" if config.api_key_secret is None else require_secret(config.api_key_secret)
    judge = config.judge
    judge_root = args.local_root / "judge"
    if judge is not None:
        judge_root.mkdir()
    with (
        preserved_model_traffic(args.local_root, output),
        preserved_model_traffic(judge_root, output / "judge") if judge is not None else nullcontext(),
        budget_relay(
            endpoint=config.endpoint,
            model=config.api_model,
            api_key=key,
            tokenizer_model=config.identity.tokenizer,
            tokenizer_revision=config.identity.tokenizer_revision,
            parameters=request_parameters(config.settings),
            budget=RequestBudget(
                config.settings.context_tokens, config.settings.input_tokens, config.settings.output_tokens
            ),
            provider_limits=config.provider_limits,
            receipts=args.local_root / "model-requests",
        ) as relay,
        (
            budget_relay(
                endpoint=judge.endpoint,
                model=judge.api_model,
                api_key=require_secret(judge.api_key_secret) if judge.api_key_secret else "EMPTY",
                tokenizer_model=judge.identity.tokenizer,
                tokenizer_revision=judge.identity.tokenizer_revision,
                parameters={"temperature": JUDGE_TEMPERATURE},
                budget=None,
                provider_limits=judge.provider_limits,
                receipts=judge_root / "model-requests",
            )
            if judge
            else nullcontext()
        ) as judge_relay,
        ThreadPoolExecutor(max_workers=config.workers) as pool,
    ):
        for source in sources:
            requested = next(entry for entry in action.sources if entry.source_id == source.sample.source_id)
            tasks = selected_tasks(source, requested)
            meter = GpuAccounting(config.gpu_count, time.monotonic)
            meter.advance()
            futures = [
                pool.submit(
                    execute_task,
                    task,
                    source,
                    config,
                    relay,
                    args.local_root,
                    output,
                    judge_relay,
                    args.ledger,
                    meter,
                    provider_secrets,
                    host_capabilities,
                )
                for task in tasks
            ]
            for future in as_completed(futures):
                future.result()
            source_key = hashlib.sha256(source.sample.source_id.encode()).hexdigest()
            append_rows(
                args.ledger,
                [
                    LedgerRow(
                        population_revision(source.sample.model_dump(mode="json")),
                        source.sample.source_id,
                        None,
                        config.solver,
                        digest([str(output), source_key, "idle"]),
                        Outcome.COST,
                        None,
                        str(output.parent),
                        meter.idle_gpu_hours(),
                        datetime.now(UTC).isoformat(),
                    )
                ],
            )
            traffic = archive_model_traffic(
                args.local_root / "model-requests",
                tasks,
                config.attempts_per_task,
                source_key,
                args.local_root / source_key / "model-requests.tar.gz",
                output / source_key / "model-requests.tar.gz",
            )
            judge_traffic = None
            if judge is not None and source_has_judge_traffic(tasks):
                judge_traffic = archive_model_traffic(
                    judge_root / "model-requests",
                    tasks,
                    config.attempts_per_task,
                    source_key,
                    args.local_root / source_key / "judge-requests.tar.gz",
                    output / source_key / "judge-requests.tar.gz",
                )
            run_seal.record(source_key, source.sample.source_id, meter.total_hours, traffic, judge_traffic)
            write_replaced_json_evidence(output / "run-seal.json", run_seal.payload())
            seal_model_traffic(args.local_root, output)
            if judge is not None:
                seal_model_traffic(judge_root, output / "judge")
    write_json_evidence(output.parent / "worker-completion.json", {"action_id": action_id(action), "worker_exit": 0})


if __name__ == "__main__":
    main()
