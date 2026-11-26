# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish and score the sealed Micro210 rows on the TaskSpec launcher."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
from evalrecords.eval_stats import DEFAULT_MIN_COVERAGE
from evalrecords.records import (
    EvalRunRecord,
    EvaluationOverrides,
    JudgeIdentity,
    PromptFormat,
    TaskEvidence,
    TaskMachineBackend,
    TaskMachineSettings,
)
from finestore.rl.status_taxonomy import DataRestriction
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import TaskRolloutSpec
from taskcompendium.evaluation.corpus import chat_item, file_sha256
from taskcompendium.evaluation.spec import (
    AgentEpisode,
    EpisodeInputs,
    EpisodeLimits,
    EvaluationItem,
    EvaluationProtocol,
    OperationKind,
    PiAgent,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, RawRow

from experiments.evaluation.benchmark import EvaluationBenchmark, ProtocolVariant
from experiments.evaluation.benchmarks.harbor import EPISODE_LABEL, REWARD, harbor_item
from experiments.evaluation.micro210 import STOP_TOKEN_CONTENT, native_binding_code_sha256
from experiments.evaluation.micro210_inputs import native_rollout
from experiments.evaluation.micro210_policy import (
    POLICY_REFERENCE,
    PREPARED_COMMIT,
    PREPARED_REPO,
    Micro210PolicyReference,
    read_micro210_policy,
)
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat


@dataclass(frozen=True)
class Cohort:
    name: str
    agent: Literal["single-turn", "pi"]
    population: int
    smoke_rows: tuple[int, ...]


COHORTS = (Cohort("micro210-chat", "single-turn", 340, (0,)), Cohort("micro210-pi", "pi", 94, (1, 12)))
POPULATIONS = {cohort.name: cohort.population for cohort in COHORTS}
FULL_POPULATION = sum(POPULATIONS.values())
FULL_PROTOCOL = "sealed"
SOURCE_FAMILY = "micro210_family"
IMAGE_LOCK = Path(__file__).parents[1] / "micro210_images.json"
CORPUS_REVISION = f"{POLICY_REFERENCE.revision}-launcher.1"


@cache
def sealed_policy(reference: Micro210PolicyReference) -> dict:
    """Load the pinned protocol on first use of a Micro210 benchmark."""
    return read_micro210_policy(reference)


def read_rows(path: StoragePath, _context: ConversionContext, *, sha256: str) -> Iterator[dict[str, Any]]:
    if file_sha256(path) != sha256:
        raise ValueError("Canonical Micro210 prepared task bytes changed")
    with path.open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        for index, batch in enumerate(
            parquet.iter_batches(batch_size=1, columns=["agent", "benchmark", "task_rollout_spec"])
        ):
            yield {**batch.to_pylist()[0], "index": index}


def select_row(row: dict[str, Any], _context: ConversionContext, *, agent: str, rows: tuple[int, ...]) -> bool:
    return row["agent"] == agent and (not rows or row["index"] in rows)


def convert_row(row: RawRow, _context: ConversionContext, *, policy: dict, images: dict[str, str]) -> TaskSpec:
    native = native_rollout(TaskRolloutSpec.model_validate_json(row.data["task_rollout_spec"]), policy, images)
    task = native.task
    labels = {**task.labels, SOURCE_FAMILY: row.data["benchmark"]}
    if native.harness == "pi":
        machine = native.task_machine
        assert machine is not None
        labels[EPISODE_LABEL] = EpisodeInputs(
            agent_timeout=native.limits.total_turn_timeout,
            verifier_timeout=native.limits.verifier_timeout,
            cpus=machine.cpus,
            memory_mb=machine.memory_mb,
            storage_mb=machine.storage_mb,
            gpus=machine.gpus,
            network=machine.network,
            startup_timeout=machine.startup_timeout,
        ).model_dump_json()
    # The curation row identifies this occurrence of a task, including repeated frozen tasks.
    return task.model_copy(
        update={"id": row.id, "source": row.source, "labels": labels, "tags": tuple(dict.fromkeys((*task.tags, "eval")))}
    )


def build_item(task: TaskSpec, *, agent: AgentEpisode | None) -> EvaluationItem:
    item = chat_item(task) if agent is None else harbor_item(task, agent=agent)
    return item.model_copy(update={"groups": {"family": task.labels[SOURCE_FAMILY]}})


def build_source(cohort: Cohort, variant: ProtocolVariant) -> tuple[RlDataSource[EvaluationCorpusRecipe], ...]:
    policy = sealed_policy(POLICY_REFERENCE)
    evaluation = policy["evaluation"]
    config = evaluation["config"]
    generator, context = config["generator"], config["context_budget"]
    sampling, judge = generator["eval_sampling_params"], evaluation["judge"]
    harbor, installed = (config["terminal_bench"][key] for key in ("harbor", "installed_agent"))
    rows = cohort.smoke_rows if variant.protocol == "smoke" else ()
    count = len(rows) if rows else cohort.population
    pi = cohort.agent == "pi"
    version = f"{CORPUS_REVISION}-{native_binding_code_sha256()[:12]}"
    kwargs = installed["agent"]["kwargs"]
    agent = (
        AgentEpisode(
            harness=PiAgent(
                version=kwargs["version"],
                thinking=kwargs["thinking"],
                compaction_reserve_tokens=kwargs["compaction_reserve_tokens"],
                setup_timeout=installed["setup_timeout"],
            ),
            limits=EpisodeLimits(
                max_turns=context["max_turns"],
                max_total_tokens=None,
                model_turn_timeout=installed["run_timeout"],
                total_turn_timeout=harbor["override_timeout_sec"],
                verifier_timeout=None,
                attempt_timeout=harbor["trial_attempt_timeout_sec"],
            ),
            max_input_tokens=kwargs["model_info"]["max_input_tokens"],
        )
        if pi
        else None
    )
    protocol = EvaluationProtocol(
        benchmark=cohort.name,
        version=version,
        operation=OperationKind.AGENT_EPISODE if pi else OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling.get("top_k"),
            max_tokens=kwargs["model_info"]["max_output_tokens"] if pi else context["max_new_tokens_per_turn"],
            stop=tuple(STOP_TOKEN_CONTENT[token_id] for token_id in sampling["stop_token_ids"]),
            chat_template_kwargs=generator.get("chat_template_kwargs", {}),
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(REWARD,),
        primary_metric=REWARD.name,
        requires_judge=True,
        summary=SummaryKind.MEAN,
        expected_population=count,
        context_window=context["request_window_tokens"],
    )
    run = EvaluationOverrides(
        seed=evaluation["seed"],
        max_attempts=1 + harbor["max_retries"],
        concurrency=harbor["n_concurrent_trials"] if pi else None,
        task_machine=TaskMachineSettings(image_build=harbor["daytona_provisioning"]),
        judge=JudgeIdentity(
            deployment="together",
            models={judge["model"]: f"provider-managed:{judge['model']}"},
            default_model=judge["model"],
        ),
        judge_endpoint=judge["base_url"],
        judge_api_key=(judge["credential"],),
    )
    source = HfSource(
        PREPARED_REPO,
        PREPARED_COMMIT,
        (evaluation["data_relative_path"],),
        SourceFormat.PARQUET,
        read=partial(read_rows, sha256=policy["prepared_rollouts_sha256"]),
        select=partial(select_row, agent=cohort.agent, rows=rows),
    )
    return (
        RlDataSource(
            info=SourceInfo(
                origin=SourceOrigin.OPENATHENA,
                upstream_dataset=PREPARED_REPO,
                upstream_name=cohort.name,
                variant=variant.protocol,
                build_path="Micro210",
                data_restriction=DataRestriction.PRIVATE_ALWAYS,
                count=count,
                task_format=TaskFormat.TASKCOMPENDIUM,
            ),
            config=EvaluationCorpusRecipe(
                recipe=CurationRecipe(
                    name=f"{cohort.name}-{variant.protocol}",
                    source=source,
                    convert=partial(convert_row, policy=policy, images=json.loads(IMAGE_LOCK.read_text())),
                    version=version,
                    intended_use=IntendedUse.EVAL,
                    # The sealed chat rows include verifier resources larger than 1 MB.
                    resource_budget_bytes=2_000_000,
                ),
                protocol=protocol,
                item_builder=partial(build_item, agent=agent),
                protocol_name=variant.protocol,
                task_evidence=TaskEvidence.REFERENCE,
                agent=agent,
                run_defaults=run,
                required_task_machine_backend=TaskMachineBackend.DAYTONA,
            ),
            pipeline=prepare_evaluation,
        ),
    )


def benchmarks() -> tuple[EvaluationBenchmark, ...]:
    return tuple(
        EvaluationBenchmark(
            cohort.name,
            tuple(
                ProtocolVariant(protocol, None, PromptFormat.CHAT if cohort.agent == "single-turn" else None)
                for protocol in (FULL_PROTOCOL, "smoke")
            ),
            partial(build_source, cohort),
        )
        for cohort in COHORTS
    )


def print_pooled_score(records: Sequence[EvalRunRecord]) -> None:
    """Report the row-weighted reward for paired full Micro210 records in one launch."""
    launches: dict[tuple[str, str, str], dict[str, tuple[str, EvalRunRecord]]] = {}
    for record in records:
        curated = record.evaluation.curated
        selection = None if curated is None else curated.selection
        candidate = record.model.config_digest
        if (
            selection is not None
            and selection.protocol == FULL_PROTOCOL
            and selection.benchmark in POPULATIONS
            and candidate
        ):
            launch = (record.group_id, record.created_at, candidate)
            launches.setdefault(launch, {})[record.run_id] = (selection.benchmark, record)
    for (group, _, _), runs in sorted(launches.items()):
        pair = dict(runs.values())
        if pair.keys() != POPULATIONS.keys():
            continue
        if len(runs) != len(pair):
            print(f"{group}  micro210  multiple records for one cohort")
            continue
        scored = 0
        total = 0.0
        for name, record in pair.items():
            coverage = record.coverage.get(name)
            reward = record.metrics.get(name, {}).get(REWARD.name)
            if coverage is not None and coverage.n_benchmark == POPULATIONS[name] and reward is not None:
                scored += coverage.n_scored
                total += coverage.n_scored * reward
        if scored < DEFAULT_MIN_COVERAGE * FULL_POPULATION:
            print(f"{group}  micro210  scored {scored}/{FULL_POPULATION} (requires {DEFAULT_MIN_COVERAGE:.0%})")
        else:
            print(f"{group}  micro210  reward {total / scored:.4f}  scored {scored}/{FULL_POPULATION}")
