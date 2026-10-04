# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a Snowball Harmony SFT corpus from completed native Qwen collections."""

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import click
from fray.types import ResourceConfig
from marin.datakit.sft import SftTokenStore
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.bfcl_rl.collect import MODELS, NATIVE_AGENT_PROFILES, ModelSource, complement_data_step
from experiments.post_training.bfcl_rl.data import BFCLPartition
from experiments.post_training.bfcl_rl.offline_collect import TEACHER_MODEL, TEACHER_REVISION
from experiments.post_training.bfcl_rl.offline_data import (
    NativeModelTrace,
    build_verified_sft_store,
    native_model_trace,
)
from experiments.post_training.bfcl_rl.preferences import RolloutOutcome
from experiments.post_training.bfcl_rl.recovery_data import generation_collection_receipt, load_audited_partition
from experiments.post_training.bfcl_rl.retained_preferences import (
    CollectionIdentity,
    RetainedRollout,
    retained_archive_records,
    retained_rollout,
)


@dataclass(frozen=True)
class OfflineCollectionInput:
    terminal_uri: str
    teacher_source: str
    seed: int


@dataclass(frozen=True)
class NativeCollectionInput:
    terminal_uri: str
    model: ModelSource
    seed: int


@dataclass(frozen=True)
class NativeCollectionEvidence:
    identity: CollectionIdentity
    retained: RetainedRollout
    record: dict
    retained_uri: str
    trial: dict
    native_uri: str
    literal_entries: list[dict]
    served_model_alias: str


@dataclass(frozen=True)
class OfflineCorpusConfig:
    collections: tuple[OfflineCollectionInput, ...]
    data_root: str
    student_tokenizer: str
    max_length: int
    output_path: str
    seed: int
    num_shards: int
    max_workers: int


@dataclass(frozen=True)
class LiteralSpan:
    path: str
    offset: int
    length: int


def collection_native_evidence(
    source: NativeCollectionInput, partition: BFCLPartition, audit_path: str
) -> Iterator[NativeCollectionEvidence]:
    """Stream complete native collection evidence for either model and every verifier outcome."""
    terminal = json.loads(StoragePath(source.terminal_uri).read_text())
    resolved = json.loads(StoragePath(terminal["config"]["artifacts"]["resolved_config_uri"]).read_text())
    receipt = generation_collection_receipt(
        terminal,
        resolved,
        model=source.model,
        harness="native",
        partition=partition,
    )
    skyrl = resolved["config"]["skyrl"]
    served_model_alias = Path(terminal["config"]["inputs"]["model"]["local_path"]).name
    if skyrl["trainer"]["seed"] != source.seed:
        raise ValueError("Model seed differs from the declared collection")
    harbor = skyrl["terminal_bench_config"]["harbor"]
    if harbor["agent_profiles"] != list(NATIVE_AGENT_PROFILES):
        raise ValueError("Model collection differs from the fixed native harness panel")
    if receipt.task_names != frozenset(task.name for task in partition.complement):
        raise ValueError("Offline corpus requires a full complement collection")
    trials = {}
    trace_root = StoragePath(terminal["config"]["artifacts"]["attempts_root"]) / "trace_jobs"
    for path in sorted((trace_root / "eval_sessions" / "*" / "*" / "result.json").glob(), key=str):
        trial = json.loads(path.read_text())
        task = trial["task_name"]
        if task not in receipt.task_names or task in trials:
            raise ValueError(f"Unexpected or duplicate canonical native trial: {task}")
        trials[task] = (str(path), trial)
    if set(trials) != receipt.task_names:
        raise ValueError("Canonical native results do not cover the complement")
    scored_ids = {
        trial["agent_result"]["metadata"]["rollout_correlation_id"]
        for _, trial in trials.values()
        if trial["exception_info"] is None and trial["verifier_result"]["rewards"] in ({"reward": 1.0}, {"reward": 0.0})
    }
    literal_root = StoragePath(terminal["config"]["runtime"]["experiments_dir"]) / "logs"
    spans: dict[str, list[LiteralSpan]] = defaultdict(list)
    literal_paths = sorted((literal_root / "*_literal.jsonl").glob(), key=str)
    for path in literal_paths:
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                entry = json.loads(line)
                if entry["trial_id"] in scored_ids and entry["literal"] is not None:
                    spans[entry["trial_id"]].append(LiteralSpan(str(path), offset, len(line)))
    archives = sorted(
        str(path) for path in (StoragePath(receipt.trajectory_root) / "schema_v6" / "archives" / "**" / "*.zip").glob()
    )
    dispositions: Counter[str] = Counter()
    seen = set()
    task_indices = {name: index for index, name in enumerate(sorted(receipt.task_names))}
    with ExitStack() as resources:
        literal_files = {str(path): resources.enter_context(path.open("rb")) for path in literal_paths}
        for uri, record in retained_archive_records(archives):
            task = record["trajectory"]["instance_id"]
            if task not in trials or task in seen or record["trajectory"]["repetition_id"] != 0:
                raise ValueError(f"Unexpected or duplicate retained native task: {task}")
            seen.add(task)
            native_uri, trial = trials[task]
            profile = NATIVE_AGENT_PROFILES[task_indices[task] % len(NATIVE_AGENT_PROFILES)]
            identity = replace(receipt.identity, harness=f"{profile['name']}@{profile['version']}")
            retained = retained_rollout(record, identity=identity, partition=partition, trajectory_uri=uri)
            if trial["exception_info"] is not None:
                retained = replace(retained, rollout=replace(retained.rollout, outcome=RolloutOutcome.UNSCORED))
            dispositions[f"{identity.harness}/{retained.rollout.outcome.value}"] += 1
            entries = []
            if retained.rollout.outcome is not RolloutOutcome.UNSCORED:
                correlation_id = trial["agent_result"]["metadata"]["rollout_correlation_id"]
                for span in spans[correlation_id]:
                    stream = literal_files[span.path]
                    stream.seek(span.offset)
                    entries.append(json.loads(stream.read(span.length)))
            yield NativeCollectionEvidence(
                identity, retained, record, uri, trial, native_uri, entries, served_model_alias
            )
    if seen != receipt.task_names:
        raise ValueError("Retained native evidence does not cover the complement")
    StoragePath(audit_path).write_text(
        json.dumps(
            {
                "terminal_uri": source.terminal_uri,
                "model_revision": source.model.revision,
                "model_source": source.model.uri,
                "served_model_alias": served_model_alias,
                "identity": asdict(receipt.identity),
                "conditions_digest": receipt.conditions_digest,
                "runtime_commit": terminal["config"]["runtime"]["launcher_commit"],
                "seed": source.seed,
                "canonical_trials": len(trials),
                "retained_tasks": len(seen),
                "dispositions": dict(dispositions),
                "literal_paths": [str(path) for path in literal_paths],
                "archives": archives,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def collection_teacher_traces(
    source: OfflineCollectionInput, partition: BFCLPartition, audit_path: str
) -> Iterator[NativeModelTrace]:
    """Select correct teacher branches from the shared native evidence stream."""
    collection = NativeCollectionInput(
        source.terminal_uri, ModelSource(TEACHER_MODEL, TEACHER_REVISION, source.teacher_source, "pinned"), source.seed
    )
    for evidence in collection_native_evidence(collection, partition, audit_path):
        if evidence.retained.rollout.outcome is not RolloutOutcome.CORRECT:
            continue
        yield native_model_trace(
            identity=evidence.identity,
            seed=source.seed,
            retained_record=evidence.record,
            retained_uri=evidence.retained_uri,
            native_trace_uri=evidence.native_uri,
            trial_result=evidence.trial,
            literal_entries=evidence.literal_entries,
            partition=partition,
            assistant_prefill="<think>\n",
        )


def build_offline_corpus(config: OfflineCorpusConfig) -> SftTokenStore:
    partition = load_audited_partition(config.data_root)
    root = StoragePath(config.output_path)
    (root / "collections").mkdirs()
    traces = (
        trace
        for index, source in enumerate(config.collections)
        for trace in collection_teacher_traces(source, partition, str(root / "collections" / f"{index:03d}.json"))
    )
    return build_verified_sft_store(
        traces,
        partition=partition,
        output_path=config.output_path,
        student_tokenizer=config.student_tokenizer,
        max_length=config.max_length,
        seed=config.seed,
        num_shards=config.num_shards,
        max_workers=config.max_workers,
    )


def dispatch_offline_corpus(config: OfflineCorpusConfig) -> SftTokenStore:
    remote(build_offline_corpus, resources=ResourceConfig.with_cpu(cpu=4, ram="32Gi", disk="64Gi"))(config)
    return read_artifact(str(StoragePath(config.output_path) / "student-store"), SftTokenStore).model_copy(
        update={"path": config.output_path}
    )


def offline_corpus_step(collections: tuple[tuple[str, int], ...], teacher_source: str) -> ArtifactStep[SftTokenStore]:
    data = complement_data_step()
    inputs = tuple(
        ArtifactStep.adopt(
            user_owned_name(f"inputs/bfcl-rl-qwen36-collection-{seed}"), Path(path).name, path, kind=Artifact
        )
        for path, seed in collections
    )
    student = MODELS["student"]
    tokenizer = f"{student.model}@{student.revision}"
    name = user_owned_name("data/bfcl-rl-qwen36-harmony")

    def build_config(ctx: StepContext) -> OfflineCorpusConfig:
        sources = tuple(
            OfflineCollectionInput(str(StoragePath(ctx.artifact_path(step)) / "terminal.json"), teacher_source, seed)
            for step, (_, seed) in zip(inputs, collections, strict=True)
        )
        return OfflineCorpusConfig(sources, ctx.artifact_path(data), tokenizer, 40960, ctx.output_path, 42, 8, 4)

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=SftTokenStore,
        run=dispatch_offline_corpus,
        build_config=build_config,
        deps=(*inputs, data),
    )


@click.command(help=__doc__)
@click.option(
    "--collection", type=(str, int), multiple=True, required=True, help="Completed artifact root and teacher seed."
)
@click.option("--teacher-source", required=True)
@rl_build_options
def main(collection: tuple[tuple[str, int], ...], teacher_source: str) -> ArtifactStep:
    return offline_corpus_step(collection, teacher_source)


if __name__ == "__main__":
    main()
