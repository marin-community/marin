# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a Snowball Harmony SFT corpus from completed native Qwen collections."""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path

import click
from fray.types import ResourceConfig
from marin.datakit.download.rollout_transforms import LiteralToolCallFormat
from marin.datakit.sft import SftTokenStore
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.bfcl_rl.collect import MODELS, NATIVE_AGENT_PROFILES, ModelSource, complement_data_step
from experiments.post_training.bfcl_rl.data import PARTITION_MANIFEST_SHA256, BFCLPartition
from experiments.post_training.bfcl_rl.offline_collect import TEACHER_MODEL, TEACHER_REVISION
from experiments.post_training.bfcl_rl.offline_data import (
    NativeModelTrace,
    build_verified_sft_store,
    native_model_trace,
)
from experiments.post_training.bfcl_rl.preferences import RolloutOutcome
from experiments.post_training.bfcl_rl.recovery_data import (
    generation_collection_config_receipt,
    generation_collection_receipt,
    load_audited_partition,
)
from experiments.post_training.bfcl_rl.retained_preferences import (
    CollectionIdentity,
    RetainedRollout,
    canonical_native_outcome,
    retained_archive_records,
    retained_rollout,
)


@dataclass(frozen=True)
class OfflineCollectionInput:
    terminal_uri: str
    teacher_source: str
    seed: int


class NativeCollectionScope(StrEnum):
    COMPLETE_RUN = "complete_run"
    SEALED_BATCHES = "sealed_batches"


@dataclass(frozen=True)
class NativeCollectionInput:
    manifest_uri: str
    model: ModelSource
    seed: int
    scope: NativeCollectionScope


def validate_snapshot_files(files: list[dict]) -> None:
    """Check frozen object identity before consuming a sealed collection snapshot."""
    for file in files:
        fs, key = filesystem_for(file["uri"])
        info = fs.info(key)
        if info["size"] != file["bytes"]:
            raise ValueError(f"Snapshot object size changed: {file['uri']}")
        if file["fingerprint_type"] == "etag":
            fingerprint = info["ETag"]
        elif file["fingerprint_type"] == "sha256":
            digest = hashlib.sha256()
            with StoragePath(file["uri"]).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            fingerprint = digest.hexdigest()
        else:
            raise ValueError("Unknown snapshot object fingerprint type")
        if fingerprint != file["fingerprint"]:
            raise ValueError(f"Snapshot object fingerprint changed: {file['uri']}")


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
    tool_call_format: LiteralToolCallFormat


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
    """Stream audited native evidence from a complete run or immutable sealed batches."""
    manifest = json.loads(StoragePath(source.manifest_uri).read_text())
    config = manifest["config"]
    if source.scope is NativeCollectionScope.COMPLETE_RUN:
        resolved = json.loads(StoragePath(config["artifacts"]["resolved_config_uri"]).read_text())
        receipt = generation_collection_receipt(
            manifest, resolved, model=source.model, harness="native", partition=partition
        )
        expected_tasks = receipt.task_names
        trial_paths = sorted(
            (StoragePath(config["artifacts"]["attempts_root"]) / "trace_jobs/eval_sessions/*/*/result.json").glob(),
            key=str,
        )
        literal_paths = sorted(
            (StoragePath(config["runtime"]["experiments_dir"]) / "logs/*_literal.jsonl").glob(), key=str
        )
        archives = sorted(
            str(path) for path in (StoragePath(receipt.trajectory_root) / "schema_v6/archives/**/*.zip").glob()
        )
    else:
        if manifest["schema_version"] != 1 or manifest["state"] != "sealed":
            raise ValueError("Native snapshot is not sealed")
        if manifest["partition_manifest_sha256"] != PARTITION_MANIFEST_SHA256:
            raise ValueError("Native snapshot uses a different BFCL partition")
        resolved = manifest["resolved"]
        receipt = generation_collection_config_receipt(
            config, resolved, model=source.model, harness="native", partition=partition
        )
        expected_tasks = frozenset(manifest["task_names"])
        if not expected_tasks or not expected_tasks <= receipt.task_names:
            raise ValueError("Native snapshot contains tasks outside the configured complement")
        files = [*manifest["canonical_results"], *manifest["literal_logs"], *manifest["archives"]]
        validate_snapshot_files(files)
        trial_paths = [StoragePath(file["uri"]) for file in manifest["canonical_results"]]
        literal_paths = [StoragePath(file["uri"]) for file in manifest["literal_logs"]]
        archives = [file["uri"] for file in manifest["archives"]]
    skyrl = resolved["config"]["skyrl"]
    served_model_alias = Path(config["inputs"]["model"]["local_path"]).name
    if skyrl["trainer"]["seed"] != source.seed:
        raise ValueError("Model seed differs from the declared collection")
    harbor = skyrl["terminal_bench_config"]["harbor"]
    profiles = harbor["agent_profiles"]
    if not profiles or any(profile not in NATIVE_AGENT_PROFILES for profile in profiles):
        raise ValueError("Model collection uses an unregistered native harness")
    if source.scope is NativeCollectionScope.COMPLETE_RUN and receipt.task_names != frozenset(
        task.name for task in partition.complement
    ):
        raise ValueError("Offline corpus requires a full complement collection")
    trials = {}
    for path in trial_paths:
        trial = json.loads(path.read_text())
        task = trial["task_name"]
        if task not in expected_tasks or task in trials:
            raise ValueError(f"Unexpected or duplicate canonical native trial: {task}")
        trials[task] = (str(path), trial)
    if set(trials) != expected_tasks:
        raise ValueError("Canonical native results do not cover the declared tasks")
    scored_ids = {
        trial["agent_result"]["metadata"]["rollout_correlation_id"]
        for _, trial in trials.values()
        if canonical_native_outcome(trial) is not RolloutOutcome.UNSCORED
    }
    spans: dict[str, list[LiteralSpan]] = defaultdict(list)
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
    dispositions: Counter[str] = Counter()
    seen = set()
    task_indices = {name: index for index, name in enumerate(sorted(receipt.task_names))}
    with ExitStack() as resources:
        literal_files = {str(path): resources.enter_context(path.open("rb")) for path in literal_paths}
        for uri, record in retained_archive_records(archives):
            task = record["trajectory"]["instance_id"]
            if source.scope is NativeCollectionScope.SEALED_BATCHES and task not in expected_tasks:
                if task not in receipt.task_names:
                    raise ValueError(f"Snapshot archive contains a task outside the complement: {task}")
                continue
            if task not in trials or task in seen or record["trajectory"]["repetition_id"] != 0:
                raise ValueError(f"Unexpected or duplicate retained native task: {task}")
            seen.add(task)
            native_uri, trial = trials[task]
            profile = profiles[task_indices[task] % len(profiles)]
            identity = replace(receipt.identity, harness=f"{profile['name']}@{profile['version']}")
            retained = retained_rollout(record, identity=identity, partition=partition, trajectory_uri=uri)
            if canonical_native_outcome(trial) is RolloutOutcome.UNSCORED:
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
                identity,
                retained,
                record,
                uri,
                trial,
                native_uri,
                entries,
                served_model_alias,
                LiteralToolCallFormat(skyrl["generator"]["engine_init_kwargs"]["tool_call_parser"]),
            )
    if seen != expected_tasks:
        raise ValueError("Retained native evidence does not cover the declared tasks")
    StoragePath(audit_path).write_text(
        json.dumps(
            {
                "manifest_uri": source.manifest_uri,
                "scope": source.scope,
                "producer": manifest.get("producer", manifest.get("result")),
                "model_revision": source.model.revision,
                "model_source": source.model.uri,
                "served_model_alias": served_model_alias,
                "identity": asdict(receipt.identity),
                "conditions_digest": receipt.conditions_digest,
                "runtime_commit": config["runtime"]["launcher_commit"],
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
        source.terminal_uri,
        ModelSource(TEACHER_MODEL, TEACHER_REVISION, source.teacher_source, "pinned"),
        source.seed,
        NativeCollectionScope.COMPLETE_RUN,
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
            model_tokenizer=f"{collection.model.model}@{collection.model.revision}",
            tool_call_format=evidence.tool_call_format,
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
