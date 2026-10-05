# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build correctness-paired native preferences through shared Harmony curation."""

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace

import click
import numpy as np
from fray.types import ResourceConfig
from levanter.data.text.formats import ChatProcessor
from levanter.store.cache import TreeCache, write_levanter_cache
from levanter.tokenizers import load_tokenizer
from marin.datakit.chat_normalize import RepeatedToolCallPolicy, normalize_chat_to_parquet
from marin.datakit.chat_render import chat_training_record
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.datakit.normalize import DedupMode
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import load_parquet

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION, MODELS, ModelSource, complement_data_step
from experiments.post_training.bfcl_rl.data import PARTITION_MANIFEST_SHA256, BFCLPartition
from experiments.post_training.bfcl_rl.launch import recovered_model
from experiments.post_training.bfcl_rl.offline_collect import TEACHER_MODEL, TEACHER_REVISION
from experiments.post_training.bfcl_rl.offline_curate import NativeCollectionInput, collection_native_evidence
from experiments.post_training.bfcl_rl.offline_data import native_chat_document, native_model_trace, native_prompt_sha256
from experiments.post_training.bfcl_rl.preferences import RolloutOutcome, VerifiedRollout, select_training_pairs
from experiments.post_training.bfcl_rl.recovery_data import (
    RecoveryPreferenceCache,
    load_audited_partition,
    recovery_cache_value,
    write_recovery_cache,
)


@dataclass(frozen=True)
class NativePreferenceConfig:
    teachers: tuple[NativeCollectionInput, ...]
    student: NativeCollectionInput
    data_root: str
    student_tokenizer: str
    max_length: int
    output_path: str
    max_workers: int
    student_model_alias: str


@dataclass(frozen=True)
class NativePreferenceBranch:
    rollout: VerifiedRollout
    source_id: str
    initial_prompt_sha256: str


def opencode_student_identity(messages: list[dict], source_alias: str, student_alias: str) -> list[dict]:
    """Translate only OpenCode's exact system model-identification line to the student."""
    original = f"You are powered by the model named {source_alias}. " f"The exact model ID is hosted_vllm/{source_alias}"
    replacement = (
        f"You are powered by the model named {student_alias}. " f"The exact model ID is hosted_vllm/{student_alias}"
    )
    translated = []
    matches = 0
    for message in messages:
        if message["role"] != "system" or not isinstance(message["content"], str):
            translated.append(message)
            continue
        lines = message["content"].splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.rstrip("\r\n") == original:
                lines[index] = replacement + line[len(original) :]
                matches += 1
        translated.append({**message, "content": "".join(lines)})
    if matches != 1:
        raise ValueError("OpenCode context lacks one exact served-model identification line")
    return translated


def build_native_preference_cache(config: NativePreferenceConfig, partition: BFCLPartition) -> RecoveryPreferenceCache:
    """Retokenize both audited native branches and publish only sole-correct preference pairs."""
    if not config.teachers:
        raise ValueError("Native preferences require at least one teacher collection")
    if len({source.terminal_uri for source in config.teachers}) != len(config.teachers):
        raise ValueError("Native preferences require distinct teacher collections")
    root = StoragePath(config.output_path)
    raw = root / "native-chat"
    raw.mkdirs()
    branches: list[list[NativePreferenceBranch]] = []
    reports = []
    identity_adaptations = []
    with (raw / "branches.jsonl").open("w") as destination:
        sources = (*(("teacher", source) for source in config.teachers), ("student", config.student))
        for index, (role, source) in enumerate(sources):
            selected = []
            audit_path = str(root / f"{role}-collection-{index:04d}.json")
            for evidence in collection_native_evidence(source, partition, audit_path):
                if role == "student" and evidence.served_model_alias != config.student_model_alias:
                    raise ValueError("Student collection's served-model alias differs from the saved preference config")
                source_id = f"{evidence.identity.run_id}/{evidence.retained.record_id}"
                initial_prompt = ""
                if evidence.retained.rollout.outcome is not RolloutOutcome.UNSCORED:
                    trace = native_model_trace(
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
                    initial_prompt = trace.initial_prompt_sha256
                    if evidence.identity.harness.startswith("opencode@"):
                        messages = opencode_student_identity(
                            trace.messages, evidence.served_model_alias, config.student_model_alias
                        )
                        initial_messages = opencode_student_identity(
                            trace.initial_messages, evidence.served_model_alias, config.student_model_alias
                        )
                        initial_prompt = native_prompt_sha256(
                            initial_messages, trace.initial_tools, trace.assistant_prefill
                        )
                        identity_adaptations.append(
                            {
                                "source_id": source_id,
                                "source_alias": evidence.served_model_alias,
                                "student_alias": config.student_model_alias,
                                "original_initial_prompt_sha256": trace.initial_prompt_sha256,
                                "student_initial_prompt_sha256": initial_prompt,
                            }
                        )
                        trace = replace(trace, messages=messages)
                    destination.write(json.dumps(native_chat_document(trace), ensure_ascii=False) + "\n")
                selected.append(NativePreferenceBranch(evidence.retained.rollout, source_id, initial_prompt))
            branches.append(selected)
            reports.append(json.loads(StoragePath(audit_path).read_text()))
    if any(report["conditions_digest"] != reports[-1]["conditions_digest"] for report in reports[:-1]):
        raise ValueError("Native preference collections used different harness or sampling conditions")
    selections = [
        selection
        for teacher_branches in branches[:-1]
        for selection in select_training_pairs(
            [branch.rollout for branch in teacher_branches],
            [branch.rollout for branch in branches[-1]],
            complement_source_ids=frozenset(task.source_id for task in partition.complement),
            parity_source_ids=frozenset(task.source_id for task in partition.parity),
        )
    ]
    normalized = normalize_chat_to_parquet(
        input_path=str(raw),
        output_path=str(root / "harmony"),
        file_extensions=(".jsonl",),
        max_workers=config.max_workers,
        dedup_mode=DedupMode.NONE,
        repeated_tool_call_policy=RepeatedToolCallPolicy.RETAIN,
    )
    processor = ChatProcessor(
        load_tokenizer(config.student_tokenizer),
        chat_template=MARIN_CHAT_TEMPLATE,
        system_prompt_field=None,
        mask_user_turns=True,
    )
    branch_indices: dict[str, int] = {}
    overlength = []
    normalized_ids = {}

    def tokenized_branches() -> Iterator[dict]:
        for path in sorted((StoragePath(normalized.main_output_dir) / "*.parquet").glob(), key=str):
            for row in load_parquet(str(path)):
                source_id = row["source_id"]
                if source_id in normalized_ids:
                    raise ValueError("Duplicate native source identity after Harmony normalization")
                normalized_ids[source_id] = row["id"]
                encoded = processor([chat_training_record(row)])[0]
                if len(encoded["input_ids"]) > config.max_length:
                    overlength.append(source_id)
                    continue
                if not np.any(encoded["assistant_masks"]):
                    raise ValueError("Native preference branch has no trainable assistant tokens")
                branch_indices[source_id] = len(branch_indices)
                yield encoded

    branch_path = str(root / "branches")
    write_levanter_cache(
        tokenized_branches(),
        branch_path,
        metadata={"student_tokenizer": config.student_tokenizer, "chat_template": MARIN_CHAT_TEMPLATE},
        batch_size=128,
    )
    cache = TreeCache.load(branch_path, {"input_ids": np.zeros(0, np.int32), "assistant_masks": np.zeros(0, np.int32)})
    by_uri = {branch.rollout.trajectory_uri: branch for group in branches for branch in group}
    accepted = []
    excluded = []
    rows = []
    student_source_ids = {branch.source_id for branch in branches[-1]}
    paired_student_ids = set()
    for selection in selections:
        pair = selection.pair
        if pair is None:
            continue
        chosen = by_uri[pair.chosen.trajectory_uri]
        rejected = by_uri[pair.rejected.trajectory_uri]
        student_source_id = chosen.source_id if chosen.source_id in student_source_ids else rejected.source_id
        reason = ""
        if chosen.initial_prompt_sha256 != rejected.initial_prompt_sha256:
            reason = "initial_context_mismatch"
        elif chosen.source_id not in normalized_ids or rejected.source_id not in normalized_ids:
            reason = "normalization_rejected"
        elif chosen.source_id not in branch_indices or rejected.source_id not in branch_indices:
            reason = "overlength"
        elif student_source_id in paired_student_ids:
            reason = "duplicate_student_counterpart"
        if reason:
            excluded.append({"reason": reason, "pair": asdict(pair)})
            continue
        encoded = {}
        for role, branch in (("chosen", chosen), ("rejected", rejected)):
            value = cache[branch_indices[branch.source_id]]
            encoded[f"{role}_input_ids"] = np.asarray(value["input_ids"]).tolist()
            encoded[f"{role}_assistant_masks"] = np.asarray(value["assistant_masks"]).tolist()
        rows.append(encoded)
        accepted.append(asdict(pair))
        paired_student_ids.add(student_source_id)
    report = {
        "dataset_commit": partition.dataset_commit,
        "partition_manifest_sha256": PARTITION_MANIFEST_SHA256,
        "teachers": [report["identity"] for report in reports[:-1]],
        "student": reports[-1]["identity"],
        "collections": reports,
        "student_tokenizer": config.student_tokenizer,
        "conditions_digest": reports[0]["conditions_digest"],
        "dispositions": dict(Counter(selection.disposition.value for selection in selections)),
        "preferences": accepted,
        "excluded_preferences": excluded,
        "overlength_branches": overlength,
        "normalized_branch_ids": normalized_ids,
        "model_identity_adaptations": identity_adaptations,
        "max_length": config.max_length,
        "repeated_tool_call_policy": RepeatedToolCallPolicy.RETAIN,
        "scoring": "student-retokenized normalized branch transcripts; assistant-only loss",
        "pair_selection": "first usable sole-correct pair in saved teacher order; at most one per student trajectory",
    }
    write_recovery_cache(rows, report, config.output_path)
    return recovery_cache_value(config.output_path)


def run_native_preference_cache(config: NativePreferenceConfig) -> RecoveryPreferenceCache:
    return build_native_preference_cache(config, load_audited_partition(config.data_root))


def dispatch_native_preference_cache(config: NativePreferenceConfig) -> RecoveryPreferenceCache:
    resources = ResourceConfig.with_cpu(
        cpu=4, ram="32Gi", disk="64Gi", target_cluster=COLLECTION_EXECUTION.target_cluster
    )
    return remote(run_native_preference_cache, resources=resources)(config)


def native_preference_step(
    teacher_collections: tuple[tuple[str, int], ...],
    student_collection_root: str,
    teacher_source: str,
    seed: int,
    recovery_version: str,
    policy_export_version: str,
    policy_checkpoint_step: int,
) -> ArtifactStep[RecoveryPreferenceCache]:
    teachers = tuple(
        ArtifactStep.adopt(
            user_owned_name(f"inputs/bfcl-rl-native-teacher-{teacher_seed}"),
            collection_root.rsplit("/", 1)[-1],
            collection_root,
            kind=SkyRLRun,
        )
        for collection_root, teacher_seed in teacher_collections
    )
    student = ArtifactStep.adopt(
        user_owned_name(f"inputs/bfcl-rl-native-student-{seed}"),
        student_collection_root.rsplit("/", 1)[-1],
        student_collection_root,
        kind=SkyRLRun,
    )
    policy = replace(
        recovered_model(recovery_version, policy_export_version), relative_path=f"hf/step-{policy_checkpoint_step}"
    )
    data = complement_data_step()
    name = user_owned_name(f"data/bfcl-rl-native-preferences-seed-{seed}")

    def build_config(ctx: StepContext) -> NativePreferenceConfig:
        original = MODELS["student"]
        student_source = policy.resolve(ctx).uri
        return NativePreferenceConfig(
            tuple(
                NativeCollectionInput(
                    str(StoragePath(ctx.artifact_path(teacher)) / "terminal.json"),
                    ModelSource(TEACHER_MODEL, TEACHER_REVISION, teacher_source, "pinned"),
                    teacher_seed,
                )
                for teacher, (_, teacher_seed) in zip(teachers, teacher_collections, strict=True)
            ),
            NativeCollectionInput(
                str(StoragePath(ctx.artifact_path(student)) / "terminal.json"),
                ModelSource(original.model, original.revision, student_source, policy_export_version),
                seed,
            ),
            ctx.artifact_path(data),
            f"{original.model}@{original.revision}",
            40960,
            ctx.output_path,
            4,
            "bfcl-rl-recovered-policy",
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=RecoveryPreferenceCache,
        run=dispatch_native_preference_cache,
        build_config=build_config,
        deps=(*teachers, student, policy.step, data),
        runtime_args={"execution": COLLECTION_EXECUTION},
    )


@click.command(help=__doc__)
@click.option(
    "--teacher-collection",
    "teacher_collections",
    type=(str, click.IntRange(min=0, max=2**31 - 1)),
    multiple=True,
    required=True,
)
@click.option("--student-collection-root", required=True)
@click.option("--teacher-source", required=True)
@click.option("--seed", type=click.IntRange(min=0, max=2**31 - 1), required=True)
@click.option("--recovery-version", required=True)
@click.option("--policy-export-version", required=True)
@click.option("--policy-checkpoint-step", type=click.IntRange(min=0), required=True)
@rl_build_options
def main(
    teacher_collections: tuple[tuple[str, int], ...],
    student_collection_root: str,
    teacher_source: str,
    seed: int,
    recovery_version: str,
    policy_export_version: str,
    policy_checkpoint_step: int,
) -> ArtifactStep:
    return native_preference_step(
        teacher_collections,
        student_collection_root,
        teacher_source,
        seed,
        recovery_version,
        policy_export_version,
        policy_checkpoint_step,
    )


if __name__ == "__main__":
    main()
