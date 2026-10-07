# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine disjoint native BFCL preference caches without retokenizing their rows."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass

import click
import numpy as np
from fray.types import ResourceConfig
from levanter.store.cache import CacheLedger, TreeCache, write_levanter_cache
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION, complement_data_step
from experiments.post_training.bfcl_rl.data import PARTITION_MANIFEST_SHA256, BFCLPartition
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache, load_audited_partition

READ_BATCH_SIZE = 16
PREFERENCE_COLUMNS = (
    "chosen_input_ids",
    "chosen_assistant_masks",
    "rejected_input_ids",
    "rejected_assistant_masks",
)


@dataclass(frozen=True)
class PreferenceUnionConfig:
    input_paths: tuple[str, ...]
    data_root: str
    output_path: str


def combine_preference_caches(
    inputs: tuple[RecoveryPreferenceCache, ...], partition: BFCLPartition, output_path: str
) -> RecoveryPreferenceCache:
    """Validate native cache provenance, reject overlapping students, and stream exact rows."""
    if not inputs:
        raise ValueError("A preference union requires inputs")
    exemplar = {name: np.zeros(0, dtype=np.int32) for name in PREFERENCE_COLUMNS}
    complement = {task.source_id: task.digest for task in partition.complement}
    parity = {task.source_id for task in partition.parity}
    manifests = []
    sources = []
    caches = []
    student_keys = set()
    preferences = []
    expected_tokens = 0
    for source in inputs:
        raw = StoragePath(source.selection_manifest_uri).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        report = json.loads(raw)
        cache_path = str(StoragePath(source.path) / "train")
        ledger = CacheLedger.load(cache_path)
        metadata = ledger.metadata.preprocessor_metadata
        if not ledger.is_finished or not (StoragePath(cache_path) / ".success").exists():
            raise ValueError(f"Unfinished preference cache: {source.path}")
        if metadata is None or metadata.get("preference_provenance_sha256") != digest:
            raise ValueError(f"Selection hash differs from cache ledger: {source.path}")
        if metadata.get("preference_provenance_uri") != source.selection_manifest_uri:
            raise ValueError(f"Selection locator differs from cache ledger: {source.path}")
        count = len(report["preferences"])
        stats = json.loads((StoragePath(cache_path) / ".stats.json").read_text())
        if not count or not count == ledger.total_num_rows == source.num_preferences == stats["total_elements"]:
            raise ValueError(f"Preference row counts disagree: {source.path}")
        if report["partition_manifest_sha256"] != PARTITION_MANIFEST_SHA256:
            raise ValueError("Preference cache uses a different audited partition")
        if report["dataset_commit"] != partition.dataset_commit:
            raise ValueError("Preference cache uses a different dataset revision")
        if report["student_tokenizer"] != f"{source.tokenizer_uri}@{source.tokenizer_revision}":
            raise ValueError("Preference tokenizer differs from artifact metadata")
        if report["max_length"] != source.max_length:
            raise ValueError("Preference context limit differs from artifact metadata")
        if manifests:
            for field in (
                "student_tokenizer",
                "max_length",
                "conditions_digest",
                "scoring",
                "repeated_tool_call_policy",
            ):
                if report[field] != manifests[0][field]:
                    raise ValueError(f"Incompatible preference cache {field}")
            for field in ("model_source_identity", "model_revision"):
                if report["student"][field] != manifests[0]["student"][field]:
                    raise ValueError(f"Incompatible student {field}")
        collections = report["collections"]
        if any(item["conditions_digest"] != report["conditions_digest"] for item in collections):
            raise ValueError("Collection conditions disagree within a preference cache")
        student = collections[-1]
        if student["identity"] != report["student"]:
            raise ValueError("Student collection identity disagrees with selection")
        prefixes = tuple(archive + "#" for archive in student["archives"])
        for pair in report["preferences"]:
            chosen, rejected = pair["chosen"], pair["rejected"]
            fields = ("task_source_id", "task_digest", "harness", "repetition")
            if any(chosen[field] != rejected[field] for field in fields):
                raise ValueError("Preference branches do not share a task and harness")
            if chosen["outcome"] != "correct" or rejected["outcome"] != "incorrect":
                raise ValueError("Preference must distinguish correct from incorrect")
            task = chosen["task_source_id"]
            if task in parity or complement.get(task) != chosen["task_digest"]:
                raise ValueError("Preference is outside the audited complement")
            student_branches = [branch for branch in (chosen, rejected) if branch["trajectory_uri"].startswith(prefixes)]
            if len(student_branches) != 1:
                raise ValueError("Preference must contain exactly one student branch")
            branch = student_branches[0]
            # Snapshot paths may differ for the same retained record. Use the original run identity.
            key = (student["identity"]["run_id"], *(branch[field] for field in fields))
            if key in student_keys:
                raise ValueError("Overlapping student trajectory across preference caches")
            student_keys.add(key)
            preferences.append(pair)
        caches.append(TreeCache.load_from_ledger(cache_path, exemplar, ledger))
        manifests.append(report)
        sources.append({"manifest_uri": source.selection_manifest_uri, "sha256": digest, "num_preferences": count})
        expected_tokens += stats["total_tokens"]

    report = {
        field: manifests[0][field]
        for field in (
            "dataset_commit",
            "partition_manifest_sha256",
            "student_tokenizer",
            "max_length",
            "conditions_digest",
            "scoring",
            "repeated_tool_call_policy",
        )
    }
    report.update(sources=sources, preferences=preferences, ordering="input cache order, then original row order")
    root = StoragePath(output_path)
    if (root / "train" / ".success").exists() or (root / "selection.json").exists():
        raise FileExistsError(output_path)
    root.mkdirs()
    text = json.dumps(report, sort_keys=True, indent=2) + "\n"
    selection_uri = str(root / "selection.json")
    (root / "selection.json").write_text(text)

    def rows() -> Iterator[dict[str, np.ndarray]]:
        total_tokens = 0
        for cache, source in zip(caches, inputs, strict=True):
            for start in range(0, source.num_preferences, READ_BATCH_SIZE):
                for row in cache.get_batch_sync(range(start, min(start + READ_BATCH_SIZE, source.num_preferences))):
                    for role in ("chosen", "rejected"):
                        ids, mask = row[f"{role}_input_ids"], row[f"{role}_assistant_masks"]
                        if not 0 < len(ids) <= source.max_length or ids.shape != mask.shape:
                            raise ValueError("Preference token/mask lengths disagree")
                        if not np.isin(mask, (0, 1)).all() or not mask.any():
                            raise ValueError("Preference assistant mask is invalid")
                        total_tokens += len(ids)
                    yield row
        if total_tokens != expected_tokens:
            raise ValueError("Preference token counts disagree with source statistics")

    write_levanter_cache(
        rows(),
        str(root / "train"),
        metadata={
            "preference_provenance_uri": selection_uri,
            "preference_provenance_sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    )
    (root / "train" / ".stats.json").write_text(
        json.dumps(
            {
                "total_elements": len(preferences),
                "total_tokens": expected_tokens,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return RecoveryPreferenceCache(
        path=output_path,
        num_preferences=len(preferences),
        selection_manifest_uri=selection_uri,
        tokenizer_uri=inputs[0].tokenizer_uri,
        tokenizer_revision=inputs[0].tokenizer_revision,
        max_length=inputs[0].max_length,
    )


def run_preference_union(config: PreferenceUnionConfig) -> RecoveryPreferenceCache:
    return combine_preference_caches(
        tuple(RecoveryPreferenceCache.raw_load(path) for path in config.input_paths),
        load_audited_partition(config.data_root),
        config.output_path,
    )


def preference_union_step(cache_inputs: tuple[tuple[str, str], ...]) -> ArtifactStep[RecoveryPreferenceCache]:
    inputs = tuple(
        ArtifactStep.adopt(
            user_owned_name(name) + "-input", version, f"{user_owned_name(name)}/{version}", kind=RecoveryPreferenceCache
        )
        for name, version in cache_inputs
    )
    data = complement_data_step()
    name = user_owned_name("data/bfcl-rl-native-preference-union")

    def build_config(ctx: StepContext) -> PreferenceUnionConfig:
        return PreferenceUnionConfig(
            tuple(ctx.artifact_path(step) for step in inputs), ctx.artifact_path(data), ctx.output_path
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=RecoveryPreferenceCache,
        build_config=build_config,
        deps=(*inputs, data),
        run=remote(run_preference_union, resources=ResourceConfig.with_cpu(cpu=4, ram="32Gi", disk="64Gi")),
        runtime_args={"execution": COLLECTION_EXECUTION},
    )


@click.command(help=__doc__)
@click.option(
    "--cache", "cache_inputs", type=(str, str), multiple=True, required=True, help="Artifact name and version."
)
@rl_build_options
def main(cache_inputs: tuple[tuple[str, str], ...]) -> ArtifactStep[RecoveryPreferenceCache]:
    return preference_union_step(cache_inputs)


if __name__ == "__main__":
    main()
