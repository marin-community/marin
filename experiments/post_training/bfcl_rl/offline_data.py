# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curate verifier-correct BFCL traces through Datakit's Harmony SFT pipeline."""

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from marin.datakit.chat_normalize import normalize_chat_to_parquet
from marin.datakit.download.rollout_transforms import openai_chat_document
from marin.datakit.sft import SftInput, SftTokenStore, build_sft_store
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.bfcl_rl.data import PARTITION_MANIFEST_SHA256, BFCLPartition
from experiments.post_training.bfcl_rl.preferences import RolloutOutcome
from experiments.post_training.bfcl_rl.retained_preferences import CollectionIdentity, retained_rollout


@dataclass(frozen=True)
class NativeTeacherTrace:
    """Native message evidence joined to an audited generation-only retained record."""

    identity: CollectionIdentity
    seed: int
    retained_record: Mapping[str, Any]
    retained_uri: str
    native_trace_uri: str
    messages: list[dict]
    tools: list[dict]


def verifier_selected_chat(trace: NativeTeacherTrace, partition: BFCLPartition) -> dict | None:
    """Validate complement provenance and hand correct native messages to the shared adapter.

    The collection reader joins messages and tool definitions from the native
    trace to its retained record. Teacher token IDs remain evidence only;
    Datakit subsequently renders and tokenizes Harmony for the student.
    """
    retained = retained_rollout(
        trace.retained_record,
        identity=trace.identity,
        partition=partition,
        trajectory_uri=trace.retained_uri,
    )
    if retained.rollout.outcome is not RolloutOutcome.CORRECT:
        return None
    return openai_chat_document(
        trace.messages,
        f"bfcl-complement/{trace.identity.harness}",
        source_id=f"{trace.identity.run_id}/{retained.record_id}",
        chat_template_kwargs={"tools": trace.tools},
    )


def build_verified_sft_store(
    traces: Iterable[NativeTeacherTrace],
    *,
    partition: BFCLPartition,
    output_path: str,
    student_tokenizer: str,
    max_length: int,
    seed: int,
    num_shards: int,
    max_workers: int,
) -> SftTokenStore:
    """Reuse Harmony normalization, deduplication and masked student token-store construction.

    The selection manifest retains every accepted trace's run, seed, model,
    harness, task digest and raw evidence locations across content deduplication.
    Datakit supplies normalization quarantine and overlength accounting.
    """
    root = StoragePath(output_path)
    raw = root / "selected-chat"
    raw.mkdirs()
    dispositions: Counter[str] = Counter()
    selected = []
    seen = set()
    with (raw / "traces.jsonl").open("w") as destination:
        for trace in traces:
            record_id = trace.retained_record["record_id"]
            identity = (trace.identity.run_id, record_id)
            if identity in seen:
                raise ValueError(f"Duplicate teacher evidence: {identity}")
            seen.add(identity)
            document = verifier_selected_chat(trace, partition)
            if document is None:
                dispositions["incorrect_or_unscored"] += 1
                continue
            destination.write(json.dumps(document, ensure_ascii=False) + "\n")
            task_name = trace.retained_record["trajectory"]["instance_id"]
            task = next(task for task in partition.complement if task.name == task_name)
            selected.append(
                {
                    "source_id": document["source_id"],
                    "identity": asdict(trace.identity),
                    "seed": trace.seed,
                    "task": asdict(task),
                    "retained_uri": trace.retained_uri,
                    "native_trace_uri": trace.native_trace_uri,
                }
            )
            dispositions["verifier_correct"] += 1
    report = {
        "dataset_commit": partition.dataset_commit,
        "partition_manifest_sha256": PARTITION_MANIFEST_SHA256,
        "student_tokenizer": student_tokenizer,
        "max_length": max_length,
        "seed": seed,
        "dispositions": dict(dispositions),
        "selected": selected,
    }
    (root / "selection.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if not selected:
        raise ValueError("No verifier-correct complement traces; no offline update")
    normalized = normalize_chat_to_parquet(
        input_path=str(raw),
        output_path=prefix_join(output_path, "harmony"),
        file_extensions=(".jsonl",),
        max_workers=max_workers,
    )
    return build_sft_store(
        (SftInput("bfcl-complement", str(normalized.main_output_dir)),),
        output_path=prefix_join(output_path, "student-store"),
        tokenizer=student_tokenizer,
        max_length=max_length,
        seed=seed,
        num_shards=num_shards,
        max_workers=max_workers,
    )
