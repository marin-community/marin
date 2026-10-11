# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project a fixed DPO exposure into identically masked, unpacked chosen-only SFT."""

import asyncio
import hashlib
import json
from dataclasses import dataclass

import jax.random as jrandom
import numpy as np
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.main.train_dpo import _derive_training_keys
from levanter.store.cache import CacheLedger, TreeCache, write_levanter_cache
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.execution.artifact import Artifact, write_artifact
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.bfcl_rl.data import PARTITION_MANIFEST_SHA256, BFCLPartition
from experiments.post_training.bfcl_rl.preference_union import PREFERENCE_COLUMNS
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache, load_audited_partition


class MatchedChosenCache(Artifact):
    cache_path: str
    source_cache_path: str
    source_selection_sha256: str
    tokenizer: str
    max_length: int
    seed: int
    presentations: int
    row_indices: list[int]


@dataclass(frozen=True)
class MatchedChosenConfig:
    source_path: str
    data_root: str
    seed: int
    presentations: int
    output_path: str


def dpo_exposure_indices(count: int, presentations: int, seed: int) -> list[int]:
    """Use the control trainer's shuffle key and single-component mixture order."""
    data_key, *_ = _derive_training_keys(seed)
    indices = ListAsyncDataset(list(range(count))).shuffle(data_key, perm_type="feistel")
    mix_key, _ = jrandom.split(data_key)
    mixture = MixtureDataset(
        {"bfcl_complement": indices},
        {"bfcl_complement": 1.0},
        block_size=2048,
        key=mix_key,
        stop_strategy="restart",
    )
    return list(asyncio.run(mixture.get_batch(list(range(presentations)))))


def matched_sft_data_config(cache_path: str, tokenizer: str) -> LmDataConfig:
    return LmDataConfig(
        tokenizer=tokenizer,
        auto_build_caches=False,
        shuffle=False,
        # The control exposure is already ordered; do not permute it again within a mixture block.
        mixture_block_size=1,
        components={
            "bfcl_complement": DatasetComponent(
                cache_dir=cache_path,
                split="train",
                flat_cache=True,
                format=ChatLmDatasetFormat(
                    chat_template=MARIN_CHAT_TEMPLATE, pack=False, mask_user_turns=True, slice_strategy="raise"
                ),
            )
        },
    )


def project_chosen_exposure(
    source: RecoveryPreferenceCache,
    partition: BFCLPartition,
    *,
    seed: int,
    presentations: int,
    output_path: str,
) -> MatchedChosenCache:
    """Verify source provenance and copy selected chosen tokens/masks without rerendering."""
    source_path = prefix_join(source.path, "train")
    ledger = CacheLedger.load(source_path)
    raw = StoragePath(source.selection_manifest_uri).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    metadata = ledger.metadata.preprocessor_metadata
    if not ledger.is_finished or not (StoragePath(source_path) / ".success").exists():
        raise ValueError("Matched SFT requires a finished preference cache")
    if metadata is None or metadata.get("preference_provenance_sha256") != digest:
        raise ValueError("Preference selection differs from its cache ledger")
    if metadata.get("preference_provenance_uri") != source.selection_manifest_uri:
        raise ValueError("Preference selection locator differs from its cache ledger")
    if len(manifest["preferences"]) != ledger.total_num_rows or source.num_preferences != ledger.total_num_rows:
        raise ValueError("Preference row counts disagree")
    if manifest["partition_manifest_sha256"] != PARTITION_MANIFEST_SHA256:
        raise ValueError("Preference partition differs from the audited BFCL split")
    if manifest["dataset_commit"] != partition.dataset_commit:
        raise ValueError("Preference dataset differs from the audited BFCL split")
    tokenizer = f"{source.tokenizer_uri}@{source.tokenizer_revision}"
    if manifest["student_tokenizer"] != tokenizer or manifest["max_length"] != source.max_length:
        raise ValueError("Preference tokenizer or context differs from its artifact")
    complement = {task.source_id: task.digest for task in partition.complement}
    parity = {task.source_id for task in partition.parity}
    for pair in manifest["preferences"]:
        chosen, rejected = pair["chosen"], pair["rejected"]
        if chosen["outcome"] != "correct" or rejected["outcome"] != "incorrect":
            raise ValueError("Source preferences must distinguish verifier-correct from incorrect")
        for branch in (chosen, rejected):
            if branch["task_source_id"] in parity or complement.get(branch["task_source_id"]) != branch["task_digest"]:
                raise ValueError("Preference source is outside the BFCL complement")
    indices = dpo_exposure_indices(source.num_preferences, presentations, seed)
    exemplar = {name: np.zeros(0, np.int32) for name in PREFERENCE_COLUMNS}
    original = TreeCache.load_from_ledger(source_path, exemplar, ledger)
    exposed = original.get_batch_sync(indices)
    projected = [
        {"input_ids": row["chosen_input_ids"].copy(), "assistant_masks": row["chosen_assistant_masks"].copy()}
        for row in exposed
    ]
    for row in projected:
        if len(row["input_ids"]) > source.max_length or not np.any(row["assistant_masks"]):
            raise ValueError("Chosen row must fit the context and contain supervised tokens")
    root = StoragePath(output_path)
    if (root / "selection.json").exists() or (root / "train" / ".success").exists():
        raise FileExistsError(output_path)
    root.mkdirs()
    report = {
        "source_cache": source.path,
        "source_selection_sha256": digest,
        "seed": seed,
        "row_indices": indices,
        "ordering": "Exact DPO exposure order; one chosen conversation per SFT example",
        "preferences": [manifest["preferences"][index] for index in indices],
    }
    (root / "selection.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    cache_path = str(root / "train")
    written = write_levanter_cache(projected, cache_path, metadata=report)
    if written["count"] != presentations:
        raise ValueError("Chosen exposure count differs from the requested control presentations")
    actual = TreeCache.load(cache_path, {"input_ids": np.zeros(0, np.int32), "assistant_masks": np.zeros(0, np.int32)})
    for expected, observed in zip(projected, actual.get_batch_sync(list(range(presentations))), strict=True):
        for field in ("input_ids", "assistant_masks"):
            np.testing.assert_array_equal(expected[field], observed[field])
    result = MatchedChosenCache(
        path=output_path,
        cache_path=cache_path,
        source_cache_path=source.path,
        source_selection_sha256=digest,
        tokenizer=tokenizer,
        max_length=source.max_length,
        seed=seed,
        presentations=presentations,
        row_indices=indices,
    )
    write_artifact(result.result_payload(), output_path)
    return result


def run_matched_chosen(config: MatchedChosenConfig) -> MatchedChosenCache:
    return project_chosen_exposure(
        RecoveryPreferenceCache.raw_load(config.source_path),
        load_audited_partition(config.data_root),
        seed=config.seed,
        presentations=config.presentations,
        output_path=config.output_path,
    )
