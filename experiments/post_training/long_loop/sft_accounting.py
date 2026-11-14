# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct committed SFT samples through the native data loader for archiving."""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import draccus
import jax
import numpy as np
from haliax import Axis
from levanter.data.dataset import MappedAsyncDataset, PermutationDataset, SlicedAsyncDataset
from levanter.data.mixture import ConcatDataset, MixtureDataset
from levanter.data.packing import pack_documents
from levanter.data.text.datasets import CausalLmDataset, PackedTokenDataset, TokenSeqDataset, _resolve_pack_config
from levanter.main.train_lm import TrainLmConfig
from levanter.utils.thread_utils import blocking_wait
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.archive import ArchiveInput, digest, write_once


@dataclass(frozen=True)
class DocumentOrigin:
    row: int
    original_tokens: int
    real_tokens: int


@dataclass(frozen=True)
class SampleOrigin:
    child: str
    original_tokens: int
    real_tokens: int
    is_sft: bool
    documents: tuple[DocumentOrigin, ...]


async def sample_origin(dataset, index: int, *, child: str = "") -> SampleOrigin:
    """Resolve the loader's real permutation/slice/packing indices without guessing its order."""
    if isinstance(dataset, PermutationDataset):
        permutation = await dataset._get_permutation()
        return await sample_origin(dataset.dataset, int(permutation(index)), child=child)
    if isinstance(dataset, SlicedAsyncDataset):
        return await sample_origin(dataset.dataset, index + dataset.start_index, child=child)
    if isinstance(dataset, ConcatDataset):
        for name, source in dataset.datasets.items():
            length = await source.async_len()
            if index < length:
                return await sample_origin(source, index, child=name)
            index -= length
        raise IndexError("SFT accounting index exceeds the concatenated source")
    if isinstance(dataset, PackedTokenDataset):
        packed = dataset.packed
        if packed.slice_strategy != "left":
            raise ValueError("SFT accounting requires the explicit left-slice view")
        rows = packed._pack_indices[index]
        documents = tuple(
            DocumentOrigin(
                row,
                int(packed._lengths["input_ids"][row]),
                min(int(packed._lengths["input_ids"][row]), dataset.Pos.size),
            )
            for row in rows
        )
        return SampleOrigin(
            child,
            sum(doc.original_tokens for doc in documents),
            sum(doc.real_tokens for doc in documents),
            True,
            documents,
        )
    if isinstance(dataset, CausalLmDataset) and isinstance(dataset.dataset, TokenSeqDataset):
        length = dataset.dataset.seq_len
        return SampleOrigin(child, length, length, False, ())
    if isinstance(dataset, MappedAsyncDataset):
        return await sample_origin(dataset.dataset, index, child=child)
    raise TypeError(f"Unsupported SFT accounting source: {type(dataset).__name__}")


def sequence_statistics(
    rows: list[dict], sequence_length: int, batch_size: int, step_seconds: float, *, max_segments: int
) -> dict:
    """Summarize actual sample lengths; packing and compute savings are estimates."""
    lengths = np.asarray([row["real_tokens"] for row in rows], dtype=np.int64)
    positions = len(rows) * sequence_length
    real = int(lengths.sum())
    original = sum(row["original_tokens"] for row in rows)
    assistant = sum(row["loss_tokens"] for row in rows if row["is_sft"])
    replay = [row for row in rows if not row["is_sft"]]
    replay_positions = len(replay) * sequence_length
    replay_real = sum(row["real_tokens"] for row in replay)
    replay_loss = sum(row["loss_tokens"] for row in replay)
    all_loss = assistant + replay_loss
    documents = [doc for row in rows for doc in row["documents"]]
    document_lengths = [doc["real_tokens"] for doc in documents]
    document_lengths.extend(row["real_tokens"] for row in rows if not row["is_sft"])
    packed = (
        len(
            pack_documents(
                np.asarray(document_lengths),
                sequence_length,
                max_segments_per_example=max_segments,
                slice_strategy="left",
            )
        )
        if rows
        else 0
    )
    unpacked_positions = len(document_lengths) * sequence_length
    return {
        "sequences": len(rows),
        "sft_traces_seen": len(documents),
        "left_truncated_traces": sum(doc["original_tokens"] > doc["real_tokens"] for doc in documents),
        "original_tokens": original,
        "real_tokens": real,
        "lost_tokens": original - real,
        "positions": positions,
        "pad_tokens": positions - real,
        "pad_fraction": (positions - real) / positions if positions else 0.0,
        "real_length_quantiles": (
            dict(zip(("p10", "p50", "p90", "p99"), np.quantile(lengths, (0.1, 0.5, 0.9, 0.99)).tolist(), strict=True))
            if rows
            else {}
        ),
        "loss_bearing_assistant_tokens": assistant,
        "assistant_fraction_of_positions": assistant / positions if positions else 0.0,
        "all_loss_tokens": sum(row["loss_tokens"] for row in rows),
        "replay": {
            "positions": replay_positions,
            "real_tokens": replay_real,
            "loss_tokens": replay_loss,
            "fraction_of_positions": replay_positions / positions if positions else 0.0,
            "fraction_of_real_tokens": replay_real / real if real else 0.0,
            "fraction_of_loss_tokens": replay_loss / all_loss if all_loss else 0.0,
        },
        "chat": {
            "positions": positions - replay_positions,
            "real_tokens": real - replay_real,
            "assistant_loss_tokens": assistant,
            "fraction_of_positions": 1 - replay_positions / positions if positions else 0.0,
            "fraction_of_real_tokens": 1 - replay_real / real if real else 0.0,
            "fraction_of_loss_tokens": assistant / all_loss if all_loss else 0.0,
            "assistant_fraction_of_chat_real_tokens": assistant / (real - replay_real) if real > replay_real else 0.0,
        },
        "positions_per_second": batch_size * sequence_length / step_seconds,
        "estimated_real_tokens_per_second": (
            batch_size * sequence_length / step_seconds * real / positions if positions else 0.0
        ),
        "position_proportional_compute_fraction_on_padding": (positions - real) / positions if positions else 0.0,
        "counterfactual_max_segments": max_segments,
        "greedy_packed_retained_sequences": packed,
        "greedy_packed_update_equivalent": packed / batch_size,
        "ideal_packed_retained_sequence_lower_bound": (real + sequence_length - 1) // sequence_length,
        "full_original_token_sequence_lower_bound": (original + sequence_length - 1) // sequence_length,
        "unpacked_sequences": len(document_lengths),
        "unpacked_update_equivalent": len(document_lengths) / batch_size,
        "unpacked_positions": unpacked_positions,
        "unpacked_pad_fraction": (unpacked_positions - real) / unpacked_positions if unpacked_positions else 0.0,
        "document_real_length_quantiles": (
            dict(
                zip(
                    ("p10", "p50", "p90", "p99"),
                    np.quantile(document_lengths, (0.1, 0.5, 0.9, 0.99)).tolist(),
                    strict=True,
                )
            )
            if document_lengths
            else {}
        ),
    }


async def sampled_rows(config: TrainLmConfig, steps: int):
    """Replay precisely the global sample range consumed by committed native updates."""
    if config.data.prior_context_phases or config.data.mixture_start_step is not None:
        raise ValueError("SFT accounting requires the experiment's fresh, fixed-context run")
    jax.config.update("jax_threefry_partitionable", config.trainer.jax_config["jax_threefry_partitionable"])
    data_key = jax.random.split(jax.random.PRNGKey(config.trainer.seed), 4)[0]
    if config.data_seed is not None:
        data_key = jax.random.PRNGKey(config.data_seed)
    length = config.train_seq_len or config.model.max_seq_len
    mixture = config.data.train_mixture(Axis("position", length), config.trainer.batch_schedule, key=data_key)
    if not isinstance(mixture, MixtureDataset):
        raise TypeError("SFT accounting requires the native mixture loader")
    total = config.trainer.batch_schedule.global_data_offset_by_step(steps)
    for first in range(0, total, 64):
        indices = list(range(first, min(first + 64, total)))
        examples = await mixture.get_batch(indices)
        for index, example in zip(indices, examples, strict=True):
            block_id, slot = divmod(index, mixture.block_size)
            source_id, source_index = mixture._index_into_dataset_for_id(mixture._get_block(block_id)[slot], block_id)
            source_name = mixture.dataset_index[source_id]
            source = mixture.datasets[source_name]
            source_index = (await mixture._remap_indices(source, [source_index]))[0]
            origin = await sample_origin(source, source_index)
            tokens = np.asarray(example.tokens)
            yield {
                "global_index": index,
                "source": source_name,
                **asdict(origin),
                "loss_tokens": int(np.count_nonzero(np.asarray(example.loss_weight))),
                "retained_tokens_sha256": hashlib.sha256(tokens[: origin.real_tokens].tobytes()).hexdigest(),
            }


def write_sft_accounting(*, output: str, trained: str, settings: dict) -> None:
    """Persist population pins and an exact-loader replay bounded by the committed checkpoint."""
    trained_path = StoragePath(trained)
    selected = json.loads((trained_path / "selected.json").read_text())
    consumed = (trained_path / "consumed-config.json").read_bytes()
    document = json.loads(consumed)
    if document["format"] != "SFTSpec":
        raise ValueError("SFT accounting requires the native SFTSpec producer")
    config = draccus.decode(TrainLmConfig, document["train_config"])
    steps = selected["completed_updates"]
    metadata = (StoragePath(selected["checkpoint"]) / "metadata.json").read_bytes()
    if json.loads(metadata)["step"] != selected["step"] or selected["step"] + 1 != steps:
        raise ValueError("SFT accounting checkpoint and selection disagree")
    length = config.train_seq_len or config.model.max_seq_len
    batch = config.trainer.train_batch_size
    if not isinstance(batch, int):
        raise ValueError("SFT accounting requires the run's constant batch size")
    files = []
    for name, pin in settings["pinned_evidence"].items():
        content = StoragePath(pin["source"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != pin["sha256"]:
            raise ValueError(f"Pinned SFT accounting evidence changed: {name}")
        write_once(str(StoragePath(output) / name), content)
        files.append({"name": name, "sha256": pin["sha256"]})
    rows = []
    with TemporaryDirectory(prefix="sft-sample-manifest-") as temporary:
        manifest = Path(temporary) / "realized-samples.jsonl"

        async def collect():
            with manifest.open("w") as stream:
                async for row in sampled_rows(config, steps):
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.append(row)

        blocking_wait(collect())
        content = manifest.read_bytes()
    write_once(str(StoragePath(output) / "realized-samples.jsonl"), content)
    files.append({"name": "realized-samples.jsonl", "sha256": hashlib.sha256(content).hexdigest()})
    max_segments, _ = _resolve_pack_config(settings["pack"])
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    report = {
        "method": (
            "Replay of the actual native loader/permutations over the committed checkpoint's global sample range; "
            "excludes prefetched and uncommitted steps."
        ),
        "checkpoint": selected["checkpoint"],
        "committed_steps": steps,
        "checkpoint_metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "consumed_config_sha256": hashlib.sha256(consumed).hexdigest(),
        "sequence_length": length,
        "batch_size": batch,
        "step_seconds": settings["measured_step_seconds"],
        "timing_source": settings["timing_source"],
        "limitations": (
            "Padding compute/MFU share assumes cost proportional to positions; not a hardware MFU measurement. "
            "Repacking combines retained sampled documents into one counterfactual pool at the configured cap; "
            "it does not preserve source-specific grouping or change the training data. Unpacking is also estimated. "
            "Replay loss is causal and is not labeled assistant loss."
        ),
        "overall": sequence_statistics(
            rows, length, batch, settings["measured_step_seconds"], max_segments=max_segments
        ),
        "per_source": {
            name: sequence_statistics(items, length, batch, settings["measured_step_seconds"], max_segments=max_segments)
            for name, items in sorted(by_source.items())
        },
    }
    for statistics in report["per_source"].values():
        statistics["positions_per_second"] = statistics["positions"] / (steps * settings["measured_step_seconds"])
        statistics["estimated_real_tokens_per_second"] = statistics["real_tokens"] / (
            steps * settings["measured_step_seconds"]
        )
    content = json.dumps(report, indent=2, sort_keys=True).encode()
    write_once(str(StoragePath(output) / "realized-truncation-padding.json"), content)
    files.append({"name": "realized-truncation-padding.json", "sha256": hashlib.sha256(content).hexdigest()})
    write_once(
        str(StoragePath(output) / "receipt.json"),
        json.dumps(
            {
                "files": files,
                "representation": {"sequence_length": length, "sft_packing": settings["pack"], "slice": "left"},
                "report": report,
            },
            sort_keys=True,
            indent=2,
        ).encode(),
    )


def sft_accounting_inputs(root: str) -> tuple[list[ArchiveInput], dict]:
    """Verify every accounting artifact before adding it to the durable archive."""
    receipt = json.loads((StoragePath(root) / "receipt.json").read_text())
    required = {
        "truncation-census.json",
        "population-padding.json",
        "decision.md",
        "realized-truncation-padding.json",
        "realized-samples.jsonl",
    }
    if not required <= {item["name"] for item in receipt["files"]}:
        raise ValueError("SFT accounting lacks population, realized, padding or decision evidence")
    inputs = [ArchiveInput(str(StoragePath(root) / "receipt.json"), "sft-data/receipt.json", "dataset")]
    for item in receipt["files"]:
        uri = str(StoragePath(root) / item["name"])
        if digest(uri)[1] != item["sha256"]:
            raise ValueError(f"SFT accounting artifact changed: {item['name']}")
        inputs.append(ArchiveInput(uri, "sft-data/" + item["name"], "dataset"))
    return inputs, receipt
