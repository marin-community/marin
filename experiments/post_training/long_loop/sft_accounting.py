# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct committed SFT samples through the native data loader for archiving."""

import hashlib
import json
import logging
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
from levanter.data.text.datasets import (
    CausalLmDataset,
    LmDataConfig,
    PackedTokenDataset,
    TokenSeqDataset,
    _resolve_pack_config,
)
from levanter.trainer import TrainerConfig
from levanter.utils.thread_utils import blocking_wait
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.archive import ArchiveInput, digest, write_once
from experiments.post_training.long_loop.source_loss import SourceLossMode, source_loss_metadata

ACCOUNTING_READ_BATCH_SIZE = 256
ACCOUNTING_CHECKPOINT_SAMPLES = 4096
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SftAccountingConfig:
    """Loader inputs for replay, independent of the producer's model and optimizer plugins."""

    data: LmDataConfig
    trainer: TrainerConfig
    sequence_length: int
    data_seed: int | None


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


async def sample_origin(
    dataset, index: int, *, child: str = "", loss_mode: SourceLossMode | None = None
) -> SampleOrigin:
    """Resolve the loader's real permutation/slice/packing indices without guessing its order."""
    if isinstance(dataset, PermutationDataset):
        permutation = await dataset._get_permutation()
        return await sample_origin(dataset.dataset, int(permutation(index)), child=child, loss_mode=loss_mode)
    if isinstance(dataset, SlicedAsyncDataset):
        return await sample_origin(dataset.dataset, index + dataset.start_index, child=child, loss_mode=loss_mode)
    if isinstance(dataset, ConcatDataset):
        await dataset._ensure_initialized()
        child_index = int(np.searchsorted(dataset._cumulative_lengths, index, side="right"))
        if child_index >= len(dataset._children):
            raise IndexError("SFT accounting index exceeds the concatenated source")
        offset = int(dataset._cumulative_lengths[child_index - 1]) if child_index else 0
        return await sample_origin(
            dataset._children[child_index], index - offset, child=dataset._names[child_index], loss_mode=loss_mode
        )
    if isinstance(dataset, PackedTokenDataset):
        packed = dataset.packed
        if packed.lookahead_window is not None and loss_mode is None:
            raise ValueError("Lookahead accounting requires an explicit source loss mode")
        rows = packed._pack_indices[index]
        if packed.slice_strategy != "left" and not (packed.slice_strategy == "raise" and len(rows) == 1):
            # Packed views slice left; an unpacked view ("raise") holds exactly one
            # whole document per pack because over-long documents raise at build.
            raise ValueError("SFT accounting requires the explicit left-slice view")
        lengths = packed._lengths["input_ids"]
        documents = tuple(
            DocumentOrigin(
                int(row),
                int(lengths[row]),
                min(int(lengths[row]), dataset.Pos.size),
            )
            for row in rows
        )
        return SampleOrigin(
            child,
            sum(doc.original_tokens for doc in documents),
            sum(doc.real_tokens for doc in documents),
            loss_mode != SourceLossMode.FULL_TOKEN,
            documents,
        )
    if isinstance(dataset, CausalLmDataset) and isinstance(dataset.dataset, TokenSeqDataset):
        length = dataset.dataset.seq_len
        return SampleOrigin(child, length, length, False, ())
    if isinstance(dataset, MappedAsyncDataset):
        return await sample_origin(dataset.dataset, index, child=child, loss_mode=loss_mode)
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
    document_lengths.extend(row["real_tokens"] for row in rows if not row["documents"])
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
        "sft_traces_seen": sum(len(row["documents"]) for row in rows if row["is_sft"]),
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


async def sampled_rows(
    config: SftAccountingConfig, steps: int, *, start_index: int = 0, loss_modes: dict[str, SourceLossMode] | None = None
):
    """Replay precisely the global sample range consumed by committed native updates."""
    if config.data.prior_context_phases or config.data.mixture_start_step is not None:
        raise ValueError("SFT accounting requires the experiment's fresh, fixed-context run")
    total = config.trainer.batch_schedule.global_data_offset_by_step(steps)
    if not 0 <= start_index <= total:
        raise ValueError("Accounting prefix exceeds the committed global sample range")
    jax.config.update("jax_threefry_partitionable", config.trainer.jax_config["jax_threefry_partitionable"])
    data_key = jax.random.split(jax.random.PRNGKey(config.trainer.seed), 4)[0]
    if config.data_seed is not None:
        data_key = jax.random.PRNGKey(config.data_seed)
    length = config.sequence_length
    mixture = config.data.train_mixture(Axis("position", length), config.trainer.batch_schedule, key=data_key)
    logger.info("Accounting loader ready; replaying samples %d through %d", start_index, total)
    if not isinstance(mixture, MixtureDataset):
        raise TypeError("SFT accounting requires the native mixture loader")
    if loss_modes is not None and set(loss_modes) != set(config.data.components):
        raise ValueError("Accounting supervision modes do not cover the exact consumed mixture")
    for first in range(start_index, total, ACCOUNTING_READ_BATCH_SIZE):
        indices = list(range(first, min(first + ACCOUNTING_READ_BATCH_SIZE, total)))
        examples = await mixture.get_batch(indices)
        source_rows = defaultdict(list)
        for position, index in enumerate(indices):
            block_id, slot = divmod(index, mixture.block_size)
            source_id, source_index = mixture._index_into_dataset_for_id(mixture._get_block(block_id)[slot], block_id)
            source_name = mixture.dataset_index[source_id]
            source_rows[source_name].append((position, source_index))
        origins: dict[int, tuple[str, SampleOrigin]] = {}
        for source_name, positions in source_rows.items():
            source = mixture.datasets[source_name]
            remapped = await mixture._remap_indices(source, [index for _, index in positions])
            for (position, _), source_index in zip(positions, remapped, strict=True):
                origins[position] = (
                    source_name,
                    await sample_origin(
                        source, source_index, loss_mode=None if loss_modes is None else loss_modes[source_name]
                    ),
                )
        for position, (index, example) in enumerate(zip(indices, examples, strict=True)):
            source_name, origin = origins[position]
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
    train = document["train_config"]
    config = SftAccountingConfig(
        data=draccus.decode(LmDataConfig, train["data"]),
        trainer=draccus.decode(TrainerConfig, train["trainer"]),
        sequence_length=train["train_seq_len"] or train["model"]["max_seq_len"],
        data_seed=train["data_seed"],
    )
    # Resolve report controls before replaying a potentially large dataset.
    step_seconds = settings["measured_step_seconds"]
    timing_source = settings["timing_source"]
    pack = settings["pack"]
    modes = None
    if "source_loss_metadata" in settings:
        pin = settings["source_loss_metadata"]
        sources = source_loss_metadata(pin["source"], pin["sha256"])
        modes = {name: source.loss_mode for name, source in sources.items()}
    steps = selected["completed_updates"]
    metadata = (StoragePath(selected["checkpoint"]) / "metadata.json").read_bytes()
    if json.loads(metadata)["step"] != selected["step"] or selected["step"] + 1 != steps:
        raise ValueError("SFT accounting checkpoint and selection disagree")
    length = config.sequence_length
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
    resume = settings.get("resume_samples")
    prefix = b""
    if resume is not None:
        if (
            resume["consumed_config_sha256"] != hashlib.sha256(consumed).hexdigest()
            or resume["checkpoint_metadata_sha256"] != hashlib.sha256(metadata).hexdigest()
        ):
            raise ValueError("Accounting prefix belongs to another consumed configuration or checkpoint")
        prefix = StoragePath(resume["source"]).read_bytes()
        if hashlib.sha256(prefix).hexdigest() != resume["sha256"] or not prefix.endswith(b"\n"):
            raise ValueError("Accounting prefix changed or has an incomplete final record")
        rows = [json.loads(line) for line in prefix.splitlines()]
        if len(rows) != resume["rows"] or any(row["global_index"] != index for index, row in enumerate(rows)):
            raise ValueError("Accounting prefix must cover the contiguous initial global sample range")
    with TemporaryDirectory(prefix="sft-sample-manifest-") as temporary:
        manifest = Path(temporary) / "realized-samples.jsonl"

        async def collect():
            with manifest.open("w") as stream:
                stream.write(prefix.decode())
                async for row in sampled_rows(config, steps, start_index=len(rows), loss_modes=modes):
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.append(row)
                    if len(rows) % ACCOUNTING_CHECKPOINT_SAMPLES == 0:
                        stream.flush()
                        content = manifest.read_bytes()
                        source = str(StoragePath(output) / f"progress/samples-{len(rows):09d}.jsonl")
                        write_once(source, content)
                        pin = {
                            "source": source,
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "rows": len(rows),
                            "consumed_config_sha256": hashlib.sha256(consumed).hexdigest(),
                            "checkpoint_metadata_sha256": hashlib.sha256(metadata).hexdigest(),
                        }
                        write_once(source + ".pin.json", json.dumps(pin, sort_keys=True).encode())
                        logger.info("Accounting saved %d committed samples: %s", len(rows), source)

        blocking_wait(collect())
        content = manifest.read_bytes()
    write_once(str(StoragePath(output) / "realized-samples.jsonl"), content)
    files.append({"name": "realized-samples.jsonl", "sha256": hashlib.sha256(content).hexdigest()})
    max_segments, _ = _resolve_pack_config(pack)
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
        "step_seconds": step_seconds,
        "timing_source": timing_source,
        "resumed_prefix": resume,
        "limitations": (
            "Padding compute/MFU share assumes cost proportional to positions; not a hardware MFU measurement. "
            "Repacking combines retained sampled documents into one counterfactual pool at the configured cap; "
            "it does not preserve source-specific grouping or change the training data. Unpacking is also estimated. "
            "Replay loss is causal and is not labeled assistant loss."
        ),
        "overall": sequence_statistics(rows, length, batch, step_seconds, max_segments=max_segments),
        "per_source": {
            name: sequence_statistics(items, length, batch, step_seconds, max_segments=max_segments)
            for name, items in sorted(by_source.items())
        },
    }
    for statistics in report["per_source"].values():
        statistics["positions_per_second"] = statistics["positions"] / (steps * step_seconds)
        statistics["estimated_real_tokens_per_second"] = statistics["real_tokens"] / (steps * step_seconds)
    content = json.dumps(report, indent=2, sort_keys=True).encode()
    write_once(str(StoragePath(output) / "realized-truncation-padding.json"), content)
    files.append({"name": "realized-truncation-padding.json", "sha256": hashlib.sha256(content).hexdigest()})
    write_once(
        str(StoragePath(output) / "receipt.json"),
        json.dumps(
            {
                "files": files,
                "representation": {"sequence_length": length, "sft_packing": pack, "slice": "left"},
                "report": report,
            },
            sort_keys=True,
            indent=2,
        ).encode(),
    )


def sft_accounting_inputs(root: str, population_evidence: dict | None = None) -> tuple[list[ArchiveInput], dict]:
    """Verify every accounting artifact before adding it to the durable archive."""
    receipt = json.loads((StoragePath(root) / "receipt.json").read_text())
    required = {
        "truncation-census.json",
        "population-padding.json",
        "decision.md",
        "realized-truncation-padding.json",
        "realized-samples.jsonl",
    }
    supplemental = population_evidence or {}
    if not required <= {item["name"] for item in receipt["files"]} | supplemental.keys():
        raise ValueError("SFT accounting lacks population, realized, padding or decision evidence")
    inputs = [ArchiveInput(str(StoragePath(root) / "receipt.json"), "sft-data/receipt.json", "dataset")]
    for item in receipt["files"]:
        uri = str(StoragePath(root) / item["name"])
        if digest(uri)[1] != item["sha256"]:
            raise ValueError(f"SFT accounting artifact changed: {item['name']}")
        inputs.append(ArchiveInput(uri, "sft-data/" + item["name"], "dataset"))
    for name, pin in supplemental.items():
        if name in {item["name"] for item in receipt["files"]}:
            raise ValueError(f"Supplemental accounting evidence duplicates a producer file: {name}")
        if digest(pin["source"])[1] != pin["sha256"]:
            raise ValueError(f"Supplemental accounting evidence changed: {name}")
        inputs.append(ArchiveInput(pin["source"], "sft-data/" + name, "dataset"))
    if supplemental:
        receipt = {**receipt, "supplemental_population_evidence": supplemental}
    return inputs, receipt
