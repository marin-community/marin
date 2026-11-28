# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct committed SFT samples through the native data loader for archiving."""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import timedelta
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
from levanter.utils.datetime_utils import encode_timedelta
from levanter.utils.thread_utils import blocking_wait
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.archive import ArchiveInput, digest, write_once
from experiments.post_training.long_loop.source_loss import SourceLossMode, source_loss_metadata

ACCOUNTING_READ_BATCH_SIZE = 256


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
    config: TrainLmConfig, steps: int, *, start_index: int = 0, loss_modes: dict[str, SourceLossMode] | None = None
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
    length = config.train_seq_len or config.model.max_seq_len
    mixture = config.data.train_mixture(Axis("position", length), config.trainer.batch_schedule, key=data_key)
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


MP_POLICY_KEYS = {"compute_dtype": "compute", "param_dtype": "params", "output_dtype": "output"}


def _mp_policy_string(value: dict) -> str:
    """Convert a draccus-encoded jmp.Policy mapping to its identical canonical policy string."""
    parts = {}
    for field, key in MP_POLICY_KEYS.items():
        entry = value.get(field)
        if isinstance(entry, dict) and "__dtype__" in entry:
            parts[key] = entry["__dtype__"]
        elif isinstance(entry, str):
            parts[key] = entry
    return ",".join(f"{k}={v}" for k, v in sorted(parts.items()))


def _decode_friendly(value):
    """Convert draccus's expanded encodings to the shapes the native dataclass decode parses.

    The launcher's request config serializes runtime objects (timedeltas, dtype policies) as
    plain mappings; the native dataclass decode expects their canonical string forms. The
    conversions are identity-preserving: the decoded values equal the encoded ones.
    """
    if isinstance(value, dict):
        if set(value) == {"days", "seconds", "microseconds"} and all(
            isinstance(entry, (int, float)) and not isinstance(entry, bool) for entry in value.values()
        ):
            return encode_timedelta(timedelta(**value))
        return {key: _decode_friendly(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_decode_friendly(entry) for entry in value]
    return value


CHOICE_DISCRIMINATORS = {
    # The launcher strips draccus choice-type discriminators from the request config. The
    # request's model and optimizer fields are the Snowball architecture's (identical to the
    # Exp6 SFTSpec consumed config, which records model.type='snowball' and
    # optimizer.type='grug_moe_hero_ep_muonh_v1').
    "model": "snowball",
    "optimizer": "grug_moe_hero_ep_muonh_v1",
}


def _native_train_config(document: dict) -> dict:
    """The request config's native trainer mapping, with expanded encodings stringified.

    The launcher strips the choice-type discriminators and renames the special-token control
    (its parity test, experiments/sft/test_launcher.py, documents both mappings); the
    request's model and optimizer fields are the Snowball architecture's, so the
    discriminators are restored for the native decode.
    """
    config = _decode_friendly(json.loads(json.dumps(document["train_config"])))
    for field, discriminator in CHOICE_DISCRIMINATORS.items():
        section = config.get(field)
        if isinstance(section, dict) and "type" not in section:
            section["type"] = discriminator
    initial_eval = config.get("initial_eval_match")
    if isinstance(initial_eval, dict):
        # The C4 reference-gate dataset format is a choice type too; the same-lineage SFTSpec
        # consumed config records type 'prebuilt' for it.
        components = (initial_eval.get("data") or {}).get("components") or {}
        for component in components.values():
            if isinstance(component, dict) and isinstance(component.get("format"), dict):
                component["format"].setdefault("type", "prebuilt")
    snowball_sft = config.get("snowball_sft")
    if isinstance(snowball_sft, dict) and "special_token_update_multiplier" in snowball_sft:
        # The launcher's parity test maps special_token_update_multiplier (LLOM preset) to
        # special_token_lr_multiplier (the prototype's SnowballSftConfig field).
        snowball_sft.setdefault("special_token_lr_multiplier", snowball_sft.pop("special_token_update_multiplier"))
    trainer_mp = (config.get("trainer") or {}).get("mp")
    if isinstance(trainer_mp, dict):
        config["trainer"]["mp"] = _mp_policy_string(trainer_mp)
    return config


def accounting_train_config(document: dict) -> tuple[TrainLmConfig, str]:
    """Decode the native trainer config from either consumed-config shape.

    An externally stopped SFT preserves the launcher's submitted request config instead of
    an SFTSpec consumed config; its ``train_config`` is the same native producer config the
    trainer consumed, with mapping-form dtype policies that decode to the identical jmp
    policy strings. Recorded as the launcher-request shape, never fabricated.
    """
    if document.get("format") == "SFTSpec":
        return draccus.decode(TrainLmConfig, document["train_config"]), "SFTSpec"
    if "train_config" in document:
        return draccus.decode(TrainLmConfig, _native_train_config(document)), "launcher-request"
    raise ValueError("SFT accounting requires the native SFTSpec producer")


def write_sft_accounting(*, output: str, trained: str, settings: dict) -> None:
    """Persist population pins and an exact-loader replay bounded by the committed checkpoint."""
    trained_path = StoragePath(trained)
    selected = json.loads((trained_path / "selected.json").read_text())
    consumed = (trained_path / "consumed-config.json").read_bytes()
    document = json.loads(consumed)
    config, consumed_config_shape = accounting_train_config(document)
    modes = None
    if "source_loss_metadata" in settings:
        pin = settings["source_loss_metadata"]
        sources = source_loss_metadata(pin["source"], pin["sha256"])
        modes = {name: source.loss_mode for name, source in sources.items()}
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
        "consumed_config_shape": consumed_config_shape,
        "sequence_length": length,
        "batch_size": batch,
        "step_seconds": settings["measured_step_seconds"],
        "timing_source": settings["timing_source"],
        "resumed_prefix": resume,
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
