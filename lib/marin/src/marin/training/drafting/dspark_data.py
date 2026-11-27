# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare assistant-masked conversations and inspect a frozen Grug teacher."""

import json
import struct
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.compat.hf_checkpoints import SAFE_TENSORS_INDEX_NAME, SAFE_TENSORS_MODEL
from levanter.data.text.formats import ChatProcessor
from levanter.models.snowball import GRUG_MOE_ARCHITECTURE, GRUG_MOE_MODEL_TYPE
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from openai_harmony import Message
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from safetensors import safe_open
from transformers.utils import CONFIG_NAME

GRUG_MASK_TOKEN = "<|finetune_right_pad_id|>"
PREPARED_DIR = "prepared"
ARROW_DATASET_DIR = "dataset"
DRAFT_CONFIG_FILE = "draft_config.json"
DRAFT_LAYER_TYPE = "qwen3"
TOKEN_FREQUENCIES_FILE = "token_frequencies.json"
SPLITS_FILE = "splits.json"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
_MAX_SAFETENSORS_HEADER = 100_000_000
EMBEDDING = "model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
FINAL_NORM = "model.norm.weight"
GATE_DOWN = "model.final_gated_norm.down_proj.weight"
GATE_UP = "model.final_gated_norm.up_proj.weight"


@dataclass(frozen=True)
class GrugShape:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    sconv: bool = False

    @classmethod
    def from_hf_config(cls, config: dict) -> "GrugShape":
        """Read the dimensions needed by the draft from HF metadata."""
        return cls(
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            vocab_size=config["vocab_size"],
            sconv=config.get("sconv", False),
        )


@dataclass(frozen=True)
class DraftDataConfig:
    """Teacher and sequence settings that determine the prepared Arrow data."""

    shape: GrugShape
    mask_token_id: int
    max_sequence_length: int

    def __post_init__(self) -> None:
        if not 0 <= self.mask_token_id < self.shape.vocab_size:
            raise ValueError("The mask token must have an embedding in the teacher")


@dataclass(frozen=True)
class PreparedSplits:
    """Prepared Arrow split sizes and the ratio passed to Speculators."""

    train: int
    validation: int
    train_ratio: float

    @classmethod
    def from_counts(cls, train: int, validation: int) -> "PreparedSplits":
        # Speculators floors dataset size times the ratio. Use the midpoint
        # between split boundaries so rounding cannot move a row into validation.
        return cls(train, validation, (train + 0.5) / (train + validation))


def supervised_token_ids(input_ids: list[int], loss_mask: list[bool]) -> Iterator[int]:
    return (token for token, supervised in zip(input_ids, loss_mask, strict=True) if supervised)


def _weight_map(index: Path | StoragePath) -> dict[str, str]:
    return json.loads(index.read_text())["weight_map"]


def tensor_files(root: Path) -> dict[str, str]:
    """Map tensor names to relative safetensors filenames."""

    index = root / SAFE_TENSORS_INDEX_NAME
    if index.is_file():
        return _weight_map(index)
    with safe_open(root / SAFE_TENSORS_MODEL, framework="np") as weights:
        return dict.fromkeys(weights.keys(), SAFE_TENSORS_MODEL)


def _tensor_header(path: StoragePath) -> dict:
    fs, location = url_to_fs(str(path))
    length = cast(bytes, fs.cat_file(location, start=0, end=8))
    (size,) = struct.unpack("<Q", length)
    if not 0 < size <= _MAX_SAFETENSORS_HEADER:
        raise ValueError(f"Invalid safetensors header size in {path}: {size}")
    return json.loads(fs.cat_file(location, start=8, end=8 + size))


def _check_teacher_tensors(root: StoragePath, shape: GrugShape) -> None:
    index = root / SAFE_TENSORS_INDEX_NAME
    if index.isfile():
        weights = _weight_map(index)
        headers = {}
    else:
        name = SAFE_TENSORS_MODEL
        headers = {name: _tensor_header(root / name)}
        weights = dict.fromkeys(headers[name].keys() - {"__metadata__"}, name)
    for shard in set(weights.values()):
        if Path(shard).name != shard or not (root / shard).isfile():
            raise ValueError(f"Missing or invalid HF weight shard {shard!r} below {root}")
    for layer in range(shape.num_hidden_layers):
        if f"model.layers.{layer}.input_layernorm.weight" not in weights:
            raise ValueError(f"Teacher export is missing layer {layer}")

    def tensor_shape(tensor: str) -> tuple[int, ...]:
        shard = weights[tensor]
        if shard not in headers:
            headers[shard] = _tensor_header(root / shard)
        return tuple(headers[shard][tensor]["shape"])

    gate_down = tensor_shape(GATE_DOWN)
    if len(gate_down) != 2 or gate_down[1] != shape.hidden_size:
        raise ValueError(f"Unexpected teacher final gate shape {gate_down}")
    for name, expected in {
        EMBEDDING: (shape.vocab_size, shape.hidden_size),
        LM_HEAD: (shape.vocab_size, shape.hidden_size),
        FINAL_NORM: (shape.hidden_size,),
        GATE_UP: (shape.hidden_size, gate_down[0]),
    }.items():
        if tensor_shape(name) != expected:
            raise ValueError(f"Unexpected shape for teacher tensor {name}, expected {expected}")


def _check_teacher_config(target: dict, config: DraftDataConfig) -> None:
    if (
        target["model_type"] != GRUG_MOE_MODEL_TYPE
        or target["architectures"] != [GRUG_MOE_ARCHITECTURE]
        or target.get("quantization_config")
    ):
        raise ValueError("Expected an unquantized GrugMoE HF SFT export")
    shape = GrugShape.from_hf_config(target)
    if shape != config.shape:
        raise ValueError(f"Teacher shape {shape} disagrees with the selected target {config.shape}")
    if config.max_sequence_length >= target["max_position_embeddings"]:
        raise ValueError("Teacher needs context for the full prefill row and one output token")


def draft_decoder_config(
    target: dict, config: DraftDataConfig, *, layers: int, intermediate_size: int
) -> dict[str, str | int | float | bool | None]:
    """Build the decoder configuration for a particular draft trained on the prepared data."""
    _check_teacher_config(target, config)
    shape = config.shape
    return {
        "model_type": DRAFT_LAYER_TYPE,
        "hidden_size": shape.hidden_size,
        "num_attention_heads": shape.num_attention_heads,
        "num_key_value_heads": shape.num_key_value_heads,
        "head_dim": shape.head_dim,
        "vocab_size": shape.vocab_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": layers,
        "max_position_embeddings": target["max_position_embeddings"],
        "rms_norm_eps": target["rms_norm_eps"],
        "initializer_range": target["initializer_range"],
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "use_sliding_window": False,
        "sliding_window": None,
    }


def _teacher_inputs(
    root: StoragePath, config: DraftDataConfig, tokenizer_source: str | None = None
) -> tuple[dict, MarinTokenizer]:
    target = json.loads((root / CONFIG_NAME).read_text())
    _check_teacher_config(target, config)
    _check_teacher_tensors(root, config.shape)
    tokenizer = load_tokenizer(tokenizer_source or str(root))
    if not tokenizer.chat_template or tokenizer.vocab_size > config.shape.vocab_size:
        raise ValueError("Teacher export needs its matching tokenizer and chat template")
    if tokenizer.encode(GRUG_MASK_TOKEN) != [config.mask_token_id]:
        raise ValueError("Teacher tokenizer disagrees with the DSpark mask token")
    return target, tokenizer


def inspect_teacher(
    root: StoragePath,
    config: DraftDataConfig,
    *,
    layers: int,
    intermediate_size: int,
    tokenizer_source: str | None = None,
) -> dict[str, str | int | float | bool | None]:
    """Validate pinned metadata and weight headers and return the draft decoder configuration."""

    target, tokenizer = _teacher_inputs(root, config, tokenizer_source)
    ChatProcessor(tokenizer)  # Require a template that marks supervised spans.
    return draft_decoder_config(target, config, layers=layers, intermediate_size=intermediate_size)


def _chat_rows(files: list[StoragePath]) -> Iterator[dict]:
    for path in files:
        with path.open("rb") as stream:
            parquet = pq.ParquetFile(stream)
            columns = ["messages"]
            message_type = parquet.schema_arrow.field("messages").type
            harmony = (
                pa.types.is_list(message_type)
                and pa.types.is_struct(message_type.value_type)
                and message_type.value_type.get_field_index("recipient") >= 0
            )
            if harmony:
                from marin.datakit.chat_render import to_template_messages  # noqa: PLC0415 - Datakit extra
            if "chat_template_kwargs" in parquet.schema_arrow.names:
                columns.append("chat_template_kwargs")
            for batch in parquet.iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    kwargs = row.get("chat_template_kwargs") or {}
                    if isinstance(kwargs, str):
                        kwargs = json.loads(kwargs)
                    if "chat_template" in kwargs:
                        raise ValueError(f"Per-row template override in {path}")
                    if harmony:
                        row["messages"] = list(to_template_messages([Message.from_dict(m) for m in row["messages"]]))
                    row["chat_template_kwargs"] = kwargs
                    yield row


def _tokenized_rows(
    files: list[StoragePath], processor: ChatProcessor, config: DraftDataConfig
) -> Iterator[tuple[bytes, dict]]:
    """Yield a serialized token sequence and its training row."""

    for batch in batched(_chat_rows(files), 64):
        for row in processor(batch):
            ids = row["input_ids"][: config.max_sequence_length]
            mask = row["assistant_masks"][: config.max_sequence_length].astype(bool)
            if not mask[1:].any():
                continue
            if ids.min() < 0 or ids.max() >= config.shape.vocab_size:
                raise ValueError("Rendered token ID is outside the teacher vocabulary")
            key = ids.astype("<i4", copy=False).tobytes()
            yield key, {"input_ids": ids.tolist(), "loss_mask": mask.tolist(), "seq_len": len(ids)}


def inspect_mix(root: StoragePath) -> tuple[list[StoragePath], list[StoragePath]]:
    """Find the train and validation Parquet shards of a completed mix."""
    train, validation = (
        sorted((root / split / "**" / "*.parquet").glob(), key=str) for split in (TRAIN_SPLIT, VALIDATION_SPLIT)
    )
    if not train or not validation:
        raise ValueError(f"The mix at {root} needs Parquet shards under train/ and validation/")
    return train, validation


def _write_masked_dataset(
    prepared: Path,
    train_files: list[StoragePath],
    val_files: list[StoragePath],
    tokenizer: MarinTokenizer,
    config: DraftDataConfig,
) -> PreparedSplits:
    processor = ChatProcessor(tokenizer)
    frequencies: Counter[int] = Counter()
    validation_keys: set[bytes] = set()

    def generate_validation() -> Iterator[dict]:
        for key, row in _tokenized_rows(val_files, processor, config):
            validation_keys.add(key)
            yield row

    def generate_train() -> Iterator[dict]:
        for key, row in _tokenized_rows(train_files, processor, config):
            if key in validation_keys:
                raise ValueError("A rendered training sequence also appears in validation")
            frequencies.update(supervised_token_ids(row["input_ids"], row["loss_mask"]))
            yield row

    features = datasets.Features(
        {
            "input_ids": datasets.Sequence(datasets.Value("int32")),
            "loss_mask": datasets.Sequence(datasets.Value("bool")),
            "seq_len": datasets.Value("int32"),
        }
    )
    val = datasets.Dataset.from_generator(generate_validation, features=features, cache_dir=str(prepared.parent))
    train = datasets.Dataset.from_generator(generate_train, features=features, cache_dir=str(prepared.parent))
    if not len(train) or not len(val):
        raise ValueError("Each split needs at least one usable row")
    # Speculators compares the reloaded input IDs directly with teacher tensors.
    dataset = datasets.concatenate_datasets([train, val]).with_format("torch")
    dataset.save_to_disk(str(prepared / ARROW_DATASET_DIR))

    (prepared / TOKEN_FREQUENCIES_FILE).write_text(json.dumps(frequencies, sort_keys=True) + "\n")
    return PreparedSplits.from_counts(len(train), len(val))


def prepare_data(
    *,
    output: str,
    teacher: str,
    teacher_subdirectory: str,
    mix: str,
    config: DraftDataConfig,
    tokenizer_source: str | None = None,
) -> None:
    """Write the Speculators Arrow dataset with a trailing, disjoint holdout."""

    destination = StoragePath(output) / PREPARED_DIR
    _, tokenizer = _teacher_inputs(StoragePath(teacher) / teacher_subdirectory, config, tokenizer_source)
    train_files, val_files = inspect_mix(StoragePath(mix))

    with TemporaryDirectory(prefix="dspark-prepare-") as directory:
        prepared = Path(directory) / PREPARED_DIR
        prepared.mkdir()
        splits = _write_masked_dataset(prepared, train_files, val_files, tokenizer, config)
        for file in sorted(prepared.rglob("*")):
            if file.is_file():
                target = destination / file.relative_to(prepared).as_posix()
                target.parent.mkdirs()
                target.upload_from(str(file))
        (destination / SPLITS_FILE).write_text(json.dumps(asdict(splits), indent=2) + "\n")
