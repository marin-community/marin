# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare assistant-masked conversations and inspect a frozen Grug teacher."""

import hashlib
import json
import struct
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

import datasets
import pyarrow.parquet as pq
from levanter.data.text.formats import ChatProcessor
from levanter.models.snowball import GRUG_MOE_ARCHITECTURE, GRUG_MOE_MODEL_TYPE
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from safetensors import safe_open

GRUG_MASK_TOKEN = "<|finetune_right_pad_id|>"
PREPARED_DIR = "prepared"
ARROW_DATASET_DIR = "dataset"
DRAFT_CONFIG_FILE = "draft_config.json"
TOKEN_FREQUENCIES_FILE = "token_frequencies.json"
SAFETENSORS_INDEX = "model.safetensors.index.json"
_SAFETENSORS_FILE = "model.safetensors"
SPLITS_FILE = "splits.json"
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


@dataclass(frozen=True)
class DraftDataConfig:
    """Teacher and sequence settings that determine the prepared Arrow data."""

    shape: GrugShape
    mask_token_id: int
    max_sequence_length: int
    draft_layers: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.mask_token_id < self.shape.vocab_size:
            raise ValueError("The mask token must have an embedding in the teacher")


@dataclass(frozen=True)
class DraftRecipe:
    """DSpark model and optimizer settings."""

    data: DraftDataConfig
    feature_taps: tuple[int, ...]
    max_anchors: int
    max_steps: int
    trainer_ranks: int
    teacher_replicas: int
    cross_entropy_weight: float
    total_variation_weight: float
    optimizer: Literal["muon", "adamw"]
    lr_schedule: Literal["linear", "cosine", "none"]
    draft_vocab_size: int = 32000
    block_size: int = 8
    markov_rank: int = 256
    learning_rate: float = 3e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.feature_taps or tuple(sorted(set(self.feature_taps))) != self.feature_taps:
            raise ValueError("Feature taps must be strictly increasing")
        if self.feature_taps[0] < 0 or self.feature_taps[-1] >= self.data.shape.num_hidden_layers:
            raise ValueError("Feature taps must precede the final pre-norm distillation tap")
        if not 0 < self.draft_vocab_size < self.data.shape.vocab_size:
            raise ValueError("The draft vocabulary must be a strict subset of the teacher vocabulary")
        max_length = self.data.max_sequence_length
        if not max_length > self.block_size or not 0 < self.max_anchors <= max_length:
            raise ValueError("The sequence length and anchor budget must fit the draft block")
        if min(self.max_steps, self.trainer_ranks, self.teacher_replicas) <= 0:
            raise ValueError("Step budget and GPU counts must be positive")

    @property
    def extraction_taps(self) -> tuple[int, ...]:
        return (*self.feature_taps, self.data.shape.num_hidden_layers)


def file_sha256(path: Path | StoragePath) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_sha256(value: str) -> None:
    """Require a lowercase hexadecimal SHA-256 digest."""

    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Expected a lowercase SHA-256 digest, got {value!r}")


def verify_sha256(path: Path | StoragePath, expected: str) -> None:
    require_sha256(expected)
    if file_sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def _bundle_files(root: StoragePath, files: dict[str, str]) -> dict[str, StoragePath]:
    result = {}
    for name, digest in files.items():
        require_sha256(digest)
        path = Path(name)
        if not path.parts or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Expected a relative file below {root}, got {name!r}")
        source = root / name
        if not source.isfile():
            raise FileNotFoundError(str(source))
        result[name] = source
    return result


def _manifest(path: StoragePath, expected: str) -> dict:
    require_sha256(expected)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return json.loads(payload)


def tensor_files(root: Path) -> dict[str, str]:
    """Map tensor names to relative safetensors filenames."""

    index = root / SAFETENSORS_INDEX
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    with safe_open(root / _SAFETENSORS_FILE, framework="np") as weights:
        return dict.fromkeys(weights.keys(), _SAFETENSORS_FILE)


def _teacher_files(
    root: StoragePath, manifest: StoragePath, manifest_sha256: str
) -> tuple[dict[str, StoragePath], dict[str, str]]:
    hashes = _manifest(manifest, manifest_sha256)
    files = _bundle_files(root, hashes)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if name not in files:
            raise ValueError(f"Teacher manifest must include {name}")
    return files, hashes


def _tensor_header(path: StoragePath) -> dict:
    fs, location = url_to_fs(str(path))
    length = cast(bytes, fs.cat_file(location, start=0, end=8))
    (size,) = struct.unpack("<Q", length)
    if not 0 < size <= _MAX_SAFETENSORS_HEADER:
        raise ValueError(f"Invalid safetensors header size in {path}: {size}")
    return json.loads(fs.cat_file(location, start=8, end=8 + size))


def _check_teacher_tensors(files: dict[str, StoragePath], metadata: Path, shape: GrugShape) -> None:
    index = metadata / SAFETENSORS_INDEX
    if index.is_file():
        weights = json.loads(index.read_text())["weight_map"]
        headers = {}
    else:
        name = _SAFETENSORS_FILE
        headers = {name: _tensor_header(files[name])}
        weights = dict.fromkeys(headers[name], name)
    missing = set(weights.values()) - files.keys()
    if missing:
        raise ValueError(f"Teacher manifest is missing weight shards: {sorted(missing)}")
    for layer in range(shape.num_hidden_layers):
        if f"model.layers.{layer}.input_layernorm.weight" not in weights:
            raise ValueError(f"Teacher export is missing layer {layer}")

    def tensor_shape(tensor: str) -> tuple[int, ...]:
        shard = weights[tensor]
        if shard not in headers:
            headers[shard] = _tensor_header(files[shard])
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


def _stage_file(source: StoragePath, destination: Path, digest: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.download_to(str(destination))
    verify_sha256(destination, digest)


def stage_teacher(root: StoragePath, manifest: StoragePath, manifest_sha256: str, destination: Path) -> None:
    """Stage and verify only the files in the frozen HF export's manifest."""

    files, hashes = _teacher_files(root, manifest, manifest_sha256)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_stage_file, source, destination / name, hashes[name]) for name, source in files.items()]
        for future in futures:
            future.result()


def _teacher_decoder(target: dict, config: DraftDataConfig) -> dict[str, str | int | float | bool | None]:
    if (
        target["model_type"] != GRUG_MOE_MODEL_TYPE
        or target["architectures"] != [GRUG_MOE_ARCHITECTURE]
        or target.get("quantization_config")
    ):
        raise ValueError("Expected an unquantized GrugMoE HF SFT export")
    shape = config.shape
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
    ):
        if target[key] != getattr(shape, key):
            raise ValueError(f"Teacher {key}={target[key]} disagrees with selected target shape {getattr(shape, key)}")
    if bool(target.get("sconv", False)) != shape.sconv:
        raise ValueError("Teacher ShortConv configuration disagrees with selected target")
    if config.max_sequence_length >= target["max_position_embeddings"]:
        raise ValueError("Teacher needs context for the full prefill row and one output token")

    return {
        "model_type": "qwen3",
        "hidden_size": shape.hidden_size,
        "num_attention_heads": shape.num_attention_heads,
        "num_key_value_heads": shape.num_key_value_heads,
        "head_dim": shape.head_dim,
        "vocab_size": shape.vocab_size,
        "intermediate_size": 3 * shape.hidden_size // 2,
        "num_hidden_layers": config.draft_layers,
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
    root: StoragePath, manifest: StoragePath, manifest_sha256: str, config: DraftDataConfig
) -> tuple[dict[str, str | int | float | bool | None], MarinTokenizer]:
    files, hashes = _teacher_files(root, manifest, manifest_sha256)
    with TemporaryDirectory(prefix="dspark-metadata-") as directory:
        metadata = Path(directory)
        for name, source in files.items():
            if source.name.endswith(".safetensors"):
                continue
            _stage_file(source, metadata / name, hashes[name])
        decoder = _teacher_decoder(json.loads((metadata / "config.json").read_text()), config)
        _check_teacher_tensors(files, metadata, config.shape)
        tokenizer = load_tokenizer(str(metadata))
        if not tokenizer.chat_template or tokenizer.vocab_size > config.shape.vocab_size:
            raise ValueError("Teacher export needs its matching tokenizer and chat template")
        if tokenizer.encode(GRUG_MASK_TOKEN) != [config.mask_token_id]:
            raise ValueError("Teacher tokenizer disagrees with the DSpark mask token")
        return decoder, tokenizer


def inspect_teacher(
    root: StoragePath, manifest: StoragePath, manifest_sha256: str, config: DraftDataConfig
) -> dict[str, str | int | float | bool | None]:
    """Validate pinned metadata and weight headers and return the draft decoder configuration."""

    decoder, tokenizer = _teacher_inputs(root, manifest, manifest_sha256, config)
    ChatProcessor(tokenizer)  # Require a template that marks supervised spans.
    return decoder


def _chat_rows(files: dict[str, StoragePath]) -> Iterator[dict]:
    for path in files.values():
        with path.open("rb") as stream:
            parquet = pq.ParquetFile(stream)
            columns = ["messages"]
            if "chat_template_kwargs" in parquet.schema_arrow.names:
                columns.append("chat_template_kwargs")
            for batch in parquet.iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    kwargs = row.get("chat_template_kwargs") or {}
                    if isinstance(kwargs, str):
                        kwargs = json.loads(kwargs)
                    if "chat_template" in kwargs:
                        raise ValueError(f"Per-row template override in {path}")
                    row["chat_template_kwargs"] = kwargs
                    yield row


def _tokenized_rows(
    files: dict[str, StoragePath], processor: ChatProcessor, config: DraftDataConfig
) -> Iterator[tuple[bytes, dict]]:
    """Yield the clipped token-ID digest alongside its row for overlap checks."""

    for batch in batched(_chat_rows(files), 64):
        for row in processor(batch):
            ids = row["input_ids"][: config.max_sequence_length]
            mask = row["assistant_masks"][: config.max_sequence_length].astype(bool)
            if not mask[1:].any():
                continue
            if ids.min() < 0 or ids.max() >= config.shape.vocab_size:
                raise ValueError("Rendered token ID is outside the teacher vocabulary")
            key = hashlib.sha256(ids.astype("<i4", copy=False).tobytes()).digest()
            yield key, {"input_ids": ids.tolist(), "loss_mask": mask.tolist(), "seq_len": len(ids)}


def inspect_mix(
    root: StoragePath, manifest_sha256: str, *, verify_shards: bool = False
) -> tuple[dict[str, StoragePath], dict[str, StoragePath]]:
    """Read the pinned train and validation shard lists."""

    splits = _manifest(root / "dspark-mix.json", manifest_sha256)
    train_files = _bundle_files(root, splits["train"])
    val_files = _bundle_files(root, splits["validation"])
    if not train_files or not val_files or set(splits["train"].values()) & set(splits["validation"].values()):
        raise ValueError("The mix requires nonempty, distinct train and validation inputs")
    if verify_shards:
        for split, paths in (("train", train_files), ("validation", val_files)):
            for name, path in paths.items():
                verify_sha256(path, splits[split][name])
    return train_files, val_files


def _write_masked_dataset(
    prepared: Path,
    train_files: dict[str, StoragePath],
    val_files: dict[str, StoragePath],
    tokenizer: MarinTokenizer,
    config: DraftDataConfig,
) -> dict[str, int | float]:
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
            frequencies.update(
                token for token, supervised in zip(row["input_ids"], row["loss_mask"], strict=True) if supervised
            )
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
    # Speculators floors len(dataset) * train_data_ratio. Use the midpoint
    # between split boundaries so rounding cannot move a row into validation.
    train_ratio = (len(train) + 0.5) / (len(train) + len(val))
    # Speculators compares the reloaded input IDs directly with teacher tensors.
    dataset = datasets.concatenate_datasets([train, val]).with_format("torch")
    dataset.save_to_disk(str(prepared / ARROW_DATASET_DIR))

    (prepared / TOKEN_FREQUENCIES_FILE).write_text(json.dumps(frequencies, sort_keys=True) + "\n")
    return {"train": len(train), "validation": len(val), "train_ratio": train_ratio}


def prepare_data(
    *,
    output: str,
    teacher: str,
    teacher_manifest: str,
    teacher_sha256: str,
    mix: str,
    mix_sha256: str,
    config: DraftDataConfig,
) -> None:
    """Write the Speculators Arrow dataset with a trailing, disjoint holdout."""

    destination = StoragePath(output) / PREPARED_DIR
    if (destination / SPLITS_FILE).isfile():
        return

    decoder, tokenizer = _teacher_inputs(StoragePath(teacher), StoragePath(teacher_manifest), teacher_sha256, config)
    train_files, val_files = inspect_mix(StoragePath(mix), mix_sha256, verify_shards=True)

    with TemporaryDirectory(prefix="dspark-prepare-") as directory:
        prepared = Path(directory) / PREPARED_DIR
        prepared.mkdir()
        splits = _write_masked_dataset(prepared, train_files, val_files, tokenizer, config)
        (prepared / DRAFT_CONFIG_FILE).write_text(json.dumps(decoder, indent=2) + "\n")
        for file in sorted(prepared.rglob("*")):
            if file.is_file():
                target = destination / file.relative_to(prepared).as_posix()
                target.parent.mkdirs()
                target.upload_from(str(file))
        # The artifact runner serializes writers. Publish the split descriptor last.
        (destination / SPLITS_FILE).write_text(json.dumps(splits, indent=2) + "\n")
