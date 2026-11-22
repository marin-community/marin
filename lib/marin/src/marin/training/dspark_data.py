# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare assistant-masked conversations and inspect a frozen Grug teacher."""

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import datasets
import pyarrow.parquet as pq
from levanter.data.text.formats import ChatProcessor
from levanter.tokenizers import load_tokenizer
from safetensors import safe_open

GRUG_ARCHITECTURE = "GrugMoeForCausalLM"
GRUG_MASK_TOKEN = "<|finetune_right_pad_id|>"
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
class DraftRecipe:
    """DSpark model and optimizer settings shared by preflight and training."""

    shape: GrugShape
    feature_taps: tuple[int, ...]
    mask_token_id: int
    max_sequence_length: int
    max_anchors: int
    max_steps: int
    trainer_ranks: int
    teacher_replicas: int
    draft_vocab_size: int = 32000
    draft_layers: int = 3
    block_size: int = 8
    markov_rank: int = 256
    learning_rate: float = 3e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.feature_taps or tuple(sorted(set(self.feature_taps))) != self.feature_taps:
            raise ValueError("Feature taps must be strictly increasing")
        if self.feature_taps[0] < 0 or self.feature_taps[-1] >= self.shape.num_hidden_layers:
            raise ValueError("Feature taps must precede the final pre-norm distillation tap")
        if not 0 < self.draft_vocab_size < self.shape.vocab_size:
            raise ValueError("The draft vocabulary must be a strict subset of the teacher vocabulary")
        if not 0 <= self.mask_token_id < self.shape.vocab_size:
            raise ValueError("The mask token must have an embedding in the teacher")
        if not self.max_sequence_length > self.block_size or not 0 < self.max_anchors <= self.max_sequence_length:
            raise ValueError("The sequence length and anchor budget must fit the draft block")
        if min(self.max_steps, self.trainer_ranks, self.teacher_replicas) <= 0:
            raise ValueError("Step budget and GPU counts must be positive")

    @property
    def extraction_taps(self) -> tuple[int, ...]:
        return (*self.feature_taps, self.shape.num_hidden_layers)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Expected a lowercase SHA-256 digest, got {value!r}")


def verify_sha256(path: Path, expected: str) -> None:
    _require_sha256(expected)
    if file_sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def _relative_file(root: Path, name: str) -> Path:
    path = Path(name)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative file below {root}, got {name!r}")
    result = root / path
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _bundle_files(root: Path, files: dict[str, str]) -> dict[str, Path]:
    result = {}
    for name, digest in files.items():
        _require_sha256(digest)
        result[name] = _relative_file(root, name)
    return result


def tensor_files(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    with safe_open(root / "model.safetensors", framework="np") as weights:
        return dict.fromkeys(weights.keys(), "model.safetensors")


def _tensor_shape(root: Path, files: dict[str, str], name: str) -> tuple[int, ...]:
    with safe_open(_relative_file(root, files[name]), framework="np") as weights:
        return tuple(weights.get_slice(name).get_shape())


def inspect_teacher(
    root: Path, manifest_sha256: str, recipe: DraftRecipe, *, verify_weights: bool = False
) -> dict[str, Any]:
    """Validate the exported checkpoint and build its Qwen3 draft decoder config."""

    manifest = root / "dspark-teacher.json"
    verify_sha256(manifest, manifest_sha256)
    hashes = json.loads(manifest.read_text())
    files = _bundle_files(root, hashes)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if name not in files:
            raise ValueError(f"Teacher manifest must include {name}")
    # These files influence Transformers/vLLM even when they are not named in config.json.
    for pattern in ("*.json", "*.jinja", "*.safetensors", "*.model", "vocab.*", "merges.*", "chat_templates/*"):
        for path in root.glob(pattern):
            if path.is_file() and path != manifest and path.relative_to(root).as_posix() not in files:
                raise ValueError(f"Unpinned teacher file: {path}")
    for name, path in files.items():
        if verify_weights or path.suffix != ".safetensors":
            verify_sha256(path, hashes[name])

    target = json.loads(files["config.json"].read_text())
    if (
        target["model_type"] != "grug_moe"
        or target["architectures"] != [GRUG_ARCHITECTURE]
        or target.get("quantization_config")
    ):
        raise ValueError("Expected an unquantized GrugMoE HF SFT export")
    shape = recipe.shape
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
    if recipe.max_sequence_length >= target["max_position_embeddings"]:
        raise ValueError("Teacher needs context for the full prefill row and one output token")

    weights = tensor_files(root)
    missing = set(weights.values()) - files.keys()
    if missing:
        raise ValueError(f"Teacher manifest is missing weight shards: {sorted(missing)}")
    # Check the unsharded index entries for every layer and the frozen logits head.
    for layer in range(shape.num_hidden_layers):
        if f"model.layers.{layer}.input_layernorm.weight" not in weights:
            raise ValueError(f"Teacher export is missing layer {layer}")
    gate_down = _tensor_shape(root, weights, GATE_DOWN)
    if len(gate_down) != 2 or gate_down[1] != shape.hidden_size:
        raise ValueError(f"Unexpected teacher final gate shape {gate_down}")
    for name, expected in {
        EMBEDDING: (shape.vocab_size, shape.hidden_size),
        LM_HEAD: (shape.vocab_size, shape.hidden_size),
        FINAL_NORM: (shape.hidden_size,),
        GATE_UP: (shape.hidden_size, gate_down[0]),
    }.items():
        if _tensor_shape(root, weights, name) != expected:
            raise ValueError(f"Unexpected shape for teacher tensor {name}, expected {expected}")

    tokenizer = load_tokenizer(str(root))
    if not tokenizer.chat_template or tokenizer.vocab_size > shape.vocab_size:
        raise ValueError("Teacher export needs its matching tokenizer and chat template")
    if tokenizer.encode(GRUG_MASK_TOKEN) != [recipe.mask_token_id]:
        raise ValueError("Teacher tokenizer disagrees with the DSpark mask token")

    return {
        "model_type": "qwen3",
        "hidden_size": shape.hidden_size,
        "num_attention_heads": shape.num_attention_heads,
        "num_key_value_heads": shape.num_key_value_heads,
        "head_dim": shape.head_dim,
        "vocab_size": shape.vocab_size,
        "intermediate_size": 3 * shape.hidden_size // 2,
        "num_hidden_layers": recipe.draft_layers,
        "max_position_embeddings": target["max_position_embeddings"],
        "rms_norm_eps": target["rms_norm_eps"],
        "initializer_range": target["initializer_range"],
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "use_sliding_window": False,
        "sliding_window": None,
    }


def _chat_rows(files: dict[str, Path]) -> Iterator[dict]:
    for path in files.values():
        parquet = pq.ParquetFile(path)
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
    files: dict[str, Path], processor: ChatProcessor, recipe: DraftRecipe
) -> Iterator[tuple[bytes, dict]]:
    for batch in batched(_chat_rows(files), 64):
        for row in processor(batch):
            ids = row["input_ids"][: recipe.max_sequence_length]
            mask = row["assistant_masks"][: recipe.max_sequence_length].astype(bool)
            if not mask[1:].any():
                continue
            if ids.min() < 0 or ids.max() >= recipe.shape.vocab_size:
                raise ValueError("Rendered token ID is outside the teacher vocabulary")
            key = hashlib.sha256(ids.astype("<i4", copy=False).tobytes()).digest()
            yield key, {"input_ids": ids.tolist(), "loss_mask": mask.tolist(), "seq_len": len(ids)}


def inspect_mix(
    root: Path, manifest_sha256: str, *, verify_shards: bool = False
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Read the pinned train and validation shard lists."""

    manifest = root / "dspark-mix.json"
    verify_sha256(manifest, manifest_sha256)
    splits = json.loads(manifest.read_text())
    train_files = _bundle_files(root, splits["train"])
    val_files = _bundle_files(root, splits["validation"])
    if not train_files or not val_files or set(splits["train"].values()) & set(splits["validation"].values()):
        raise ValueError("The mix requires nonempty, distinct train and validation inputs")
    if verify_shards:
        for split, paths in (("train", train_files), ("validation", val_files)):
            for name, path in paths.items():
                verify_sha256(path, splits[split][name])
    return train_files, val_files


def prepare_data(
    *, output: str, teacher: str, teacher_sha256: str, mix: str, mix_sha256: str, recipe: DraftRecipe
) -> None:
    """Write the Speculators Arrow dataset with a trailing, disjoint holdout."""

    decoder = inspect_teacher(Path(teacher), teacher_sha256, recipe)
    train_files, val_files = inspect_mix(Path(mix), mix_sha256, verify_shards=True)

    tokenizer = load_tokenizer(teacher)
    processor = ChatProcessor(tokenizer)
    frequencies: Counter[int] = Counter()
    validation_keys: set[bytes] = set()

    def generate_validation() -> Iterator[dict]:
        for key, row in _tokenized_rows(val_files, processor, recipe):
            validation_keys.add(key)
            yield row

    def generate_train() -> Iterator[dict]:
        for key, row in _tokenized_rows(train_files, processor, recipe):
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
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "prepared").exists():
        return
    with TemporaryDirectory(dir=root, prefix=".prepare-") as directory:
        prepared = Path(directory) / "prepared"
        prepared.mkdir()
        val = datasets.Dataset.from_generator(generate_validation, features=features, cache_dir=directory)
        train = datasets.Dataset.from_generator(generate_train, features=features, cache_dir=directory)
        if len(train) < recipe.trainer_ranks or len(val) < recipe.trainer_ranks:
            raise ValueError("Each split needs at least one usable row per trainer rank")
        # Speculators floors len(dataset) * train_data_ratio. Use the midpoint
        # between split boundaries so rounding cannot move a row into validation.
        train_ratio = (len(train) + 0.5) / (len(train) + len(val))
        datasets.concatenate_datasets([train, val]).save_to_disk(str(prepared / "dataset"))

        (prepared / "draft_config.json").write_text(json.dumps(decoder, indent=2) + "\n")
        (prepared / "token_frequencies.json").write_text(json.dumps(frequencies, sort_keys=True) + "\n")
        (prepared / "splits.json").write_text(
            json.dumps({"train": len(train), "validation": len(val), "train_ratio": train_ratio}, indent=2) + "\n"
        )
        prepared.rename(root / "prepared")
