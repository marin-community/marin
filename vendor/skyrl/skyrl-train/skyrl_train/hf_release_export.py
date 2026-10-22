"""Apply the shared HF release contract to Megatron Bridge output."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import PreTrainedTokenizerBase

from marin_hf_release import (
    PrecisionMode,
    ReleasePolicy,
    ReleaseRequest,
    TensorSpec,
    canonical_grug_config,
    exporter_revision,
    grug_tensor_specs,
    persist_request,
    resolve_metadata,
    tokenizer_special_token_ids,
    validate_release_metadata,
    validate_release_weights,
    write_completion_manifest,
)
from marin_hf_release.policy import dtype_size
from marin_hf_release.validation import INDEX_NAME, read_safetensors_tensors
from marinskyrl.hf_model import normalize_fast_tokenizer_metadata

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM


@dataclass(frozen=True)
class ReleaseExportOptions:
    precision: PrecisionMode
    source_checkpoint: str
    source_step: int | None
    checkpoint_identity: str
    source_model: str | None
    source_model_commit: str | None
    eos_token_ids: list[int] | None
    generation_controls: dict[str, object]
    chat_template: str | None


def _expected_grug_shapes(config: dict[str, object]) -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        model = GrugMoeForCausalLM(GrugMoeConfig(**config))
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def _bridge_tensor_specs(root: Path, expected_shapes: dict[str, tuple[int, ...]]) -> dict[str, TensorSpec]:
    tensors = {}
    for shard in root.glob("*.safetensors"):
        for name, spec in read_safetensors_tensors(shard).items():
            if name in tensors:
                raise ValueError(f"duplicate Bridge tensor {name}")
            tensors[name] = spec
    if tensors.keys() != expected_shapes.keys():
        raise ValueError(
            f"Bridge tensor names differ from Grug schema: "
            f"missing={sorted(expected_shapes.keys() - tensors.keys())[:5]}, "
            f"extra={sorted(tensors.keys() - expected_shapes.keys())[:5]}"
        )
    for name, spec in tensors.items():
        if spec.shape != expected_shapes[name]:
            raise ValueError(f"Bridge tensor {name} has shape {spec.shape}, expected {expected_shapes[name]}")
    return tensors


def _rewrite_precision(root: Path, specs: dict[str, TensorSpec], policy: ReleasePolicy) -> None:
    for shard in sorted(root.glob("*.safetensors")):
        headers = read_safetensors_tensors(shard)
        if all(policy.target_dtype(name, specs[name]) == spec.source_dtype for name, spec in headers.items()):
            continue
        tensors = load_file(shard, device="cpu")
        for name, tensor in tensors.items():
            target = policy.target_dtype(name, specs[name])
            if target != headers[name].source_dtype:
                tensors[name] = tensor.to({"BF16": torch.bfloat16, "F32": torch.float32}[target])
        staged = shard.with_suffix(".staged")
        save_file(tensors, staged, metadata={"format": "pt"})
        os.replace(staged, shard)


def _write_index(root: Path) -> None:
    weight_map = {}
    total_size = 0
    for shard in sorted(root.glob("*.safetensors")):
        for name, spec in read_safetensors_tensors(shard).items():
            if name in weight_map:
                raise ValueError(f"duplicate Bridge tensor {name}")
            weight_map[name] = shard.name
            count = 1
            for dim in spec.shape:
                count *= dim
            total_size += count * dtype_size(spec.source_dtype)
    (root / INDEX_NAME).write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, sort_keys=True, indent=2) + "\n"
    )


def finalize_grug_release(
    root: Path,
    config: dict[str, object],
    tokenizer: PreTrainedTokenizerBase,
    options: ReleaseExportOptions,
) -> ReleaseRequest:
    """Validate and complete a Grug release after Bridge writes its HF shards."""
    canonical = canonical_grug_config(config)
    policy = ReleasePolicy(options.precision)
    expected = _expected_grug_shapes(canonical)
    specs = grug_tensor_specs(canonical, _bridge_tensor_specs(root, expected))
    for name, spec in specs.items():
        policy.target_dtype(name, spec)
    scalar_dtype = policy.config_dtype(specs)
    if scalar_dtype is None:
        canonical.pop("dtype", None)
    else:
        canonical["dtype"] = scalar_dtype
    canonical = json.loads(json.dumps(canonical))
    explicit = {"generation_controls": options.generation_controls}
    if options.eos_token_ids is not None:
        explicit["eos_token_ids"] = options.eos_token_ids
    if options.chat_template is not None:
        explicit["chat_template"] = options.chat_template
    checkpoint = {
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "chat_template": tokenizer.chat_template,
        "special_token_ids": tokenizer_special_token_ids(tokenizer),
    }
    metadata, generation = resolve_metadata(explicit, checkpoint, {}, require_chat_template=True)
    assert metadata.chat_template is not None
    canonical["eos_token_id"] = metadata.eos_token_id
    if metadata.bos_token_id is not None:
        canonical["bos_token_id"] = metadata.bos_token_id
    if "pad_token" in metadata.special_token_ids:
        canonical["pad_token_id"] = metadata.special_token_ids["pad_token"]
    request = ReleaseRequest(
        policy=policy,
        schema_id=f"grug-moe-v{canonical['grugmoe_artifact_schema_version']}",
        tensors=specs,
        config=canonical,
        generation=generation,
        tokenizer=metadata,
        source_checkpoint=options.source_checkpoint,
        source_step=options.source_step,
        checkpoint_identity=options.checkpoint_identity,
        source_model=options.source_model,
        source_model_commit=options.source_model_commit,
        exporter_revision=exporter_revision(Path(__file__)),
    )
    persist_request(root, request)
    _rewrite_precision(root, specs, policy)
    _write_index(root)
    tokenizer.chat_template = metadata.chat_template
    tokenizer.save_pretrained(root)
    normalize_fast_tokenizer_metadata(root)
    tokenizer_config_path = root / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config["chat_template"] = metadata.chat_template
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, sort_keys=True, indent=2) + "\n")
    (root / "chat_template.jinja").write_text(metadata.chat_template)
    (root / "config.json").write_text(json.dumps(canonical, sort_keys=True, indent=2) + "\n")
    (root / "generation_config.json").write_text(json.dumps(generation, sort_keys=True, indent=2) + "\n")
    hashes = validate_release_weights(root, specs, policy)
    hashes.update(validate_release_metadata(root, request))
    write_completion_manifest(root, request, hashes)
    return request
