# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Investigate LoRA-derived HF exports for Marin vLLM serving failures."""

import argparse
import json
import logging
import os
import struct
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass

import fsspec
import jax
from transformers import AutoModelForCausalLM
from transformers import LlamaConfig as HfLlamaConfig

from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import _patch_hf_hub_download, load_tokenizer
from levanter.lora import LoraConfig, combine_lora_params, loraize, partition_lora_params, save_merged_hf_model
from levanter.models.llama import LlamaConfig
from levanter.utils.jax_utils import local_cpu_mesh

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlamaShapeConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @classmethod
    def from_json(cls, payload: dict) -> "LlamaShapeConfig":
        return cls(
            vocab_size=int(payload["vocab_size"]),
            hidden_size=int(payload["hidden_size"]),
            intermediate_size=int(payload["intermediate_size"]),
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            num_key_value_heads=int(payload["num_key_value_heads"]),
            head_dim=int(payload["head_dim"]),
        )


def _read_json(path: str) -> dict:
    with fsspec.open(path, "r") as handle:
        return json.load(handle)


def _read_text(path: str) -> str:
    with fsspec.open(path, "r") as handle:
        return handle.read()


def _read_optional_json(path: str) -> dict | None:
    fs, plain_path = fsspec.core.url_to_fs(path)
    if not fs.exists(plain_path):
        return None
    return _read_json(path)


def _read_safetensors_header(path: str) -> dict:
    with fsspec.open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = handle.read(header_len)
    return json.loads(header.decode("utf-8"))


def _iter_weight_files(index_payload: dict, base_path: str) -> Iterable[str]:
    seen = set()
    for shard_name in index_payload["weight_map"].values():
        if shard_name in seen:
            continue
        seen.add(shard_name)
        yield f"{base_path}/{shard_name}"


def _expected_shape(config: LlamaShapeConfig, tensor_name: str) -> tuple[int, ...] | None:
    if tensor_name == "model.embed_tokens.weight":
        return (config.vocab_size, config.hidden_size)
    if tensor_name == "lm_head.weight":
        return (config.vocab_size, config.hidden_size)
    if tensor_name == "model.norm.weight":
        return (config.hidden_size,)

    parts = tensor_name.split(".")
    if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
        return None

    suffix = ".".join(parts[3:])
    if suffix in {"input_layernorm.weight", "post_attention_layernorm.weight"}:
        return (config.hidden_size,)
    if suffix == "self_attn.q_proj.weight":
        return (config.num_attention_heads * config.head_dim, config.hidden_size)
    if suffix in {"self_attn.k_proj.weight", "self_attn.v_proj.weight"}:
        return (config.num_key_value_heads * config.head_dim, config.hidden_size)
    if suffix == "self_attn.o_proj.weight":
        return (config.hidden_size, config.hidden_size)
    if suffix in {"mlp.gate_proj.weight", "mlp.up_proj.weight"}:
        return (config.intermediate_size, config.hidden_size)
    if suffix == "mlp.down_proj.weight":
        return (config.hidden_size, config.intermediate_size)
    return None


def _copy_if_exists(source_path: str, destination_path: str) -> bool:
    source_fs, source_plain = fsspec.core.url_to_fs(source_path)
    if not source_fs.exists(source_plain):
        return False

    destination_fs, destination_plain = fsspec.core.url_to_fs(destination_path)
    destination_dir = os.path.dirname(destination_plain)
    if destination_dir:
        destination_fs.makedirs(destination_dir, exist_ok=True)

    with (
        source_fs.open(source_plain, "rb") as source_handle,
        destination_fs.open(destination_plain, "wb") as dest_handle,
    ):
        dest_handle.write(source_handle.read())
    return True


def _write_json(path: str, payload: dict) -> None:
    with fsspec.open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def sync_chat_template_into_tokenizer_config(model_path: str) -> bool:
    """Embed `chat_template.jinja` into `tokenizer_config.json` when present."""
    tokenizer_config_path = f"{model_path}/tokenizer_config.json"
    tokenizer_config = _read_optional_json(tokenizer_config_path)
    if tokenizer_config is None:
        logger.warning("No tokenizer_config.json found at %s", tokenizer_config_path)
        return False

    chat_template_path = f"{model_path}/chat_template.jinja"
    chat_fs, chat_plain_path = fsspec.core.url_to_fs(chat_template_path)
    if not chat_fs.exists(chat_plain_path):
        logger.info("No chat_template.jinja found at %s", chat_template_path)
        return False

    chat_template = _read_text(chat_template_path)
    if tokenizer_config.get("chat_template") == chat_template:
        logger.info("tokenizer_config.json already embeds chat_template at %s", tokenizer_config_path)
        return False

    tokenizer_config["chat_template"] = chat_template
    _write_json(tokenizer_config_path, tokenizer_config)
    logger.info("Embedded chat_template into %s", tokenizer_config_path)
    return True


def _stage_model_dir_locally(path: str) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    protocol, _ = fsspec.core.split_protocol(path)
    if protocol in (None, "file"):
        return path, None

    fs, plain_path = fsspec.core.url_to_fs(path)
    staging_dir = tempfile.TemporaryDirectory()
    logger.info("Staging remote model directory %s to %s", path, staging_dir.name)
    fs.get(os.path.join(plain_path, "*"), staging_dir.name, recursive=True)
    return staging_dir.name, staging_dir


def _build_lora_config_and_model_config(
    model_path: str,
    *,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_target_modules: list[str] | None,
) -> tuple[LlamaConfig, LoraConfig]:
    config_payload = _read_json(f"{model_path}/config.json")
    hf_config = HfLlamaConfig.from_dict(config_payload)
    model_config = LlamaConfig.from_hf_config(hf_config)
    lora_config = LoraConfig(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=lora_target_modules,
    )
    return model_config, lora_config


def run_hf_load(model_path: str) -> None:
    local_model_path, staging_dir = _stage_model_dir_locally(model_path)
    try:
        logger.info("Loading tokenizer from %s", local_model_path)
        tokenizer = load_tokenizer(local_model_path)
        logger.info("Tokenizer class: %s", type(tokenizer).__name__)

        logger.info("Loading model from %s with plain Transformers", local_model_path)
        with _patch_hf_hub_download():
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
    finally:
        if staging_dir is not None:
            staging_dir.cleanup()

    param_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Model class: %s", type(model).__name__)
    logger.info("Parameter count: %s", param_count)
    logger.info("Config type: %s", type(model.config).__name__)

    for name, parameter in list(model.named_parameters())[:10]:
        logger.info("Param preview: %s %s", name, tuple(parameter.shape))


def run_shape_check(model_path: str) -> None:
    config_payload = _read_json(f"{model_path}/config.json")
    index_payload = _read_json(f"{model_path}/model.safetensors.index.json")
    config = LlamaShapeConfig.from_json(config_payload)

    mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    unchecked: list[str] = []
    tensor_count = 0

    for shard_path in _iter_weight_files(index_payload, model_path):
        logger.info("Reading safetensors header: %s", shard_path)
        metadata = _read_safetensors_header(shard_path)
        for tensor_name, tensor_meta in metadata.items():
            if tensor_name == "__metadata__":
                continue
            tensor_count += 1
            actual_shape = tuple(int(x) for x in tensor_meta["shape"])
            expected = _expected_shape(config, tensor_name)
            if expected is None:
                unchecked.append(tensor_name)
                continue
            if actual_shape != expected:
                mismatches.append((tensor_name, expected, actual_shape))

    logger.info("Checked %d tensors", tensor_count)
    logger.info("Unchecked tensors: %d", len(unchecked))
    if unchecked:
        for tensor_name in unchecked[:20]:
            logger.info("Unchecked tensor: %s", tensor_name)

    if mismatches:
        for tensor_name, expected, actual in mismatches:
            logger.error("Shape mismatch: %s expected=%s actual=%s", tensor_name, expected, actual)
        raise RuntimeError(f"Found {len(mismatches)} shape mismatches")

    logger.info("No config-derived shape mismatches found")


def run_reexport_merged(
    *,
    source_hf_path: str,
    raw_checkpoint_path: str,
    checkpoint_subpath: str,
    output_path: str,
    base_model_ref: str,
    tokenizer_path: str,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_target_modules: list[str] | None,
    verify_shapes: bool,
) -> None:
    logger.info("Building LoRA exemplar from %s", source_hf_path)
    model_config, lora_config = _build_lora_config_and_model_config(
        source_hf_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )

    generation_config = _read_optional_json(f"{source_hf_path}/generation_config.json")
    tokenizer = load_tokenizer(tokenizer_path)
    converter = model_config.hf_checkpoint_converter(ref_checkpoint=base_model_ref).replaced(tokenizer=tokenizer)

    logger.info("Loading base model from %s", base_model_ref)
    with local_cpu_mesh():
        base_model = converter.load_pretrained(
            model_config.model_type,
            ref=base_model_ref,
            config=model_config,
        )
        lora_model = loraize(base_model, lora_config, key=jax.random.PRNGKey(1))
        base_params, lora_params = partition_lora_params(lora_model)

        logger.info("Loading LoRA checkpoint subtree %s from %s", checkpoint_subpath, raw_checkpoint_path)
        loaded_lora_params = load_checkpoint(
            lora_params,
            raw_checkpoint_path,
            subpath=checkpoint_subpath,
            discover_latest=False,
        )
        loaded_lora_model = combine_lora_params(base_params, loaded_lora_params)

        save_merged_hf_model(
            loaded_lora_model,
            converter,
            output_path,
            generation_config=generation_config,
            save_reference_code=False,
            save_tokenizer=True,
        )

    copied = _copy_if_exists(f"{source_hf_path}/chat_template.jinja", f"{output_path}/chat_template.jinja")
    if copied:
        logger.info("Copied chat_template.jinja to %s", output_path)
    sync_chat_template_into_tokenizer_config(output_path)

    if verify_shapes:
        logger.info("Running post-export shape audit for %s", output_path)
        run_shape_check(output_path)


def run_shape_inventory(root_path: str) -> None:
    fs, path = fsspec.core.url_to_fs(root_path)
    entries = sorted(fs.ls(path, detail=False))
    passed: list[str] = []
    failed: list[str] = []

    for entry in entries:
        model_path = entry if entry.startswith("gs://") else f"gs://{entry}"
        logger.info("Inventory checking %s", model_path)
        try:
            run_shape_check(model_path)
        except Exception:
            logger.exception("Inventory failure for %s", model_path)
            failed.append(model_path)
            continue
        passed.append(model_path)

    logger.info("Inventory complete: %d passed, %d failed", len(passed), len(failed))
    for model_path in passed:
        logger.info("INVENTORY PASS %s", model_path)
    for model_path in failed:
        logger.info("INVENTORY FAIL %s", model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF checkpoint path, including gs:// object-store paths.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("hf-load", help="Load tokenizer and model with plain Transformers.")
    subparsers.add_parser("shape-check", help="Inspect safetensors headers and compare shapes to config.")
    inventory_parser = subparsers.add_parser(
        "inventory-shape-check", help="Run shape-check across every immediate child under a root path."
    )
    inventory_parser.add_argument("--root-path", required=True, help="Root directory containing per-step HF exports.")
    reexport_parser = subparsers.add_parser(
        "reexport-merged", help="Re-export a merged HF checkpoint from raw LoRA trainer state."
    )
    reexport_parser.add_argument(
        "--raw-checkpoint-path", required=True, help="Raw Levanter checkpoint path, e.g. checkpoints/step-1699."
    )
    reexport_parser.add_argument(
        "--checkpoint-subpath",
        default="model",
        help="Subtree within the raw checkpoint to load. For adapter-only DPO LoRA checkpoints this is usually model.",
    )
    reexport_parser.add_argument("--output-path", required=True, help="Destination HF export path.")
    reexport_parser.add_argument(
        "--base-model-ref", required=True, help="Base HF model ID/path used to initialize the LoRA run."
    )
    reexport_parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer source path. Defaults to --model-path if omitted.",
    )
    reexport_parser.add_argument("--lora-r", type=int, required=True, help="LoRA rank used during training.")
    reexport_parser.add_argument("--lora-alpha", type=float, required=True, help="LoRA alpha used during training.")
    reexport_parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout used during training.")
    reexport_parser.add_argument(
        "--lora-target-module",
        action="append",
        default=None,
        help="Optional LoRA target module suffix. Repeat for multiple values. Omit to target all compatible linears.",
    )
    reexport_parser.add_argument(
        "--verify-shapes",
        action="store_true",
        help="Run the config-derived shape audit on the output export after writing it.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.command == "hf-load":
        run_hf_load(args.model_path)
        return 0
    if args.command == "shape-check":
        run_shape_check(args.model_path)
        return 0
    if args.command == "inventory-shape-check":
        run_shape_inventory(args.root_path)
        return 0
    if args.command == "reexport-merged":
        tokenizer_path = args.tokenizer_path or args.model_path
        run_reexport_merged(
            source_hf_path=args.model_path,
            raw_checkpoint_path=args.raw_checkpoint_path,
            checkpoint_subpath=args.checkpoint_subpath,
            output_path=args.output_path,
            base_model_ref=args.base_model_ref,
            tokenizer_path=tokenizer_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_module,
            verify_shapes=args.verify_shapes,
        )
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
