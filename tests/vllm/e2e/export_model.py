# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared June 67B BF16 Hugging Face export implementation."""

import dataclasses
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import safetensors
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer
from rigging.filesystem import StoragePath, url_to_fs
from rigging.timing import retry_with_backoff

from experiments.grug.moe.model import GrugModelConfig, Transformer

from .june_67b import (
    VendoredTransformer,
    apply_pending_qb_betas,
    decode_vendored_config,
    load_checkpoint,
    read_executor_info,
)
from .reference import EXPORT_TREE_SHA256, storage_tree_sha256, tree_sha256


def _decode_main_config(model_config: dict[str, Any]) -> GrugModelConfig:
    main_fields = {field.name for field in dataclasses.fields(GrugModelConfig)}
    return draccus.decode(
        GrugModelConfig,
        {name: value for name, value in model_config.items() if name in main_fields},
    )


def _to_main_model(params: VendoredTransformer, config: GrugModelConfig) -> Transformer:
    assert params.stacked_blocks is not None
    source = cast(Any, params)
    return Transformer(
        token_embed=source.token_embed,
        embed_norm=source.embed_norm,
        embed_gated_norm=source.embed_gated_norm,
        output_proj=source.output_proj,
        blocks=tuple(source.stacked_blocks.unstacked()),
        final_norm=source.final_norm,
        final_gated_norm=source.final_gated_norm,
        config=config,
    )


def _assert_vllm_bf16(export_dir: Path) -> None:
    exported_config = json.loads((export_dir / "config.json").read_text())
    assert exported_config["architectures"] == ["GrugMoeForCausalLM"]
    assert exported_config["model_type"] == "grug_moe"
    assert exported_config["dtype"] == "bfloat16"

    tensor_dtypes: set[str] = set()
    for shard_path in export_dir.glob("model-*.safetensors"):
        with safetensors.safe_open(shard_path, framework="numpy") as tensors:
            tensor_dtypes.update(tensors.get_slice(name).get_dtype() for name in tensors.keys())
    assert tensor_dtypes == {"BF16"}


def _copy_storage_tree(staging_uri: str, publish_uri: str) -> None:
    staging_root = StoragePath(staging_uri)
    publish_root = StoragePath(publish_uri)
    if staging_root.scheme != publish_root.scheme:
        raise ValueError("staging_uri and publish_uri must use the same filesystem")
    publish_root.mkdirs(exist_ok=True)
    sources = [
        directory / filename for directory, _subdirectories, filenames in staging_root.walk() for filename in filenames
    ]

    def publish(source: StoragePath) -> None:
        destination = publish_root / source.relative_to(staging_root)
        destination.parent.mkdirs(exist_ok=True)
        source_fs, source_path = url_to_fs(str(source))
        destination_fs, destination_path = url_to_fs(str(destination))
        if type(source_fs) is not type(destination_fs):
            raise ValueError("Source and destination must use the same filesystem")
        retry_with_backoff(
            lambda: source_fs.copy(source_path, destination_path),
            max_attempts=4,
            max_elapsed=120,
            operation=f"publish {source} -> {destination}",
        )

    # atomic_rename handles one object. This immutable sharded tree instead uses
    # idempotent server-side copies and becomes visible only when its marker is written.
    with ThreadPoolExecutor(max_workers=min(16, len(sources))) as executor:
        list(executor.map(publish, sources))


def _save_verified_remote_export(
    converter: Any,
    export_model: Transformer,
    *,
    staging_uri: str,
    publish_uri: str,
    completion_uri: str,
) -> str:
    converter.save_pretrained(export_model, staging_uri, dtype=jnp.bfloat16)
    actual_sha256 = storage_tree_sha256(staging_uri)
    assert actual_sha256 == EXPORT_TREE_SHA256, actual_sha256
    _copy_storage_tree(staging_uri, publish_uri)
    StoragePath(completion_uri).write_text(f"{actual_sha256}\n")
    return actual_sha256


def _save_verified_local_export(converter: Any, export_model: Transformer) -> str:
    with tempfile.TemporaryDirectory(prefix="june-67b-bf16-export-") as export_dir_str:
        export_dir = Path(export_dir_str)
        converter.save_pretrained(export_model, export_dir_str, dtype=jnp.bfloat16)
        _assert_vllm_bf16(export_dir)
        actual_sha256 = tree_sha256(export_dir)
        assert actual_sha256 == EXPORT_TREE_SHA256, actual_sha256
        return actual_sha256


def export_checkpoint_bf16(
    *,
    executor_info_path: str,
    checkpoint_path: str,
    staging_uri: str | None = None,
    publish_uri: str | None = None,
    completion_uri: str | None = None,
) -> str:
    executor_info = read_executor_info(executor_info_path)
    model_config = executor_info["config"]["model"]
    vendored_config = decode_vendored_config(executor_info)
    main_config = _decode_main_config(model_config)
    tokenizer_name = executor_info["config"]["data"]["tokenizer"]

    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        params, pending_qb_betas = load_checkpoint(vendored_config, mesh, checkpoint_path)
        params = apply_pending_qb_betas(params, pending_qb_betas)
        del pending_qb_betas

        params = jax.tree.map(
            lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
            params,
        )
        jax.block_until_ready(params)

        tokenizer = load_tokenizer(tokenizer_name)
        converter = (
            main_config.hf_checkpoint_converter()
            .replaced(tokenizer=tokenizer)
            .with_config_overrides({"dtype": "bfloat16"})
        )
        export_model = _to_main_model(params, main_config)

        if staging_uri is not None:
            if publish_uri is None or completion_uri is None:
                raise ValueError("publish_uri and completion_uri are required with staging_uri")
            return _save_verified_remote_export(
                converter,
                export_model,
                staging_uri=staging_uri,
                publish_uri=publish_uri,
                completion_uri=completion_uri,
            )
        return _save_verified_local_export(converter, export_model)
