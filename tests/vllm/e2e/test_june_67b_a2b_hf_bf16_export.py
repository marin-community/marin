# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the June 67B A2B BF16 export and verify its persisted digest.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/vllm/e2e/test_june_67b_a2b_hf_bf16_export.py -o addopts= -vv -s
"""

import dataclasses
import hashlib
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any, cast

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import safetensors
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from haliax.partitioning import set_mesh
from iris.client import IrisClient
from iris.rpc import job_pb2
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer

from experiments.grug.moe.model import GrugModelConfig, Transformer

from .june_67b_a2b import (
    JUNE_67B_A2B,
    VendoredTransformer,
    apply_pending_qb_betas,
    decode_vendored_config,
    load_checkpoint,
    read_executor_info,
)
from .remote_job import run_remote_test_job

PENDING_TIMEOUT = 5 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


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


def _tree_sha256(export_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in export_dir.rglob("*") if path.is_file()):
        digest.update(path.relative_to(export_dir).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as file:
            digest.update(hashlib.file_digest(file, "sha256").digest())
    return digest.hexdigest()


def assert_checkpoint_reproduces_bf16_export() -> None:
    executor_info = read_executor_info()
    model_config = executor_info["config"]["model"]
    vendored_config = decode_vendored_config(executor_info)
    main_config = _decode_main_config(model_config)
    tokenizer_name = executor_info["config"]["data"]["tokenizer"]

    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        params, pending_qb_betas = load_checkpoint(vendored_config, mesh)
        params = apply_pending_qb_betas(params, pending_qb_betas)
        del pending_qb_betas

        # Avoid keeping the full FP32 checkpoint alongside temporary BF16 export buffers in GPU memory.
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

        with tempfile.TemporaryDirectory(prefix="june-67b-bf16-export-") as export_dir_str:
            export_dir = Path(export_dir_str)
            converter.save_pretrained(
                export_model,
                export_dir_str,
                dtype=jnp.bfloat16,
            )
            _assert_vllm_bf16(export_dir)
            actual_sha256 = _tree_sha256(export_dir)
            assert actual_sha256 == JUNE_67B_A2B.export_sha256, actual_sha256


def test_h100_node_reproduces_persisted_vllm_bf16_export(marin_gpu_client: IrisClient) -> None:
    run_remote_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"june-67b-bf16-export-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(assert_checkpoint_reproduces_bf16_export),
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="512g", disk="256g"),
            environment=create_environment(extras=["gpu"], sync_packages=["marin-levanter"]),
            # These e2es are manually triggered and highly interactive, so they use production priority.
            # Routine or automated workloads should not copy this priority.
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
