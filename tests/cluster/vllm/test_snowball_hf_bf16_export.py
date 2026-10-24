# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate a Snowball BF16 release from its native checkpoint.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/cluster/vllm/test_snowball_hf_bf16_export.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s
"""

import dataclasses
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
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import validate_single_name_config
from levanter.tokenizers import load_tokenizer
from marin.testing.inference.snowball_checkpoint import (
    VendoredTransformer,
    apply_pending_qb_betas,
    decode_vendored_config,
    load_checkpoint,
    read_executor_info,
)

from experiments.grug.moe.model import GrugModelConfig, Transformer
from tests.cluster.conftest import MARIN_GPU_CLUSTER

PENDING_TIMEOUT = 5 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


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


def _assert_vllm_bf16(export_dir: Path, config: GrugModelConfig) -> None:
    exported_config = json.loads((export_dir / "config.json").read_text())
    assert exported_config["architectures"] == ["GrugMoeForCausalLM"]
    assert exported_config["model_type"] == "grug_moe"
    assert exported_config["dtype"] == "bfloat16"
    # Checked on the real config_overrides-merged artifact, which could otherwise reintroduce an alias.
    validate_single_name_config(exported_config, config)

    tensor_dtypes: dict[str, str] = {}
    for shard_path in export_dir.glob("model-*.safetensors"):
        with safetensors.safe_open(shard_path, framework="numpy") as tensors:
            tensor_dtypes.update({name: tensors.get_slice(name).get_dtype() for name in tensors.keys()})
    assert tensor_dtypes
    assert all(
        dtype == ("F32" if name.endswith(".mlp.router.bias") else "BF16") for name, dtype in tensor_dtypes.items()
    )
    assert (export_dir / "hf_release_manifest.json").is_file()


def assert_checkpoint_exports_bf16_release() -> None:
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

        # Keep the FP32 router state while reducing the other parameters' GPU footprint.
        assert params.stacked_blocks is not None
        router_bias = params.stacked_blocks.stacked.mlp.router_bias
        params = jax.tree.map(
            lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
            params,
        )
        params = eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, params, router_bias)
        jax.block_until_ready(params)

        tokenizer = load_tokenizer(tokenizer_name)
        converter = (
            main_config.hf_checkpoint_converter()
            .replaced(tokenizer=tokenizer)
            .with_config_overrides({"dtype": "bfloat16"})
        )
        export_model = _to_main_model(params, main_config)

        with tempfile.TemporaryDirectory(prefix="snowball-bf16-export-") as export_dir_str:
            export_dir = Path(export_dir_str)
            converter.save_pretrained(
                export_model,
                export_dir_str,
                release_precision="bf16",
            )
            _assert_vllm_bf16(export_dir, main_config)


def test_snowball_checkpoint_validates_bf16_release(marin_gpu_client: IrisClient, run_test_job) -> None:
    run_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"snowball-bf16-export-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(assert_checkpoint_exports_bf16_release),
            resources=ResourceConfig.with_gpu(
                "H100", count=8, cpu=64, ram="512g", disk="256g", target_cluster=MARIN_GPU_CLUSTER
            ),
            environment=create_environment(extras=["gpu"], sync_packages=["marin-core", "marin-levanter"]),
            # These e2es are manually triggered and highly interactive, so they use production priority.
            # Routine or automated workloads should not copy this priority.
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
