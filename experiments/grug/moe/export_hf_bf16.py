# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a native array-stacked Grug checkpoint as vLLM-compatible BF16 HF shards."""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import Any, cast

import click
import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.partitioning import set_mesh
from levanter.checkpoint import load_checkpoint as load_levanter_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer
from rigging.filesystem.cluster_config import check_gcs_paths_same_region
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as ArrayStackedGrugModelConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as ArrayStackedTransformer

logger = logging.getLogger(__name__)

_NON_ARCHITECTURAL_MODEL_FIELDS = frozenset({"hybrid_attention_flops_accounting"})


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_path: str
    output_path: str
    max_seq_len: int
    qk_mult: float
    executor_info_path: str | None = None

    @property
    def resolved_executor_info_path(self) -> str:
        if self.executor_info_path is not None:
            return self.executor_info_path
        run_root, separator, _ = self.checkpoint_path.rstrip("/").partition("/checkpoints/")
        if not separator:
            raise ValueError("checkpoint_path must contain '/checkpoints/' when executor_info_path is omitted")
        return f"{run_root}/.executor_info"


@dataclass(frozen=True)
class ExportPlan:
    config: ExportConfig
    tokenizer: str
    checkpoint_config: ArrayStackedGrugModelConfig
    inference_config: GrugModelConfig


def _gcs_bucket(path: str) -> str:
    parsed = StoragePath.parse(path)
    if parsed.scheme != "gs" or parsed.bucket is None:
        raise ValueError(f"Grug export paths must be GCS URLs, got {path!r}")
    return parsed.bucket


def _validate_same_bucket(config: ExportConfig) -> None:
    paths = (config.checkpoint_path, config.resolved_executor_info_path, config.output_path)
    buckets = {_gcs_bucket(path) for path in paths}
    if len(buckets) != 1:
        raise ValueError(f"checkpoint, executor metadata, and output must use one GCS bucket, got {sorted(buckets)}")


def _decode_checkpoint_config(raw_model_config: dict[str, Any]) -> ArrayStackedGrugModelConfig:
    checkpoint_fields = {field.name for field in dataclasses.fields(ArrayStackedGrugModelConfig)}
    unsupported = set(raw_model_config) - checkpoint_fields - _NON_ARCHITECTURAL_MODEL_FIELDS
    if unsupported:
        raise ValueError(f"checkpoint model config has unsupported fields: {sorted(unsupported)}")
    return draccus.decode(
        ArrayStackedGrugModelConfig,
        {name: value for name, value in raw_model_config.items() if name in checkpoint_fields},
    )


def inference_config(
    raw_model_config: dict[str, Any],
    *,
    max_seq_len: int,
    qk_mult: float,
) -> GrugModelConfig:
    inference_fields = {field.name for field in dataclasses.fields(GrugModelConfig)}
    decoded = draccus.decode(
        GrugModelConfig,
        {name: value for name, value in raw_model_config.items() if name in inference_fields},
    )
    return dataclasses.replace(decoded, max_seq_len=max_seq_len, qk_mult=qk_mult)


def export_plan(config: ExportConfig) -> ExportPlan:
    _validate_same_bucket(config)
    executor_info = json.loads(StoragePath(config.resolved_executor_info_path).read_text())
    raw_config = executor_info["config"]
    raw_model_config = raw_config["model"]
    tokenizer = raw_config["data"]["tokenizer"]
    return ExportPlan(
        config=config,
        tokenizer=tokenizer,
        checkpoint_config=_decode_checkpoint_config(raw_model_config),
        inference_config=inference_config(
            raw_model_config,
            max_seq_len=config.max_seq_len,
            qk_mult=config.qk_mult,
        ),
    )


def apply_pending_qb_betas(
    model: ArrayStackedTransformer,
    pending_qb_betas: jax.Array,
) -> ArrayStackedTransformer:
    """Apply the checkpoint's deferred QB router-bias update before export."""

    assert model.stacked_blocks is not None
    router_bias = -pending_qb_betas
    router_bias -= jnp.mean(router_bias, axis=-1, keepdims=True)
    return eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, model, router_bias)


def inference_model(
    checkpoint_model: ArrayStackedTransformer,
    config: GrugModelConfig,
) -> Transformer:
    """Adapt the array-stacked training model to the canonical HF exporter model."""

    assert checkpoint_model.stacked_blocks is not None
    source = cast(Any, checkpoint_model)
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


def save_hf_bf16(
    model: Transformer,
    config: GrugModelConfig,
    output_path: str,
    *,
    tokenizer: Any | None,
) -> None:
    converter = (
        config.hf_checkpoint_converter().replaced(tokenizer=tokenizer).with_config_overrides({"dtype": "bfloat16"})
    )
    converter.save_pretrained(
        model,
        output_path,
        save_tokenizer=tokenizer is not None,
        dtype=jnp.bfloat16,
    )


def export_checkpoint(config: ExportConfig) -> None:
    plan = export_plan(config)
    check_gcs_paths_same_region(config, local_ok=False)

    # Export is a single model replica. Use every chip for parameter sharding on
    # multi-host TPU slices instead of replicating the model once per host.
    mesh = compact_grug_mesh(replica_axis_size=1)
    with set_mesh(mesh):
        template = eqx.filter_eval_shape(
            ArrayStackedTransformer.init,
            plan.checkpoint_config,
            key=jax.random.PRNGKey(0),
        )
        checkpoint_state = load_levanter_checkpoint(
            {
                "params": template,
                "pending_qb_betas": jax.ShapeDtypeStruct(
                    (plan.checkpoint_config.num_layers, plan.checkpoint_config.num_experts),
                    jnp.float32,
                ),
            },
            config.checkpoint_path,
            mesh=mesh,
        )
        checkpoint_model = apply_pending_qb_betas(
            checkpoint_state["params"],
            checkpoint_state["pending_qb_betas"],
        )
        checkpoint_model = jax.tree.map(
            lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
            checkpoint_model,
        )
        jax.block_until_ready(checkpoint_model)

        tokenizer = load_tokenizer(plan.tokenizer)
        save_hf_bf16(
            inference_model(checkpoint_model, plan.inference_config),
            plan.inference_config,
            config.output_path,
            tokenizer=tokenizer,
        )

    exported_config = json.loads(StoragePath(f"{config.output_path.rstrip('/')}/config.json").read_text())
    if exported_config.get("max_position_embeddings") != config.max_seq_len:
        raise ValueError("exported config.json has the wrong max_position_embeddings")
    if exported_config.get("qk_mult") != config.qk_mult:
        raise ValueError("exported config.json has the wrong qk_mult")


@click.command()
@click.option("--checkpoint-path", required=True)
@click.option("--output-path", required=True)
@click.option("--max-seq-len", required=True, type=click.IntRange(min=1))
@click.option("--qk-mult", required=True, type=click.FloatRange(min=0.0, min_open=True))
@click.option("--executor-info-path", default=None)
@click.option("--dry-run", is_flag=True, help="Resolve and validate the export without loading checkpoint weights.")
def cli(
    checkpoint_path: str,
    output_path: str,
    max_seq_len: int,
    qk_mult: float,
    executor_info_path: str | None,
    dry_run: bool,
) -> None:
    config = ExportConfig(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        max_seq_len=max_seq_len,
        qk_mult=qk_mult,
        executor_info_path=executor_info_path,
    )
    if dry_run:
        plan = export_plan(config)
        click.echo(
            json.dumps(
                {
                    "checkpoint_path": config.checkpoint_path,
                    "executor_info_path": config.resolved_executor_info_path,
                    "output_path": config.output_path,
                    "tokenizer": plan.tokenizer,
                    "max_seq_len": plan.inference_config.max_seq_len,
                    "qk_mult": plan.inference_config.qk_mult,
                },
                sort_keys=True,
            )
        )
        return
    export_checkpoint(config)


if __name__ == "__main__":
    cli()
