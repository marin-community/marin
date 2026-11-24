# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score fixed token histories with the matching JAX training implementation."""

import argparse
import hashlib
import inspect
import json
import logging
import os
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import draccus
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from haliax import Axis
from iris.jax.init import initialize_jax
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballLMHeadModel
from levanter.utils.jax_utils import leaf_key_paths
from marin.inference.backend import ModelSpec
from marin.inference.config import LevanterEngineConfig
from marin.inference.levanter_backend import LevanterBackend
from rigging.filesystem.cross_region import record_transfer
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.export_cpu import CpuCheckpointWeights, CpuExportConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.sample import next_logits

from .snowball_routers import restore as restore_snowball_routers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    case_id: str
    token_ids: tuple[int, ...]
    position: int


def read_predictions(path: Path) -> tuple[Prediction, ...]:
    """Expand supplied positions without generating continuation tokens."""
    cases = json.loads(path.read_text())["cases"]
    return tuple(
        Prediction(case["id"], tuple(case["token_ids"]), position) for case in cases for position in case["positions"]
    )


def padded_batch(predictions: tuple[Prediction, ...], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Repeat the last row to fill the mesh; the caller discards padding rows."""
    assert 0 < len(predictions) <= batch_size
    length = max(len(row.token_ids) for row in predictions)
    rows = predictions + (predictions[-1],) * (batch_size - len(predictions))
    tokens = np.zeros((batch_size, length), dtype=np.int32)
    positions = np.empty(batch_size, dtype=np.int32)
    for index, row in enumerate(rows):
        assert 0 <= row.position < len(row.token_ids)
        tokens[index, : len(row.token_ids)] = row.token_ids
        positions[index] = row.position
    return tokens, positions


def hero_config(request: dict, attention: str, moe: str) -> GrugModelConfig:
    config = request["spec"]["model"] | {"attention_implementation": attention, "moe_implementation": moe}
    return draccus.decode(GrugModelConfig, config)


def restore_hero_bf16(request: dict, config: GrugModelConfig, mesh: jax.sharding.Mesh) -> Transformer:
    """Restore BF16 checkpoint weights and the export's pending router-bias update."""
    checkpoint = request["checkpoint"]
    reader = CpuCheckpointWeights(
        CpuExportConfig(
            checkpoint=checkpoint["uri"],
            metadata_digest=checkpoint["metadata_digest"],
            model=config,
            destination="",
            source_revision=request["source_revision"],
        )
    )
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.PRNGKey(0))
    paths = jax.tree.leaves(leaf_key_paths(template))
    pending = -reader.pending.read().result().astype(np.float32)
    pending -= pending.mean(axis=-1, keepdims=True)

    def load(path: str, exemplar: jax.ShapeDtypeStruct) -> jax.Array:
        path = path.replace(".", "/")
        store = reader.stores[path]
        assert isinstance(exemplar.sharding, NamedSharding), (path, exemplar.sharding)
        sharding = NamedSharding(mesh, exemplar.sharding.spec)

        def read(index):
            if path == "stacked_blocks/stacked/mlp/router_bias":
                return pending[index].astype(jnp.bfloat16)
            selection = store[index]
            record_transfer(selection.size * np.dtype(store.dtype.numpy_dtype).itemsize, checkpoint["uri"])
            return ts.cast(selection, ts.bfloat16).read().result()

        logger.info("Restore %s %s %s", path, exemplar.shape, sharding.spec)
        return jax.make_array_from_callback(exemplar.shape, sharding, read, dtype=jnp.bfloat16)

    leaves = [load(path, exemplar) for path, exemplar in zip(paths, jax.tree.leaves(template), strict=True)]
    model = jax.tree.unflatten(jax.tree.structure(template), leaves)
    jax.block_until_ready(model)
    return model


@hax.named_jit
def snowball_logprobs(model: SnowballLMHeadModel, tokens: hax.NamedArray, positions: hax.NamedArray) -> jax.Array:
    hidden = model.activations(tokens).rearrange(("batch", "position", "embed")).array
    last = hidden.at[jnp.arange(hidden.shape[0]), positions.array].get(out_sharding=P("data"))
    logits = jnp.einsum("bh,hv->bv", last, model.transformer.output_proj, out_sharding=P("data"))
    return jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)


@eqx.filter_jit
def hero_logits_and_overflow(model: Transformer, tokens: jax.Array, positions: jax.Array):
    hidden, stats = model(tokens)
    last = hidden.at[jnp.arange(tokens.shape[0]), positions].get(out_sharding=P())
    logits = jnp.einsum("bh,hv->bv", last, model.output_proj, preferred_element_type=jnp.float32)
    return jax.sharding.reshard(logits, P()), stats["capacity_overflow_per_layer"]


def score_hero(
    predictions: tuple[Prediction, ...], request: dict, config: GrugModelConfig, expert_parallel_size: int = 1
) -> np.ndarray:
    mesh = compact_grug_mesh(expert_axis_size=expert_parallel_size, replica_axis_size=1)
    batch_size = jax.device_count()
    rows = []
    with jax.set_mesh(mesh):
        model = restore_hero_bf16(request, config, mesh)
        sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert")))
        for start in range(0, len(predictions), batch_size):
            batch = predictions[start : start + batch_size]
            tokens, positions = padded_batch(batch, batch_size)
            ids = jax.make_array_from_callback(tokens.shape, sharding, lambda index, value=tokens: value[index])
            last = jax.make_array_from_callback(positions.shape, sharding, lambda index, value=positions: value[index])
            if expert_parallel_size > 1:
                logits, overflow = hero_logits_and_overflow(model, ids, last)
                assert not np.any(np.asarray(overflow)), "Reference dropped MoE assignments"
            else:
                logits = next_logits(model, ids, last)
            rows.append(np.asarray(jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1))[: len(batch)])
    return np.concatenate(rows)


def score_snowball(
    predictions: tuple[Prediction, ...], weights: str, revision: str, attention: str, moe: str, router_weights: Path
) -> np.ndarray:
    spec = ModelSpec(
        weights=weights,
        revision=None if "://" in weights else revision,
        tokenizer_revision=revision,
        api_model="snowball-reference",
        num_chips=jax.device_count(),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=max(len(row.token_ids) for row in predictions),
        chat_template_content=None,
    )
    rows = []
    batch_size = jax.device_count()
    with LevanterBackend(LevanterEngineConfig()).load_model(
        spec, config_overrides={"attention_implementation": attention, "moe_implementation": moe}
    ) as loaded:
        assert isinstance(loaded.model, SnowballLMHeadModel)
        model = restore_snowball_routers(loaded.model, router_weights, weights, revision)
        for start in range(0, len(predictions), batch_size):
            batch = predictions[start : start + batch_size]
            tokens, positions = padded_batch(batch, batch_size)
            axes = (Axis("batch", batch_size), Axis("position", tokens.shape[1]))
            with hax.axis_mapping(loaded.trainer.compute_axis_mapping):
                scores = snowball_logprobs(
                    model,
                    hax.named(jnp.asarray(tokens), axes),
                    hax.named(jnp.asarray(positions), (axes[0],)),
                )
            rows.append(np.asarray(scores)[: len(batch)])
    return np.concatenate(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["hero", "snowball"])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--weights")
    parser.add_argument("--revision")
    parser.add_argument("--router-weights", type=Path)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--moe", default="sonic")
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--single-host", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    jax.config.update("jax_default_matmul_precision", "highest")
    if not args.single_host:
        # A one-shot validation job must let Iris collect outputs before completion.
        initialize_jax()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / f"reference-start-{jax.process_index()}.json").write_text(
        json.dumps({"arguments": {key: str(value) for key, value in vars(args).items()}, "devices": str(jax.devices())})
        + "\n"
    )
    assert jax.default_backend() == "gpu"
    predictions = read_predictions(args.cases)
    if args.model == "hero":
        assert args.export_manifest is not None
        request = json.loads(args.export_manifest.read_text())["request"]
        config = hero_config(request, args.attention, args.moe)
        if args.expert_parallel_size > 1:
            # A reference must evaluate every selected expert, as the local path does.
            config = replace(
                config,
                capacity_factor=max(config.capacity_factor, float(args.expert_parallel_size)),
                report_capacity_overflow=True,
            )
        scores = score_hero(predictions, request, config, args.expert_parallel_size)
        weights = request["checkpoint"]
    else:
        assert args.weights and args.revision and args.router_weights
        scores = score_snowball(predictions, args.weights, args.revision, args.attention, args.moe, args.router_weights)
        weights = {"uri": args.weights, "revision": None if "://" in args.weights else args.revision}
        if "://" in args.weights:
            weights["source_revision"] = args.revision
    assert np.isfinite(scores).all()
    if jax.process_index() == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        np.save(args.output / "logprobs.npy", scores)
        metadata = {
            "model": args.model,
            "jax": jax.__version__,
            "devices": str(jax.devices()),
            "device_kind": sorted({device.device_kind for device in jax.devices()}),
            "node": os.environ.get("IRIS_NODE_NAME"),
            "bundle_id": os.environ.get("IRIS_BUNDLE_ID"),
            "process_count": jax.process_count(),
            "expert_parallel_size": args.expert_parallel_size if args.model == "hero" else None,
            "precision": {
                "weights": "bfloat16",
                "hidden": "bfloat16",
                "lm_projection_output": "float32" if args.model == "hero" else "bfloat16",
                "log_softmax": "float32",
                "router_weights": "bfloat16" if args.model == "hero" else "float32",
                "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            },
            "case_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "rows": [
                {
                    "case_id": row.case_id,
                    "position": row.position,
                    "history_sha256": (
                        hashlib.sha256(
                            np.asarray(row.token_ids[: row.position + 1], dtype=np.int32).tobytes()
                        ).hexdigest()
                    ),
                }
                for row in predictions
            ],
            "weights": weights,
            "attention": args.attention,
            "moe": args.moe,
            "router_weights_sha256": (
                hashlib.sha256(args.router_weights.read_bytes()).hexdigest() if args.model == "snowball" else None
            ),
            "source_sha256": {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in (
                    ("hero_model", inspect.getfile(Transformer)),
                    ("snowball_model", inspect.getfile(SnowballLMHeadModel)),
                    ("snowball_routers", inspect.getfile(restore_snowball_routers)),
                    ("reference", __file__),
                )
            },
        }
        (args.output / "scores.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if jax.distributed.is_initialized():
        sync_global_devices("reference-output-written")
        jax.distributed.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if output_dir := os.environ.get("IRIS_OUTPUT_DIR"):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "reference-error.txt").write_text(traceback.format_exc())
        raise
