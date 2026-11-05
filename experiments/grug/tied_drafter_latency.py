# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen full-width Belay drafter restrictions against a saved EMA checkpoint."""

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.model import GrugModelConfig, Transformer
from experiments.grug.fast_track.train import build_train_loader
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.audit import linear_weight_equivalents
from experiments.grug.tied_drafter.calibration import MODEL_SOURCE_SHA256
from experiments.grug.tied_drafter.coordinates import coordinate_map, restrict_model
from experiments.grug.tied_drafter.runtime import sequence_losses
from experiments.grug.tied_drafter_activation import model as activation_model
from experiments.grug.tied_drafter_approximations import modify_blocks
from experiments.grug.tied_replica_sanity import output_difference
from experiments.grug.tied_replica_sizes import comparison, size_config

POSITIONS = (31, 255, 511, 1023, 1535, 2047, 3071, 4094)


@dataclass(frozen=True)
class Variant:
    name: str
    active_experts: int
    gain: float
    expert_width: int
    selection: str
    dispatch_dtype: str = "bfloat16"
    residual_width: int = 512
    approximation: str = "none"
    skip_layers: tuple[int, ...] = ()


VARIANTS = (
    Variant("identity", 8, 1.0, 384, "prefix"),
    Variant("fp8_dispatch", 8, 1.0, 384, "prefix", "float8_e4m3"),
    *(Variant(f"k{k}_gain{gain:g}", k, gain, 384, "prefix") for k in (6, 4, 2) for gain in (1.0, 0.75, 0.5)),
    Variant("ffn_half_prefix", 8, 1.0, 192, "prefix"),
    Variant("ffn_half_norm", 8, 1.0, 192, "norm"),
    Variant("ffn_quarter_prefix", 8, 1.0, 96, "prefix"),
    Variant("ffn_quarter_norm", 8, 1.0, 96, "norm"),
    Variant("half_compute_baseline", 8, 1.0, 128, "prefix", residual_width=384),
    Variant("quarter_compute_baseline", 8, 1.0, 48, "prefix", residual_width=256),
    Variant("k1_gain0.75", 1, 0.75, 384, "prefix"),
    Variant("hash_k8", 8, 1.0, 384, "prefix", approximation="hash"),
    Variant("hash_k2", 2, 0.75, 384, "prefix", approximation="hash"),
    Variant("drop_shared", 8, 1.0, 384, "prefix", approximation="drop_shared"),
    Variant("gate_half", 8, 1.0, 384, "prefix", approximation="gate_half"),
    Variant("gate_identity", 8, 1.0, 384, "prefix", approximation="gate_identity"),
    Variant("rms_gain_only", 8, 1.0, 384, "prefix", approximation="rms_gain_only"),
    Variant("fp4_expert_inputs", 8, 1.0, 384, "prefix", approximation="fp4_expert_inputs"),
    Variant("k4_fp4_inputs", 4, 0.75, 384, "prefix", approximation="fp4_expert_inputs"),
    Variant("activation_identity", 8, 1.0, 384, "prefix", approximation="activation_identity"),
    Variant("routed_relu", 8, 1.0, 384, "prefix", approximation="routed_relu"),
    Variant("shared_relu", 8, 1.0, 384, "prefix", approximation="shared_relu"),
    Variant("all_experts_relu", 8, 1.0, 384, "prefix", approximation="all_experts_relu"),
    Variant("fp4_expert_weights", 8, 1.0, 384, "prefix", approximation="fp4_expert_weights"),
    Variant("k4_fp4_weights", 4, 0.75, 384, "prefix", approximation="fp4_expert_weights"),
    *(Variant(f"depth_drop{i}", 8, 1.0, 384, "prefix", approximation="depth", skip_layers=(i,)) for i in range(6)),
    *(
        Variant(
            f"depth_keep{''.join(map(str, keep))}",
            8,
            1.0,
            384,
            "prefix",
            approximation="depth",
            skip_layers=tuple(i for i in range(6) if i not in keep),
        )
        for keep in ((0, 2, 3, 5), (0, 1, 4, 5), (0, 2, 5), (0, 3, 5), (0, 1, 2), (3, 4, 5))
    ),
)


def variant_config(cfg, variant):
    if variant.residual_width != cfg.hidden_dim:
        cfg = size_config(cfg, width=variant.residual_width, expert_width=variant.expert_width)
    cfg = replace(
        cfg,
        num_experts_per_token=variant.active_experts,
        routing_renorm_sum=cfg.routing_renorm_sum * variant.gain,
        intermediate_dim=variant.expert_width,
        moe_fp8_dispatch=variant.dispatch_dtype == "float8_e4m3",
    )
    if variant.approximation == "hash":
        cfg = replace(cfg, moe_hash_layers=tuple(range(cfg.num_layers)))
    if variant.approximation in ("activation_identity", "routed_relu", "shared_relu", "all_experts_relu", "depth"):
        cfg = draccus.decode(activation_model.GrugModelConfig, draccus.encode(cfg))
        cfg = replace(cfg, drafter_skip_layers=variant.skip_layers)
        if variant.approximation in ("routed_relu", "all_experts_relu"):
            cfg = replace(cfg, moe_ungated_activation=activation_model.UngatedExpertActivation.RELU)
        if variant.approximation in ("shared_relu", "all_experts_relu"):
            cfg = replace(cfg, shared_ungated_activation=activation_model.UngatedExpertActivation.RELU)
    return cfg


def variant_template(config, key):
    model_type = activation_model.Transformer if isinstance(config, activation_model.GrugModelConfig) else Transformer
    return eqx.filter_eval_shape(model_type.init, config, key=key)


def sampled_predictions(params, batch, mp):
    positions = jnp.asarray([(row, pos) for row in range(128) for pos in POSITIONS])
    model = mp.cast_to_compute(params)
    return model.position_predictions(batch.tokens, positions, mask=batch.attn_mask, k=5)


def selected_weight_blocks(up, down, *, width):
    score = jnp.sum(up**2, axis=-2) ** 2 * jnp.sum(down**2, axis=-1)
    _, indices = jax.lax.top_k(score, width)
    indices = jnp.sort(indices, axis=-1)
    return (
        jnp.take_along_axis(up, indices[..., None, :], axis=-1),
        jnp.take_along_axis(down, indices[..., :, None], axis=-2),
    )


def norm_selected_experts(hero, selected, width):
    """Rank Belay's ReLU-squared expert channels by input/output weight norms.

    This is a weight-only structured heuristic, not activation-calibrated Wanda.
    The score is ||up||^4 * ||down||^2, reflecting the squared activation.
    Corresponding up columns and down rows always use the same indices.
    """
    source = {jax.tree_util.keystr(path): leaf for path, leaf in jax.tree_util.tree_flatten_with_path(hero)[0]}
    selected_paths, structure = jax.tree_util.tree_flatten_with_path(selected)
    replacements = {}
    for path, _leaf in selected_paths:
        name = jax.tree_util.keystr(path)
        if "expert_mlp" not in name or not name.endswith(".w_down"):
            continue
        prefix = name.removesuffix("w_down")
        if prefix + "w_gate" in source:
            raise ValueError("This selection rule is specific to ungated ReLU-squared experts")
        up, down = (source[prefix + field] for field in ("w_up", "w_down"))
        # On the native mesh only the expert axis has size >1. Each device owns
        # complete channel sets for its experts, so selection needs no collective.
        specs = (jax.typeof(up).sharding.spec, jax.typeof(down).sharding.spec)
        pruned_up, pruned_down = jax.shard_map(
            partial(selected_weight_blocks, width=width), in_specs=specs, out_specs=specs
        )(up, down)
        replacements[prefix + "w_up"] = pruned_up
        replacements[name] = pruned_down
    if not replacements:
        raise AssertionError("No routed expert tensors selected")
    leaves = [replacements.get(jax.tree_util.keystr(path), leaf) for path, leaf in selected_paths]
    return jax.tree.unflatten(structure, leaves)


def zero_skipped_blocks(model, skipped):
    """Erase bypassed physical blocks to check they cannot affect predictions."""
    replacements = []
    for stack, indices in zip(model.layer_stacks(), model.stack_layer_indices(), strict=True):
        value = stack.stacked
        for local, physical in enumerate(indices):
            if physical in skipped:
                value = jax.tree.map(lambda array, local=local: array.at[local].set(0), value)
        replacements.append(value)
    return eqx.tree_at(lambda m: [stack.stacked for stack in m.layer_stacks()], model, replacements)


def restored_leaf(stream, template, mesh):
    value = np.load(stream, allow_pickle=False)
    if value.shape != template.shape or value.dtype != template.dtype:
        raise ValueError(f"Checkpoint shape/dtype mismatch: {value.shape}/{value.dtype}, {template}")
    # Unannotated initialization constants (for example ve_lambda) are replicated.
    spec = P() if template.sharding is None else template.sharding.spec
    return jax.device_put(value, NamedSharding(mesh, spec))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--variants", nargs="+", choices=[v.name for v in VARIANTS], required=True)
    parser.add_argument("--batch-start", type=int, required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--timing-repeats", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    record = json.loads(REFERENCE.read_text())["config"]
    for key, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(key, value)
    cfg = load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    if args.model_config is not None:
        cfg = draccus.decode(GrugModelConfig, json.loads(args.model_config.read_text()))
    devices = jax.devices("gpu")
    if len(devices) != 8:
        raise ValueError("Use the native eight-H100 mesh")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    key = jax.random.PRNGKey(0)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "initializing",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "approximation_source_sha256": (
            hashlib.sha256(Path(__file__).with_name("tied_drafter_approximations.py").read_bytes()).hexdigest()
        ),
        "activation_source_sha256": hashlib.sha256(Path(activation_model.__file__).read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint,
        "model_config": draccus.encode(cfg),
        "model_config_sha256": (
            hashlib.sha256(args.model_config.read_bytes()).hexdigest() if args.model_config is not None else None
        ),
        "batch_indices": list(range(args.batch_start, args.batch_start + args.batches)),
        "sample_positions_per_row": POSITIONS,
        "hardware": "8 H100, sequential full-batch forwards, FP32 weights/BF16 compute",
        "scope": "Teacher-forced token agreement and loss-forward latency; no updates, decoding or training speedup",
        "replicas": [],
    }

    def save():
        (args.output / "latency.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    with jax.set_mesh(mesh):
        template = eqx.filter_eval_shape(Transformer.init, cfg, key=key)
        fs, path = filesystem_for(args.checkpoint)
        result["checkpoint_metadata"] = {"size": fs.size(path)}
        began = time.perf_counter()
        with fs.open(path, "rb") as stream:
            hero = eqx.tree_deserialise_leaves(
                stream, template, filter_spec=lambda stream, leaf: restored_leaf(stream, leaf, mesh)
            )
            if stream.read(1):
                raise ValueError("Checkpoint contains unexpected trailing data")
        jax.block_until_ready(hero)
        result["restore_seconds"] = time.perf_counter() - began
        dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
        loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=2)
        iterator = iter(loader.iter_from_step(args.batch_start))
        try:
            batches = [next(iterator) for _ in range(args.batches)]
        finally:
            iterator._batches.stop()
        forward = eqx.filter_jit(sequence_losses)
        prediction = eqx.filter_jit(sampled_predictions)
        hero_forward = forward.lower(hero, batches[0], mp).compile()
        hero_prediction = prediction.lower(hero, batches[0], mp).compile()
        reference = []
        sample_masks = []
        for batch in batches:
            loss = jax.device_get(hero_forward(hero, batch, mp))
            repeated = jax.device_get(hero_forward(hero, batch, mp))
            result.setdefault("hero_repeatability", []).append(output_difference(loss[0], repeated[0]))
            reference.append((loss, jax.device_get(hero_prediction(hero, batch, mp))))
            sample_masks.append(np.asarray(jax.device_get(batch.loss_weight))[:, POSITIONS].reshape(-1) > 0)
        full_cost = linear_weight_equivalents(hero)
        full_count = sum(x.size for x in jax.tree.leaves(hero))
        result["hero_parameters"] = full_count
        result["status"] = "comparing"
        save()
        for variant_name in args.variants:
            variant = next(v for v in VARIANTS if v.name == variant_name)
            config = variant_config(cfg, variant)
            template = variant_template(config, key)
            selected = eqx.filter_jit(restrict_model)(hero, template, coordinate_map(hero, template))
            if variant.selection == "norm":
                selected = eqx.filter_jit(norm_selected_experts)(hero, selected, variant.expert_width)
            if variant.approximation == "fp4_expert_weights":
                selected = eqx.filter_jit(modify_blocks)(selected, variant.approximation)
            if variant.approximation in (
                "drop_shared",
                "gate_half",
                "gate_identity",
                "rms_gain_only",
                "fp4_expert_inputs",
            ):
                selected = modify_blocks(selected, variant.approximation)
            jax.block_until_ready(selected)
            row = {
                **asdict(variant),
                "parameters": sum(x.size for x in jax.tree.leaves(selected)),
                "linear_compute_fraction": (
                    None if variant.skip_layers else linear_weight_equivalents(selected) / full_cost
                ),
                "executed_layer_indices": [i for i in range(cfg.num_layers) if i not in variant.skip_layers],
                "storage_note": "Bypassed block arrays remain stored; their forward computation is omitted.",
                "batches": [],
            }
            began = time.perf_counter()
            executable = forward.lower(selected, batches[0], mp).compile()
            prediction_executable = prediction.lower(selected, batches[0], mp).compile()
            row["compile_seconds"] = time.perf_counter() - began
            if variant.skip_layers:
                erased = eqx.filter_jit(zero_skipped_blocks)(selected, variant.skip_layers)
                original_loss = jax.device_get(executable(selected, batches[0], mp))[0]
                erased_loss = jax.device_get(executable(erased, batches[0], mp))[0]
                row["skipped_weight_invariance"] = output_difference(original_loss, erased_loss)
                if not row["skipped_weight_invariance"]["exact_equal"]:
                    raise AssertionError("Bypassed block weights affect the loss")
                del erased
            for batch, (before, before_prediction), mask in zip(batches, reference, sample_masks, strict=True):
                after = jax.device_get(executable(selected, batch, mp))
                after_prediction = jax.device_get(prediction_executable(selected, batch, mp))
                metrics = comparison(
                    before, after, tuple(x[mask] for x in before_prediction), tuple(x[mask] for x in after_prediction)
                )
                metrics["sampled_valid_positions"] = int(mask.sum())
                row["batches"].append(metrics)
            # Paired, alternating timings limit drift from clock and temperature changes.
            for _ in range(3):
                jax.block_until_ready(executable(selected, batches[0], mp))
                jax.block_until_ready(hero_forward(hero, batches[0], mp))
            times, control_times = [], []
            for repeat in range(args.timing_repeats):
                calls = [(executable, selected, times), (hero_forward, hero, control_times)]
                for function, model, measurements in calls[:: 1 if repeat % 2 == 0 else -1]:
                    began = time.perf_counter()
                    jax.block_until_ready(function(model, batches[0], mp))
                    measurements.append(time.perf_counter() - began)
            row["forward_seconds"] = times
            row["paired_hero_forward_seconds"] = control_times
            if variant.name in ("identity", "activation_identity") and any(
                not b["sequence_loss_difference"]["exact_equal"] or b["sampled_top1_agreement"] != 1
                for b in row["batches"]
            ):
                raise AssertionError("Restored full-width identity control failed")
            result["replicas"].append(row)
            print(
                json.dumps(
                    {
                        "event": "variant",
                        "name": variant.name,
                        "agreement": [b["sampled_top1_agreement"] for b in row["batches"]],
                        "forward_median": float(np.median(times)),
                    }
                ),
                flush=True,
            )
            save()
            del selected, executable, prediction_executable
        result["status"] = "complete"
        save()


if __name__ == "__main__":
    main()
