# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded 67B initializer proof on the frozen Exp1 slice and named C4-en input."""

import functools
import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import click
import draccus
import jax
import numpy as np
from fray.types import ResourceConfig
from haliax import Axis
from iris.cluster.client.job_info import get_job_info
from iris.rpc import job_pb2
from iris.rpc.proto_display import priority_band_value
from jax.sharding import AxisType, Mesh
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.eval import TaggedEvaluator, lm_eval_loss
from levanter.eval_audit import InitialEvalMatchConfig, evaluation_input_audit
from levanter.main.train_lm import TrainLmConfig
from levanter.models.snowball import SnowballConfig
from levanter.store.cache import SerialCacheWriter, TreeCache
from levanter.utils.mesh import MeshConfig
from levanter.utils.thread_utils import blocking_wait
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import IrisSkyRLExecution
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.exp3_recipe import TOKEN_IDS, exp3_sft_spec, verified_initializer_model
from experiments.post_training.long_loop.exp3_smokes import certify_smoke, prepare_smoke_sources, smoke_train_config
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.stage_guard import StageGuard, run_guarded_stage
from experiments.sft.launcher import HFModel, levanter_train_request

C4_CACHE = "gs://marin-us-central2/tokenized/paloma/c4_en-c2eba6/validation"
COMPILATION_CACHE = "s3://marin-us-east-08a/tmp/ttl=30d/compilation-cache"
CONTEXT = 262144
REFERENCE_LOSS = 2.3137418889651173
REFERENCE_TOLERANCE = 0.5
RESOURCES = ResourceConfig.with_gpu("H100", count=8, replicas=4, cpu=64, ram="512GB", disk="512GB")
MESH = MeshConfig(
    axes={"data": 1, "expert": 8, "model": 1, "replica": 1},
    dcn_axes={"context": 4, "replica_dcn": 1},
    compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"},
)


def freeze_c4_reference(*, output: str) -> None:
    """Freeze a bounded C4 stream from a CPU host with access to the original native cache."""
    cache = TreeCache.load(C4_CACHE, {"input_ids": np.empty(0, dtype=np.int32)})
    rows, selected, count = [], [], 0
    for first in range(0, cache.ledger.total_num_rows, 64):
        batch = blocking_wait(cache.get_batch(list(range(first, min(first + 64, cache.ledger.total_num_rows)))))
        for offset, row in enumerate(batch):
            tokens = np.asarray(row["input_ids"], dtype=np.int32)
            rows.append(tokens)
            selected.append(
                {"row": first + offset, "tokens": len(tokens), "sha256": hashlib.sha256(tokens.tobytes()).hexdigest()}
            )
            count += len(tokens)
            if count >= CONTEXT:
                break
        if count >= CONTEXT:
            break
    if count < CONTEXT:
        raise ValueError("The named C4 cache has fewer than 262,144 tokens")
    tokens = np.concatenate(rows)[:CONTEXT]
    weights = np.ones(CONTEXT, dtype=np.float32)
    weights[-1] = 0
    with SerialCacheWriter(
        prefix_join(output, "c4/validation"),
        {"input_ids": np.empty(0, dtype=np.int32), "loss_weights": np.empty(0, dtype=np.float32)},
    ) as writer:
        writer.write_batch([{"input_ids": tokens, "loss_weights": weights}])
    data = LmDataConfig(
        tokenizer=prefix_join(output, "tokenizer"),
        components={
            "c4_en": DatasetComponent(
                cache_dir=prefix_join(output, "c4"), format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights")
            )
        },
        auto_build_caches=False,
    )
    audit = c4_input_audit(data)
    reference = {
        "metrics": {"eval/loss": REFERENCE_LOSS},
        "input_audit": audit,
        "reference_kind": "estimate, not the inaccessible upstream final loss",
        "absolute_tolerance": REFERENCE_TOLERANCE,
        "proxy_run": "marin_moe/moe_67b_a2b_d2560_ctxext_qk175_skew8_step157k_paloma_lm_eval_rolling4096",
        "derivation": {"c4_en_bits_per_byte": 0.6971565402567951, "measured_bytes_per_token": 4.788055130168453},
        "limitations": (
            "Proxy uses rolling4096 and different sampling; tolerance is provisional, not a confidence interval."
        ),
        "native_input": {
            "cache": C4_CACHE,
            "source_rows": selected,
            "source_tokens_read": count,
            "selected_tokens": CONTEXT,
        },
    }
    (StoragePath(output) / "c4-reference.json").write_text(json.dumps(reference, indent=2))
    (StoragePath(output) / "c4-data.json").write_text(json.dumps(draccus.encode(data), indent=2))


def c4_input_audit(data: LmDataConfig) -> dict:
    """Hash the actual evaluation loader without loading a model or allocating GPUs."""
    axes = ("data", "expert", "context", "model", "replica", "replica_dcn")
    mesh = Mesh(
        np.asarray(jax.devices("cpu")[:1]).reshape((1,) * len(axes)), axes, axis_types=(AxisType.Explicit,) * len(axes)
    )
    batch = Axis("batch", 1)
    with jax.set_mesh(mesh):
        evaluator = TaggedEvaluator(
            EvalBatch=batch,
            tagged_eval_sets=data.tagged_eval_sets(Axis("position", CONTEXT)),
            loss_fn=functools.partial(lm_eval_loss, EvalBatch=batch),
            tokenizer=data.the_tokenizer,
            device_mesh=mesh,
            axis_mapping={"batch": "data"},
            max_examples_per_dataset=1,
        )
        return evaluation_input_audit(evaluator, TOKEN_IDS)


def prepare_initializer_smoke(
    *,
    output: str,
    reference_bundle: str,
    reference_bundle_sha256: str,
    initializer_receipt: str,
    initializer_receipt_sha256: str,
) -> None:
    """Adopt an inventoried regional reference; production workers never fetch the GCS source."""
    verified_initializer_model(initializer_receipt, initializer_receipt_sha256)
    (StoragePath(output) / "initializer.json").write_text(
        json.dumps(
            {
                "initializer_receipt": initializer_receipt,
                "initializer_receipt_sha256": initializer_receipt_sha256,
            },
            indent=2,
        )
    )
    prepare_smoke_sources(output=output)
    raw = StoragePath(reference_bundle).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference_bundle_sha256:
        raise ValueError("C4 reference bundle manifest SHA256 mismatch")
    manifest = json.loads(raw)
    for item in manifest["files"]:
        content = StoragePath(item["uri"]).read_bytes()
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"C4 reference bundle member changed: {item['uri']}")
    reference = json.loads(StoragePath(manifest["reference_uri"]).read_text())
    data = LmDataConfig(
        tokenizer=prefix_join(output, "tokenizer"),
        components={
            "c4_en": DatasetComponent(
                cache_dir=manifest["cache_uri"], format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights")
            )
        },
        auto_build_caches=False,
    )
    if c4_input_audit(data) != reference["input_audit"]:
        raise ValueError("Regional C4 token stream differs from the frozen reference")
    reference["regional_bundle"] = {"uri": reference_bundle, "sha256": reference_bundle_sha256, "manifest": manifest}
    reference["initializer_stage"] = {
        "receipt_uri": initializer_receipt,
        "receipt_sha256": initializer_receipt_sha256,
        "receipt": json.loads(StoragePath(initializer_receipt).read_text()),
    }
    (StoragePath(output) / "c4-reference.json").write_text(json.dumps(reference, indent=2))
    (StoragePath(output) / "c4-data.json").write_text(json.dumps(draccus.encode(data), indent=2))


def initializer_recipe(*, prepared: str, output: str, artifact_version: str, policy: StageGuard) -> dict:
    data = json.loads((StoragePath(prepared) / "data.json").read_text())
    tiny = smoke_train_config(
        draccus.decode(LmDataConfig, data), output=output, prepared=prepared, artifact_version=artifact_version
    )
    c4 = draccus.decode(LmDataConfig, json.loads((StoragePath(prepared) / "c4-data.json").read_text()))
    reference = prefix_join(prepared, "c4-reference.json")
    gate = InitialEvalMatchConfig(c4, reference, CONTEXT, 1, TOKEN_IDS, REFERENCE_TOLERANCE, reference)
    model = draccus.encode(
        SnowballConfig(max_seq_len=CONTEXT, moe_implementation="ring_gather_combine", moe_capacity_factor=1)
    )
    model.pop("type", None)
    return {
        **json.loads((StoragePath(prepared) / "initializer.json").read_text()),
        "profile": "snowball_exp3",
        "data": data,
        "evaluation_data": data,
        "source_loss_metadata": prefix_join(prepared, "source-loss.json"),
        "source_loss_metadata_sha256": (
            hashlib.sha256((StoragePath(prepared) / "source-loss.json").read_bytes()).hexdigest()
        ),
        "packing_cap": 256,
        "packing_lookahead": 128,
        "trainer": draccus.encode(
            replace(
                tiny.trainer,
                num_train_steps=2,
                mesh=MESH,
                max_eval_batches=0,
                offload_opt_state=True,
                jax_compilation_cache_dir=COMPILATION_CACHE,
                log_jaxprs=False,
                log_xla_hlo=False,
                watch=replace(tiny.trainer.watch, include_per_parameter_norms=False),
                checkpointer=CheckpointerConfig(save_interval=None, keep=None),
                progress_watchdog=ProgressWatchdogConfig(
                    startup_timeout=timedelta(minutes=45),
                    step_timeout=timedelta(minutes=10),
                    process_timeout=timedelta(minutes=30),
                    startup_grace_period=timedelta(0),
                ),
            )
        ),
        "initial_eval_match": draccus.encode(gate),
        "model": model,
        "weight_decay": 0.1,
        "data_seed": 20261115,
        "tensorstore_cache_bytes": 100_000_000,
        "mesh": draccus.encode(MESH),
        "resources": {key: value for key, value in draccus.encode(RESOURCES).items() if value is not None},
        "seq_len": 2048,
        "batch_size": 8,
        "per_device_parallelism": 1,
        "stop_after_steps": 2,
        "wandb_entity": "marin-community",
        "wandb_project": "marin-long-loop",
        "xla_flags": ["--xla_gpu_shard_autotuning=false"],
        "task_timeout_seconds": 259200,
        "stage_guard": vars(policy),
    }


def run_initializer_smoke(*, output: str, prepared: str, artifact_version: str, policy: StageGuard) -> None:
    configure_coreweave_s3()
    recipe = initializer_recipe(prepared=prepared, output=output, artifact_version=artifact_version, policy=policy)
    (StoragePath(output) / "recipe.json").write_text(json.dumps(recipe, indent=2))
    spec = exp3_sft_spec("exp3-initializer-proof", artifact_version, recipe)
    assert isinstance(spec.model, HFModel) and spec.prepared_data is not None
    context = StepContext.for_run(output, output)
    config = spec.model.build_train_config(context, spec, spec.prepared_data, RESOURCES, 2)
    train_config = config.train_config
    assert isinstance(train_config, TrainLmConfig)
    config = replace(
        config,
        train_config=replace(
            train_config,
            hf_save_steps=None,
            trainer=replace(train_config.trainer, native_telemetry_root=prefix_join(output, "telemetry")),
        ),
    )
    # Exercise the worker wire boundary on the regional CPU coordinator before allocating GPUs.
    config = replace(
        config,
        train_config=draccus.decode(TrainLmConfig, json.loads(json.dumps(draccus.encode(config.train_config)))),
    )
    request = levanter_train_request(config, task_timeout_seconds=259200, priority=priority_band_value("production"))
    receipt = run_guarded_stage(request, policy, prefix_join(output, "guard"))
    (StoragePath(output) / "guard-result.json").write_text(json.dumps(receipt, indent=2))


@click.command(help=__doc__)
@click.option("--h100-hour-limit", type=float, required=True)
@click.option("--source-revision", required=True)
@click.option("--reference-bundle", required=True)
@click.option("--reference-bundle-sha256", required=True)
@click.option("--initializer-receipt", required=True)
@click.option("--initializer-receipt-sha256", required=True)
@rl_build_options
def main(
    h100_hour_limit: float,
    source_revision: str,
    reference_bundle: str,
    reference_bundle_sha256: str,
    initializer_receipt: str,
    initializer_receipt_sha256: str,
) -> ArtifactStep:
    configure_coreweave_s3()
    if get_job_info() is None:
        require_clean_revision(Path(__file__).resolve().parents[3], source_revision)
    source = runtime_source_manifest(Path(__file__).resolve().parents[3], source_revision)
    version = resolve_version("long-loop-exp3/initializer-proof", None)
    prepared = apply(
        "long-loop-exp3/initializer-proof-inputs",
        prepare_initializer_smoke,
        version=version,
        output=OUT,
        reference_bundle=reference_bundle,
        reference_bundle_sha256=reference_bundle_sha256,
        initializer_receipt=initializer_receipt,
        initializer_receipt_sha256=initializer_receipt_sha256,
    )
    trained = apply(
        "long-loop-exp3/initializer-proof",
        run_initializer_smoke,
        version=version,
        output=OUT,
        prepared=prepared,
        artifact_version=version,
        policy=StageGuard(h100_hour_limit, 30, 2),
    )
    terminal = apply(
        "long-loop-exp3/initializer-proof-archive",
        certify_smoke,
        version=version,
        output=OUT,
        trained=trained,
        prepared=prepared,
        source=source,
        completed_updates=2,
        reference_scope="Full67B initializer; estimated C4-en reference at262144 context; two2048-token updates.",
        reference_filename="c4-reference.json",
        producers=tuple(f"rank-{rank}" for rank in range(RESOURCES.replicas)),
    )
    execution = IrisSkyRLExecution(
        cluster="cw-rno2a",
        cluster_config="lib/iris/config/cw-rno2a.yaml",
        cpu=4,
        memory="16GB",
        disk="64GB",
        priority="production",
        max_retries=0,
        target_cluster="cw-rno2a",
        parent_cluster_config="lib/iris/config/marin.yaml",
        coordinator_timeout_hours=72,
        coordinator_name="exp3-initializer-proof-" + version,
        container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        required_env_vars=("MARIN_PREFIX", "MARIN_TEMP_PREFIX", "HF_TOKEN"),
    )
    return replace(terminal, runtime_args={"skyrl_execution": execution})


if __name__ == "__main__":
    main()
