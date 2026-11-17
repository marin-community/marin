# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal native Exp3 wiring smoke on a pinned slice of Exp1 data."""

import hashlib
import json
import logging
import math
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import draccus
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from haliax import Axis
from iris.cluster.client.job_info import get_job_info
from iris.rpc import job_pb2
from iris.rpc.proto_display import priority_band_value
from jax.sharding import AxisType, Mesh
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig, save_checkpoint
from levanter.checkpoint_manifest import read_manifest
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.data.text.serialized_chat import mask_serialized_chat
from levanter.distributed import DistributedConfig
from levanter.eval_audit import InitialEvalMatchConfig
from levanter.main.eval_lm import EvalLmConfig
from levanter.main.eval_lm import main as evaluate
from levanter.main.train_lm import TrainLmConfig
from levanter.models.snowball import SnowballConfig
from levanter.store.cache import SerialCacheWriter, TreeCache
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from levanter.utils.thread_utils import blocking_wait
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import IrisSkyRLExecution
from marin.training.levanter import apply_output_path, resolve_training_env, run_levanter_train_lm
from marin.training.run_environment import extras_for_resources
from marin.training.training import TrainLmOnPodConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credential_options
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import Duration
from rigging.training_journal import verify_training_journals
from transformers import AutoTokenizer

from experiments.post_training.long_loop.archive import ArchiveInput, seal_archive
from experiments.post_training.long_loop.exp3_recipe import TOKEN_IDS, exp3_controls, exp3_optimizer
from experiments.post_training.long_loop.recipe_bundle import (
    require_clean_revision,
    runtime_source_manifest,
    write_recipe_bundle,
)
from experiments.post_training.long_loop.sft import TOKENIZER_REPO, TOKENIZER_REVISION
from experiments.post_training.long_loop.source_loss import prepared_supervision_data, source_loss_metadata
from experiments.post_training.long_loop.stage_guard import StageGuard, run_guarded_stage

SOURCES = {
    "chat": "s3://marin-us-east-02a/marin/datakit/store/sft/nemotron_sft_v3/safety_v2/train_89506f41/train",
    "replay": "s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/replay/c00q1-train/2026.10.24.1/lte_64k",
}
SEQUENCE_LENGTH = 2048
SOURCE_DOCUMENTS = 64
GPU_RESOURCES = ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="256GB", disk="256GB")
logger = logging.getLogger(__name__)


def smoke_model() -> SnowballConfig:
    return SnowballConfig(
        vocab_size=128256,
        hidden_dim=128,
        intermediate_dim=64,
        shared_expert_intermediate_dim=128,
        num_experts=16,
        num_experts_per_token=4,
        num_layers=2,
        num_heads=4,
        num_kv_heads=1,
        head_dim=32,
        max_seq_len=SEQUENCE_LENGTH,
        sliding_window=128,
    )


def prepare_smoke_sources(*, output: str) -> None:
    """Freeze complete short documents; record every selected source row and byte hash."""
    configure_coreweave_s3()
    credentials = s3_credential_options()
    os.environ["AWS_ACCESS_KEY_ID"] = credentials["key"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["secret"]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)
    prefix = tuple(tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False))
    with TemporaryDirectory() as temporary:
        tokenizer.save_pretrained(temporary)
        for path in Path(temporary).iterdir():
            (StoragePath(output) / "tokenizer" / path.name).write_bytes(path.read_bytes())
    components, metadata, receipts = {}, {}, {}
    for name, uri in SOURCES.items():
        logger.info("Reading Exp1 smoke source %s", name)
        cache = TreeCache.load(uri, {"input_ids": np.empty(0, dtype=np.int32)})
        selected, identities, rejected = [], [], []
        for first in range(0, min(cache.ledger.total_num_rows, 8192), 64):
            rows = blocking_wait(cache.get_batch(list(range(first, min(first + 64, cache.ledger.total_num_rows)))))
            for offset, row in enumerate(rows):
                tokens = np.asarray(row["input_ids"], dtype=np.int32)
                if not 1 < len(tokens) <= SEQUENCE_LENGTH:
                    rejected.append(first + offset)
                    continue
                weights = np.ones(len(tokens), dtype=np.float32)
                weights[-1] = 0
                if name == "chat":
                    example = mask_serialized_chat(
                        GrugLmExample.causal(
                            jnp.asarray(np.pad(tokens, (0, SEQUENCE_LENGTH - len(tokens)))),
                            loss_weight=jnp.asarray(np.pad(weights, (0, SEQUENCE_LENGTH - len(tokens)))),
                        ),
                        message_start_id=128006,
                        assistant_prefix=prefix,
                    )
                    weights = np.asarray(example.loss_weight, dtype=np.float32)[: len(tokens)].copy()
                if weights.sum() == 0:
                    rejected.append(first + offset)
                    continue
                selected.append({"input_ids": tokens, "loss_weights": weights})
                identities.append(
                    {
                        "row": first + offset,
                        "tokens": len(tokens),
                        "loss_tokens": int(weights.sum()),
                        "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
                    }
                )
                if len(selected) == SOURCE_DOCUMENTS:
                    break
            if len(selected) == SOURCE_DOCUMENTS:
                break
            logger.info("Source %s: scanned %d rows, selected %d", name, first + len(rows), len(selected))
        if len(selected) != SOURCE_DOCUMENTS:
            raise ValueError(f"Insufficient complete short smoke documents in {name}")
        directory = prefix_join(output, "cache/" + name)
        with SerialCacheWriter(
            directory, {"input_ids": np.empty(0, dtype=np.int32), "loss_weights": np.empty(0, dtype=np.float32)}
        ) as writer:
            writer.write_batch(selected)
        components[name] = DatasetComponent(
            cache_dir=directory, flat_cache=True, format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights")
        )
        receipts[name] = {
            "source": uri,
            "selected": identities,
            "excluded_from_smoke": rejected,
            "truncated_tokens": 0,
            "original_pool_documents": cache.ledger.total_num_rows,
        }
        metadata[name] = {
            "loss_mode": "chat_assistant_only" if name == "chat" else "full_token",
            "loss_weights_key": "loss_weights",
            "loss_weights_indexing": "prediction",
            "teacher_only": False,
            "source_revision": uri,
        }
    raw = json.dumps({"version": 1, "sources": metadata}, sort_keys=True).encode()
    sidecar = StoragePath(output) / "source-loss.json"
    sidecar.write_bytes(raw)
    modes = source_loss_metadata(str(sidecar), hashlib.sha256(raw).hexdigest())
    data = prepared_supervision_data(
        LmDataConfig(
            tokenizer=prefix_join(output, "tokenizer"),
            components=components,
            auto_build_caches=False,
            train_weights={"chat": 0.8, "replay": 0.2},
            num_validation_sequences={"chat": 1, "replay": 1},
            mixture_block_size=8,
            shuffle=True,
            permutation_type="feistel",
        ),
        modes,
        packing_cap=256,
        lookahead=128,
    )
    (StoragePath(output) / "data.json").write_text(json.dumps(draccus.encode(data), indent=2))
    (StoragePath(output) / "source-receipts.json").write_text(json.dumps(receipts, indent=2))


def prepare_smoke_initializer(*, output: str) -> None:
    """Create the tiny initializer and its independent single-process evaluation."""
    data = draccus.decode(LmDataConfig, json.loads((StoragePath(output) / "data.json").read_text()))
    axes = ("data", "expert", "context", "model", "replica", "replica_dcn")
    mesh = Mesh(
        np.asarray(jax.devices("cpu")[:1]).reshape((1,) * len(axes)), axes, axis_types=(AxisType.Explicit,) * len(axes)
    )
    with jax.set_mesh(mesh):
        model = smoke_model().build(Axis("vocab", 128256), key=jax.random.PRNGKey(20261115))
        save_checkpoint({"model": model}, 0, prefix_join(output, "initializer/checkpoints/step-0"), is_temporary=False)
    evaluate(
        EvalLmConfig(
            checkpoint_path=prefix_join(output, "initializer/checkpoints/step-0"),
            model=smoke_model(),
            data=data,
            max_eval_length=SEQUENCE_LENGTH,
            trainer=TrainerConfig(
                id="exp3-cpu-reference",
                train_batch_size=1,
                per_device_parallelism=1,
                per_device_eval_parallelism=1,
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
                max_eval_batches=1,
                tracker=WandbConfig(mode="disabled"),
                mesh=MeshConfig(axes={"data": 1, "expert": 1, "context": 1, "model": 1, "replica": 1}),
                use_explicit_mesh_axes=True,
                mp=jmp.get_policy("p=f32,c=bfloat16"),
            ),
            snowball_sft=exp3_controls(),
            audit_token_ids=TOKEN_IDS,
            result_path=prefix_join(output, "tiny-reference.json"),
        )
    )


def prepare_smoke(*, output: str) -> None:
    prepare_smoke_sources(output=output)
    prepare_smoke_initializer(output=output)


def smoke_train_config(data: LmDataConfig, *, output: str, prepared: str, artifact_version: str) -> TrainLmConfig:
    trainer = TrainerConfig(
        id="exp3-native-smoke-" + artifact_version,
        train_batch_size=8,
        num_train_steps=3,
        per_device_parallelism=1,
        per_device_eval_parallelism=1,
        max_eval_batches=1,
        steps_per_eval=1,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(
            axes={"data": 1, "expert": 8, "context": 1, "model": 1, "replica": 1},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"},
        ),
        watch=WatchConfig(watch_targets=["params", "updates", "grads"], interval=1),
        tracker=WandbConfig(entity="marin-community", project="marin-long-loop", tags=["exp3", "smoke"]),
        native_telemetry_root=prefix_join(output, "telemetry"),
        checkpointer=CheckpointerConfig(),
    )
    config = TrainLmConfig(
        data=data,
        model=smoke_model(),
        trainer=trainer,
        train_seq_len=SEQUENCE_LENGTH,
        optimizer=exp3_optimizer(weight_decay=0),
        optimizer_num_train_steps=4750,
        z_loss_weight=1e-4,
        global_gradient_clip=1,
        snowball_sft=exp3_controls(),
        token_weighted_accumulation=True,
        initialize_model_from_checkpoint_path=prefix_join(prepared, "initializer/checkpoints/step-0"),
        initial_eval_match=InitialEvalMatchConfig(
            data, prefix_join(prepared, "tiny-reference.json"), SEQUENCE_LENGTH, 1, TOKEN_IDS, 0.01
        ),
        hf_save_steps=4,
        hf_save_dtype="bfloat16",
    )
    return apply_output_path(config, output)


def run_native_smoke(*, output: str, prepared: str, artifact_version: str, policy: StageGuard) -> None:
    configure_coreweave_s3()
    data = draccus.decode(LmDataConfig, json.loads((StoragePath(prepared) / "data.json").read_text()))
    config = smoke_train_config(data, output=output, prepared=prepared, artifact_version=artifact_version)
    (StoragePath(output) / "consumed-config.json").write_text(json.dumps(draccus.encode(config), indent=2))
    environment = resolve_training_env(None, GPU_RESOURCES)
    request = JobRequest(
        name="exp3-native-smoke-" + artifact_version,
        entrypoint=Entrypoint.from_callable(
            run_levanter_train_lm,
            args=[TrainLmOnPodConfig(train_config=config, resources=GPU_RESOURCES, output_path=output)],
        ),
        resources=GPU_RESOURCES,
        environment=create_environment(extras=extras_for_resources(GPU_RESOURCES), env_vars=environment),
        priority=priority_band_value("production"),
        timeout=Duration.from_hours(3 * 24),
    )
    receipt = run_guarded_stage(request, policy, prefix_join(output, "guard"))
    (StoragePath(output) / "guard-result.json").write_text(json.dumps(receipt, indent=2))


def certify_smoke(
    *,
    output: str,
    trained: str,
    prepared: str,
    source: dict,
    completed_updates: int,
    reference_scope: str,
    reference_filename: str,
    producers: tuple[str, ...],
) -> None:
    telemetry = verify_training_journals(
        prefix_join(trained, "telemetry"),
        set(producers),
        {"init", "compile", "first_step", "steady_state", "checkpoint_drain"},
    )
    checkpoint = prefix_join(trained, f"checkpoints/step-{completed_updates - 1}")
    manifest = read_manifest(checkpoint)
    if manifest is None:
        raise ValueError("Smoke checkpoint has no array manifest")
    parameters = [array for array in manifest.arrays if array.path.startswith("model/")]
    if not parameters or any(array.dtype != "float32" for array in parameters):
        raise ValueError("Smoke did not retain FP32 model parameters")
    histories = {}
    for attempt in telemetry:
        history = StoragePath(attempt["source"]).parent / "history.jsonl"
        for line in history.read_text().splitlines():
            row = json.loads(line)
            if row["step"] is not None:
                histories.setdefault(row["step"], {}).update(row["metrics"])
    ratios = []
    for step in range(completed_updates):
        metrics = histories[step]
        ratio = metrics["updates/norm/total"] / metrics["params/norm/total"]
        if not math.isfinite(ratio) or not 0 < ratio < 0.01:
            raise ValueError(f"Smoke update/parameter ratio is outside (0, 0.01): step={step}, ratio={ratio}")
        ratios.append(
            {
                "step": step,
                "update_parameter_ratio": ratio,
                "positions": metrics["train/positions"],
                "padding_positions": metrics["train/padding_positions"],
                "padding_fraction": metrics["train/padding_positions"] / metrics["train/positions"],
                "scored_tokens": metrics["train/scored_tokens"],
                "sender_dropped_assignments": metrics["train/moe/sender_dropped"],
                "receiver_dropped_assignments": metrics["train/moe/receiver_dropped"],
            }
        )
    proof = {
        "fp32_parameter_arrays": len(parameters),
        "update_ratios": ratios,
        "reference_scope": reference_scope,
        "initial_evaluation": json.loads((StoragePath(trained) / "initial-eval-match.json").read_text()),
    }
    (StoragePath(output) / "smoke-proof.json").write_text(json.dumps(proof, indent=2))
    inputs = [
        ArchiveInput(prefix_join(trained, "consumed-config.json"), "consumed-config.json", "config"),
        ArchiveInput(prefix_join(trained, "guard/policy.json"), "stage-policy.json", "config"),
        ArchiveInput(prefix_join(trained, "guard-result.json"), "stage-guard-result.json", "timing"),
        ArchiveInput(prefix_join(prepared, "source-receipts.json"), "source-receipts.json", "dataset"),
        ArchiveInput(prefix_join(prepared, "source-loss.json"), "source-loss.json", "dataset"),
        ArchiveInput(prefix_join(prepared, reference_filename), reference_filename, "config"),
        ArchiveInput(prefix_join(output, "smoke-proof.json"), "smoke-proof.json", "config"),
    ]
    recipe, receipt = write_recipe_bundle(
        prefix_join(output, "recipe-bundle"), inputs, source, Path(__file__).resolve().parents[3]
    )
    seal_archive(
        output,
        inputs + recipe,
        {
            "recipe_bundle": receipt,
            "telemetry": telemetry,
            "scope": reference_scope,
        },
    )


@click.command(help=__doc__)
@click.option("--h100-hour-limit", type=float, required=True)
@click.option("--source-revision", required=True)
@rl_build_options
def main(h100_hour_limit: float, source_revision: str) -> ArtifactStep:
    configure_coreweave_s3()
    if get_job_info() is None:
        require_clean_revision(Path(__file__).resolve().parents[3], source_revision)
    source = runtime_source_manifest(Path(__file__).resolve().parents[3], source_revision)
    version = resolve_version("long-loop-exp3/smoke", None)
    prepared = apply("long-loop-exp3/smoke-inputs", prepare_smoke, version=version, output=OUT)
    trained = apply(
        "long-loop-exp3/native-smoke",
        run_native_smoke,
        version=version,
        output=OUT,
        prepared=prepared,
        artifact_version=version,
        policy=StageGuard(h100_hour_limit, 30, 2),
    )
    terminal = apply(
        "long-loop-exp3/smoke-archive",
        certify_smoke,
        version=version,
        output=OUT,
        trained=trained,
        prepared=prepared,
        source=source,
        completed_updates=3,
        reference_scope="Independent CPU evaluation of the tiny initializer; not the upstream67B loss.",
        reference_filename="tiny-reference.json",
        producers=("rank-0",),
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
        coordinator_name="exp3-smoke-" + version,
        container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        required_env_vars=("MARIN_PREFIX", "MARIN_TEMP_PREFIX", "HF_TOKEN"),
    )
    return replace(terminal, runtime_args={"skyrl_execution": execution})


if __name__ == "__main__":
    main()
