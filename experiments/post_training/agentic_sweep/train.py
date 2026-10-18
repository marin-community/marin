"""Prepare or run one authorized Snowball v0 learning-rate sweep member."""

import argparse
import hashlib
import json
import os
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
from fray.current_client import current_client, set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import ANY_REGION, Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import get_job_info
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.config import AdamConfig
from levanter.store.cache import TreeCache
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, materialized_config, run
from marin.processing.tokenize.tokenize import TokenizeConfig, TokenizedCache, tokenize
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, run_levanter_train_lm
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import Duration

from experiments.post_training.agentic_sweep.callbacks import EpochTrackerConfig, SweepExportsConfig
from experiments.post_training.agentic_sweep.export import ExportConfig, export_checkpoint
from experiments.post_training.agentic_sweep.validate import OFFLINE_MESH, OfflineValidationConfig, evaluate_checkpoint
from experiments.sft.launcher import HFModel, SFTSpec

MODEL = "open-athena/Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12"
REVISION = "4eeb756b404db25bc04e4daf49dfb9817bceb741"
TOKENIZER = f"hf://{MODEL}@{REVISION}"
VERSION = "2026.10.16.1"
TRAIN_VERSION = "2026.10.17.4"
LEARNING_RATES = (3e-7, 1e-6, 3e-6, 1e-5)
EPOCHS = 3
BATCH_SIZE = 16
CONTEXT = 32768
GPU_LOG_ENV = {"NCCL_DEBUG": "WARN", "NCCL_DEBUG_SUBSYS": ""}
FORMAT = PrebuiltLmDatasetFormat(input_ids_key="input_ids", loss_weights_key="loss_weights")


def optimizer(learning_rate: float) -> AdamConfig:
    return AdamConfig(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
        warmup=0,
        lr_schedule="constant",
    )


def cached_data(cache_path: str, split: str) -> LmDataConfig:
    component = DatasetComponent(
        cache_dir=prefix_join(cache_path, "train") if split == "train" else cache_path,
        format=FORMAT,
        split=split,
        flat_cache=split == "train",
        pack=False,
    )
    return LmDataConfig(
        tokenizer=TOKENIZER,
        components={"pilot": component},
        train_weights={"pilot": 1.0},
        enforce_eos=False,
        auto_build_caches=False,
        shuffle=split == "train",
        block_cross_document_attention=False,
    )


def verify_cache(cache_paths: dict[str, str], report: dict) -> None:
    exemplar = {"input_ids": np.zeros(0, dtype=np.int32), "loss_weights": np.zeros(0, dtype=np.float32)}
    splits = [
        ("train", report["training_rows"] + report["padding_rows"], report["supervised_tokens"]),
    ]
    splits.extend(
        (
            family,
            report["validation_rows"][family] + report["validation_padding_rows"][family],
            report["validation_supervised_tokens"][family],
        )
        for family in report["validation_files"]
    )
    for split, rows, tokens in splits:
        cache = TreeCache.load(cache_paths[split], exemplar)
        if len(cache) != rows or rows % BATCH_SIZE:
            raise ValueError(f"Wrong {split} cache row count: {len(cache)}; expected {rows}, divisible by 16")
        supervised = 0
        for start in range(0, rows, BATCH_SIZE):
            for row in cache.get_batch_sync(list(range(start, start + BATCH_SIZE))):
                weights = row["loss_weights"]
                if len(row["input_ids"]) != CONTEXT or weights.shape != row["input_ids"].shape:
                    raise ValueError(f"Wrong {split} cache sequence length or mask shape")
                if not np.isin(weights, [0, 1]).all() or weights[-1] != 0:
                    raise ValueError(f"Malformed {split} loss mask")
                supervised += int(weights.sum())
        if supervised != tokens:
            raise ValueError(f"{split} cache changed supervised tokens: {supervised} != {tokens}")
        print(json.dumps({"verified_split": split, "rows": rows, "supervised_tokens": supervised}), flush=True)


def run_worker(config: TrainLmOnPodConfig) -> None:
    os.environ["WANDB_DIR"] = os.environ["IRIS_OUTPUT_DIR"]
    run_levanter_train_lm(config)


def run_training(config: TrainLmOnPodConfig, report: dict) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("A nonempty W&B key is required before requesting GPUs")
    training = cast(TrainLmConfig, config.train_config)
    component = training.data.components["pilot"]
    assert isinstance(component, DatasetComponent) and component.cache_dir is not None
    validation = cast(SweepExportsConfig, training.adapter).validation_data
    cache_paths = {"train": component.cache_dir}
    for family, validation_component in validation.components.items():
        assert isinstance(validation_component, DatasetComponent) and validation_component.cache_dir is not None
        cache_paths[family] = prefix_join(validation_component.cache_dir, "validation")
    verify_cache(cache_paths, report)
    environment = {key: os.environ[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET", "MARIN_PREFIX", "WANDB_API_KEY")}
    environment.update(GPU_LOG_ENV)
    if "HF_TOKEN" in os.environ:
        environment["HF_TOKEN"] = os.environ["HF_TOKEN"]
    request = JobRequest(
        name="snowball-v0-pilot-sft",
        entrypoint=Entrypoint.from_callable(run_worker, args=(config,)),
        resources=config.resources,
        environment=create_environment(extras=["gpu"], env_vars=environment),
        timeout=Duration.from_seconds(12 * 60 * 60),
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    current_client().submit(request, adopt_existing=False).wait(raise_on_failure=True)


def run_validation(config: TrainLmOnPodConfig, checkpoint_path: str, output_path: str) -> None:
    training = cast(TrainLmConfig, config.train_config)
    data = cast(SweepExportsConfig, training.adapter).validation_data
    converter = (
        cast(HFCompatConfig, training.model)
        .hf_checkpoint_converter()
        .replaced(reference_checkpoint=training.initialize_from_hf, tokenizer=data.the_tokenizer)
    )
    model = converter.config_from_hf_config(converter.default_hf_config)
    trainer = replace(
        training.trainer,
        id=training.trainer.id + "-offline-validation",
        mesh=OFFLINE_MESH,
        per_device_eval_parallelism=1,
        tracker=JsonLoggerConfig(),
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    validation = OfflineValidationConfig(checkpoint_path, output_path, model, data, trainer, CONTEXT)
    environment = {key: os.environ[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET", "MARIN_PREFIX", "HF_TOKEN")}
    environment.update(GPU_LOG_ENV)
    request = JobRequest(
        name="offline-validation",
        entrypoint=Entrypoint.from_callable(evaluate_checkpoint, args=(validation,)),
        resources=ResourceConfig.with_gpu(
            "H100", count=8, replicas=1, cpu=64, ram="1024g", disk="512g", regions=[ANY_REGION], preemptible=False
        ),
        environment=create_environment(extras=["gpu"], env_vars=environment),
        timeout=Duration.from_seconds(2 * 60 * 60),
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    current_client().submit(request, adopt_existing=False).wait(raise_on_failure=True)


def run_export(config: TrainLmOnPodConfig, checkpoint_path: str, output_path: str) -> None:
    training = cast(TrainLmConfig, config.train_config)
    export = ExportConfig(
        trainer=replace(
            training.trainer,
            id=training.trainer.id + "-offline-export",
            mesh=OFFLINE_MESH,
            tracker=JsonLoggerConfig(),
            log_jaxprs=False,
            log_xla_hlo=False,
        ),
        model=cast(HFCompatConfig, training.model),
        reference_checkpoint=training.initialize_from_hf,
        tokenizer=TOKENIZER,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        eos_token_ids=training.hf_generation_eos_token_ids,
    )
    environment = {key: os.environ[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET", "MARIN_PREFIX", "HF_TOKEN")}
    environment.update(GPU_LOG_ENV)
    request = JobRequest(
        name="offline-export",
        entrypoint=Entrypoint.from_callable(export_checkpoint, args=(export,)),
        resources=ResourceConfig.with_gpu(
            "H100", count=8, replicas=1, cpu=64, ram="1024g", disk="512g", regions=[ANY_REGION], preemptible=False
        ),
        environment=create_environment(extras=["gpu"], env_vars=environment),
        timeout=Duration.from_seconds(2 * 60 * 60),
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    current_client().submit(request, adopt_existing=False).wait(raise_on_failure=True)


def pipeline(
    report: dict, data_source: str, parent_uri: str, learning_rate: float, resume_checkpoint: str | None = None
) -> dict[str, ArtifactStep]:
    if report["student_revision"] != REVISION or report["student_model"] != MODEL:
        raise ValueError("Prepared data belongs to a different parent")
    if report["max_sequence_tokens"] != CONTEXT or report["batch_size"] != BATCH_SIZE:
        raise ValueError("Prepared data changes the frozen batch or context")
    if learning_rate not in LEARNING_RATES:
        raise ValueError("Learning rate is outside the authorized sweep")
    if parent_uri != report["student_regional_copy"]:
        raise ValueError("Parent URI differs from the verified regional copy")
    rows = report["training_rows"] + report["padding_rows"]
    if rows % BATCH_SIZE or report["supervised_tokens"] <= 0:
        raise ValueError("Empty or non-batch-aligned training data")
    steps_per_epoch = rows // BATCH_SIZE
    steps = EPOCHS * steps_per_epoch
    start_step = 0
    if resume_checkpoint is not None:
        metadata = json.loads(StoragePath(prefix_join(resume_checkpoint, "metadata.json")).read_text())
        start_step = int(metadata["step"]) + 1
        if start_step != steps_per_epoch or learning_rate != 1e-6:
            raise ValueError("Only the approved 1e-6 first-epoch continuation may resume")
    source = ArtifactStep.adopt("datasets/karan/v0-pilot-203", VERSION, data_source, config=report)
    parent = ArtifactStep.adopt(
        "checkpoints/karan/v0-pilot-parent", VERSION, parent_uri, config={"model": MODEL, "revision": REVISION}
    )

    def cache_config(ctx: StepContext, family: str = "train") -> TokenizeConfig:
        return TokenizeConfig(
            train_paths=[prefix_join(ctx.artifact_path(source), "train-prebuilt.jsonl.gz")] if family == "train" else [],
            validation_paths=(
                [] if family == "train" else [prefix_join(ctx.artifact_path(source), report["validation_files"][family])]
            ),
            allow_test_in_train=False,
            cache_path=ctx.output_path,
            tokenizer=TOKENIZER,
            format=FORMAT,
            max_workers=1,
            levanter_batch_size=8,
            worker_resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g"),
            tags=["v0-agentic-pilot", "task203"],
        )

    cache = ArtifactStep(
        name="tokenized/karan/v0-pilot-203-32k",
        version=VERSION,
        artifact_type=TokenizedCache,
        run=tokenize,
        build_config=cache_config,
        deps=(source,),
    )
    validation_caches = {
        family: ArtifactStep(
            name=f"tokenized/karan/v0-pilot-203-32k-{family}",
            version=VERSION,
            artifact_type=TokenizedCache,
            run=tokenize,
            build_config=partial(cache_config, family=family),
            deps=(source,),
        )
        for family in report["validation_files"]
    }
    model = HFModel(model_ref=f"{MODEL}@{REVISION}", tokenizer_path=TOKENIZER, model_type="snowball")
    resources = ResourceConfig.with_gpu(
        "H100", count=8, replicas=4, cpu=32, ram="512g", disk="512g", regions=[ANY_REGION], preemptible=False
    )
    name = f"v0-agentic-pilot-lr{learning_rate:g}"
    spec = SFTSpec(
        name=f"checkpoints/karan/{name}",
        version=TRAIN_VERSION,
        model=model,
        chat_template="",
        datasets=[],
        optimizer=optimizer(learning_rate),
        mesh=MeshConfig(
            axes={"data": 1, "replica": 1, "model": 1, "expert": 8},
            dcn_axes={"replica_dcn": 1, "context": 4},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
        ),
        seq_len=CONTEXT,
        pack=False,
        batch_size=BATCH_SIZE,
        num_train_steps=steps,
        hf_save_dtype="bfloat16",
        wandb_project="snowball-v0-agentic-sft-sweep",
    )

    def train_config(ctx: StepContext) -> TrainLmOnPodConfig:
        cache_path = ctx.artifact_path(cache)
        validation = replace(
            cached_data(cache_path, "validation"),
            components={
                family: DatasetComponent(
                    cache_dir=ctx.artifact_path(artifact),
                    format=FORMAT,
                    split="validation",
                    flat_cache=False,
                    pack=False,
                )
                for family, artifact in validation_caches.items()
            },
            train_weights=None,
        )
        pod = replace(model, model_ref=ctx.artifact_path(parent)).build_train_config(
            ctx, spec, cached_data(cache_path, "train"), resources, steps
        )
        config = cast(TrainLmConfig, pod.train_config)
        wandb = replace(
            cast(WandbConfig, config.trainer.tracker),
            entity="marin-community",
            name=name,
            resume="never",
            replicate_path=ctx.output_path,
            mode="online",
            save_code=False,
        )
        trainer = replace(
            config.trainer,
            id=name + "-20261017-r4",
            seed=42,
            load_checkpoint=resume_checkpoint is not None,
            load_checkpoint_path=resume_checkpoint,
            per_device_parallelism=2,
            per_device_eval_parallelism=2,
            tracker=(
                wandb,
                EpochTrackerConfig(ctx.output_path, steps_per_epoch, report["supervised_tokens"], start_step),
                JsonLoggerConfig(),
            ),
            checkpointer=replace(config.trainer.checkpointer, keep=[], save_interval=None),
        )
        return replace(
            pod,
            train_config=replace(
                config,
                trainer=trainer,
                data_seed=42,
                adapter=SweepExportsConfig(validation, steps_per_epoch, CONTEXT),
            ),
            env_vars={"WANDB_MODE": "online", **GPU_LOG_ENV},
        )

    trained = ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=LevanterCheckpoint,
        run=partial(run_training, report=report),
        build_config=train_config,
        deps=(cache, parent, *validation_caches.values()),
    )
    return {"cache": cache, "train": trained}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--parent-uri", required=True)
    parser.add_argument("--learning-rate", type=float, choices=LEARNING_RATES, required=True)
    parser.add_argument("--only", choices=("cache", "train", "validate", "export"), default="cache")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--validation-checkpoint")
    parser.add_argument("--validation-output")
    parser.add_argument("--export-checkpoint")
    parser.add_argument("--export-output")
    args = parser.parse_args()
    report_bytes = StoragePath(args.report).read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != args.report_sha256:
        raise ValueError("Prepared manifest hash mismatch")
    report = json.loads(report_bytes)
    data_source = args.report.rsplit("/", 1)[0]
    artifacts = pipeline(report, data_source, args.parent_uri, args.learning_rate, args.resume_checkpoint)
    selected = artifacts["train" if args.only in ("validate", "export") else args.only]
    if args.only == "validate" and (not args.validation_checkpoint or not args.validation_output):
        raise ValueError("Offline validation requires an explicit checkpoint and output")
    if args.only == "export" and (not args.export_checkpoint or not args.export_output):
        raise ValueError("Offline export requires an explicit checkpoint and output")
    if not args.run:
        print(lower(selected))
        config = materialized_config(artifacts["train"], "/tmp/preflight")
        print("mesh", config.train_config.trainer.mesh.axis_shapes(num_devices=32, num_slices=4))
        print("completed_updates", config.train_config.trainer.num_train_steps)
        return
    for filename in ("train-prebuilt.jsonl.gz", *report["validation_files"].values()):
        digest = hashlib.sha256(StoragePath(prefix_join(data_source, filename)).read_bytes()).hexdigest()
        if digest != report["output_hashes"][filename]:
            raise ValueError(f"Prepared data hash mismatch: {filename}")
    info = get_job_info()
    assert info is not None and info.controller_address is not None
    with IrisClient.in_cluster(info.controller_address, workspace=Path.cwd()) as client:
        context = IrisContext.from_job_info(info, client=client)
        with iris_ctx_scope(context), set_current_client(FrayIrisClient.from_iris_client(client)):
            if args.only == "validate":
                config = materialized_config(artifacts["train"], os.environ["MARIN_PREFIX"])
                run_validation(config, args.validation_checkpoint, args.validation_output)
                return
            if args.only == "export":
                config = materialized_config(artifacts["train"], os.environ["MARIN_PREFIX"])
                run_export(config, args.export_checkpoint, args.export_output)
                return
            result = run(selected, max_concurrent=1)[0]
    print(json.dumps({"completed": args.only, "output": result.path}), flush=True)


if __name__ == "__main__":
    main()
