# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the plain or rejection-sampled SFT transfer arm from the frozen baseline."""

import argparse
import json
import os
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

from fray.current_client import current_client, set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import ANY_REGION, Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import get_job_info
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, materialized_config, run
from marin.processing.tokenize.tokenize import TokenizeConfig, TokenizedCache, tokenize
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, run_levanter_train_lm
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import Duration

from experiments.post_training.cruxeval_transfer.cache_integrity import verify_cache
from experiments.post_training.cruxeval_transfer.constants import PARENT_DESTINATION, STUDENT_MODEL, STUDENT_REVISION
from experiments.sft.launcher import HFModel, SFTSpec

TOKENIZER = f"hf://{STUDENT_MODEL}@{STUDENT_REVISION}"
FORMAT = PrebuiltLmDatasetFormat(input_ids_key="input_ids", loss_weights_key="loss_weights")
VERSION = "2026.10.08.1"


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
        min_lr_ratio=0.1,
    )


def run_training(config: TrainLmOnPodConfig, report: dict) -> None:
    assert config.env_vars is not None
    if config.env_vars["WANDB_MODE"] == "online" and not os.environ.get("WANDB_API_KEY"):
        raise ValueError("Online W&B requires a nonempty WANDB_API_KEY before requesting GPUs")
    training = cast(TrainLmConfig, config.train_config)
    verify_cache(training.data.components["cruxeval"].cache_dir, report)
    environment = {key: os.environ[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET", "MARIN_PREFIX")}
    if "WANDB_API_KEY" in os.environ:
        environment["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    request = JobRequest(
        name="cruxeval-snowball-sft",
        entrypoint=Entrypoint.from_callable(run_worker, args=(config,)),
        resources=config.resources,
        environment=create_environment(extras=["gpu"], env_vars=environment),
        timeout=Duration.from_seconds(12 * 60 * 60),
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    current_client().submit(request, adopt_existing=False).wait(raise_on_failure=True)


def run_worker(config: TrainLmOnPodConfig) -> None:
    os.environ["WANDB_DIR"] = os.environ["IRIS_OUTPUT_DIR"]
    run_levanter_train_lm(config)


def pipeline(
    report: dict, learning_rate: float, entity: str, wandb_mode: str, data_source: str, arm: str
) -> dict[str, ArtifactStep]:
    if report["student_revision"] != STUDENT_REVISION or not report["test_exposed"]:
        raise ValueError("Wrong parent or data protocol")
    steps = 4 * report["steps_per_epoch"]
    source = ArtifactStep.adopt(f"datasets/karan/cruxeval-{arm}-e", VERSION, data_source, config=report)
    parent = ArtifactStep.adopt(
        "checkpoints/karan/cruxeval-parent",
        VERSION,
        "s3://" + PARENT_DESTINATION,
        config={"model": STUDENT_MODEL, "revision": STUDENT_REVISION},
    )

    def cache_config(ctx: StepContext) -> TokenizeConfig:
        return TokenizeConfig(
            train_paths=[prefix_join(ctx.artifact_path(source), "train-prebuilt.jsonl.gz")],
            validation_paths=[],
            allow_test_in_train=True,
            cache_path=ctx.output_path,
            tokenizer=TOKENIZER,
            format=FORMAT,
            max_workers=1,
            levanter_batch_size=8,
            worker_resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g"),
            tags=["cruxeval", "test-exposed", f"{arm}-sft-e"],
        )

    cache = ArtifactStep(
        name=f"tokenized/karan/cruxeval-{arm}-e-32k",
        version=VERSION,
        artifact_type=TokenizedCache,
        run=tokenize,
        build_config=cache_config,
        deps=(source,),
    )
    model = HFModel(model_ref=f"{STUDENT_MODEL}@{STUDENT_REVISION}", tokenizer_path=TOKENIZER, model_type="snowball")
    resources = ResourceConfig.with_gpu(
        "H100", count=8, replicas=4, cpu=32, ram="512g", disk="512g", regions=[ANY_REGION], preemptible=False
    )
    name = f"cruxeval-{arm}-e-from-b-lr{learning_rate:g}"
    spec = SFTSpec(
        name=f"checkpoints/karan/{name}",
        version=VERSION,
        model=model,
        chat_template="",
        datasets=[],
        optimizer=optimizer(learning_rate),
        mesh=MeshConfig(
            axes={"data": 1, "replica": 1, "model": 1, "expert": 8},
            dcn_axes={"replica_dcn": 1, "context": 4},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
        ),
        seq_len=report["max_sequence_tokens"],
        pack=False,
        batch_size=report["batch_size"],
        num_train_steps=steps,
        hf_save_dtype="bfloat16",
        wandb_project="cruxeval-transfer-133",
    )

    def train_config(ctx: StepContext) -> TrainLmOnPodConfig:
        cache_path = ctx.artifact_path(cache)
        component = DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=[], cache_dir=cache_path, format=FORMAT),
            cache_dir=cache_path,
            format=FORMAT,
            split="train",
        )
        data = LmDataConfig(
            tokenizer=TOKENIZER,
            components={"cruxeval": component},
            train_weights={"cruxeval": 1.0},
            enforce_eos=False,
            auto_build_caches=False,
            shuffle=True,
            block_cross_document_attention=False,
        )
        regional_model = replace(model, model_ref=ctx.artifact_path(parent))
        pod = regional_model.build_train_config(ctx, spec, data, resources, steps)
        config = cast(TrainLmConfig, pod.train_config)
        trainer = replace(
            config.trainer,
            id=name + "-20261008",
            seed=42,
            load_checkpoint=False,
            tracker=(
                replace(
                    cast(WandbConfig, config.trainer.tracker),
                    entity=entity,
                    name=name,
                    resume="never",
                    replicate_path=ctx.output_path,
                    mode=wandb_mode,
                    save_code=False,
                ),
                JsonLoggerConfig(),
            ),
            checkpointer=replace(config.trainer.checkpointer, keep=[{"every": report["steps_per_epoch"]}]),
        )
        return replace(
            pod,
            train_config=replace(
                config,
                trainer=trainer,
                hf_save_steps=report["steps_per_epoch"],
                hf_save_reference_code=False,
                data_seed=42,
            ),
            env_vars={"WANDB_MODE": wandb_mode},
        )

    trained = ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=LevanterCheckpoint,
        run=partial(run_training, report=report),
        build_config=train_config,
        deps=(cache, parent),
    )
    return {"cache": cache, "train": trained}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--arm", choices=("plain", "rs"), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), required=True)
    parser.add_argument("--only", choices=("cache", "train"), default="cache")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    report = json.loads(StoragePath(args.report).read_text())
    data_source = str(StoragePath(args.report).parent)
    artifacts = pipeline(report, args.learning_rate, args.wandb_entity, args.wandb_mode, data_source, args.arm)
    selected = artifacts[args.only]
    if not args.run:
        print(lower(selected))
        config = materialized_config(artifacts["train"], "/tmp/preflight")
        shapes = config.train_config.trainer.mesh.axis_shapes(num_devices=32, num_slices=4)
        print("mesh", shapes, "steps", config.train_config.trainer.num_train_steps)
        return
    if args.only == "train":
        receipt = json.loads((StoragePath(f"s3://{PARENT_DESTINATION}") / "task133-readiness.json").read_text())
        if receipt["objects"] != 87 or receipt["bytes"] != 134175117778:
            raise ValueError("Regional parent not verified")
    info = get_job_info()
    assert info is not None and info.controller_address is not None
    with IrisClient.in_cluster(info.controller_address, workspace=Path.cwd()) as client:
        context = IrisContext.from_job_info(info, client=client)
        with iris_ctx_scope(context), set_current_client(FrayIrisClient.from_iris_client(client)):
            result = run(selected, max_concurrent=1)[0]
    print(json.dumps({"completed": args.only, "output": result.path}), flush=True)


if __name__ == "__main__":
    main()
