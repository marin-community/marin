# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train, export, resume, and evaluate the nested SkyRL runtime on two H100s."""

from dataclasses import dataclass, replace

import click
import yaml
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)

from experiments.evaluation.pipeline import EvaluationResult
from experiments.models import qwen2_5_0_5b_instruct
from experiments.post_training.cat_count_canary.data import TRAIN_FILENAME, VALIDATION_FILENAME, cat_count_data_step
from experiments.post_training.cat_count_canary.launcher import role_plan, training_config
from experiments.post_training.skyrl_evaluation import SKYRL_POLICY_LOCATION, skyrl_eval_step

CLUSTER = "cw-rno2a"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 17
TRAIN_COUNTS = (4, 6, 8, 12)
BATCH_SIZE = 4
GROUP_SIZE = 4
MICRO_BATCH_SIZE = 1


@dataclass(frozen=True)
class Smoke:
    train: ArtifactStep[SkyRLRun]
    resume: ArtifactStep[SkyRLRun]
    evaluate: ArtifactStep[EvaluationResult]


def build_smoke(version: str) -> Smoke:
    data = cat_count_data_step(
        user_owned_name("documents/skyrl-port-smoke"),
        version,
        train_ns=TRAIN_COUNTS,
        train_rows=64,
        seed=SEED,
    )
    plan = replace(
        role_plan(batch_size=BATCH_SIZE, group_size=GROUP_SIZE, micro_train_batch_size=MICRO_BATCH_SIZE),
        colocate_all=True,
    )
    execution = IrisSkyRLExecution(
        cluster=CLUSTER,
        cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
        cpu=32,
        memory="128GB",
        disk="200GB",
        priority="interactive",
        max_retries=0,
        target_cluster=CLUSTER,
        parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
        coordinator_timeout_hours=2,
        job_timeout_seconds=2400,
    )

    def training_step(name: str, max_steps: int) -> ArtifactStep[SkyRLRun]:
        config = training_config(
            preset="dry",
            lane="sync",
            batch_size=BATCH_SIZE,
            group_size=GROUP_SIZE,
            micro_train_batch_size=MICRO_BATCH_SIZE,
            train_ns=TRAIN_COUNTS,
            checkpoint=True,
            export=True,
            settings=(f"trainer.max_steps={max_steps}", "trainer.logger=console", "trainer.update_epochs_per_batch=1"),
        )
        _materialize_role_plan_config(config, plan)
        config["generator"]["gpu_memory_utilization"] = 0.4
        config["trajectory_runner"] = {
            "max_concurrent_tasks": 4,
            "rollout_workers": {"num_workers": 2, "cpus_per_worker": 2},
        }
        config["trainer"]["callbacks"] = [
            callback for callback in config["trainer"]["callbacks"] if callback["type"] != "database_registration"
        ]
        for callback in config["trainer"]["callbacks"]:
            if callback["type"] == "evaluation":
                callback["additional_evaluations"] = {}
                callback["metric_groups"] = {}
        return skyrl_step(
            SkyRLSpec(
                name=user_owned_name(f"checkpoints/skyrl-port-smoke/{name}"),
                version=version,
                config_yaml=yaml.safe_dump(config),
                model=ArtifactHfModel(
                    step=qwen2_5_0_5b_instruct,
                    relative_path="",
                    tokenizer_uri=MODEL,
                    tokenizer_revision=MODEL_REVISION,
                ),
                train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
                validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
                topology=SkyRLTopology(num_nodes=1, gpus_per_node=2, gpu_variant="H100", role_plan=plan),
                retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
                seed=SEED,
            ),
            execution,
            export_hf=True,
        )

    train = training_step("train", 2)
    resume = training_step("resume", 4)
    build_resume = resume.build_config

    def resume_config(ctx: StepContext) -> SkyRLRunConfig:
        config = build_resume(ctx)
        checkpoint_root = (
            f"<resume-checkpoints:{artifact_identity(train)}>"
            if ctx.is_fingerprint
            else ctx.resolved(train).checkpoint_root
        )
        launch = yaml.safe_load(config.launch_config_yaml)
        launch["artifacts"]["checkpoint_root"] = checkpoint_root
        return replace(
            config,
            launch_config_yaml=yaml.safe_dump(launch),
            output=replace(config.output, checkpoint_root=checkpoint_root),
        )

    resume = replace(resume, build_config=resume_config, deps=(*resume.deps, train))
    evaluate = skyrl_eval_step(
        resume,
        ModelConfig(
            name="skyrl-port-smoke",
            location=SKYRL_POLICY_LOCATION,
            tokenizer=MODEL,
            tokenizer_revision=MODEL_REVISION,
            apply_chat_template=True,
            resource_hint=ResourceHint(gpu={"H100": 1}),
            serve=ServeConfig(tensor_parallel_size=1, max_model_len=4096, max_num_seqs=16),
            generation=GenerationConfig(max_gen_toks=256),
        ),
        "gsm8k-smoke",
        version=version,
        accelerator="H100x1",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    return Smoke(train, resume, evaluate)


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(("train", "resume", "evaluate")), default="evaluate")
@rl_build_options
def main(stage: str) -> ArtifactStep:
    smoke = build_smoke(resolve_version("skyrl-port-smoke", None))
    return {"train": smoke.train, "resume": smoke.resume, "evaluate": smoke.evaluate}[stage]


if __name__ == "__main__":
    main()
