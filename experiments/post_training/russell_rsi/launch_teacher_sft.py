# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect verified Russell teacher conversations and run one Snowball SFT update."""

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, TypeVar, cast

import click
import httpx
from fray.types import ResourceConfig
from levanter.callbacks.watch import WatchConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.config import AdamConfig
from levanter.tokenizers import load_tokenizer
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle, install_runtime_bundle
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.parquet import read_tasks

from experiments.evaluation.pipeline import eval_step
from experiments.post_training.glm import resolve_glm_base_url
from experiments.post_training.russell_rsi.bootstrap_loop import checkpoint_score, promotes, write_once
from experiments.post_training.russell_rsi.collection_recovery import CollectionRecovery, StudentContextAmendment
from experiments.post_training.russell_rsi.launch import CLUSTER, evaluation_model
from experiments.post_training.russell_rsi.launch_dose_comparison import selected_dose
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.rollout_eval import qemu_factory
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_collection import (
    STUDENT_CONTEXT_TOKENS,
    STUDENT_ROWS,
    TeacherModelConfig,
    collect_teacher_rows,
    selected_teacher_tasks,
)
from experiments.sft.launcher import ArtifactDatasetSpec, PreparedModel, SFTSpec, sft_step

SFT_NODES = 4
SFT_LEARNING_RATE = 1e-6
TEACHER_PIP_PACKAGES = ("./lib/rolloutengine", "./lib/taskcompendium", "./lib/shellbox")
CollectionConfig = TypeVar("CollectionConfig")


@dataclass(frozen=True)
class TeacherCollectionConfig:
    selection: dict
    dose_decision_uri: str
    dose_decision_sha256: str
    bank_path: str
    parent_path: str
    parent_identity: str
    tokenizer_files: dict[str, str]
    runtime_bundle: RuntimeBundle
    relay_job: str
    output_path: str


def require_teacher_condition(config: TeacherCollectionConfig) -> dict:
    """Require the frozen dose choice to fail the unchanged parent promotion gate."""
    decision = json.loads(pinned_bytes(config.dose_decision_uri, config.dose_decision_sha256))
    parent = checkpoint_score(decision["parent"])
    candidate = checkpoint_score(decision["selected"])
    selected = selected_dose(checkpoint_score(decision["four"]), checkpoint_score(decision["eight"]))
    if candidate != selected or parent.checkpoint_identity != config.parent_identity:
        raise ValueError("Teacher condition differs from the frozen dose choice or parent")
    if promotes(candidate, parent):
        raise ValueError("Dose improved the parent; the conditional teacher experiment cannot start")
    return decision


def run_teacher_collection(config: TeacherCollectionConfig) -> None:
    """Verify frozen inputs before the first teacher request and emit eight JSONL rows."""
    collect_teacher_dataset(config, require_teacher_condition(config))


@dataclass(frozen=True)
class StudentTrainingTemplate:
    uri: str
    sha256: str


def collect_teacher_dataset(
    config: TeacherCollectionConfig,
    decision: dict,
    training_template: StudentTrainingTemplate | None = None,
    condition_hash_field: str = "dose_decision_sha256",
    recovery: CollectionRecovery | None = None,
    context_amendment: StudentContextAmendment | None = None,
) -> None:
    """Collect the frozen dataset after the caller verifies its study condition."""
    selection = config.selection
    bank_path = StoragePath(config.bank_path)
    bank = json.loads(pinned_bytes(str(bank_path / "bank.json"), selection["bank_sha256"]))
    train_bytes = pinned_bytes(str(bank_path / "train.parquet"), selection["train_sha256"])
    ordered_ids = tuple(row["task_id"] for row in selection["selected"])
    selected = selected_teacher_tasks(bank, selection["capabilities"], ordered_ids)
    output = StoragePath(config.output_path)
    with tempfile.TemporaryDirectory(prefix="russell-teacher-inputs-") as temporary:
        root = Path(temporary)
        tasks_path = root / "train.parquet"
        tasks_path.write_bytes(train_bytes)
        tasks = {task.id: task for task in read_tasks(str(tasks_path))}
        for name, digest in config.tokenizer_files.items():
            if Path(name).name != name:
                raise ValueError("Tokenizer inputs must be files at the parent export root")
            (root / name).write_bytes(pinned_bytes(str(StoragePath(config.parent_path) / name), digest))
        if training_template is not None:
            if "training_chat_template.jinja" in config.tokenizer_files:
                raise ValueError("The external student template cannot also be a parent tokenizer input")
            (root / "training_chat_template.jinja").write_bytes(
                pinned_bytes(training_template.uri, training_template.sha256)
            )
        tokenizer = load_tokenizer(str(root))
        if (root / "training_chat_template.jinja").read_text() != MARIN_CHAT_TEMPLATE:
            raise ValueError("Student template differs from the pinned parent training template")
        runtime = install_runtime_bundle(config.runtime_bundle)
        tokenizer_inputs = {"parent_identity": config.parent_identity, "files": config.tokenizer_files}
        if training_template is not None:
            tokenizer_inputs["student_training_template"] = {
                "uri": training_template.uri,
                "sha256": training_template.sha256,
            }
        tokenizer_identity = compact_json_sha256(tokenizer_inputs)
        write_once(
            output / "inputs.json",
            {
                "selection_sha256": compact_json_sha256(selection),
                condition_hash_field: compact_json_sha256(decision),
                "tokenizer_files": config.tokenizer_files,
                "parent_identity": config.parent_identity,
                "student_tokenizer_identity": tokenizer_identity,
            },
        )
        model = selection["teacher_model"]

        async def collect() -> dict:
            async with httpx.AsyncClient(
                timeout=600,
                headers={"Authorization": f"Bearer {os.environ[GLM_TOKEN_ENV]}"},
                transport=httpx.AsyncHTTPTransport(retries=0),
            ) as client:
                return await collect_teacher_rows(
                    selected,
                    tasks,
                    selection["capabilities"],
                    tokenizer,
                    tokenizer_identity,
                    client,
                    lambda: resolve_glm_base_url(config.relay_job),
                    TeacherModelConfig(
                        compact_json_sha256(selection),
                        model["max_tokens"],
                        model["temperature"],
                        model["reasoning_effort"],
                    ),
                    {
                        EnvironmentKind.SHELLSIM: ShellSimMachineFactory(),
                        EnvironmentKind.DOCKER: qemu_factory(runtime, config.runtime_bundle),
                    },
                    output,
                    recovery=recovery,
                    context_amendment=context_amendment,
                )

        result = asyncio.run(collect())
    if result["status"] != "passed":
        raise ValueError("The frozen teacher budget did not produce eight qualified student rows")
    content = "".join(json.dumps(row["row"], sort_keys=True) + "\n" for row in result["accepted"])
    rows_path = output / "train.jsonl"
    if rows_path.exists():
        if rows_path.read_text() != content:
            raise ValueError("Student rows differ from the saved eight-row dataset")
    else:
        rows_path.write_text(content)
    write_once(
        output / "dataset.json",
        {
            "rows": len(result["accepted"]),
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "collection_sha256": compact_json_sha256(result),
        },
    )


def run_teacher_collection_remote(config: TeacherCollectionConfig) -> None:
    remote(
        run_teacher_collection,
        resources=ResourceConfig.with_cpu(cpu=8, ram="64GB", disk="64GB"),
        pip_packages=list(TEACHER_PIP_PACKAGES),
        env_vars={GLM_TOKEN_ENV: os.environ[GLM_TOKEN_ENV]},
    )(config)


def teacher_sft_workflow(config: dict) -> dict[str, ArtifactStep]:
    """Bind verified conversations and the pinned parent to one standard SFT update."""
    return teacher_sft_steps(config, CollectionBinding(lambda base: base, run_teacher_collection_remote), 1, "teacher")


@dataclass(frozen=True)
class CollectionBinding(Generic[CollectionConfig]):
    build: Callable[[TeacherCollectionConfig], CollectionConfig]
    run: Callable[[CollectionConfig], None]


def teacher_sft_steps(
    config: dict,
    collection: CollectionBinding[CollectionConfig],
    updates: int,
    namespace: str,
    context_tokens: int = STUDENT_CONTEXT_TOKENS,
) -> dict[str, ArtifactStep]:
    version = config["version"]
    parent_spec = config["parent"]
    parent = ArtifactStep.adopt(
        parent_spec["name"],
        parent_spec["version"],
        parent_spec["uri"],
        kind=LevanterCheckpoint,
        config=parent_spec["identity_config"],
    )
    bank_spec = config["bank"]
    bank = ArtifactStep.adopt(
        bank_spec["name"], bank_spec["version"], bank_spec["uri"], config=bank_spec["identity_config"]
    )

    def collection_config(ctx: StepContext) -> CollectionConfig:
        base = TeacherCollectionConfig(
            config["selection"],
            config["dose_decision_uri"],
            config["dose_decision_sha256"],
            ctx.artifact_path(bank),
            ctx.artifact_path(parent),
            artifact_identity(parent),
            config["tokenizer_files"],
            RuntimeBundle(**config["runtime_bundle"]),
            config["relay_job"],
            ctx.output_path,
        )
        return collection.build(base)

    collected = ArtifactStep(
        name=f"documents/russell-rsi-{namespace}-conversations",
        version=version,
        artifact_type=Artifact,
        deps=(parent, bank),
        build_config=collection_config,
        run=collection.run,
    )
    spec = SFTSpec(
        name=f"checkpoints/russell-rsi-{namespace}-sft",
        version=version,
        model=PreparedModel(parent, model_type="snowball"),
        chat_template=MARIN_CHAT_TEMPLATE,
        datasets=(ArtifactDatasetSpec("russell-teacher", collected, "train.jsonl", 1.0),),
        optimizer=AdamConfig(
            learning_rate=SFT_LEARNING_RATE,
            lr_schedule="constant",
            warmup=0,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            weight_decay=0,
            max_grad_norm=1,
        ),
        mesh=MeshConfig(
            axes={"data": 1, "replica": 1, "model": 1, "context": SFT_NODES, "expert": 8},
            dcn_axes={"replica_dcn": 1},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
        ),
        seq_len=context_tokens,
        pack=False,
        batch_size=STUDENT_ROWS,
        num_train_steps=updates,
        hf_save_dtype="bfloat16",
        wandb_project="russell-rsi",
    )
    trained = sft_step(
        spec,
        ResourceConfig.with_gpu(
            "H100",
            count=8,
            cpu=32,
            ram="512GB",
            disk="512GB",
            replicas=SFT_NODES,
            preemptible=False,
        ),
    )
    original_build_config = trained.build_config

    def training_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod = cast(TrainLmOnPodConfig, original_build_config(ctx))
        train = cast(TrainLmConfig, pod.train_config)
        watch = WatchConfig(watch_targets=["grads", "updates"], include_per_parameter_norms=False, interval=1)
        trainer = replace(train.trainer, id=f"russell-rsi-{namespace}-sft-{version}", watch=watch)
        env_vars = pod.env_vars
        if "student_context_amendment" in config:
            trainer = replace(trainer, metrics_start_step=0, tracker=(JsonLoggerConfig(),))
            env_vars = {**(env_vars or {}), "WANDB_MODE": "disabled"}
        return replace(
            pod,
            env_vars=env_vars,
            train_config=replace(
                train,
                data=replace(train.data, mixture_block_size=STUDENT_ROWS),
                trainer=trainer,
            ),
        )

    trained = replace(trained, build_config=training_config)
    reload_model = evaluation_model(f"russell-rsi-{namespace}-sft-reload", "<completed-sft-export>", None)
    reload = eval_step(
        reload_model,
        "mmlu-smoke",
        version=version,
        deps=(trained,),
        resolve_model=lambda ctx: replace(
            reload_model,
            location=prefix_join(ctx.artifact_path(trained), f"hf/step-{updates - 1}"),
            identity=artifact_identity(trained),
        ),
        limit=1,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    return {"collect": collected, "train": trained, "reload": reload}


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--stage", type=click.Choice(["collect", "train", "reload"]), required=True)
@build_options
def main(config_uri: str, config_sha256: str, stage: str) -> dict[str, ArtifactStep]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if resolve_version("russell-rsi-teacher-sft", None) != config["version"]:
        raise click.UsageError("Teacher config and artifact version differ")
    return {stage: teacher_sft_workflow(config)[stage]}


if __name__ == "__main__":
    main()
