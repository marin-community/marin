# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local inputs for each long-loop runtime-document producer."""

import hashlib
import json
from functools import partial
from pathlib import Path

import draccus
import yaml
from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, SkyRLRolePlan
from marin.training.training import LevanterCheckpoint

from experiments.post_training.long_loop import evaluation, rl
from experiments.post_training.long_loop.nvl72 import NVL72RunSpec, run_from_spec
from experiments.post_training.long_loop.nvl72_evaluation import NVL72EvaluationSpec, evaluation_from_spec
from experiments.post_training.long_loop.spec_pipeline import runtime_execution, stage_execution

SPECS = Path(__file__).resolve().parents[2] / "experiments/post_training/long_loop/specs"
VERSION = "2026.10.24"


def configured_training(root: Path):
    recipe = yaml.safe_load((SPECS / "rl.yaml").read_text())
    coordinator = runtime_execution(yaml.safe_load((SPECS / "execution.yaml").read_text())["coordinator"])
    model = ArtifactStep.adopt("census/long-loop-model", VERSION, str(root / "policy"), kind=LevanterCheckpoint)
    data = ArtifactStep.adopt("census/long-loop-data", VERSION, str(root / "data"))
    return rl.configured_rl_step(
        version=VERSION,
        model=ArtifactHfModel(model, tokenizer_uri="fixture/tokenizer", tokenizer_revision="a" * 40),
        train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
        validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
        config=recipe["config"],
        plan=draccus.decode(SkyRLRolePlan, recipe["role_plan"]),
        execution=stage_execution(recipe["execution"], coordinator),
        name=recipe["name"],
        retained=recipe["retained"],
        seed=recipe["seed"],
        ray_timeouts=recipe["ray_timeouts"],
    )


def configured_evaluation(root: Path):
    recipe = yaml.safe_load((SPECS / "evaluation.yaml").read_text())
    coordinator = runtime_execution(yaml.safe_load((SPECS / "execution.yaml").read_text())["coordinator"])
    return evaluation.configured_evaluation_step(
        VERSION,
        configured_training(root),
        ArtifactStep.adopt("census/eval-data", VERSION, str(root / "data")),
        name=recipe["name"],
        config=recipe["config"],
        role_plan=draccus.decode(SkyRLRolePlan, recipe["role_plan"]),
        execution=stage_execution(recipe["execution"], coordinator),
        judge=draccus.decode(ModelConfig, recipe["judge"]),
        judge_gpu_count=recipe["judge_gpu_count"],
        seed=recipe["seed"],
        expected_step=2,
        gpu_variant=recipe["gpu_variant"],
        capability_origin=recipe["capability_origin"],
        data_relative_path=recipe["data_relative_path"],
        ray_timeouts=recipe["ray_timeouts"],
    )


def nvl72_training(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    raw = b"local training task fixture"
    (root / "train.parquet").write_bytes(raw)
    spec = NVL72RunSpec(
        format="nvl72_rl",
        marin_revision="a" * 40,
        output_prefix=str(root),
        profile="snowball-one-rack",
        model_uri=str(root / "policy"),
        tokenizer_uri="fixture/tokenizer",
        tokenizer_revision="a" * 40,
        data_uri=(str(root),),
        data_sha256=hashlib.sha256(raw).hexdigest(),
        input_version=VERSION,
        name="census/nvl72-train",
        max_steps=2,
    )
    return run_from_spec(spec, smoke=True, config_output=root / "nvl72-train.yaml")


def nvl72_evaluation(root: Path):
    model = root / "nvl72-model"
    model.mkdir(parents=True, exist_ok=True)
    manifest = json.dumps({"request": {"config": {"checkpoint": "fixture/step-2"}}}).encode()
    config = json.dumps({"num_experts": 384, "max_position_embeddings": 262144}).encode()
    (model / "export-manifest.json").write_bytes(manifest)
    (model / "config.json").write_bytes(config)
    raw = b"local evaluation task fixture"
    (root / "rollouts.parquet").write_bytes(raw)
    spec = NVL72EvaluationSpec(
        format="nvl72_eval",
        marin_revision="a" * 40,
        output_prefix=str(root),
        name="census/nvl72-eval",
        model_uri=str(model),
        model_step=2,
        model_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        model_config_sha256=hashlib.sha256(config).hexdigest(),
        tokenizer_uri="fixture/tokenizer",
        tokenizer_revision="a" * 40,
        data_uri=str(root),
        data_sha256=hashlib.sha256(raw).hexdigest(),
        input_version=VERSION,
        job_timeout_hours=2,
        context_tokens=65536,
        max_new_tokens=16384,
    )
    return evaluation_from_spec(spec, root / "nvl72-eval.yaml")


def launch_builders(root: Path):
    return {
        f"long-loop/{build.__name__}": partial(build, root)
        for build in (configured_training, configured_evaluation, nvl72_training, nvl72_evaluation)
    }
