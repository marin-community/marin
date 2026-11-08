# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Micro210 preparation and Snowball evaluation using the existing Mini runtime."""

import os
import subprocess
from dataclasses import dataclass, replace

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from fray.types import ResourceConfig
from huggingface_hub import hf_hub_download
from iris.rpc import job_pb2
from marin.evaluation.model_config import load_model_config
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, artifact_identity
from marin.execution.remote import remote
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    IrisSkyRLExecution,
    ResolvedModelLocator,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)
from rigging.config_discovery import find_project_root
from rigging.filesystem.buckets import filesystem_for

from experiments.post_training.long_loop.eval_runtime.judge import JudgedEvaluation, run_judged
from experiments.post_training.long_loop.eval_runtime.profile import DATASET, REVISION
from experiments.post_training.long_loop.eval_runtime.recipe import recipe, validate_environment_preflight
from experiments.post_training.long_loop.eval_runtime.repair_release import repaired_table
from experiments.post_training.long_loop.sft import read_json

WHEELS_REVISION = "d5dd315d58f0f47514dd0a2208e90e2a48541328"


def prepare_micro(*, output: str, dataset: str, revision: str, wheels_revision: str) -> None:
    path = hf_hub_download(dataset, "data/validation.parquet", repo_type="dataset", revision=revision)
    table = pq.read_table(path)
    if table.num_rows != 210:
        raise ValueError("Pinned Micro210 must contain 210 unique tasks")
    chosen = []
    seen = set()
    for i, row in enumerate(table.to_pylist()):
        if row["benchmark"] not in seen:
            chosen.append(i)
            seen.add(row["benchmark"])
    if len(chosen) != 21:
        raise ValueError("Micro smoke must cover all 21 benchmark paths")
    smoke = repaired_table(table.take(pa.array(chosen)), wheels_revision)
    fs, key = filesystem_for(output + "/rollouts.parquet")
    with fs.open(key, "wb") as target:
        pq.write_table(smoke, target)
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "--with-editable",
            "./vendor/harbor",
            "--with",
            "daytona==0.200.2",
            "--prerelease",
            "allow",
            "python",
            "-m",
            "experiments.post_training.long_loop.eval_runtime.environment_preflight",
            "--smoke",
            "--credential-env",
            "DAYTONA_EVAL_API_KEY",
            "--data-uri",
            output,
            "--output-prefix",
            output,
            "--report-name",
            "environment.json",
        ],
        check=True,
    )
    validate_environment_preflight(read_json(output + "/environment.json"), output)


@dataclass(frozen=True)
class TerminalPolicy:
    step: ArtifactStep[SkyRLRun]

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ResolvedModelLocator:
        if ctx.is_fingerprint:
            return ResolvedModelLocator(
                uri=artifact_identity(self.step) + "/terminal-export",
                identity=artifact_identity(self.step),
                local_path="models/long-loop-terminal",
                tokenizer_uri="<terminal-tokenizer>",
                tokenizer_revision="<terminal-tokenizer-revision>",
            )
        result = ctx.resolved(self.step)
        if result.hf_model_uri is None or result.global_step != 3:
            raise ValueError("Micro smoke requires the committed final step-3 RL export")
        return ResolvedModelLocator(
            uri=result.hf_model_uri,
            identity=artifact_identity(self.step),
            local_path="models/long-loop-terminal",
            tokenizer_uri=result.tokenizer_uri,
            tokenizer_revision=result.tokenizer_revision,
        )


def smoke_evaluation(version: str, trained: ArtifactStep[SkyRLRun], data: ArtifactStep) -> ArtifactStep:
    config, plan = recipe({}, {}, {}, "runtime://minimax-judge", "DAYTONA_EVAL_API_KEY")
    plan = replace(
        plan,
        policy_num_gpus_per_node=8,
        inference_engine_tensor_parallel_size=1,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
    )
    _materialize_role_plan_config(config, plan)
    for role in ("policy", "ref"):
        config["trainer"][role]["megatron_config"].update(tensor_model_parallel_size=1, expert_model_parallel_size=8)
    config["generator"]["engine_init_kwargs"].update(tool_call_parser="hermes")
    config["generator"]["sampling_params"].update(skip_special_tokens=False, stop_token_ids=[128001, 128009])
    config["generator"]["eval_sampling_params"].update(skip_special_tokens=False, stop_token_ids=[128001, 128009])
    step = skyrl_step(
        SkyRLSpec(
            name="long-loop/micro-smoke",
            version=version,
            config_yaml=yaml.safe_dump(config),
            model=TerminalPolicy(trained),
            train_data=(),
            validation_data=(ArtifactDataSource(data, relative_path="rollouts.parquet"),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=20261030,
        ),
        IrisSkyRLExecution(
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            cpu=32,
            memory="512GB",
            disk="256GB",
            priority="interactive",
            max_retries=0,
            target_cluster="cw-rno2a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=36,
            coordinator_name=f"long-loop-plumbing-{version}",
            job_timeout_seconds=86400,
            required_env_vars=("DAYTONA_EVAL_API_KEY", "DAYTONA_TARGET", "TOGETHER_API_KEY", "MARIN_PREFIX"),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=False,
    )
    root = find_project_root()
    assert root is not None, "Run from the Marin worktree"
    judge = load_model_config(
        root / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/"
        "olympiadbench_judge/MiniMaxAI-MiniMax-M3-MXFP8-131k.yaml"
    )
    cluster = yaml.safe_load((root / IRIS_HUB_CLUSTER_CONFIG).read_text())

    def build_config(ctx: StepContext) -> JudgedEvaluation:
        run = step.build_config(ctx)
        if not ctx.is_fingerprint:
            manifest = read_json(ctx.artifact_path(data) + "/environment.json")
            validate_environment_preflight(manifest, ctx.artifact_path(data))
            launch = yaml.safe_load(run.launch_config_yaml)
            native = launch["skyrl"]["terminal_bench"]["installed_agent"]
            native["environment_overrides"] = manifest["environment_overrides"]
            native["agent_mcp_servers"] = manifest["agent_mcp_servers"]
            # Reuse the runtime's credential-variable lowering rather than serialize secrets.
            native["verifier_env"] = recipe(
                manifest["environment_overrides"],
                manifest["agent_mcp_servers"],
                manifest["verifier_env"],
                "runtime://minimax-judge",
                manifest["credential_env"],
            )[0]["terminal_bench"]["installed_agent"]["verifier_env"]
            run = replace(run, launch_config_yaml=yaml.safe_dump(launch))
        return JudgedEvaluation(run, judge, cluster["dashboard_url"])

    return replace(step, run=run_judged, build_config=build_config)


def micro_inputs(version: str) -> ArtifactStep:
    return apply(
        "long-loop/micro-inputs",
        remote(
            prepare_micro,
            name=f"long-loop-micro-inputs-{version}",
            env_vars={
                name: os.environ[name]
                for name in ("DAYTONA_EVAL_API_KEY", "DAYTONA_TARGET", "TOGETHER_API_KEY")
                if name in os.environ
            },
            resources=ResourceConfig.with_gpu("H100", count=1, cpu=4, ram="16GB", disk="64GB", target_cluster=None),
        ),
        version=version,
        output=OUT,
        dataset=DATASET,
        revision=REVISION,
        wheels_revision=WHEELS_REVISION,
    )
