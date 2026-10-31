# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile the actual Mini840 recipe through Hydra on CPUs before model allocation."""

import argparse
import json
import tempfile
from pathlib import Path

import yaml
from cloud.iris.launch_config import load_launch_config, validate_launch_config
from huggingface_hub import hf_hub_download
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import (
    ArtifactHfModel,
    IrisSkyRLExecution,
    ResolvedDirectoryDataSource,
    ResolvedModelLocator,
    SkyRLOutputPaths,
    SkyRLRetentionPolicy,
    SkyRLSpec,
    SkyRLTopology,
    _launch_config_yaml,
)
from omegaconf import OmegaConf
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy_runtime.artifacts import publish
from experiments.evaluation.mini_policy_runtime.cohort import cohort_model
from experiments.evaluation.mini_policy_runtime.launch import recipe
from experiments.evaluation.mini_policy_runtime.profile import SEED

VALIDATION_VERSION = "config-check-dev"


def main(manifest_uri: str, model_key: str):
    fs, key = filesystem_for(manifest_uri)
    with fs.open(key) as reader:
        manifest = json.load(reader)
    model = cohort_model(model_key)
    config, plan = recipe(
        model,
        manifest["environment_overrides"],
        manifest["agent_mcp_servers"],
        manifest["verifier_env"],
        "http://judge.example.invalid/v1",
        manifest["credential_env"],
    )
    model_directory = str(Path(hf_hub_download(model.repo, "config.json", revision=model.revision)).parent)
    model_artifact = ArtifactStep.adopt(f"mini840-{model.key}-validation-model", VALIDATION_VERSION, model.repo)
    spec = SkyRLSpec(
        name=f"mini840-config-validation-{model.key}",
        version=VALIDATION_VERSION,
        config_yaml=yaml.safe_dump(config),
        model=ArtifactHfModel(
            model_artifact, relative_path="", tokenizer_uri=model.repo, tokenizer_revision=model.revision
        ),
        train_data=(),
        validation_data=(),
        topology=SkyRLTopology(num_nodes=1, gpus_per_node=2, gpu_variant="H100", role_plan=plan),
        retention=SkyRLRetentionPolicy(),
        seed=SEED,
    )
    execution = IrisSkyRLExecution(
        cluster="cw-rno2a",
        cluster_config="lib/iris/config/cw-rno2a.yaml",
        cpu=32,
        memory="256GB",
        disk="256GB",
        priority="production",
        max_retries=0,
        target_cluster=None,
        parent_cluster_config=None,
        coordinator_timeout_hours=3,
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "launch.yaml"
        path.write_text(
            _launch_config_yaml(
                spec,
                execution,
                export_hf=False,
                run_id="mini840-cpu-validation",
                attempt_id="cpu-validation",
                model=ResolvedModelLocator(model.repo, model.revision, model_directory, model.repo, model.revision),
                train_data=(),
                validation_data=(
                    ResolvedDirectoryDataSource(manifest["data_uri"], "mini840-input", "/tmp/data", "rollouts.parquet"),
                ),
                output=SkyRLOutputPaths(
                    "/tmp/checkpoints", "/tmp/exports", "/tmp/attempts", "/tmp/resolved.yaml", "/tmp/terminal.json"
                ),
            )
        )
        resolved = load_launch_config(path)
        validate_launch_config(resolved)
    detached = OmegaConf.create(OmegaConf.to_container(resolved, resolve=True))
    publish("hydra-config-validation.json", OmegaConf.to_container(detached, resolve=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.environment_manifest, args.model)
