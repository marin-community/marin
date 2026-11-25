# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch frozen Micro210 evaluation with only a model binding and run identity."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import draccus
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLGenerationPlan,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    run_skyrl,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import resolve_secret_spec
from transformers import AutoTokenizer

from experiments.evaluation.judge_config import JudgeConfig
from experiments.evaluation.micro210_inputs import native_inputs
from experiments.evaluation.micro210_policy import (
    MODEL_IDENTITY_PREFIX,
    PREPARED_COMMIT,
    PREPARED_REPO,
    Micro210Evaluation,
    Micro210Model,
    micro210_evaluation,
    runtime_policy_config,
)
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME

# The prepared table re-hosts benchmark items that must not be public; tasks read it
# with the standard HF_TOKEN.
PREPARED_INPUTS = f"hf://datasets/{PREPARED_REPO}@{PREPARED_COMMIT}"

# Native task bindings have their own artifact identity alongside the frozen source.
PREPARED_INPUTS_VERSION = "2026.11.24.1"

STOP_TOKEN_CONTENT = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}


@dataclass(frozen=True)
class Micro210Run:
    skyrl: SkyRLRunConfig
    evaluation: Micro210Evaluation
    canonical_launch_sha256: str
    smoke_rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class Micro210Smoke:
    """Scale overrides for a diagnostic run of the production protocol."""

    rows: tuple[int, ...]
    num_inference_engines: int


def judge_runtime_config(config: dict, judge: JudgeConfig) -> dict:
    """Bind all runtime judge routes from the typed policy without changing other settings."""
    config = deepcopy(config)
    config["environment"]["task_sessions"]["remote_judge"].update(base_url=judge.base_url, api_key_env="JUDGE_API_KEY")
    config["extra_env"]["JUDGE_BASE_URL"] = judge.base_url
    return config


def prepare_native_inputs(
    source: str, destination: str, policy: dict, image_digests: dict[str, str], rows: tuple[int, ...]
) -> None:
    """Verify the frozen source before writing its native runtime binding."""
    raw = (StoragePath(source) / policy["evaluation"]["data_relative_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != policy["prepared_rollouts_sha256"]:
        raise ValueError("Canonical Micro210 prepared task bytes changed")
    with TemporaryDirectory(prefix="micro210-inputs-") as directory:
        original, native = Path(directory) / "source.parquet", Path(directory) / "rollouts.parquet"
        original.write_bytes(raw)
        native_inputs(str(original), str(native), policy, image_digests, rows=rows)
        output = StoragePath(destination)
        (output / "rollouts.parquet").write_bytes(native.read_bytes())
        (output / "binding.json").write_text(
            json.dumps(
                {
                    "source_sha256": policy["prepared_rollouts_sha256"],
                    "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                    "image_digests": image_digests,
                    "smoke_rows": rows,
                },
                indent=2,
            )
        )


def validate_inputs(evaluation: Micro210Evaluation, inputs: str) -> None:
    """Reject changed prepared tasks or export metadata before GPU allocation."""
    protocol = evaluation.policy()
    table = (StoragePath(inputs) / protocol["evaluation"]["data_relative_path"]).read_bytes()
    if hashlib.sha256(table).hexdigest() != protocol["prepared_rollouts_sha256"]:
        raise ValueError("Canonical Micro210 prepared task bytes changed")
    manifest = (StoragePath(evaluation.model.uri) / MODEL_MANIFEST_FILENAME).read_bytes()
    if MODEL_IDENTITY_PREFIX + hashlib.sha256(manifest).hexdigest() != evaluation.model.identity:
        raise ValueError("Micro210 model export manifest differs from the model binding")
    tokenizer = AutoTokenizer.from_pretrained(
        evaluation.model.tokenizer_uri, revision=evaluation.model.tokenizer_revision, trust_remote_code=False
    )
    generator = protocol["evaluation"]["config"]["generator"]
    for sampling in ("sampling_params", "eval_sampling_params"):
        for token_id in generator[sampling]["stop_token_ids"]:
            if tokenizer.convert_ids_to_tokens(token_id) != STOP_TOKEN_CONTENT[token_id]:
                raise ValueError(f"Canonical Micro210 tokenizer has incompatible stop token ID: {token_id}")


def run_micro210(config: Micro210Run) -> SkyRLRun:
    if hashlib.sha256(config.skyrl.launch_config_yaml.encode()).hexdigest() != config.canonical_launch_sha256:
        raise ValueError("Canonical Micro210 refuses modified field: launch_config_yaml")
    validate_inputs(config.evaluation, PREPARED_INPUTS)
    (StoragePath(config.evaluation.output_uri) / "canonical-evaluation.json").write_text(
        json.dumps({**asdict(config.evaluation.record()), "smoke_rows": config.smoke_rows}, indent=2)
    )
    evaluation_policy = config.evaluation.policy()["evaluation"]
    judge = draccus.decode(JudgeConfig, evaluation_policy["judge"])
    api_key = resolve_secret_spec(("env:TOGETHER_API_KEY", judge.credential)).value
    return run_skyrl(config.skyrl, env_vars={"JUDGE_API_KEY": api_key, "OPENAI_API_KEY": api_key})


def micro210_step(
    model: Micro210Model, *, name: str, version: str, smoke: Micro210Smoke | None = None
) -> ArtifactStep[SkyRLRun]:
    """Build the canonical evaluation graph for a model and run identity."""
    binding = micro210_evaluation(model, run_id=f"{name}@{version}", output_uri="<resolved-at-build>")
    frozen = binding.policy()
    evaluation_policy = frozen["evaluation"]
    model_artifact = ArtifactStep.adopt(
        f"{name}/model", version, model.uri, kind=LevanterCheckpoint, config=asdict(model)
    )
    frozen_inputs = ArtifactStep.adopt(
        "evaluation/micro210/prepared",
        "2026.11.23",
        PREPARED_INPUTS,
        config={"sha256": frozen["prepared_rollouts_sha256"]},
    )
    inputs = apply(
        "evaluation/micro210/native" if smoke is None else f"{name}/native-inputs",
        prepare_native_inputs,
        source=frozen_inputs,
        destination=OUT,
        policy=frozen,
        image_digests=json.loads(Path(__file__).with_name("micro210_images.json").read_text()),
        rows=() if smoke is None else smoke.rows,
        version=PREPARED_INPUTS_VERSION if smoke is None else version,
    )
    role_policy = evaluation_policy["role_plan"]
    runtime = runtime_policy_config(frozen)
    if smoke is not None:
        role_policy = {**role_policy, "num_inference_engines": smoke.num_inference_engines}
        runtime["generator"]["num_inference_engines"] = smoke.num_inference_engines
        runtime["trainer"]["eval_batch_size"] = len(smoke.rows)
        runtime["trajectory_runner"]["max_concurrent_tasks"] = len(smoke.rows)
    roles = SkyRLGenerationPlan(
        gpus_per_node=role_policy["policy_num_gpus_per_node"],
        **{
            key: role_policy[key]
            for key in (
                "num_inference_engines",
                "inference_engine_tensor_parallel_size",
                "inference_engine_pipeline_parallel_size",
                "inference_engine_data_parallel_size",
                "inference_engine_expert_parallel_size",
                "n_samples_per_prompt",
            )
        },
    )
    execution = IrisSkyRLExecution(
        **{
            **evaluation_policy["execution"],
            "required_env_vars": tuple(evaluation_policy["execution"]["required_env_vars"]),
        },
        coordinator_timeout_hours=evaluation_policy["execution"]["job_timeout_seconds"] // 3600 + 1,
    )
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(
                judge_runtime_config(runtime, draccus.decode(JudgeConfig, evaluation_policy["judge"]))
            ),
            model=ArtifactHfModel(model_artifact, model.tokenizer_uri, model.tokenizer_revision, relative_path=""),
            train_data=(),
            validation_data=(ArtifactDataSource(inputs, relative_path=evaluation_policy["data_relative_path"]),),
            topology=SkyRLTopology(roles.num_nodes, roles.gpus_per_node, evaluation_policy["gpu_variant"], roles),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=evaluation_policy["seed"],
        ),
        execution,
        export_hf=False,
    )
    builder = step.build_config

    def build(context: StepContext) -> Micro210Run:
        run = builder(context)
        document = yaml.safe_load(run.launch_config_yaml)
        document["ray"].update(evaluation_policy["ray_timeouts"])
        document["skyrl"]["generator"]["trajectory_retention"]["run_id"] = binding.run_id
        rendered = yaml.safe_dump(document)
        return Micro210Run(
            replace(run, launch_config_yaml=rendered),
            replace(binding, output_uri=context.output_path),
            hashlib.sha256(rendered.encode()).hexdigest(),
            () if smoke is None else smoke.rows,
        )

    configured = replace(step, run=run_micro210, build_config=build, expected_fingerprint=None)
    return replace(configured, expected_fingerprint=configured.fingerprint())


@click.command(help=__doc__)
@click.option("--model-uri", required=True)
@click.option("--model-manifest-sha256", required=True)
@click.option("--tokenizer-uri", required=True)
@click.option("--tokenizer-revision", required=True)
@click.option("--name", required=True)
@click.option(
    "--smoke-row", multiple=True, type=click.IntRange(min=0), help="Prepared row index for a diagnostic smoke."
)
@click.option("--smoke-engines", default=1, type=click.IntRange(min=1))
@rl_build_options
def main(
    model_uri: str,
    model_manifest_sha256: str,
    tokenizer_uri: str,
    tokenizer_revision: str,
    name: str,
    smoke_row: tuple[int, ...],
    smoke_engines: int,
) -> ArtifactStep:
    configure_coreweave_s3()
    model = Micro210Model(model_uri, MODEL_IDENTITY_PREFIX + model_manifest_sha256, tokenizer_uri, tokenizer_revision)
    smoke = Micro210Smoke(smoke_row, smoke_engines) if smoke_row else None
    return micro210_step(model, name=name, version=resolve_version(name, None), smoke=smoke)


if __name__ == "__main__":
    main()
