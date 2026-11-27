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
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import resolve_secret_spec
from taskcompendium.evaluation.corpus import file_sha256
from transformers import AutoTokenizer

from experiments.evaluation.judge_config import JudgeConfig
from experiments.evaluation.micro210.inputs import native_inputs
from experiments.evaluation.micro210.policy import (
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
NATIVE_INPUTS_VERSION = "2026.11.27"
NATIVE_INPUTS_FINGERPRINT = "a1adbf11"

NATIVE_ROLLOUTS = "rollouts.parquet"
"""The published native table name; the launch reads the derived artifact at this fixed
location rather than the frozen source's policy-relative layout."""

STOP_TOKEN_CONTENT = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}


class NativeInputsBinding(BaseModel):
    """The receipt recorded beside a derived native task table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_sha256: str
    converter_sha256: str
    rollouts_sha256: str
    """Digest of the ``rollouts.parquet`` beside this receipt, computed when it was written."""
    image_digests: dict[str, str]
    smoke_rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class Micro210Run:
    skyrl: SkyRLRunConfig
    evaluation: Micro210Evaluation
    canonical_launch_sha256: str


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


def native_binding_code_sha256() -> str:
    sources = (Path(__file__).with_name(name) for name in ("inputs.py", "policy.py"))
    digests = "".join(file_sha256(StoragePath(str(path))) for path in sources)
    return hashlib.sha256(digests.encode()).hexdigest()


def image_lock() -> dict[str, str]:
    """The pinned container image digests native task provisioning runs."""
    return json.loads(Path(__file__).with_name("images.json").read_text())


def native_inputs_step_name(rows: tuple[int, ...]) -> str:
    """The native inputs artifact a row selection builds; equal selections share one conversion."""
    if not rows:
        return "evaluation/micro210/native"
    return "evaluation/micro210/native-smoke-" + "-".join(map(str, rows))


def prepare_native_inputs(
    source: str,
    destination: str,
    policy: dict,
    image_digests: dict[str, str],
    rows: tuple[int, ...],
    converter_sha256: str,
) -> None:
    """Verify the frozen source before writing its native runtime binding."""
    if converter_sha256 != native_binding_code_sha256():
        raise ValueError("Micro210 native converter differs from its artifact binding")
    raw = (StoragePath(source) / policy["evaluation"]["data_relative_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != policy["prepared_rollouts_sha256"]:
        raise ValueError("Canonical Micro210 prepared task bytes changed")
    with TemporaryDirectory(prefix="micro210-inputs-") as directory:
        original, native = Path(directory) / "source.parquet", Path(directory) / "rollouts.parquet"
        original.write_bytes(raw)
        native_inputs(str(original), str(native), policy, image_digests, rows=rows)
        output = StoragePath(destination)
        (output / NATIVE_ROLLOUTS).write_bytes(native.read_bytes())
        binding = NativeInputsBinding(
            source_sha256=policy["prepared_rollouts_sha256"],
            converter_sha256=converter_sha256,
            rollouts_sha256=file_sha256(StoragePath(str(native))),
            image_digests=image_digests,
            smoke_rows=rows,
        )
        (output / "binding.json").write_text(binding.model_dump_json(indent=2))


def validate_inputs(evaluation: Micro210Evaluation) -> None:
    """Reject changed model export or tokenizer metadata before GPU allocation."""
    protocol = evaluation.policy()
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


def verified_native_evaluation(evaluation: Micro210Evaluation, inputs: str) -> Micro210Evaluation:
    """Verify the native inputs the launch consumes and bind their identity to the run record.

    The launch checks the artifact's binding receipt — its frozen source, converter, and image
    pins — against this evaluation; the artifact system owns the table's byte identity.
    """
    root = StoragePath(inputs)
    binding = NativeInputsBinding.model_validate_json((root / "binding.json").read_text())
    if binding.source_sha256 != evaluation.policy()["prepared_rollouts_sha256"]:
        raise ValueError("Micro210 native binding has a different frozen source")
    if binding.converter_sha256 != native_binding_code_sha256():
        raise ValueError("Micro210 native binding has a different converter")
    if binding.image_digests != image_lock():
        raise ValueError("Micro210 native binding has different image pins")
    return replace(evaluation, native_inputs_sha256=binding.rollouts_sha256, native_inputs_uri=inputs)


def run_micro210(config: Micro210Run) -> SkyRLRun:
    if hashlib.sha256(config.skyrl.launch_config_yaml.encode()).hexdigest() != config.canonical_launch_sha256:
        raise ValueError("Canonical Micro210 refuses modified field: launch_config_yaml")
    validate_inputs(config.evaluation)
    document = yaml.safe_load(config.skyrl.launch_config_yaml)
    native_inputs_uri = document["inputs"]["validation_data"][0]["uri"]
    evaluation = verified_native_evaluation(config.evaluation, native_inputs_uri)
    (StoragePath(evaluation.output_uri) / "canonical-evaluation.json").write_text(
        json.dumps(asdict(evaluation.record()), indent=2)
    )
    evaluation_policy = config.evaluation.policy()["evaluation"]
    judge = draccus.decode(JudgeConfig, evaluation_policy["judge"])
    api_key = resolve_secret_spec(("env:TOGETHER_API_KEY", judge.credential)).value
    return run_skyrl(config.skyrl, env_vars={"JUDGE_API_KEY": api_key, "OPENAI_API_KEY": api_key})


def finalized_launch_document(run: SkyRLRunConfig, *, run_id: str, ray_timeouts: dict) -> str:
    """Rewrite the resolved launch document with the frozen ray timeouts and run identity."""
    document = yaml.safe_load(run.launch_config_yaml)
    document["ray"].update(ray_timeouts)
    document["skyrl"]["generator"]["trajectory_retention"]["run_id"] = run_id
    return yaml.safe_dump(document)


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
    rows = () if smoke is None else smoke.rows
    # Smoke tables are their own artifact, so equal selections share one conversion across runs.
    inputs = apply(
        native_inputs_step_name(rows),
        prepare_native_inputs,
        source=frozen_inputs,
        destination=OUT,
        policy=frozen,
        image_digests=image_lock(),
        rows=rows,
        converter_sha256=native_binding_code_sha256(),
        version=NATIVE_INPUTS_VERSION,
    )
    if smoke is None:
        inputs = replace(inputs, expected_fingerprint=NATIVE_INPUTS_FINGERPRINT)
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
            validation_data=(ArtifactDataSource(inputs, relative_path=NATIVE_ROLLOUTS),),
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
        rendered = finalized_launch_document(run, run_id=binding.run_id, ray_timeouts=evaluation_policy["ray_timeouts"])
        return Micro210Run(
            replace(run, launch_config_yaml=rendered),
            replace(binding, output_uri=context.output_path),
            hashlib.sha256(rendered.encode()).hexdigest(),
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
@click.option("--smoke-engines", type=click.IntRange(min=1))
@rl_build_options
def main(
    model_uri: str,
    model_manifest_sha256: str,
    tokenizer_uri: str,
    tokenizer_revision: str,
    name: str,
    smoke_row: tuple[int, ...],
    smoke_engines: int | None,
) -> ArtifactStep:
    configure_coreweave_s3()
    model = Micro210Model(model_uri, MODEL_IDENTITY_PREFIX + model_manifest_sha256, tokenizer_uri, tokenizer_revision)
    if smoke_engines is not None and not smoke_row:
        raise click.UsageError("--smoke-engines requires at least one --smoke-row")
    smoke = Micro210Smoke(smoke_row, 1 if smoke_engines is None else smoke_engines) if smoke_row else None
    return micro210_step(model, name=name, version=resolve_version(name, None), smoke=smoke)


if __name__ == "__main__":
    main()
