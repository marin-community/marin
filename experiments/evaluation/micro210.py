# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch frozen Micro210 evaluation with only a model binding and run identity."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import draccus
import yaml
from evalrecords.records import EvalRef, ModelRef, RLAttemptRef
from finestore.rl.run_records import read_attempt, read_attempt_metrics, read_evaluations
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    ResolvedModelLocator,
    SkyRLModelSource,
    SkyRLPolicyModel,
    SkyRLRetentionPolicy,
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    run_skyrl,
    skyrl_execution,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import resolve_secret_spec
from skyrlrecipe.documents import merge_mappings, set_path
from taskcompendium.evaluation.corpus import file_sha256

from experiments.evaluation import micro210_policy
from experiments.evaluation.judge_config import JudgeConfig
from experiments.evaluation.micro210_inputs import native_inputs
from experiments.evaluation.micro210_policy import (
    MODEL_IDENTITY_PREFIX,
    PREPARED_COMMIT,
    PREPARED_REPO,
    Micro210Model,
    Micro210PolicyReference,
    runtime_policy_config,
)
from experiments.post_training.rl_compose import layout_execution
from experiments.post_training.rl_presets import Generation, Recipe
from experiments.post_training.skyrl_evaluation import model_rl_attempt
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME

if TYPE_CHECKING:
    from experiments.evaluation.pipeline import EvaluationResult

# The prepared table re-hosts benchmark items that must not be public; tasks read it
# with the standard HF_TOKEN.
PREPARED_INPUTS = f"hf://datasets/{PREPARED_REPO}@{PREPARED_COMMIT}"

# Native task bindings have their own artifact identity alongside the frozen source.
PREPARED_INPUTS_VERSION = "2026.11.24.2"
NATIVE_INPUTS_FINGERPRINT = "a1adbf11"
SERVING_KEYS = frozenset(
    {
        "vllm_attention_backend",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "vllm_v1_disable_multiproc",
        "engine_init_kwargs",
    }
)
STOP_TOKEN_CONTENT = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}


@dataclass(frozen=True)
class Micro210Run:
    skyrl: SkyRLRunConfig
    canonical_launch_sha256: str
    output_uri: str
    smoke_rows: tuple[int, ...]
    expected_manifest_sha256: str | None
    policy_reference: Micro210PolicyReference
    sft_export_manifest: str | None = None
    rl_attempt: RLAttemptRef | None = None


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
    sources = (Path(__file__).with_name(name) for name in ("micro210_inputs.py", "micro210_policy.py"))
    digests = "".join(file_sha256(StoragePath(str(path))) for path in sources)
    return hashlib.sha256(digests.encode()).hexdigest()


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
        (output / "rollouts.parquet").write_bytes(native.read_bytes())
        (output / "binding.json").write_text(
            json.dumps(
                {
                    "source_sha256": policy["prepared_rollouts_sha256"],
                    "converter_sha256": converter_sha256,
                    "native_sha256": file_sha256(StoragePath(str(native))),
                    "image_digests": image_digests,
                    "smoke_rows": rows,
                },
                indent=2,
            )
        )


def validate_inputs(protocol: dict, model: ResolvedModelLocator, inputs: str) -> None:
    """Verify prepared tasks and the bound tokenizer before GPU allocation."""
    from levanter.compat.hf_checkpoints import load_tokenizer  # noqa: PLC0415

    table = (StoragePath(inputs) / protocol["evaluation"]["data_relative_path"]).read_bytes()
    if hashlib.sha256(table).hexdigest() != protocol["prepared_rollouts_sha256"]:
        raise ValueError("Canonical Micro210 prepared task bytes changed")
    tokenizer = load_tokenizer(
        model.tokenizer_uri or model.uri, revision=model.tokenizer_revision, trust_remote_code=False
    )
    generator = protocol["evaluation"]["config"]["generator"]
    for sampling in ("sampling_params", "eval_sampling_params"):
        for token_id in generator[sampling]["stop_token_ids"]:
            if tokenizer.convert_ids_to_tokens(token_id) != STOP_TOKEN_CONTENT[token_id]:
                raise ValueError(f"Canonical Micro210 tokenizer has incompatible stop token ID: {token_id}")


def export_identity(model: ResolvedModelLocator, expected: str | None, sft_manifest: str | None) -> str:
    """Identify the completed export by its manifest or immutable producer."""
    root = StoragePath(model.uri)
    manifests = (MODEL_MANIFEST_FILENAME, sft_manifest) if sft_manifest else (MODEL_MANIFEST_FILENAME,)
    for filename in manifests:
        try:
            digest = file_sha256(root / filename)
        except FileNotFoundError:
            continue
        if expected is None or digest == expected:
            return MODEL_IDENTITY_PREFIX + digest
    if expected is not None:
        raise ValueError("Micro210 model export manifest differs from the model binding")
    return model.identity


def verified_native_inputs(inputs: str, protocol: dict, smoke_rows: tuple[int, ...]) -> str:
    """Return the verified digest of the launched native table."""
    root = StoragePath(inputs)
    receipt = json.loads((root / "binding.json").read_text())
    if receipt["source_sha256"] != protocol["prepared_rollouts_sha256"]:
        raise ValueError("Micro210 native binding has a different frozen source")
    if tuple(receipt["smoke_rows"]) != smoke_rows:
        raise ValueError("Micro210 native binding has different smoke rows")
    if receipt["converter_sha256"] != native_binding_code_sha256():
        raise ValueError("Micro210 native binding has a different converter")
    if receipt["image_digests"] != json.loads(Path(__file__).with_name("micro210_images.json").read_text()):
        raise ValueError("Micro210 native binding has different image pins")
    digest = file_sha256(root / "rollouts.parquet")
    if digest != receipt["native_sha256"]:
        raise ValueError("Canonical Micro210 native task bytes changed")
    return digest


def run_micro210(config: Micro210Run) -> EvaluationResult:
    from experiments.evaluation.pipeline import EvaluationResult  # noqa: PLC0415
    from experiments.post_training.rl_evaluation_records import publish_skyrl_evaluation  # noqa: PLC0415

    if hashlib.sha256(config.skyrl.launch_config_yaml.encode()).hexdigest() != config.canonical_launch_sha256:
        raise ValueError("Canonical Micro210 refuses modified field: launch_config_yaml")
    reference = config.policy_reference
    policy = micro210_policy.read_micro210_policy(reference)
    model = config.skyrl.model
    identity = export_identity(model, config.expected_manifest_sha256, config.sft_export_manifest)
    validate_inputs(policy, model, PREPARED_INPUTS)
    document = yaml.safe_load(config.skyrl.launch_config_yaml)
    native_inputs_uri = document["inputs"]["validation_data"][0]["uri"]
    native_sha256 = verified_native_inputs(native_inputs_uri, policy, config.smoke_rows)
    expected = len(config.smoke_rows) or policy["task_changes"]["sealed_rows"]
    judge = draccus.decode(JudgeConfig, policy["evaluation"]["judge"])
    api_key = resolve_secret_spec(("env:TOGETHER_API_KEY", judge.credential)).value
    run_skyrl(config.skyrl, env_vars={"JUDGE_API_KEY": api_key, "OPENAI_API_KEY": api_key})
    export_root, attempt_id = config.skyrl.output.export_root, config.skyrl.attempt_id
    metrics = read_attempt_metrics(export_root, attempt_id).get(0, {})
    attempted, scored = metrics.get("eval/all/num_attempted", 0), metrics.get("eval/all/num_scored", 0)
    score = metrics.get("eval/all/avg_verifier_score")
    if attempted != expected or scored != expected or score is None or not math.isfinite(score):
        raise ValueError(f"Micro210 requires {expected} scored rows and one finite score, got {scored}/{attempted}")
    attempt = read_attempt(export_root, attempt_id)
    rows = read_evaluations(export_root, attempt_id=attempt_id, evaluation="eval/", step=-1)
    answers = [answer for row in rows for answer in row.answers]
    answer_scores = [answer.verifier_score for answer in answers if answer.verifier_score is not None]
    if (
        attempt is None
        or {answer.ordinal for answer in answers} != set(range(expected))
        or len(answer_scores) != expected
        or any(not math.isfinite(value) for value in answer_scores)
        or len(answers) != expected
    ):
        raise ValueError(f"Micro210 is missing the receipt or {expected} native answer rows for {attempt_id}")
    if not math.isclose(math.fsum(answer_scores) / expected, score, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"Micro210 tracker score disagrees with the native answers for {attempt_id}")
    record = publish_skyrl_evaluation(
        attempt=attempt,
        evaluation=EvalRef(
            name="micro210-smoke" if config.smoke_rows else "micro210",
            mechanism="skyrl",
            source_digest=f"sha256:{reference.sha256}",
            postprocessing={
                "policy_uri": reference.uri,
                "native_table_sha256": native_sha256,
                "native_inputs_uri": native_inputs_uri,
                "smoke_rows": json.dumps(config.smoke_rows),
                "model_identity": identity,
                "tokenizer_uri": model.tokenizer_uri or model.uri,
                "tokenizer_revision": model.tokenizer_revision or "",
                "evaluation_attempt_id": attempt_id,
            },
        ),
        model=ModelRef(
            name=document["run"].get("model") or StoragePath(model.uri).name, location=model.uri, backend="vllm"
        ),
        rows=rows,
        rl_attempt=config.rl_attempt,
        score=score,
        answer_coverage=True,
    )
    return EvaluationResult(
        path=config.output_uri,
        group_id=record.group_id,
        records_prefix=str(StoragePath(record.results_path).parent.parent),
        run_ids=(record.run_id,),
        results_paths=(record.results_path,),
    )


def micro210_step(
    model: Micro210Model | SkyRLModelSource,
    *,
    name: str,
    version: str,
    smoke: Micro210Smoke | None = None,
    serving: Recipe,
    model_name: str | None = None,
) -> ArtifactStep[EvaluationResult]:
    """Build the canonical evaluation graph for a model and run identity."""
    from experiments.evaluation.pipeline import EvaluationResult  # noqa: PLC0415

    pinned = None
    if isinstance(model, Micro210Model):
        pinned = model
        model_artifact = ArtifactStep.adopt(
            f"{name}/model", version, pinned.uri, kind=LevanterCheckpoint, config=asdict(pinned)
        )
        source = ArtifactHfModel(model_artifact, pinned.tokenizer_uri, pinned.tokenizer_revision, relative_path="")
    else:
        source = model
    run_id = f"{name}@{version}"
    smoke_rows = () if smoke is None else smoke.rows
    reference = micro210_policy.POLICY_REFERENCE
    frozen = micro210_policy.read_micro210_policy(reference)
    evaluation_policy = frozen["evaluation"]
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
        rows=smoke_rows,
        converter_sha256=native_binding_code_sha256(),
        version=PREPARED_INPUTS_VERSION if smoke is None else version,
    )
    if smoke is None:
        inputs = replace(inputs, expected_fingerprint=NATIVE_INPUTS_FINGERPRINT)
    layout = serving.layout
    if not isinstance(layout.workload, Generation) or layout.workload.plan.n_samples_per_prompt != 1:
        raise ValueError("Micro210 serving requires a generation layout with one sample per row")
    roles = layout.workload.plan
    execution = replace(
        layout_execution(layout),
        required_env_vars=tuple(evaluation_policy["execution"]["required_env_vars"]),
        wandb_entity=evaluation_policy["execution"].get("wandb_entity"),
    )
    runtime = runtime_policy_config(frozen)
    runtime["trainer"].pop("placement", None)
    patch = serving.config.to_skyrl()
    generator = patch.get("generator", {})
    if (
        patch.keys() - {"generator", "extra_env"}
        or generator.keys() - SERVING_KEYS
        or generator.get("engine_init_kwargs", {}).keys() - serving.engine_serving_keys
    ):
        raise ValueError("Micro210 serving may only change inference serving settings")
    runtime = merge_mappings(runtime, patch)
    runtime["trainer"]["logger"] = ["console", "records"]
    if smoke is not None:
        roles = replace(roles, num_inference_engines=smoke.num_inference_engines)
        runtime["trainer"]["eval_batch_size"] = len(smoke.rows)
        runtime["trajectory_runner"]["max_concurrent_tasks"] = len(smoke.rows)
    for key, value in roles.config_values().items():
        set_path(runtime, key, value)
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(
                judge_runtime_config(runtime, draccus.decode(JudgeConfig, evaluation_policy["judge"]))
            ),
            model=source,
            train_data=(),
            validation_data=(ArtifactDataSource(inputs, relative_path=evaluation_policy["data_relative_path"]),),
            topology=SkyRLTopology(roles.num_nodes, roles.gpus_per_node, layout.gpu_variant, roles),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=evaluation_policy["seed"],
            model_name=model_name,
        ),
        execution,
        export_hf=False,
    )
    builder = step.build_config

    def build(context: StepContext) -> Micro210Run:
        run = builder(context)
        output = replace(
            run.output,
            terminal_manifest_uri=prefix_join(context.output_path, f"native-attempts/{run.attempt_id}/terminal.json"),
        )
        document = yaml.safe_load(run.launch_config_yaml)
        document["artifacts"]["terminal_manifest_uri"] = output.terminal_manifest_uri
        document["ray"].update(evaluation_policy["ray_timeouts"])
        document["skyrl"]["generator"]["trajectory_retention"]["run_id"] = run_id
        rendered = yaml.safe_dump(document)
        return Micro210Run(
            replace(run, launch_config_yaml=rendered, output=output),
            hashlib.sha256(rendered.encode()).hexdigest(),
            context.output_path,
            smoke_rows,
            pinned.identity.removeprefix(MODEL_IDENTITY_PREFIX) if pinned else None,
            reference,
            serving.sft_export_manifest,
            model_rl_attempt(source, context),
        )

    configured = ArtifactStep(
        name=step.name,
        version=step.version,
        artifact_type=EvaluationResult,
        run=run_micro210,
        build_config=build,
        deps=step.deps,
        runtime_args=step.runtime_args,
    )
    return replace(configured, expected_fingerprint=configured.fingerprint())


@dataclass(frozen=True)
class Micro210SkyRL:
    serving: Recipe
    model_name: str

    def evaluated(self, model: SkyRLModelSource, *, version: str) -> ArtifactStep[EvaluationResult]:
        serving = self.serving
        if isinstance(model, SkyRLPolicyModel) and (execution := skyrl_execution(model.step)) is not None:
            serving = replace(
                serving,
                layout=replace(
                    serving.layout,
                    cluster=execution.cluster,
                    priority=execution.priority,
                    excluded_nvlink_domains=execution.excluded_nvlink_domains,
                ),
            )
        return micro210_step(
            model, name=f"{model.step.name}/micro210", version=version, serving=serving, model_name=self.model_name
        )
