"""Behavior tests for SkyRL launch, prefill and task evaluation contracts."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from iris.rpc import job_pb2
from omegaconf import OmegaConf
from rolloutengine.harness.prefill import prefill_prompt
from rolloutengine.models import ModelRequest, ModelTurn, RolloutContractError
from skyrl_gym.answer_tasks import grade_gsm8k
from skyrl_gym.task_sessions import AnswerTaskSession
from skyrl_gym.verification import VerificationResult
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import TaskDataset, source_row_task, task_prompt_token_count, write_tasks
from skyrl_train.distributed.step_policy import NonfiniteStepPolicy, nonfinite_step_policy
from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl_train.evaluate import evaluate, evaluate_step_wise, evaluation_session
from skyrl_train.rollouts.buffer import write_trajectory_batch
from skyrl_train.rollouts.finalization import finalize_trajectory_batch
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.task_inputs import task_messages
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.trajectory_retention import (
    TrajectorySink,
    execute_publication,
    parse_trajectory_retention_config,
)
from skyrl_train.trajectory_runners.trajectory_retention_publisher import InlineTrajectoryPublisher
from skyrl_train.trajectory_runners.types import (
    BatchMetadata,
    TaskRequestBatch,
    TokenProvenance,
    TrajectoryBatch,
    TrajectoryID,
)
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.grading import grade_answer
from shellbox.machine import Backend, Command, ExitReason, Result
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    PlainText,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from tests.cpu.task_specs import NATIVE_ENVIRONMENT, lowered_task, task_input, text_task, machine_runtime
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from verifyit.closure_scoring import ClosureFormat
from verifyit.spec import NumericSpec, StructureClosureSpec, MathSpec, JudgeSpec
from verifyit.modes.grade_judge import JudgeConnection

from cloud.iris import training_driver
from cloud.iris.launch_config import LaunchTopology, load_launch_config, validate_launch_config
from cloud.iris.rl_config_translation import (
    RL_CONFIG_PAYLOAD_ENV,
    compose_skyrl_config,
    materialize_launch_config,
    parse_rl_config,
)
from marinskyrl.runtime_options import reference_model_required
from marinskyrl.saved_answers import (
    AnswerAdmission,
    SavedAnswer,
    admit_answer,
    answer_uri,
    regrade_once,
    regrade_answer,
    save_answer,
)


def _raw_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {
            "id": "smoke",
            "attempt_id": "attempt-1",
            "seed": 42,
            "mode": "train",
            "submission": "wait",
            "export_hf": True,
        },
        "runtime": {
            "source_digest": "a" * 64,
            "profile": "megatron",
            "entrypoint": "skyrl_train.entrypoints.main_base",
        },
        "iris": {
            "cluster": "cw-us-east-08a",
            "cluster_config": "configs/cw-us-east-08a.yaml",
            "job_name": "smoke",
            "wandb_entity": None,
            "allocation": {
                "num_nodes": 1,
                "gpus_per_node": 8,
                "gpu_variant": "H100",
                "cpu": 48,
                "memory": "700GB",
                "disk": "4TB",
            },
        },
        "ray": {
            "spill_backend": "local",
            "spill_dir": "/tmp/skyrl-ray-spill",
            "rendezvous_dir": "s3://runs/smoke/rendezvous",
            "log_dir": "s3://runs/smoke/ray-logs",
        },
        "artifacts": {
            "checkpoint_root": "s3://runs/smoke/checkpoints",
            "export_root": "s3://runs/smoke/exports",
            "attempts_root": "s3://runs/smoke/attempts",
            "resolved_config_uri": "s3://runs/smoke/resolved.yaml",
            "terminal_manifest_uri": "s3://runs/smoke/terminal.json",
        },
        "inputs": {
            "model": {
                "uri": "s3://models/qwen",
                "identity": "sha256:model",
                "local_path": "/tmp/models/qwen",
                "tokenizer_uri": "s3://models/qwen-tokenizer",
                "tokenizer_revision": "a" * 40,
            },
            "data_kind": "tasks",
            "train_data": [],
            "validation_data": [],
        },
        "skyrl": {
            "entrypoint": "standard",
            "context_budget": {
                "request_window_tokens": 1024,
                "max_new_tokens_per_turn": 256,
                "max_turns": 1,
            },
            "model_num_attention_heads": 8,
            "trainer": {
                "seed": 42,
                "strategy": "megatron",
                "algorithm": {"use_kl_loss": False},
                "placement": {
                    "colocate_all": True,
                    "policy_num_nodes": 1,
                    "policy_num_gpus_per_node": 8,
                },
                "train_batch_size": 8,
                "policy_mini_batch_size": 8,
                "micro_train_batch_size_per_gpu": 1,
            },
            "generator": {
                "backend": "vllm",
                "run_engines_locally": True,
                "num_inference_engines": 8,
                "inference_engine_tensor_parallel_size": 1,
                "n_samples_per_prompt": 1,
            },
        },
    }


def test_launch_preserves_harbor_dataset_relative_paths(tmp_path):
    raw = _raw_config()
    task = {
        "git_url": "https://example.org/tasks.git",
        "git_commit_id": "b" * 40,
        "path": "datasets/bfcl_parity/example-task",
    }
    raw["run"]["export_hf"] = False
    raw["skyrl"]["trainer"]["callbacks"] = [
        {
            "type": "harbor_evaluation",
            "daytona_api_key_env": "DAYTONA_EVAL_API_KEY",
            "cohorts": [{"name": "bfcl_parity", "reward_key": "reward", "job": {"tasks": [task]}}],
            "timeout": 600,
            "minimum_coverage": 0.9,
        }
    ]
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_launch_config(path)
    resolved_task = config.skyrl.trainer.callbacks[0].cohorts[0].job.tasks[0]
    assert OmegaConf.to_container(resolved_task) == task


def test_launch_rejects_terminal_export_without_request_callback(tmp_path):
    raw = _raw_config()
    raw["skyrl"]["trainer"].update(export_hf_artifact=True, ckpt_interval=1, hf_save_interval=-1)
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="Terminal HF artifact export"):
        load_launch_config(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("text,reward", [("#### 12", 1.0), ("#### 12\nMore reasoning.", 0.0)])
async def test_custom_session_settings_reach_private_grading_through_the_launch_config(
    tmp_path: Path, text: str, reward: float
) -> None:
    raw = _raw_config()
    raw["skyrl"]["environment"] = {"task_sessions": {"custom_math": {"reward_method": "final_line"}}}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path)
    task = source_row_task(
        {
            "prompt": [{"role": "user", "content": "What is six plus six?"}],
            "env_class": "custom_math",
            "reward_spec": {"ground_truth": "12"},
        },
        0,
        source_name="fixture",
        environment_configs=OmegaConf.to_container(config.skyrl.environment.task_sessions, resolve=True),
    )
    session = AnswerTaskSession(task, None, grader=grade_gsm8k)
    start = await session.prepare()
    try:
        transition = await session.advance(
            ModelTurn({"role": "assistant", "content": text}, (), (1,), None, "stop", text)
        )
        assert transition.done and transition.grade.reward == reward
        assert (await session.grade((), stop_reason="stop", outputs={})).reward == reward
        assert start.messages == ({"role": "user", "content": "What is six plus six?"},)
    finally:
        await session.close()


@pytest.mark.parametrize("storage_prefix", ["s3://runs/smoke", "gs://runs/smoke"])
@pytest.mark.parametrize(("loss", "reduction"), [("regular", "token_mean"), ("gspo", "sequence_mean")])
@pytest.mark.parametrize("probe_enabled", [False, True])
def test_launch_config_composes_and_loads_as_structured_hydra(
    tmp_path: Path, loss: str, reduction: str, storage_prefix: str, probe_enabled: bool
) -> None:
    path = tmp_path / "resolved-launch.yaml"
    raw = _raw_config()
    raw["skyrl"]["trainer"]["algorithm"].update(policy_loss_type=loss, loss_reduction=reduction)
    trainer = raw["skyrl"]["trainer"]
    trainer["resume_mode"] = "from_path"
    trainer["resume_path"] = f"{storage_prefix}/checkpoints/global_step_1"
    trainer["mismatch_probe"] = {
        "archive_uri": f"{storage_prefix}/mismatch_probe",
        "reuse_probe": f"{storage_prefix}/source/mismatch_probe",
    }
    if probe_enabled:
        # The direct engine returns exact model tokens without a transport flag.
        trainer["mismatch_probe"].update(
            enabled=True, prompts={"count": 2, "samples_per_prompt": 1}, seed=17, updates=0
        )
        raw["skyrl"]["generator"].update(
            sampling_params={"temperature": 1.0, "logprobs": 0},
            engine_init_kwargs={"logprobs_mode": "processed_logprobs", "generation_config": "vllm"},
        )
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    assert config.skyrl.trainer.train_batch_size == 8
    assert validate_launch_config(config).num_nodes == 1
    assert config.skyrl.trainer.resume_path == f"{storage_prefix}/checkpoints/global_step_1"
    assert config.skyrl.trainer.mismatch_probe.archive_uri == f"{storage_prefix}/mismatch_probe"
    assert config.skyrl.trainer.mismatch_probe.reuse_probe == f"{storage_prefix}/source/mismatch_probe"
    if loss == "gspo":
        config.skyrl.trainer.algorithm.loss_reduction = "token_mean"
        with pytest.raises(ValueError, match="gspo requires trainer.algorithm.loss_reduction=sequence_mean"):
            validate_launch_config(config)
        raw["skyrl"]["trainer"]["algorithm"]["loss_reduction"] = "token_mean"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        with pytest.raises(ValueError, match="gspo requires trainer.algorithm.loss_reduction=sequence_mean"):
            load_launch_config(path)


def test_auxiliary_ftpo_launch_retains_task_objective_and_topk(tmp_path: Path) -> None:
    # Smartie's mixed objective previously failed the launcher's distillation-only top-K gate.
    raw = _raw_config()
    raw["skyrl"]["trainer"]["use_sample_packing"] = False
    raw["skyrl"]["trainer"]["algorithm"].update(policy_loss_type="regular", ftpo={"auxiliary_weight": 0.1})
    raw["skyrl"]["generator"]["sampling_params"] = {"temperature": 1.0, "logprobs": 32}
    path = tmp_path / "joint-launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_launch_config(path)

    assert config.skyrl.trainer.algorithm.policy_loss_type == "regular"
    assert config.skyrl.generator.sampling_params.logprobs == 32
    assert reference_model_required(config.skyrl.trainer.algorithm)


@pytest.mark.parametrize("switch", ["use_abs_kl", "use_kl_estimator_k3"])
def test_composed_launch_rejects_kl_switches(tmp_path: Path, switch: str) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(_raw_config()))
    config = load_launch_config(path)
    OmegaConf.update(config.skyrl.trainer.algorithm, switch, True, force_add=True)
    OmegaConf.save(config, path)

    with pytest.raises(ValueError, match="kl_estimator_type"):
        load_launch_config(path)


@pytest.mark.parametrize(
    ("entrypoint", "max_staleness_steps", "expected"),
    [
        ("standard", 0, "sync"),
        ("standard", 2, "async"),
        ("terminal_bench", 1, "async"),
        ("generate", 0, None),
    ],
)
def test_composed_launch_records_whether_training_runs_ahead_of_its_updates(
    tmp_path: Path, entrypoint: str, max_staleness_steps: int, expected: str | None
) -> None:
    raw = _raw_config()
    raw["skyrl"]["entrypoint"] = entrypoint
    raw["skyrl"]["trainer"]["placement"]["colocate_all"] = False
    raw["skyrl"]["trainer"]["rollout_buffer"] = {"max_staleness_steps": max_staleness_steps}
    raw["skyrl"]["trainer"]["algorithm"]["off_policy_correction"] = "none"
    raw["iris"]["allocation"]["num_nodes"] = 2
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    assert load_launch_config(path).runtime.training_type == expected


def test_object_store_uris_in_skyrl_config_reach_the_task_unchanged(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["skyrl"]["trainer"]["rollout_buffer"] = {"object_store_root": "s3://runs/smoke/rollouts"}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    assert config.skyrl.trainer.rollout_buffer.object_store_root == "s3://runs/smoke/rollouts"


def test_qwen_smoke_hugging_face_model_input_reaches_the_runner_without_a_model_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _raw_config()
    config["skyrl"] = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml").read_text()
    )
    config["inputs"]["model"] = {
        "uri": "Qwen/Qwen3-0.6B",
        "identity": "main",
        "local_path": "Qwen/Qwen3-0.6B",
        "tokenizer_uri": "Qwen/Qwen3-0.6B",
        "tokenizer_revision": "main",
    }
    config["inputs"]["data_kind"] = "parquet"
    path = tmp_path / "qwen-launch.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    seen = []

    class RecordingRunner:
        def __init__(self, run_config):
            seen.append(run_config)

        def setup(self):
            pass

        def run(self):
            return 0

    monkeypatch.setattr(training_driver, "LocalRLRunner", RecordingRunner)
    monkeypatch.setattr("sys.argv", ["training_driver", "--config", str(path)])

    resolved = load_launch_config(path)
    with pytest.raises(SystemExit) as result:
        training_driver.main()

    assert resolved.runtime.entrypoint == "skyrl_train.entrypoints.main_base"
    assert resolved.skyrl.trainer.policy.model.source_uri is None
    assert result.value.code == 0
    assert seen[0].model_path == "Qwen/Qwen3-0.6B"
    assert seen[0].model_source_uri is None


def test_task_materializes_the_forwarded_launch_document(tmp_path: Path) -> None:
    destination = tmp_path / "launch.yaml"
    contents = yaml.safe_dump(_raw_config()).encode()

    path = materialize_launch_config(
        str(destination),
        {RL_CONFIG_PAYLOAD_ENV: base64.b64encode(contents).decode("ascii")},
    )

    assert path == str(destination)
    assert destination.read_bytes() == contents


def test_evaluation_metric_names_survive_iris_path_resolution(tmp_path: Path) -> None:
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml").read_text())
    groups = {"eval/train/avg_score": ["eval/cat_count_n1/avg_score", "eval/cat_count_n2/avg_score"]}
    profiles = {"sampled": {"sampling_params": {"stop": ["./END"]}}}
    raw["trainer"]["callbacks"] = [{"type": "evaluation", "metric_groups": groups, "additional_evaluations": profiles}]
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = compose_skyrl_config(
        parse_rl_config(str(path)), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
    ).config

    assert config.trainer.callbacks[0].metric_groups == groups
    assert config.trainer.callbacks[0].additional_evaluations == profiles


def test_source_recipe_rejects_resolved_harbor_key_before_discarding_agent_settings(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["skyrl"]["terminal_bench_config"] = {"installed_agent": {"agent": {"name": "pi"}}}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="terminal_bench_config.*terminal_bench"):
        load_launch_config(path)


def test_installed_agent_artifacts_follow_launch_attempt_and_survive_reload(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["skyrl"]["config_groups"] = {"terminal_bench_config": "terminal_bench"}
    raw["skyrl"]["terminal_bench"] = {"installed_agent": {"agent": {"name": "pi"}}}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    resolved = load_launch_config(path)
    destination = resolved.skyrl.terminal_bench_config.installed_agent.output_path
    assert destination == "s3://runs/smoke/attempts/trajectories/installed-agents"
    OmegaConf.save(resolved, path)
    assert load_launch_config(path).skyrl.terminal_bench_config.installed_agent.output_path == destination


def test_null_nonfinite_limit_in_launch_fails_on_first_invalid_step(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["skyrl"]["trainer"]["policy"] = {"max_consecutive_nonfinite_steps": None}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    action = nonfinite_step_policy(0, config.skyrl.trainer.policy.max_consecutive_nonfinite_steps)
    assert action is NonfiniteStepPolicy.FAIL


@pytest.mark.parametrize(("key", "value"), [("use_tis", True), ("tis_imp_ratio_cap", 2.0)])
def test_composed_launch_rejects_tis_selectors(tmp_path: Path, key: str, value) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(_raw_config()))
    config = load_launch_config(path)
    OmegaConf.update(config.skyrl.trainer.algorithm, key, value, force_add=True)
    OmegaConf.save(config, path)

    with pytest.raises(ValueError, match="off_policy_correction"):
        load_launch_config(path)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"rollout_buffer.max_staleness_steps": 1}, "off-policy OLD-anchored"),
        ({"rollout_buffer.max_staleness_steps": 1, "algorithm.off_policy_correction": "none"}, None),
        ({"rollout_buffer.max_staleness_steps": 1, "algorithm.policy_loss_type": "behavior_clip"}, None),
        ({"algorithm.policy_loss_type": "behavior_clip", "algorithm.off_policy_correction": "tis"}, "OLD-anchored"),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"action": "truncate", "high": 2.0}],
            },
            "kind must be token or sequence",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"kind": "token", "action": "mask", "low": 2, "high": 1}],
            },
            "low <= high",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"kind": "token", "action": "truncate", "high": 2}] * 2,
            },
            "at most one truncate",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"kind": "token", "action": "truncate", "hig": 2}],
            },
            "hig",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"kind": "token", "action": "clamp", "high": 2}],
            },
            "action",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [
                    {"kind": "sequence", "aggregate": "average", "action": "mask", "high": 2}
                ],
            },
            "aggregate",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"kind": "token", "action": "truncate", "high": "2.0"}],
            },
            "high",
        ),
        ({"algorithm.dynamic_sampling.max_mean_reward": 0.9}, "requires dynamic_sampling.type=filter"),
        ({"algorithm.dynamic_sampling.max_mean_reward": 0.9, "algorithm.dynamic_sampling.type": "filter"}, None),
    ],
)
def test_launch_validates_correction_and_selection_contract(tmp_path: Path, overrides: dict, error: str | None):
    raw = OmegaConf.create(_raw_config())
    for key, value in overrides.items():
        OmegaConf.update(raw.skyrl.trainer, key, value, force_add=True)
    path = tmp_path / "launch.yaml"
    OmegaConf.save(raw, path)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            load_launch_config(path)
    else:
        # Explicitly uncorrected stale policies and active reward filters are valid launch contracts.
        load_launch_config(path)


@pytest.mark.parametrize(
    ("recipe", "nodes"), [("snowball_mopd_ultra_async_smoke", 8), ("snowball_mopd_ultra_async_32k_smoke", 9)]
)
def test_inherited_recipe_round_trips_as_a_self_contained_launch(tmp_path: Path, recipe: str, nodes: int):
    raw = _raw_config()
    raw["skyrl"] = {"defaults": [recipe, "_self_"], "trainer": {"max_steps": 2}}
    raw["iris"]["allocation"]["num_nodes"] = nodes
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path)
    assert config.skyrl.trainer.max_steps == 2
    assert config.skyrl.trainer.rollout_buffer.max_staleness_steps == 1
    assert config.skyrl.trainer.algorithm.off_policy_correction == "tis"
    assert config.skyrl.data.sampling.kind is None
    assert config.skyrl.generator.engine_init_kwargs.max_model_len == (8192 if nodes == 8 else 32767)
    resolved = tmp_path / "resolved.yaml"
    OmegaConf.save(config, resolved)
    reloaded = load_launch_config(resolved)
    assert OmegaConf.to_container(reloaded, resolve=True) == OmegaConf.to_container(config, resolve=True)


class WorkerTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt, **options):
        return [1, 2]


@pytest.fixture
def task_inputs():
    task = TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    config = OmegaConf.create(
        {
            "backend": "vllm",
            "max_turns": 2,
            "max_input_length": 100,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 10,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": True,
            },
        }
    )
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task), teacher_route="arithmetic", data_source="arithmetic"),),
        trajectory_ids=(TrajectoryID("arithmetic", 0),),
    )
    return config, request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,parsed_channel,reward",
    [
        ("12}", None, 1.0),
        ("12} trailing text", None, 1.0),
        ("13}", None, 0.0),
        ("12", None, 0.0),
        ("12}", {"reasoning_content": "parsed reasoning"}, None),
        ("12}", {"reasoning": "parsed reasoning alias"}, None),
        ("12}", {"tool_calls": [{"id": "call-1", "type": "function"}]}, None),
    ],
)
async def test_prefill_worker_preserves_continuation_and_rejects_parsed_channels(
    task_inputs, tmp_path, response, parsed_channel, reward
):
    config, request = task_inputs
    context = ConversationInput(
        events=(
            TextMessage(role="user", content="Complete the supplied structure"),
            TextMessage(role="assistant", content='{"value":'),
        ),
    )
    package = verifyit_package(StructureClosureSpec('{"value":', '{"value":12}', ClosureFormat.JSON))
    task = request.rows[0].spec.task.model_copy(
        update={
            "context": context,
            "answer_type": AnswerType.TEXT,
            "answer_format": PlainText(),
            "grader": package.grader,
        }
    )
    spec = lowered_task(task, "prefill", max_turns=1)
    request = replace(request, rows=(task_input(spec),), batch_metadata=BatchMetadata(0, "eval"))
    observed = []

    class Engine:
        tokenizer = WorkerTokenizer()

        async def generate(self, model_request):
            observed.append(model_request)
            return {
                "responses": [response],
                "assistant_messages": [{"role": "assistant", "content": response, **(parsed_channel or {})}],
                "prompt_ids": model_request["prompt_token_ids"],
                "response_ids": [[11, 12, 13]],
                "response_logprobs": [[-0.1, -0.2, -0.3]],
                "stop_reasons": ["stop"],
            }

    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, WorkerTokenizer()),
        DirectModelClient(Engine()),
        {},
        shutdown_timeout=30,
        eval_saved_answers_root=str(tmp_path / "answers"),
    )
    try:
        (result,) = await worker.generate(request)
    finally:
        await worker.shutdown()
    assert observed[0]["prompt_token_ids"] == [[1, 2]]
    assert observed[0].get("prompts") is None and "tools" not in observed[0]
    assert observed[0]["sampling_params"]["skip_special_tokens"] is False
    assert result.grade.reward == reward
    path = Path(answer_uri(str(tmp_path / "answers"), "arithmetic"))
    saved = SavedAnswer.model_validate_json(path.read_bytes())
    admission = AnswerAdmission.model_validate_json(Path(str(path) + ".admission.json").read_bytes())
    if parsed_channel:
        assert result.grade.status != Outcome.GRADED
        assert not admission.eligible
        with pytest.raises(ValueError, match="not admitted"):
            await regrade_once(saved, admission, spec, {}, str(tmp_path / "regrades"), timeout=2)
        return
    assert saved.messages == result.messages
    assert admission.eligible
    assert result.messages[-1] == {"role": "assistant", "content": response}
    assert result.prompt_token_ids == (1, 2)
    assert result.response_token_ids == (11, 12, 13)
    assert result.loss_mask == (1, 1, 1)
    assert result.logprobs == (-0.1, -0.2, -0.3)


@pytest.mark.asyncio
@pytest.mark.parametrize("served_prompt", ["matching", "different"])
async def test_prefill_transport_renders_serving_tokens_and_retains_raw_structural_delimiters(served_prompt):
    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "user": 1,
                "question": 2,
                "assistant": 3,
                "prefix": 4,
                "tool-description": 5,
                "closed": 6,
                "new-generation": 7,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template="{% for tool in tools or [] %}{{ tool.function.description }} {% endfor %}"
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
        "closed {% if add_generation_prompt %}new-generation{% endif %}",
    )
    task = text_task("prefill").model_copy(
        update={
            "context": ConversationInput(
                events=(TextMessage(role="user", content="question"), TextMessage(role="assistant", content="prefix"))
            ),
            "final_tools": (
                FunctionDefinition(name="function", description="tool-description", parameters={"type": "object"}),
            ),
        }
    )
    prompt = prefill_prompt(task)
    observed = []

    class Engine:
        async def generate(self, request):
            observed.append(request)
            return {
                "responses": ["</think> </tool_call> trailing"],
                "response_ids": [[8, 9, 10]],
                "response_logprobs": [[-0.1, -0.2, -0.3]],
                "stop_reasons": ["length"],
                "prompt_ids": request["prompt_token_ids"] if served_prompt == "matching" else [[1, 2]],
                "student_topk_indices": [[[8], [9], [10]]],
                "behavior_topk_logprobs": [[[-0.1], [-0.2], [-0.3]]],
                "response_policy_steps": [[4, 4, 4]],
            }

    engine = Engine()
    engine.tokenizer = tokenizer
    request = {
        "prompts": [list(prompt.messages)],
        "chat_completion_params": [dict(prompt.options)],
        "session_ids": ["one"],
        "sampling_params": {"max_generate_length": 10, "logprobs": 1},
        "max_context_length": 8,
        "max_prompt_length": 5,
    }
    if served_prompt == "different":
        with pytest.raises(RolloutContractError):
            await DirectModelClient(engine).generate(request)
        assert observed[0]["prompt_token_ids"] == [[5, 1, 2, 3, 4]]
        return
    output = await DirectModelClient(engine).generate(request)
    # Both admission and inference render the unfinished message, excluding its
    # template closer and a new assistant generation prompt, including tool schemas.
    assert output["prompt_ids"] == [[5, 1, 2, 3, 4]]
    lowered = lowered_task(task, "prefill", max_turns=1)
    assert task_prompt_token_count(tokenizer, {"task_rollout_spec": lowered.model_dump_json()}, "prompt") == 5
    assert observed[0]["sampling_params"]["max_tokens"] == 3
    assert observed[0]["sampling_params"]["skip_special_tokens"] is False
    assert "extra_body" not in observed[0]["sampling_params"]
    assert output["responses"] == ["</think> </tool_call> trailing"]
    assert output["assistant_messages"] == [{"role": "assistant", "content": "</think> </tool_call> trailing"}]
    assert output["response_ids"] == [[8, 9, 10]]
    assert output["response_logprobs"] == [[-0.1, -0.2, -0.3]]
    assert output["student_topk_indices"] == [[[8], [9], [10]]]
    assert output["response_policy_steps"] == [[4, 4, 4]]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_header,expected_ids", [(False, [1, 2]), (True, [3, 1, 2])])
async def test_prefill_admission_and_transport_share_configured_template_and_defaults(
    tmp_path, include_header, expected_ids
):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "assistant": 1, "prefix": 2, "header": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    original = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", chat_template="native template must not be used"
    )
    model_path = tmp_path / "model"
    original.save_pretrained(model_path)
    template_path = tmp_path / "serving.jinja"
    template_path.write_text(
        "{% if include_header %}header {% endif %}"
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
        "closed {% if add_generation_prompt %}new generation{% endif %}"
    )
    config = get_default_config()
    config.trainer.policy.model.tokenizer_path = str(model_path)
    config.generator.chat_template.source = "file"
    config.generator.chat_template.name_or_path = str(template_path)
    config.generator.chat_template_kwargs = {"include_header": include_header}
    tokenizer = tokenizer_from_config(config)
    task = text_task("configured-prefill").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix"),))}
    )
    spec = lowered_task(task, "prefill", max_turns=1)
    panel_path = tmp_path / "panel.parquet"
    write_tasks(panel_path, [spec])
    dataset = TaskDataset(
        [str(panel_path)],
        tokenizer,
        len(expected_ids),
        num_workers=1,
        chat_template_kwargs=dict(config.generator.chat_template_kwargs),
    )
    assert [row["input"].spec for row in dataset.collate_fn(list(dataset))] == [spec]

    class Engine:
        async def generate(self, request):
            return {
                "prompt_ids": request["prompt_token_ids"],
                "responses": ["continuation"],
                "response_ids": [[4]],
                "response_logprobs": [[-0.1]],
                "stop_reasons": ["stop"],
            }

    engine = Engine()
    engine.tokenizer = tokenizer
    prompt = prefill_prompt(task)
    output = await DirectModelClient(engine).generate(
        {
            "prompts": [list(prompt.messages)],
            "chat_completion_params": [
                {**prompt.options, "chat_template_kwargs": dict(config.generator.chat_template_kwargs)}
            ],
            "sampling_params": {"max_tokens": 1},
            "max_prompt_length": len(expected_ids),
            "max_context_length": 10,
        }
    )
    assert output["prompt_ids"] == [expected_ids]
    assert output["responses"] == ["continuation"]


def test_prefill_dataset_counts_effective_continuation_without_new_generation_prompt(tmp_path):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template="{% for m in messages %}{{ m.content }} {% endfor %}"
        "{% if add_generation_prompt %}new assistant generation prompt that exceeds the cap{% endif %}",
    )
    short = text_task("short-prefill").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix one"),))}
    )
    long = text_task("long-prefill").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix " * 20),))}
    )
    path = tmp_path / "prefill.parquet"
    write_tasks(path, [lowered_task(task, "prefill", max_turns=1) for task in (short, long)])
    dataset = TaskDataset([str(path)], tokenizer, 6, num_workers=1)
    (selected,) = dataset.collate_fn(list(dataset))
    spec = selected["input"].spec
    assert spec.task.id == "short-prefill"
    assert task_messages(spec) == [{"role": "assistant", "content": "prefix one"}]


@pytest.fixture
def dummy_config():
    return get_default_config()


class DummyRunner:
    trajectory_runner_cfg = {}
    trajectory_sink = None

    def __init__(self, outputs: dict[str, TrajectoryBatch]):
        self.outputs = outputs

    async def run(self, input_batch) -> TrajectoryBatch:
        output = {**self.outputs[input_batch.metadata[0].uid], "trajectory_ids": list(input_batch.trajectory_ids)}
        return await finalize_trajectory_batch(input_batch, output, self.trajectory_runner_cfg, self.trajectory_sink)

    async def run_task(self, task, writer) -> int:
        output = await self.run(task.request)
        return await write_trajectory_batch(task, writer, output)

    def set_trajectory_sink(self, sink) -> None:
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None = None) -> None:
        pass

    async def stop_eval_session(self) -> None:
        pass

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


def configure_eval(cfg, tmp_path):
    cfg.generator.backend = "vllm"
    cfg.generator.eval_sampling_params = OmegaConf.create(
        {
            "max_generate_length": 20,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "logprobs": None,
            "stop": None,
        }
    )
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.environment = OmegaConf.create({"env_class": "gsm8k"})
    cfg.trainer.dump_eval_results = False
    cfg.trainer.export_path = str(tmp_path)
    return cfg


class DummyStatefulDataLoader:
    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


@pytest.mark.asyncio
@pytest.mark.parametrize("step_wise", [False, True])
@pytest.mark.parametrize("scored", [False, True])
async def test_required_coverage_persists_failed_attempts_and_accepts_valid_zero(
    dummy_config, tmp_path, step_wise, scored
):
    cfg = configure_eval(dummy_config, tmp_path)
    cfg.trainer.dump_eval_results = True
    cfg.trainer.eval_required_score_coverage = 1.0
    verdict = VerificationResult.verified(0.0) if scored else VerificationResult.unavailable("provider unavailable")
    runner = DummyRunner(
        {
            "paired-task": {
                "prompt_token_ids": [[101]],
                "response_ids": [[201]],
                "rewards": [0.0],
                "loss_masks": [[1]],
                "stop_reasons": ["stop"],
                "rollout_logprobs": None,
                "verification_results": [verdict],
                "is_last_step": [True],
            }
        }
    )
    loader = DummyStatefulDataLoader(
        [[{"input": task_input(lowered_task(text_task("paired-task", source="panel"))), "uid": "paired-task"}]]
    )
    evaluation = evaluate_step_wise if step_wise else evaluate
    tokenizer = SimpleNamespace(decode=lambda _: "response")
    sink = TrajectorySink(
        parse_trajectory_retention_config(cfg.generator.trajectory_retention),
        tokenizer,
        publisher=InlineTrajectoryPublisher(execute_publication),
    )
    arguments = dict(
        eval_dataloader=loader,
        trajectory_runner=runner,
        cfg=cfg,
        global_step=0,
        tokenizer=tokenizer,
        trajectory_sink=sink,
    )
    if scored:
        metrics = await evaluation(**arguments)
        assert metrics["eval/all/avg_verifier_score"] == 0.0
        assert metrics["eval/all/verifier_score_coverage"] == 1.0
    else:
        with pytest.raises(ValueError, match="Evaluation verifier coverage"):
            await evaluation(**arguments)
    record = json.loads((tmp_path / "dumped_evals/global_step_0_evals/panel.jsonl").read_text())
    assert record["uid"] == ("eval-0-0--paired-task" if step_wise else "paired-task")
    assert record["verification"]["status"] == ("verified" if scored else "unavailable")
    assert record["verification"]["score"] == (0.0 if scored else None)


class EvaluationLifecycle:
    def __init__(self):
        self.events = []

    async def pause_and_drain(self):
        self.events.append("drained")

    def resume(self):
        self.events.append("resumed")

    def set_trajectory_sink(self, sink):
        pass

    async def start_eval_session(self, **kwargs):
        self.events.append("started")

    async def stop_eval_session(self):
        self.events.append("stopped")


@pytest.mark.asyncio
@pytest.mark.parametrize("eval_limit,expected_peak", [(2, 2), (None, 3)])
async def test_evaluation_session_caps_shared_work_independently_of_training(eval_limit, expected_peak):
    cfg = get_default_config()
    cfg.trajectory_runner.max_concurrent_tasks = 3
    cfg.trajectory_runner.eval_max_concurrent_tasks = eval_limit
    ready = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0

    async def work():
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == expected_peak:
            ready.set()
        await release.wait()
        active -= 1

    runner = EvaluationLifecycle()
    async with evaluation_session(runner, cfg, 0, object()) as pool:
        async with asyncio.TaskGroup() as tasks:
            for source in ("native", "sandbox", "callback"):
                tasks.create_task(pool.run(source, work))
            async with asyncio.timeout(5):
                await ready.wait()
            release.set()
    assert peak == expected_peak
    assert runner.events == ["started", "stopped"]


def test_eval_only_task_dataset_preserves_private_grader_and_prefill_contract(tmp_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )
    grader = ScriptGrader(argv=("sh", "/tests/private-grader.sh"), environment=NATIVE_ENVIRONMENT)
    shell_task = text_task("shell-private").model_copy(update={"grader": grader})
    native_task = text_task("native-private").model_copy(
        update={
            "grader": grader,
            "context": ConversationInput(
                events=(
                    TextMessage(role="user", content="Complete this"),
                    TextMessage(role="assistant", content="prefix"),
                ),
            ),
        }
    )
    expected = [
        lowered_task(shell_task, verifier_backend="daytona"),
        lowered_task(native_task, "prefill", max_turns=1, verifier_backend="daytona"),
    ]
    path = tmp_path / "validation.parquet"
    write_tasks(path, expected)
    experiment = object.__new__(EvalOnlyEntrypoint)
    experiment.cfg = get_default_config()
    experiment.cfg.trainer.evaluation_runner = "task"
    experiment.cfg.trainer.max_prompt_length = 8
    experiment.cfg.data.val_data = [str(path)]
    experiment.tokenizer = tokenizer

    dataset = experiment.get_eval_dataset()
    selected = dataset.collate_fn(list(dataset))

    assert [row["input"].spec for row in selected] == expected
    assert [row["uid"] for row in selected] == ["0", "1"]


@pytest.mark.asyncio
async def test_canceled_saved_answer_attempt_cannot_spend_a_second_grade(task_inputs, tmp_path, monkeypatch):
    _, request = task_inputs
    spec = request.rows[0].spec
    spec = spec.model_copy(
        update={
            "task": spec.task.model_copy(
                update={
                    "answer_type": AnswerType.TEXT,
                    "context": ConversationInput(events=(TextMessage(role="assistant", content='{"value":'),)),
                    "grader": verifyit_package(
                        StructureClosureSpec('{"value":', '{"value":12}', ClosureFormat.JSON)
                    ).grader,
                }
            ),
            "harness": "prefill",
            "limits": spec.limits.model_copy(update={"max_turns": 1}),
        }
    )
    messages = ({"role": "assistant", "content": '{"value":'},)
    turn = ModelTurn({"role": "assistant", "content": "12}"}, (1,), (2,), None, "stop", text="12}")
    root = str(tmp_path / "answers")
    save_answer(root, spec, "arithmetic", 0, ModelRequest(messages, dict(prefill_prompt(spec.task).options)), turn)
    admit_answer(root, "arithmetic", 0, (*messages, turn.message), "grade", "TimeoutError", "infra_error")
    path = Path(answer_uri(root, "arithmetic"))
    record = SavedAnswer.model_validate_json(path.read_bytes())
    admission = AnswerAdmission.model_validate_json(Path(str(path) + ".admission.json").read_bytes())
    entered, released = asyncio.Event(), asyncio.Event()
    calls = []

    async def pending_original_grade(*args, **kwargs):
        calls.append(1)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    monkeypatch.setattr("marinskyrl.saved_answers.regrade_answer", pending_original_grade)
    attempt = asyncio.create_task(regrade_once(record, admission, spec, {}, str(tmp_path / "grades"), timeout=2))
    await entered.wait()
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert released.is_set()
    resumed = await regrade_once(record, admission, spec, {}, str(tmp_path / "grades"), timeout=2)
    assert resumed["status"] == Outcome.UNAVAILABLE and resumed["reward"] is None
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["ready", "blocked"])
@pytest.mark.parametrize("harness", ["prefill", "shellbox"])
async def test_grader_failure_keeps_durable_answer_for_one_retry_without_generation(
    task_inputs, tmp_path, monkeypatch, storage, harness
):
    config, request = task_inputs
    spec = request.rows[0].spec
    task = spec.task.model_copy(
        update={
            "context": ConversationInput(
                events=(*spec.task.context.events, TextMessage(role="assistant", content='{"value":'))
            ),
            "answer_type": AnswerType.TEXT,
            "grader": verifyit_package(StructureClosureSpec('{"value":', '{"value":12}', ClosureFormat.JSON)).grader,
        }
    )
    spec = lowered_task(task, harness, max_turns=1)
    request = replace(request, rows=(task_input(spec),), batch_metadata=BatchMetadata(0, "eval"))
    root = str(tmp_path / "answers")
    if storage == "blocked":
        Path(root).write_text("storage unavailable")
    path = Path(answer_uri(root, "arithmetic"))
    generated = []
    grades = []

    class Engine:
        tokenizer = WorkerTokenizer()

        async def generate(self, model_request):
            generated.append(model_request)
            return {
                "responses": ["12}"],
                "token_provenance": TokenProvenance.ENGINE,
                "assistant_messages": [{"role": "assistant", "content": "12}"}],
                "prompt_ids": model_request.get("prompt_token_ids", [[1, 2]]),
                "response_ids": [[11, 12]],
                "response_logprobs": [[-0.1, -0.2]],
                "stop_reasons": ["stop"],
            }

    def original_grader(task, attempt):
        # The original grade cannot begin until the answer survived an exclusive filesystem write.
        record = SavedAnswer.model_validate_json(path.read_bytes())
        assert record.messages[-1]["content"] == "12}"
        grades.append(task.id)
        if len(grades) == 1:
            raise RuntimeError("temporary grader failure")
        return grade_answer(task, attempt)

    monkeypatch.setattr("rolloutengine.grading.grade_answer", original_grader)
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, WorkerTokenizer()),
        Engine(),
        {},
        shutdown_timeout=30,
        eval_saved_answers_root=root,
    )
    try:
        if storage == "blocked":
            with pytest.raises(ExceptionGroup) as failure:
                await worker.generate(request)
            assert failure.value.subgroup(OSError) is not None
            assert len(generated) == 1 and not grades
            return
        (failed,) = await worker.generate(request)
    finally:
        await worker.shutdown()
    assert failed.grade.reward is None and failed.failure is not None
    record = SavedAnswer.model_validate_json(path.read_bytes())
    admission = AnswerAdmission.model_validate_json(Path(str(path) + ".admission.json").read_bytes())
    receipt = await regrade_once(record, admission, spec, {}, str(tmp_path / "regrades"), timeout=2)
    assert receipt["status"] == Outcome.GRADED and receipt["reward"] == 1
    assert record.response_token_ids == (11, 12)
    # Reading an existing admitted result cannot spend the same retry again.
    assert await regrade_once(record, admission, spec, {}, str(tmp_path / "regrades"), timeout=2) == receipt
    assert len(generated) == 1 and len(grades) == 2
    changed_request = record.model_copy(update={"request_options": {"unadmitted": "changed"}})
    rejected = await regrade_answer(changed_request, spec, {}, timeout=2)
    assert rejected.status == Outcome.INFRA_ERROR and rejected.reward is None
    assert len(generated) == 1 and len(grades) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [MathSpec("12"), JudgeSpec(references=("12",), model="fixture")])
async def test_recovery_admission_rejects_unbounded_host_grading_before_model(task_inputs, tmp_path, unsupported):
    config, request = task_inputs
    spec = request.rows[0].spec
    task = spec.task.model_copy(
        update={
            "answer_type": AnswerType.TEXT,
            "context": ConversationInput(events=(TextMessage(role="assistant", content="Answer:"),)),
            "grader": verifyit_package(
                unsupported, environment=NATIVE_ENVIRONMENT if isinstance(unsupported, JudgeSpec) else None
            ).grader,
        }
    )
    spec = spec.model_copy(
        update={"task": task, "harness": "prefill", "limits": spec.limits.model_copy(update={"max_turns": 1})}
    )
    request = replace(request, rows=(task_input(spec),), batch_metadata=BatchMetadata(0, "eval"))
    calls = []

    class Engine:
        tokenizer = WorkerTokenizer()

        async def generate(self, request):
            calls.append(request)
            raise AssertionError("Unsupported recovery must fail admission before inference")

    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, WorkerTokenizer()),
        DirectModelClient(Engine()),
        {},
        shutdown_timeout=30,
        eval_saved_answers_root=str(tmp_path / "answers"),
        judge_connection=JudgeConnection("https://judge.invalid/v1", "unused"),
    )
    try:
        with pytest.raises(ExceptionGroup) as failure:
            await worker.generate(request)
        assert failure.value.subgroup(ValueError) is not None
    finally:
        await worker.shutdown()
    assert calls == []
    assert not (tmp_path / "answers").exists()


@pytest.mark.asyncio
async def test_saved_answer_replay_preserves_verifier_user_and_bounds_hung_close(task_inputs, tmp_path, monkeypatch):
    _, request = task_inputs
    original = request.rows[0].spec
    task = original.task.model_copy(
        update={
            "answer_type": AnswerType.TEXT,
            "context": ConversationInput(events=(TextMessage(role="assistant", content="Answer:"),)),
            "grader": ScriptGrader(argv=("true",), environment=NATIVE_ENVIRONMENT),
        }
    )
    spec = original.model_copy(
        update={
            "task": task,
            "harness": "prefill",
            "verifier_machine": machine_runtime("fixture").model_copy(update={"user": "123", "cleanup_timeout": 0.01}),
            "limits": original.limits.model_copy(update={"max_turns": 1}),
        }
    )
    prompt = prefill_prompt(task)
    root = str(tmp_path / "answers")
    turn = ModelTurn({"role": "assistant", "content": "12"}, (1,), (2,), None, "stop", text="12")
    save_answer(root, spec, "arithmetic", 0, ModelRequest(prompt.messages, dict(prompt.options)), turn)
    record = SavedAnswer.model_validate_json(Path(answer_uri(root, "arithmetic")).read_bytes())
    users = []
    closing, canceled = asyncio.Event(), asyncio.Event()

    class Machine:
        async def run(self, command):
            users.append(command.user)
            return Result(0, b"", b"", False, False, ExitReason.EXITED)

        async def close(self):
            closing.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()

    class Factory:
        backend = Backend.DAYTONA

        def check_environment(self, requirements, spec):
            pass

        async def create(self, spec):
            return Machine()

    async def original_sandbox_grade(task, attempt, factory, machine_spec, **kwargs):
        machine = await factory.create(machine_spec)
        try:
            await machine.run(Command(("true",)))
            return GradeResult(Outcome.GRADED, 1.0)
        finally:
            await machine.close()

    monkeypatch.setattr("rolloutengine.grading.grade_in_sandbox", original_sandbox_grade)
    async with asyncio.timeout(3):
        grade = await regrade_answer(record, spec, {"fixture": Factory()}, timeout=2)
        await canceled.wait()
    assert users == ["123"] and closing.is_set()
    assert grade.reward == 1.0
    assert grade.diagnostics["cleanup_errors"] == [{"operation": "machine_close", "exception_type": "TimeoutError"}]
