"""Behavior tests for the structured SkyRL launch configuration."""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from datetime import datetime, timezone
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from omegaconf import OmegaConf, open_dict
from skyrlrecipe import SkyRLRecipe
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
from rolloutengine.models import ModelTurn
from skyrl_gym.answer_tasks import grade_gsm8k
from skyrl_gym.task_sessions import AnswerTaskSession

from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from finestore.rl.run_records import read_attempt, read_attempt_metrics, read_evaluations
from cloud.iris import launch, skyrl_entrypoint, task_runtime, training_driver
from cloud.iris.launch_config import LaunchTopology, load_launch_config, validate_launch_config
from cloud.iris.role_plan import derive_num_nodes, derive_role_plan
from cloud.iris.rl_config_translation import (
    RL_CONFIG_PAYLOAD_ENV,
    compose_skyrl_config,
    materialize_launch_config,
    parse_rl_config,
)
from skyrl_train import trainer as trainer_module
from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import HFModelSaveCallback, create_default_callbacks
from skyrl_train.distributed.step_policy import NonfiniteStepPolicy, nonfinite_step_policy
from skyrl_train.dataset.tasks import source_row_task
from skyrl_train.evaluate import _EvaluationRollouts, _persist_evaluation
from skyrl_train.task_inputs import TrajectoryMetadata
from taskcompendium.grading_result import Outcome
from marinskyrl.runtime_options import reference_model_required
from marinskyrl.inference_placement import validate_expert_block_transport
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.tracking import Tracking
from skyrl_train.config.rollout_validation import EntrypointOperation, validate_rollout_launch
from skyrl_train.inference_engines import placement as inference_placement
from skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    MODEL_METADATA_PATH_KEY,
    _build_inference_engine_runtime_env,
    should_verify_inference_workers,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config


@pytest.mark.parametrize(
    "overrides", [{}, {"resume_mode": "from_path"}, {"resume_mode": "none"}, {"resume_path": "other"}]
)
def test_starting_checkpoint_requires_automatic_recovery(tmp_path: Path, overrides: dict) -> None:
    raw = raw_launch_config()
    checkpoint = {"uri": "s3://seed/checkpoints/global_step_2", "identity": "seed@2026.11.13:abc"}
    raw["inputs"]["starting_checkpoint"] = checkpoint
    raw["skyrl"]["trainer"].update(overrides)
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    if overrides:
        with pytest.raises(ValueError, match="starting_checkpoint"):
            load_launch_config(path)
    else:
        resolved = load_launch_config(path)
        assert resolved.skyrl.trainer.resume_mode == "latest"
        assert resolved.skyrl.trainer.resume_path == checkpoint["uri"]


@pytest.mark.parametrize("entrypoint", ["generate", "terminal_bench_generate"])
def test_generation_rejects_starting_checkpoint(tmp_path: Path, entrypoint: str) -> None:
    raw = raw_launch_config()
    raw["skyrl"]["entrypoint"] = entrypoint
    raw["inputs"]["starting_checkpoint"] = {"uri": "s3://seed/global_step_2", "identity": "seed@2:abc"}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="starting_checkpoint"):
        load_launch_config(path)


@pytest.mark.parametrize(
    ("trainer_gpus", "rollout_gpus", "expected_nodes"),
    [
        (32, 16, 12),
        (32, 32, 16),
        (32, 64, 24),
        (48, 16, 16),
        (48, 32, 20),
        (48, 64, 28),
        (64, 16, 20),
        (64, 32, 24),
        (64, 64, 32),
        (48, 12, 15),
        (72, 16, 22),
    ],
)
def test_cross_host_ep_launch_derives_whole_node_claims(tmp_path, trainer_gpus, rollout_gpus, expected_nodes):
    raw = raw_launch_config()
    raw["run"]["export_hf"] = False
    raw["iris"]["allocation"].update(num_nodes=expected_nodes, gpus_per_node=4)
    raw["skyrl"]["trainer"]["placement"].update(
        colocate_all=False,
        policy_num_nodes=trainer_gpus // 4,
        policy_num_gpus_per_node=4,
    )
    raw["skyrl"]["trainer"]["policy"] = {
        "megatron_config": {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 4,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
        }
    }
    raw["skyrl"]["generator"].update(
        num_inference_engines=1,
        inference_engine_data_parallel_size=rollout_gpus,
        inference_engine_expert_parallel_size=rollout_gpus,
    )
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path)
    plan = derive_role_plan(OmegaConf.to_container(config.skyrl, resolve=True))
    rollout = plan.claim("rollout")
    policy = plan.claim("policy")

    assert derive_num_nodes(plan) == expected_nodes
    assert policy.num_nodes * policy.gpus_per_node == trainer_gpus
    assert rollout.num_nodes * rollout.gpus_per_node == rollout_gpus
    assert {bundle.name: bundle.num_nodes for bundle in plan.bundles} == {
        "policy": trainer_gpus // 4,
        "rollout": rollout_gpus // 4,
    }


@pytest.mark.parametrize(
    ("gpus_per_node", "engines", "dp", "rollout_nodes"),
    [(4, 4, 3, 4), (8, 5, 3, 3), (4, 3, 10, 8), (8, 2, 12, 3)],
)
def test_rollout_pool_respects_host_local_engines_and_cross_host_workers(gpus_per_node, engines, dp, rollout_nodes):
    config = raw_launch_config()["skyrl"]
    config["trainer"]["placement"].update(colocate_all=False, policy_num_gpus_per_node=gpus_per_node)
    config["generator"].update(
        num_inference_engines=engines,
        inference_engine_data_parallel_size=dp,
        inference_engine_expert_parallel_size=dp,
    )
    plan = derive_role_plan(config)
    assert plan.claim("rollout").num_nodes == rollout_nodes
    assert derive_num_nodes(plan) == 1 + rollout_nodes


@pytest.mark.parametrize(("tp", "pp", "dp", "ep"), [(2, 1, 4, 4), (1, 2, 4, 4), (1, 1, 8, 4)])
def test_unsupported_cross_host_layouts_still_require_one_node(tp, pp, dp, ep):
    config = raw_launch_config()["skyrl"]
    config["trainer"]["placement"].update(colocate_all=False, policy_num_gpus_per_node=4)
    config["generator"].update(
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=tp,
        inference_engine_pipeline_parallel_size=pp,
        inference_engine_data_parallel_size=dp,
        inference_engine_expert_parallel_size=ep,
    )
    with pytest.raises(ValueError, match="each ray-backed rollout engine requires"):
        derive_role_plan(config)


def test_training_channels_require_compatible_weight_transport() -> None:
    config = get_default_config()
    config.trainer.algorithm.batch_invariant = True
    config.trainer.placement.colocate_all = False
    config.generator.inference_engine_tensor_parallel_size = 1
    config.generator.weight_sync_transport = "expert_block"
    # Expert-block groups inherit training channels; broadcast has its own override.
    validate_expert_block_transport(config)
    config.trainer.policy.megatron_config.nccl_max_channels = 8
    with pytest.raises(ValueError, match="nccl_max_channels"):
        validate_expert_block_transport(config)
    config.generator.weight_sync_transport = "broadcast"
    validate_expert_block_transport(config)


def test_colocated_expert_block_accepts_only_the_four_gpu_verified_layout() -> None:
    config = get_default_config()
    config.trainer.strategy = "megatron"
    config.trainer.placement.colocate_all = True
    config.trainer.placement.policy_num_nodes = 1
    config.trainer.placement.policy_num_gpus_per_node = 4
    config.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    config.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    config.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    config.trainer.policy.megatron_config.expert_model_parallel_size = 4
    config.generator.weight_sync_transport = "expert_block"
    config.generator.weight_sync_backend = "nccl"
    config.generator.backend = "vllm"
    config.generator.run_engines_locally = True
    config.generator.num_inference_engines = 1
    config.generator.inference_engine_tensor_parallel_size = 1
    config.generator.inference_engine_pipeline_parallel_size = 1
    config.generator.inference_engine_data_parallel_size = 4
    config.generator.inference_engine_expert_parallel_size = 4
    config.generator.expert_block_sync.verify = True
    with open_dict(config.generator.engine_init_kwargs):
        config.generator.engine_init_kwargs.kernel_config = {"moe_backend": "triton"}

    validate_expert_block_transport(config)

    config.generator.inference_engine_data_parallel_size = 2
    with pytest.raises(ValueError, match="colocation requires"):
        validate_expert_block_transport(config)


def test_colocated_expert_block_rejects_generation_before_ray_starts() -> None:
    config = get_default_config()
    config.trainer.placement.colocate_all = True
    config.generator.weight_sync_transport = "expert_block"

    with pytest.raises(ValueError, match="requires a training entrypoint"):
        validate_rollout_launch(config, EntrypointOperation.GENERATE)


@pytest.mark.parametrize(
    "entrypoint",
    ["skyrl_train.entrypoints.main_generate", "skyrl_train.entrypoints.taskcompendium"],
)
def test_colocated_expert_block_rejects_unverified_launch_entrypoint(tmp_path: Path, entrypoint: str) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw_launch_config()))
    config = load_launch_config(path)
    config.skyrl.generator.weight_sync_transport = "expert_block"
    config.runtime.entrypoint = entrypoint

    with pytest.raises(ValueError, match="requires the standard training entrypoint"):
        validate_launch_config(config)


@pytest.mark.parametrize(
    ("engine_sleep", "require_verified_placements", "expected"),
    [(False, False, True), (True, False, False), (True, True, True)],
)
def test_worker_verification_handles_generation_and_sleeping_training(
    engine_sleep: bool, require_verified_placements: bool, expected: bool
) -> None:
    assert (
        should_verify_inference_workers(
            tensor_parallel_size=1,
            data_parallel_size=16,
            use_mp_backend=False,
            inference_engine_enable_sleep=engine_sleep,
            require_verified_placements=require_verified_placements,
        )
        is expected
    )


def test_shared_bundle_reports_reject_duplicate_or_misrouted_receivers(monkeypatch) -> None:
    shared_group = object()
    bundle_nodes = {0: "node-a", 1: "node-b", 2: "node-a", 3: "node-b"}
    monkeypatch.setattr(
        inference_placement,
        "placement_group_table",
        lambda group: {"bundles_to_node_id": bundle_nodes},
    )
    nodes = inference_placement.colocated_inference_bundle_nodes(
        shared_group, [[2], [0], [3], [1]], num_inference_engines=1, data_parallel_size=4, pipeline_parallel_size=1
    )
    assert nodes == [["node-a", "node-a", "node-b", "node-b"]]
    reports = [
        [
            {
                "host": "host-a" if rank < 2 else "host-b",
                "gpu_uuid": f"gpu-{rank}",
                "dp_rank": rank,
                "dp_world_size": 4,
                "ep_rank": rank,
                "ep_world_size": 4,
                "torch_rank": rank,
                "torch_world_size": 4,
            }
        ]
        for rank in range(4)
    ]
    kwargs = dict(
        bundle_nodes=nodes,
        node_hosts={"node-a": "host-a", "node-b": "host-b"},
        relative_rank_offsets=[0] * 4,
        data_parallel_size=4,
        expert_parallel_size=4,
    )
    placements = inference_placement.verified_inference_replica_placements(reports, **kwargs)
    assert [row.worker.gpu_uuid for row in placements] == [f"gpu-{rank}" for rank in range(4)]

    reports[3][0]["gpu_uuid"] = "gpu-0"
    with pytest.raises(ValueError, match="distinct"):
        inference_placement.verified_inference_replica_placements(reports, **kwargs)
    reports[3][0]["gpu_uuid"] = "gpu-3"
    reports[3][0]["ep_rank"] = 2
    with pytest.raises(ValueError, match="EP rank"):
        inference_placement.verified_inference_replica_placements(reports, **kwargs)


def test_launch_preserves_harbor_dataset_relative_paths(tmp_path):
    raw = raw_launch_config()
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
            "minimum_native_coverage": 0.99,
        }
    ]
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_launch_config(path)
    resolved_task = config.skyrl.trainer.callbacks[0].cohorts[0].job.tasks[0]
    assert OmegaConf.to_container(resolved_task) == task


def test_default_hf_export_requests_only_the_terminal_checkpoint(tmp_path):
    raw = raw_launch_config()
    raw["skyrl"]["trainer"].update(ckpt_interval=1)
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_launch_config(path)
    [export_callback] = [
        callback for callback in create_default_callbacks(config.skyrl) if isinstance(callback, HFModelSaveCallback)
    ]
    state = TrainerState(global_step=1, epoch=0, total_steps=3, num_steps_per_epoch=3)

    assert not export_callback.on_step_end(state, TrainerControl()).should_save_hf_model
    assert export_callback.on_train_end(state, TrainerControl()).should_save_hf_model


@pytest.mark.asyncio
@pytest.mark.parametrize("text,reward", [("#### 12", 1.0), ("#### 12\nMore reasoning.", 0.0)])
async def test_custom_session_settings_reach_private_grading_through_the_launch_config(
    tmp_path: Path, text: str, reward: float
) -> None:
    raw = raw_launch_config()
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
    raw = raw_launch_config()
    raw["run"]["export_hf"] = not probe_enabled
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
    if probe_enabled:
        config.run.export_hf = True
        OmegaConf.save(config, path)
        with pytest.raises(ValueError, match="mismatch_probe requires the synchronous trainer"):
            load_launch_config(path)
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
    raw = raw_launch_config()
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
    path.write_text(yaml.safe_dump(raw_launch_config()))
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
    raw = raw_launch_config()
    raw["skyrl"]["entrypoint"] = entrypoint
    raw["skyrl"]["trainer"]["placement"]["colocate_all"] = False
    raw["skyrl"]["trainer"]["rollout_buffer"] = {"max_staleness_steps": max_staleness_steps}
    raw["skyrl"]["trainer"]["algorithm"]["off_policy_correction"] = "none"
    raw["iris"]["allocation"]["num_nodes"] = 1 if entrypoint == "generate" else 2
    if entrypoint == "generate":
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        raw["skyrl"]["trainer"].setdefault("policy", {}).setdefault("model", {})["lora"] = {
            "rank": 8,
            "adapter_path": str(adapter),
        }
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    assert load_launch_config(path).runtime.training_type == expected


def test_object_store_uris_in_skyrl_config_reach_the_task_unchanged(tmp_path: Path) -> None:
    raw = raw_launch_config()
    raw["skyrl"]["trainer"]["rollout_buffer"] = {"object_store_root": "s3://runs/smoke/rollouts"}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    assert config.skyrl.trainer.rollout_buffer.object_store_root == "s3://runs/smoke/rollouts"


@pytest.mark.parametrize("model", ["", " policy "])
def test_launch_rejects_a_noncanonical_model_label(tmp_path: Path, model: str) -> None:
    raw = raw_launch_config()
    raw["run"]["model"] = model
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match=r"run\.model must be a non-empty name"):
        load_launch_config(path)


def test_qwen_smoke_hugging_face_model_input_reaches_the_runner_without_a_model_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = raw_launch_config()
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


@pytest.mark.parametrize(
    ("entrypoint", "own_tokenizer", "records_enabled", "resume"),
    [
        ("standard", False, True, False),
        ("generate", False, True, False),
        ("standard", True, True, False),
        ("generate", False, False, False),
        ("standard", False, True, True),
    ],
)
def test_forwarded_launch_metadata_reaches_attempt_startup_record(
    tmp_path: Path, monkeypatch, entrypoint, own_tokenizer, records_enabled, resume
) -> None:
    destination = tmp_path / "launch.yaml"
    document = raw_launch_config()
    document["run"]["coordinator_iris_job_id"] = "/tests/coordinator"
    document["run"]["model"] = "test-policy"
    if own_tokenizer:
        document["inputs"]["model"].update(tokenizer_uri=None, tokenizer_revision=None)
    if resume:
        document["inputs"]["starting_checkpoint"] = {
            "uri": "s3://seed/checkpoints/global_step_2",
            "identity": "checkpoints/seed/resume/step-2@2026.11.13:abc",
        }
    document["iris"]["max_retries"] = 0
    backends = ["console"]
    if records_enabled:
        backends.append("records")
        if entrypoint == "generate" or resume:
            backends.append("wandb")
    document["skyrl"]["trainer"]["logger"] = backends
    document["skyrl"]["entrypoint"] = entrypoint
    document["artifacts"]["export_root"] = str(tmp_path / "exports")
    contents = yaml.safe_dump(document).encode()

    path = materialize_launch_config(
        str(destination),
        {RL_CONFIG_PAYLOAD_ENV: base64.b64encode(contents).decode("ascii")},
    )

    assert path == str(destination)
    assert destination.read_bytes() == contents

    wandb_runs = []

    def start_wandb(**kwargs):
        wandb_runs.append(kwargs)
        return SimpleNamespace(
            id=kwargs.get("id", "sdk-generated-id"),
            define_metric=lambda *args, **kwargs: None,
            log=lambda **kwargs: None,
            finish=lambda **kwargs: None,
        )

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=start_wandb, Settings=lambda **kwargs: kwargs))
    monkeypatch.setenv("WANDB_API_KEY", "fixture-api-key")
    export_root = str(tmp_path / "exports")
    first_start = datetime.now(timezone.utc)
    fail = True

    def workload(config):
        assert read_attempt(export_root, "attempt-1") is not None
        assert config.runtime.iris_job_id == "/tests/bootstrap-trainer"
        if fail:
            raise RuntimeError("workload initialization failed")
        with Tracking("project", "display-name", config.trainer.logger, config) as tracker:
            tracker.start(resumed_step=None)
            if entrypoint == "standard":
                for invalid in ({"step": 8}, {"attempt_id": "foreign"}, {"metric_attempt_ids": {}}):
                    with pytest.raises(ValueError, match="active execution|reserved for metric provenance"):
                        tracker.log(invalid, step=9)
            tracker.log({"startup/ready": 1}, step=0)
            if resume:
                batch = {"task_ids": ["task"], "response_ids": [[1]], "rewards": [1.0], "task_outcomes": [Outcome.GRADED]}
                rollouts = _EvaluationRollouts(
                    batch, ["answer"], [TrajectoryMetadata("task", "fixture")], [0], [], batch
                )
                _persist_evaluation(config, 3, None, rollouts, export_root)

    monkeypatch.setattr(skyrl_entrypoint, "get_job_info", lambda: SimpleNamespace(job_id="/tests/bootstrap-trainer"))

    def installed_version(name):
        if not fail:
            return "fixture-vllm-v2"
        if entrypoint == "generate":
            return "fixture-vllm"
        raise skyrl_entrypoint.PackageNotFoundError(name)

    monkeypatch.setattr(skyrl_entrypoint, "version", installed_version)
    module = (
        "skyrl_train.entrypoints.main_generate" if entrypoint == "generate" else "skyrl_train.entrypoints.main_base"
    )
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(run=workload))
    with pytest.raises(RuntimeError, match="workload initialization failed"):
        skyrl_entrypoint.run_config(destination)
    attempt = read_attempt(export_root, "attempt-1")
    assert attempt is not None
    runtime = attempt.native.config["skyrl"]["runtime"]
    assert {key: runtime[key] for key in ("run_id", "attempt_id", "iris_job_id", "coordinator_iris_job_id")} == {
        "run_id": "smoke",
        "attempt_id": "attempt-1",
        "iris_job_id": "/tests/bootstrap-trainer",
        "coordinator_iris_job_id": "/tests/coordinator",
    }
    assert attempt.native.config["inputs"]["model"]["tokenizer_uri"] == (
        None if own_tokenizer else "s3://models/qwen-tokenizer"
    )
    assert first_start <= datetime.fromisoformat(attempt.started_at) <= datetime.now(timezone.utc)
    assert attempt.vllm_version == ("fixture-vllm" if entrypoint == "generate" else None)
    fail = False
    skyrl_entrypoint.run_config(destination)
    final = read_attempt(export_root, "attempt-1")
    assert final is not None
    if not records_enabled:
        assert final.iris_job_id == "/tests/bootstrap-trainer" and final.metrics_started_at is None
        assert read_attempt_metrics(export_root, "attempt-1") == {}
        return
    assert final.metrics_started_at is not None
    assert final.started_at == attempt.started_at and final.vllm_version == attempt.vllm_version
    assert read_attempt_metrics(export_root, "attempt-1")[0]["startup/ready"] == 1
    assert final.iris_job_id == "/tests/bootstrap-trainer"
    assert final.native.result.iris_job_id == "/tests/bootstrap-trainer"
    if entrypoint == "standard":
        with pytest.raises(ValueError, match="fresh attempt_id"):
            skyrl_entrypoint.run_config(destination)
        assert read_attempt_metrics(export_root, "attempt-1")[0]["startup/ready"] == 1
    if entrypoint == "generate":
        assert [(run["id"], run["group"], run["resume"]) for run in wandb_runs] == [("attempt-1", "smoke", "allow")]
        assert wandb_runs[0]["tags"] == ["test-policy"]
        assert wandb_runs[0]["config"]["model"] == "test-policy"
    if resume:
        (evaluation,) = read_evaluations(export_root, attempt_id="attempt-1", evaluation="eval/")
        for field, expected in (
            ("model_source_identity", document["inputs"]["model"]["identity"]),
            ("starting_checkpoint_identity", document["inputs"]["starting_checkpoint"]["identity"]),
        ):
            assert getattr(evaluation, field) == expected
            assert wandb_runs[0]["config"]["runtime"][field] == expected


def test_evaluation_metric_names_survive_iris_path_resolution(tmp_path: Path) -> None:
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml").read_text())
    groups = {"eval/train/avg_score": ["eval/cat_count_n1/avg_score", "eval/cat_count_n2/avg_score"]}
    profiles = {"sampled": {"sampling_params": {"stop": ["./END"]}}}
    raw["trainer"]["callbacks"] = [
        {"type": "evaluation", "metric_groups": groups, "additional_evaluations": profiles},
        {
            "type": "best_checkpoint",
            "selection_metric": "eval/identity_behavior_holdout/pass_at_1",
            "initial_model_uri": "s3://models/initial",
            "expected_tasks": 4,
            "minimum_scored_tasks": 2,
        },
    ]
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = compose_skyrl_config(
        parse_rl_config(OmegaConf.load(path)), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
    ).config

    assert config.trainer.callbacks[0].metric_groups == groups
    assert config.trainer.callbacks[0].additional_evaluations == profiles
    assert config.trainer.callbacks[1].selection_metric == "eval/identity_behavior_holdout/pass_at_1"


@pytest.mark.parametrize("consumer", ["metric_groups", "stop_when"])
def test_callback_setup_rejects_final_metrics_during_periodic_evaluation(tmp_path: Path, consumer: str) -> None:
    config = get_default_config()
    metric = "eval/full_task_set/all/avg_score"
    dependency = {"eval/reporting/avg_score": [metric]} if consumer == "metric_groups" else {metric: {"minimum": 0.5}}
    config.trainer.callbacks = [
        {"type": "evaluation", consumer: dependency},
        {"type": "evaluation", "eval_steps": 0, "additional_evaluations": {"full_task_set": {"schedule": "final"}}},
    ]

    raw = raw_launch_config()
    raw["run"]["export_hf"] = False
    raw["skyrl"]["trainer"]["callbacks"] = OmegaConf.to_container(config.trainer.callbacks, resolve=True)
    path = tmp_path / "periodic-final.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=f"final-only metrics.*{metric}"):
        load_launch_config(path)
    raw["skyrl"]["trainer"]["callbacks"][0]["eval_steps"] = 0
    path.write_text(yaml.safe_dump(raw))
    create_default_callbacks(load_launch_config(path).skyrl)


@pytest.mark.parametrize(
    "case,patch,error",
    [
        ("inherit", {}, None),
        ("explicit", {"n_samples_per_prompt": 3}, None),
        ("sampling-omitted", {}, None),
        ("sampling-null", {"sampling_params": None}, None),
        ("sampling-empty", {"sampling_params": {}}, None),
        ("final-schedule", {"schedule": "final"}, None),
        ("generation-final", {"schedule": "final"}, "final evaluation profiles require training updates"),
        ("duplicate", {}, "duplicate evaluation profile: sampled"),
        ("field", {"answer_count": 2}, "answer_count"),
        ("sampling", {"sampling_params": {"temprature": 0.1}}, "temprature"),
        ("missing-profile", {}, "evaluation data has no additional evaluation profile: sampled"),
        ("two-inputs", {"data_files": ["another.parquet"]}, "both data_files and evaluation_data"),
        ("scalar-profile", 7, "additional_evaluations.sampled"),
        ("mixed-field-keys", {1: 0, "typo": 0}, "mapping keys must be strings"),
    ],
)
def test_additional_evaluation_data_is_staged_and_bound_in_training_driver(
    tmp_path: Path, monkeypatch, case, patch, error
) -> None:
    path = tmp_path / "evaluation.yaml"
    document = raw_launch_config()
    document["run"]["export_hf"] = False
    document["iris"]["max_retries"] = 0
    document["skyrl"]["entrypoint"] = "generate"
    generator = document["skyrl"].setdefault("generator", {})
    generator["eval_sampling_params"] = {
        "temperature": 0.5,
        "stop": ["base-stop"],
        "top_p": 0.73,
        "skip_special_tokens": False,
    }
    generator["eval_n_samples_per_prompt"] = 2
    profile = {
        "sampling_params": {
            "temperature": 0.0,
            "stop": None,
            "stop_token_ids": [7],
        },
        "n_samples_per_prompt": None,
    }
    profile = {**profile, **patch} if isinstance(patch, dict) else patch
    if case == "sampling-omitted":
        profile.pop("sampling_params")
    if case == "final-schedule":
        document["skyrl"]["entrypoint"] = "standard"
    document["skyrl"]["trainer"]["callbacks"] = [{"type": "evaluation", "additional_evaluations": {"sampled": profile}}]
    sources = []
    for name, identifier in (("primary", "throughput-only"), ("task_set", "task_set-only")):
        directory = tmp_path / name
        directory.mkdir()
        pq.write_table(pa.table({"task_id": [identifier]}), directory / "rows.parquet")
        sources.append(
            {
                "kind": "directory",
                "uri": directory.as_uri(),
                "identity": name,
                "local_path": str(tmp_path / ("staged-" + name)),
                "relative_path": "rows.parquet",
            }
        )
    document["inputs"]["validation_data"] = sources[:1]
    document["inputs"]["evaluation_data"] = {"sampled": sources[1:]}
    if case == "duplicate":
        document["skyrl"]["trainer"]["callbacks"].append(
            {"type": "evaluation", "additional_evaluations": {"sampled": profile}}
        )
    elif case == "missing-profile":
        document["skyrl"]["trainer"]["callbacks"] = []
    if error is not None:
        path.write_text(yaml.safe_dump(document))
        with pytest.raises(ValueError, match=error):
            load_launch_config(path)
        return
    expected_sampling = generator["eval_sampling_params"] | (profile.get("sampling_params") or {})
    authored = deepcopy(document["skyrl"])
    authored["trainer"].pop("seed", None)
    authored["trainer"]["logger"] = ["records"]
    document["skyrl"] = SkyRLRecipe.from_document(authored).to_skyrl()
    path.write_text(yaml.safe_dump(document))
    loaded = load_launch_config(path)
    recorded = loaded.skyrl.trainer.callbacks[0].additional_evaluations.sampled
    assert recorded.n_samples_per_prompt == (3 if case == "explicit" else 2)
    request = OmegaConf.to_container(recorded.sampling_params, resolve=True)
    assert {key: request[key] for key in expected_sampling} == expected_sampling
    assert recorded.schedule == ("final" if case == "final-schedule" else "every_evaluation")
    prepared = launch._prepare_config(loaded)
    task_runtime.materialize_data_sources(task_runtime._runtime_namespace(prepared).data_sources_json)
    runner = training_driver.LocalRLRunner(
        training_driver.LocalRLConfig(
            job_name="named-evaluation-data",
            model_path=str(prepared.inputs.model.local_path),
            model_source_uri=str(prepared.inputs.model.uri),
            model_source_identity=str(prepared.inputs.model.identity),
            val_data=list(prepared.inputs.validation_data),
            launch_config=prepared,
            resolved_config_uri=(tmp_path / "resolved.json").as_uri(),
        )
    )
    observed = {}

    def read_datasets(config):
        profile = config.skyrl.trainer.callbacks[0].additional_evaluations.sampled
        observed["primary"] = pq.read_table(config.skyrl.data.val_data[0])["task_id"].to_pylist()
        observed["profile"] = pq.read_table(profile.data_files[0])["task_id"].to_pylist()
        observed["request"] = OmegaConf.to_container(profile.sampling_params, resolve=True)
        observed["count"] = profile.n_samples_per_prompt
        assert config.skyrl.generator.eval_sampling_params.stop == ["base-stop"]
        return 0

    monkeypatch.setattr(runner, "_setup_environment", lambda: None)
    monkeypatch.setattr(runner, "_run_skyrl", read_datasets)
    assert runner.run() == 0
    assert observed["primary"] == ["throughput-only"]
    assert observed["profile"] == ["task_set-only"]
    assert observed["count"] == (3 if case == "explicit" else 2)
    assert {key: observed["request"][key] for key in expected_sampling} == expected_sampling

    native_request = get_sampling_params_for_backend("vllm", OmegaConf.create(observed["request"]))
    assert native_request["skip_special_tokens"] is False
    assert native_request["stop"] == expected_sampling["stop"]
    record = json.loads((tmp_path / "resolved.json").read_text())
    assert record["config"]["inputs"]["evaluation_data"]["sampled"][0]["identity"] == "task_set"
    stored = record["config"]["skyrl"]["trainer"]["callbacks"][0]["additional_evaluations"]["sampled"]
    assert stored["n_samples_per_prompt"] == observed["count"]
    assert stored["sampling_params"] == observed["request"]


def test_source_recipe_rejects_resolved_harbor_key_before_discarding_agent_settings(tmp_path: Path) -> None:
    raw = raw_launch_config()
    raw["skyrl"]["terminal_bench_config"] = {"installed_agent": {"agent": {"name": "pi"}}}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="terminal_bench_config.*terminal_bench"):
        load_launch_config(path)


def test_installed_agent_artifacts_follow_launch_attempt_and_survive_reload(tmp_path: Path) -> None:
    raw = raw_launch_config()
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
    raw = raw_launch_config()
    raw["skyrl"]["trainer"]["policy"] = {"max_consecutive_nonfinite_steps": None}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    action = nonfinite_step_policy(0, config.skyrl.trainer.policy.max_consecutive_nonfinite_steps)
    assert action is NonfiniteStepPolicy.FAIL


@pytest.mark.parametrize(("key", "value"), [("use_tis", True), ("tis_imp_ratio_cap", 2.0)])
def test_composed_launch_rejects_tis_selectors(tmp_path: Path, key: str, value) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw_launch_config()))
    config = load_launch_config(path)
    OmegaConf.update(config.skyrl.trainer.algorithm, key, value, force_add=True)
    OmegaConf.save(config, path)

    with pytest.raises(ValueError, match="off_policy_correction"):
        load_launch_config(path)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"algorithm.batch_invariant": True}, "cannot configure a remote inference server"),
        (
            {
                "policy.megatron_config.moe_router_replay": True,
                "policy.megatron_config.transformer_config_kwargs.moe_router_fusion": True,
            },
            "fused router bypasses",
        ),
        ({"rollout_buffer.max_staleness_steps": 1}, "off-policy OLD-anchored"),
        (
            {
                "rollout_buffer.max_staleness_steps": 1,
                "algorithm.off_policy_correction": "none",
                "placement.colocate_all": False,
            },
            None,
        ),
        ({"rollout_buffer.max_staleness_steps": 1, "algorithm.off_policy_correction": "none"}, "colocate_all requires"),
        (
            {
                "rollout_buffer.max_staleness_steps": 1,
                "algorithm.policy_loss_type": "behavior_clip",
                "placement.colocate_all": False,
            },
            None,
        ),
        ({"algorithm.policy_loss_type": "behavior_clip", "algorithm.off_policy_correction": "tis"}, "OLD-anchored"),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"action": "truncate", "high": 2.0}],
            },
            "discriminator 'kind'",
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
        (
            {"rollout_buffer.groups_per_family": {"math": 8}, "rollout_buffer.batch_policy": "rolling"},
            "requires batch_policy=full_batch",
        ),
        ({"rollout_buffer.groups_per_family": {"math": 7}}, "sums to 7, expected train_batch_size=8"),
        (
            {"rollout_buffer.groups_per_family": {"math": 8}, "data.sampling.kind": "domain-weighted"},
            r"requires data\.sampling\.kind to be null",
        ),
    ],
)
def test_launch_validates_correction_and_selection_contract(tmp_path: Path, overrides: dict, error: str | None):
    raw = OmegaConf.create(raw_launch_config())
    for key, value in overrides.items():
        path = key if key.startswith("data.") else f"trainer.{key}"
        OmegaConf.update(raw.skyrl, path, value, force_add=True)
    if overrides.get("placement.colocate_all") is False:
        raw.iris.allocation.num_nodes = 2
    if overrides.get("algorithm.batch_invariant"):
        raw.skyrl.generator.run_engines_locally = False
        raw.skyrl.generator.remote_inference_engine_urls = ["http://inference.test:8000"]
    path = tmp_path / "launch.yaml"
    OmegaConf.save(raw, path)
    if error is not None:
        with pytest.raises(ValueError, match=error) as raised:
            load_launch_config(path)
        assert "/tmp/" not in str(raised.value)
    else:
        # Explicitly uncorrected stale policies and active reward filters are valid launch contracts.
        load_launch_config(path)


def test_source_launch_preserves_nested_open_mappings(tmp_path):
    raw = raw_launch_config()
    raw["skyrl"]["generator"]["engine_init_kwargs"] = {"user_options": {"nested": {}}}
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert OmegaConf.to_container(load_launch_config(path).skyrl.generator.engine_init_kwargs.user_options) == {
        "nested": {}
    }


@pytest.mark.parametrize(
    ("remaining_families", "error"),
    [
        (["code"] * 7, "requires 8 rows for family 'code', found 7"),
        (["code"] * 8 + ["other"], r"names .* but training rows have families .*'other'"),
        (["code"] * 8, None),
    ],
    ids=["insufficient", "unlisted-family", "valid-pool"],
)
def test_launch_preflights_groups_per_family(tmp_path, remaining_families, error):
    data_path = tmp_path / "tasks.parquet"
    pq.write_table(pa.table({"family": ["math"] * 8 + remaining_families}), data_path)
    raw = raw_launch_config()
    raw["inputs"]["data_kind"] = "parquet"
    raw["inputs"]["train_data"] = [
        {"kind": "directory", "uri": str(tmp_path), "identity": "fixture", "local_path": str(tmp_path),
         "relative_path": "tasks.parquet"}
    ]
    raw["skyrl"]["data"] = {"kind": "parquet"}
    trainer = raw["skyrl"]["trainer"]
    trainer["rollout_buffer"] = {
        "groups_per_family": {"math": 4, "code": 4},
        "max_staleness_steps": 1,
    }
    trainer["placement"]["colocate_all"] = False
    trainer["algorithm"]["off_policy_correction"] = "none"
    raw["iris"]["allocation"]["num_nodes"] = 2
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    if error:
        with pytest.raises(ValueError, match=rf"trainer\.rollout_buffer\.groups_per_family {error}"):
            load_launch_config(path)
        return

    resolved = load_launch_config(path)
    assert resolved.skyrl.trainer.rollout_buffer.groups_per_family == {"math": 4, "code": 4}
    OmegaConf.save(resolved, path)
    data_path.unlink()
    assert load_launch_config(path) == resolved


def test_first_train_submission_validates_recipe_and_resolved_reload(tmp_path):
    raw = raw_launch_config()
    raw["runtime"]["entrypoint"] = ""
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    loaded = load_launch_config(path)
    resolved = tmp_path / "resolved.yaml"
    OmegaConf.save(loaded, resolved)
    assert OmegaConf.to_container(load_launch_config(resolved), resolve=True) == OmegaConf.to_container(
        loaded, resolve=True
    )
    rollout_buffer = raw["skyrl"]["trainer"].setdefault("rollout_buffer", {})
    rollout_buffer["groups_per_family"] = {}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="groups_per_family"):
        load_launch_config(path)
    del rollout_buffer["groups_per_family"]

    del raw["skyrl"]["context_budget"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="context_budget"):
        load_launch_config(path)


@pytest.mark.parametrize(
    ("geometry", "weight_decay", "error"),
    [
        ({"tensor_model_parallel_size": 2}, 0.0, "require tensor parallel size 1"),
        ({"expert_tensor_parallel_size": 2}, 0.0, "require expert tensor parallel size 1"),
        ({}, 0.1, "requires weight_decay=0"),
    ],
)
def test_launch_rejects_invalid_muonh_geometry_or_weight_decay(tmp_path, geometry, weight_decay, error):
    document = raw_launch_config()
    document["skyrl"]["trainer"]["policy"] = {
        "optimizer_config": {"optimizer": "MuonH", "weight_decay": weight_decay},
        "megatron_config": geometry,
    }
    path = tmp_path / "muonh.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=error):
        load_launch_config(path)


def test_source_launch_rejects_conflicting_author_values(tmp_path):
    cases = []
    for recipe_key, envelope_key in (("train_data", "train_data"), ("val_data", "validation_data")):
        raw = OmegaConf.create(raw_launch_config())
        raw.skyrl.data = {recipe_key: ["/authored/data"]}
        raw.inputs[envelope_key] = [
            {
                "uri": "s3://data/gsm8k",
                "identity": "sha256:gsm8k",
                "local_path": "/tmp/data/gsm8k",
                "relative_path": "train.parquet",
                "kind": "directory",
            }
        ]
        cases.append((raw, f"data.{recipe_key} conflicts"))
    for role in ("policy", "ref"):
        raw = OmegaConf.create(raw_launch_config())
        raw.skyrl.trainer.placement[f"{role}_num_gpus_per_node"] = 32
        cases.append((raw, rf"trainer\.placement\.{role}_num_gpus_per_node=32 exceeds"))
    raw = OmegaConf.create(raw_launch_config())
    raw.inputs.data_kind = "parquet"
    cases.append((raw, "inputs.data_kind"))
    raw = OmegaConf.create(raw_launch_config())
    raw.iris.allocation.num_nodes = 2
    raw.skyrl.trainer.placement.colocate_all = False
    raw.skyrl.generator.speculative_decoding = {
        "method": "eagle3",
        "model": {"source_uri": "s3://models/draft", "source_identity": "mutable"},
        "num_speculative_tokens": 3,
        "training": None,
    }
    cases.append((raw, "artifact sources require sha256"))
    raw = OmegaConf.create(raw_launch_config())
    raw.iris.allocation.num_nodes = 2
    cases.append((raw, "num_nodes"))
    raw = OmegaConf.create(raw_launch_config())
    raw.runtime.profile = "fsdp"
    cases.append((raw, "runtime.profile"))
    raw = OmegaConf.create(raw_launch_config())
    raw.skyrl.trainer.callbacks = [{"type": "best_checkpoint"}]
    cases.append((raw, "initial_model_uri"))
    raw = OmegaConf.create(raw_launch_config())
    raw.skyrl.checkpoint_export = {"hf_hub_repo_id": "authored/model"}
    cases.append((raw, "checkpoint_export: the launch document owns export settings"))
    path = tmp_path / "invalid.yaml"
    for raw, error in cases:
        OmegaConf.save(raw, path)
        with pytest.raises(ValueError, match=error):
            load_launch_config(path)


def test_loaded_launch_revision_selects_downloaded_tokenizer_bytes(tmp_path, monkeypatch):
    revision = "68c46c4b3498877f3ef123c856ecfde50c39f404"
    raw = raw_launch_config()
    raw["inputs"]["model"] = {
        "uri": "org/policy",
        "identity": "a" * 40,
        "local_path": "org/policy",
        "tokenizer_uri": "org/tokenizer",
        "tokenizer_revision": revision,
    }
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path)
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path / "node"))

    def download(model_id, *, revision, destination, allow_patterns):
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, model_id: 1, str(revision): 2}, unk_token="[UNK]"))
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(destination)

    monkeypatch.setattr(task_runtime, "download_hugging_face_snapshot", download)
    tokenizer = task_runtime.prepare_policy_tokenizer(task_runtime._runtime_namespace(config))

    assert json.loads((Path(tokenizer.local_path) / "tokenizer.json").read_text())["model"]["vocab"] == {
        "[UNK]": 0,
        "org/tokenizer": 1,
        revision: 2,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("local_engines,served_name", [(True, "policy"), (False, "remote-policy")])
async def test_declared_serving_alias_survives_composition_and_streamed_cache_requests(
    tmp_path, monkeypatch, local_engines, served_name
):
    raw = raw_launch_config()
    raw["skyrl"]["generator"].update(
        run_engines_locally=local_engines,
        model_loading="stream",
        engine_init_kwargs={"served_model_name": served_name},
    )
    launch_path = tmp_path / "launch.yaml"
    launch_path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(launch_path)
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path / "node"))
    cache_path = str(tmp_path / "streamed-cache" / "weights-123")
    final_path = task_runtime._write_final_config(
        config,
        policy_model=task_runtime.PreparedPolicyModel("s3://models/qwen", "sha256:model", cache_path),
        policy_tokenizer=None,
        draft_model=None,
    )
    final = OmegaConf.load(final_path).skyrl
    assert final.trainer.policy.model.path == cache_path
    submitted_models = []

    class Engine:
        async def tokenize(self, request):
            submitted_models.append(request["json"]["model"])
            return {"tokens": [1]}

        async def chat_completion(self, request):
            submitted_models.append(request["json"]["model"])
            return {
                "prompt_token_ids": [1],
                "choices": [
                    {"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop", "token_ids": [2]}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "answer": 2}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    engines = [Engine()]
    if local_engines:
        allocations = []

        def allocate_engines(**kwargs):
            allocations.append(kwargs)
            return engines

        monkeypatch.setattr(
            "skyrl_train.inference_engines.ray_wrapped_inference_engine.create_ray_wrapped_inference_engines",
            allocate_engines,
        )
        engines = create_ray_wrapped_inference_engines_from_config(final, None)
        [allocation] = allocations
        assert allocation["pretrain"] == "s3://models/qwen"
        assert allocation["engine_init_kwargs"]["load_format"] == "runai_streamer"
        assert allocation["engine_init_kwargs"][MODEL_METADATA_PATH_KEY] == cache_path
    client = DirectModelClient(InferenceEngineClient(engines, tokenizer, final))
    output = await client.generate(
        {"prompts": [[{"role": "user", "content": "hello"}]], "chat_completion_params": [{}]}
    )

    assert submitted_models == [served_name, served_name]
    assert output["response_ids"] == [[2]]


def test_loaded_launch_policy_revision_reaches_trainer_config(tmp_path, monkeypatch):
    revision = "b" * 40
    raw = raw_launch_config()
    raw["inputs"]["model"] = {
        "uri": "org/policy",
        "identity": revision,
        "local_path": "org/policy",
        "tokenizer_uri": "org/tokenizer",
        "tokenizer_revision": "c" * 40,
    }
    raw["runtime"]["task_env"] = {"HF_HUB_OFFLINE": "1"}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path)
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path / "node"))
    trainer_config = task_runtime._write_final_config(
        config, policy_model=None, policy_tokenizer=None, draft_model=None
    )

    assert OmegaConf.load(trainer_config).skyrl.trainer.policy.model.revision == revision


@pytest.mark.parametrize(("colocate_all", "colocate_policy_ref"), [(True, False), (False, True), (False, False)])
def test_resolved_nccl_buffers_override_inherited_small_buffer(
    tmp_path, monkeypatch, colocate_all, colocate_policy_ref
):
    # A stale driver environment must not restore the NVLSTree corruption setting.
    monkeypatch.setenv("NCCL_BUFFSIZE", "262144")
    raw = raw_launch_config()
    raw["skyrl"]["trainer"]["algorithm"]["use_kl_loss"] = True
    raw["skyrl"]["trainer"]["placement"].update(colocate_all=colocate_all, colocate_policy_ref=colocate_policy_ref)
    raw["iris"]["allocation"]["num_nodes"] = 1 if colocate_all else 2 if colocate_policy_ref else 3
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path).skyrl
    actor_environments = {}

    def record_actor_group(cfg, nodes, gpus, worker, **kwargs):
        actor_environments[worker] = kwargs["actor_env_vars"]
        return object()

    # Replace the remote allocation boundary, then run the actual role wiring.
    monkeypatch.setattr(trainer_module, "PPORayActorGroup", record_actor_group)
    trainer = trainer_module.RayPPOTrainer.__new__(trainer_module.RayPPOTrainer)
    trainer.cfg = config
    trainer.colocate_pg = None
    monkeypatch.setattr(trainer, "_initialize_model_actors", lambda *_args: None)
    trainer.build_models("policy", "ref", policy_pg=object())
    assert actor_environments == {
        "policy": {"NCCL_BUFFSIZE": "4194304"},
        "ref": {"NCCL_BUFFSIZE": "4194304"},
    }
    runtime_env = _build_inference_engine_runtime_env(
        nccl_buffer_size_bytes=config.trainer.policy.nccl_buffer_size_bytes
    )
    assert runtime_env["env_vars"]["NCCL_BUFFSIZE"] == "4194304"

    # Keep explicit per-role tuning representable for diagnostic controls.
    raw["skyrl"]["trainer"].setdefault("policy", {})["nccl_buffer_size_bytes"] = 8388608
    path.write_text(yaml.safe_dump(raw))
    resolved = load_launch_config(path).skyrl
    runtime_env = _build_inference_engine_runtime_env(
        nccl_buffer_size_bytes=resolved.trainer.policy.nccl_buffer_size_bytes
    )
    assert runtime_env["env_vars"]["NCCL_BUFFSIZE"] == "8388608"
