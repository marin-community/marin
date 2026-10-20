"""Behavior tests for the structured SkyRL launch configuration."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf
from rolloutengine.models import ModelTurn
from omegaconf.errors import MissingMandatoryValue
from skyrl_gym.answer_tasks import grade_gsm8k
from skyrl_gym.task_sessions import AnswerTaskSession

from cloud.iris import training_driver
from cloud.iris.launch_config import LaunchTopology, load_launch_config, validate_launch_config
from cloud.iris.rl_config_translation import (
    RL_CONFIG_PAYLOAD_ENV,
    compose_skyrl_config,
    materialize_launch_config,
    parse_rl_config,
)
from skyrl_train.distributed.step_policy import NonfiniteStepPolicy, nonfinite_step_policy
from skyrl_train.dataset.tasks import source_row_task
from marinskyrl.runtime_options import reference_model_required


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
                "strategy": "megatron",
                "algorithm": {"use_kl_loss": False},
                "placement": {
                    "colocate_all": True,
                    "policy_num_nodes": 1,
                    "ref_num_nodes": 1,
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
    raw["skyrl"]["trainer"].update(ckpt_interval=1, hf_save_interval=-1)
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


@pytest.mark.parametrize("geometry", ["colocated", "two-node", "native-defaults", "shared-ref", "separate-ref", "dpo"])
def test_source_launch_fills_allocation_without_document_changes(tmp_path, geometry):
    raw = _raw_config()
    placement = raw["skyrl"]["trainer"]["placement"]
    if geometry != "colocated":
        placement["colocate_all"] = False
        raw["iris"]["allocation"]["num_nodes"] = 2
    if geometry in {"shared-ref", "separate-ref"}:
        raw["skyrl"]["trainer"]["algorithm"]["use_kl_loss"] = True
        placement["colocate_policy_ref"] = geometry == "shared-ref"
        raw["iris"]["allocation"]["num_nodes"] = 2 if geometry == "shared-ref" else 3
    if geometry == "native-defaults":
        placement.update(policy_num_gpus_per_node=None, colocate_policy_ref=False)
        del raw["skyrl"]["trainer"]["micro_train_batch_size_per_gpu"]
        raw["skyrl"]["trainer"]["algorithm"] = {}
        del raw["skyrl"]["generator"]["backend"]
        raw["iris"]["allocation"]["num_nodes"] = 3
    if geometry == "dpo":
        raw["skyrl"].update(
            config_groups={"algorithm_recipe": "dpo"},
            environment={"env_class": "preference_pair"},
            trainer={
                "strategy": "megatron",
                "use_sample_packing": False,
                "micro_train_batch_size_per_gpu": 2,
                "placement": {**placement, "colocate_policy_ref": True},
            },
        )
        raw["skyrl"]["generator"]["n_samples_per_prompt"] = 2
        raw["iris"]["allocation"]["num_nodes"] = 1
    if geometry == "colocated":
        raw["skyrl"]["generator"]["engine_init_kwargs"] = {"user_options": {"nested": {}}}
        raw["skyrl"]["data"] = {"kind": "parquet"}
        raw["inputs"]["data_kind"] = "parquet"
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    explicit = load_launch_config(path)
    if geometry == "colocated":
        assert OmegaConf.to_container(explicit.skyrl.generator.engine_init_kwargs.user_options) == {}
        OmegaConf.save(explicit, path)
        assert load_launch_config(path).inputs.data_kind == "parquet"
    del raw["iris"]["allocation"]["num_nodes"]
    del raw["runtime"]["profile"]
    del raw["inputs"]["data_kind"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert json.dumps(OmegaConf.to_container(load_launch_config(path), resolve=True)) == json.dumps(
        OmegaConf.to_container(explicit, resolve=True)
    )


def test_source_launch_rejects_missing_geometry_and_conflicting_author_values(tmp_path):
    cases = []
    for field in (
        "trainer.placement.policy_num_nodes",
        "trainer.placement.ref_num_nodes",
        "generator.num_inference_engines",
    ):
        for explicit_null in (False, True):
            raw = OmegaConf.create(_raw_config())
            del raw.iris.allocation.num_nodes
            parent, key = field.rsplit(".", 1)
            if explicit_null:
                OmegaConf.update(raw.skyrl, field, None)
            else:
                del OmegaConf.select(raw.skyrl, parent)[key]
            cases.append((raw, field))
    for recipe_key, envelope_key in (("train_data", "train_data"), ("val_data", "validation_data")):
        raw = OmegaConf.create(_raw_config())
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
        raw = OmegaConf.create(_raw_config())
        raw.skyrl.trainer.placement[f"{role}_num_gpus_per_node"] = 32
        cases.append((raw, "gpus_per_node"))
    raw = OmegaConf.create(_raw_config())
    raw.inputs.data_kind = "parquet"
    cases.append((raw, "inputs.data_kind"))
    raw = OmegaConf.create(_raw_config())
    raw.skyrl.trainer.train_batch_size = "8"
    cases.append((raw, "train_batch_size"))
    path = tmp_path / "invalid.yaml"
    for raw, error in cases:
        OmegaConf.save(raw, path)
        with pytest.raises(ValueError, match=error):
            load_launch_config(path)
    raw = OmegaConf.create(_raw_config())
    raw.skyrl.data = {"kind": "parquet"}
    del raw.inputs.data_kind
    OmegaConf.save(raw, path)
    resolved = load_launch_config(path)
    del resolved.inputs.data_kind
    OmegaConf.save(resolved, path)
    with pytest.raises(MissingMandatoryValue, match="data_kind"):
        load_launch_config(path)


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
            "kind",
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
    raw["inputs"]["data_kind"] = "parquet"
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
