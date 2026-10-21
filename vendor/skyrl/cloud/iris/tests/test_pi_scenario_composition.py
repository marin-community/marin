"""The Iris composer selects reviewed shared-worker Pi behavior, not the retired runner."""

from pathlib import Path

import pytest
import yaml

from cloud.iris.launch_config import LaunchTopology
from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config
from cloud.iris.role_plan import ModelRoleKind, RoleExecution, derive_num_nodes, derive_role_plan
from omegaconf import OmegaConf
from skyrl_train.config.pi_training import validate_pi_evaluation_results


BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"


@pytest.mark.parametrize(
    "recipe_name",
    ("pi_teacher_distribution_smoke.yaml", "pi_teacher_trace_smoke.yaml", "pi_verifier_grpo_smoke.yaml"),
)
def test_pi_gpu_recipes_allow_tool_followup_before_rollout_engine_stops(recipe_name):
    # At two turns the engine cancels Pi during a tool follow-up, before it can
    # return validated in-memory capture; a capped trial must fail closed.
    recipe = yaml.safe_load((BASE_CONFIG.parent / recipe_name).read_text())
    assert recipe["context_budget"]["max_turns"] >= 4


def _recipe(tmp_path, preset):
    raw = yaml.safe_load(BASE_CONFIG.read_text())
    raw["scenario"] = {"preset": preset}
    raw["data"]["kind"] = "tasks"
    raw["data"]["train_data"] = ["fixture-task"]
    raw["data"]["terminal_bench_data"] = []
    raw["generator"]["trajectory_retention"] = {"output_path": str(tmp_path / "trials")}
    model = "teacher" if preset.startswith("teacher_demo") else "Qwen/Qwen2.5-1.5B-Instruct"
    raw["terminal_bench"] = {
        "installed_agent": {
            "agent": {"model_name": f"hosted_vllm/{model}", "kwargs": {"thinking_format": "qwen-chat-template"}}
        }
    }
    path = tmp_path / "pi.yaml"
    path.write_text(yaml.safe_dump(raw))
    return raw, path


def _compose(path):
    return compose_skyrl_config(
        parse_rl_config(str(path)), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
    ).config


def test_verifier_preset_selects_pi_grpo_without_manual_objective(tmp_path):
    _, path = _recipe(tmp_path, "verifier_grpo_base")
    cfg = _compose(path)
    assert set(cfg.scenario) == {"preset"}
    assert cfg.scenario.preset == "verifier_grpo_base"
    assert cfg.terminal_bench_config.installed_agent.agent.pi.mode == "literal"
    assert cfg.trainer.algorithm.policy_loss_type == "regular"
    assert cfg.generator.n_samples_per_prompt == 2
    assert cfg.generator.sampling_params.logprobs == 0
    assert cfg.trainer.algorithm.dynamic_sampling.type is None


def test_pi_student_preset_rejects_missing_sampled_vllm_scores(tmp_path):
    raw, path = _recipe(tmp_path, "verifier_grpo_base")
    raw["generator"]["sampling_params"]["logprobs"] = None
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="sampled-token logprobs"):
        _compose(path)


def test_pi_student_evaluation_requires_sampled_vllm_scores(tmp_path):
    raw, path = _recipe(tmp_path, "verifier_grpo_base")
    raw["trainer"].update(eval_interval=1, evaluation_runner="harbor")
    raw["data"]["val_data"] = ["/tmp/pi-eval-tasks"]
    raw["generator"]["eval_sampling_params"] = {"logprobs": 0}
    path.write_text(yaml.safe_dump(raw))
    assert _compose(path).generator.eval_sampling_params.logprobs == 0

    raw["generator"]["eval_sampling_params"]["logprobs"] = None
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="student evaluation requires sampled-token logprobs"):
        _compose(path)


def test_pi_student_evaluation_rejects_missing_verifier_outcomes(tmp_path):
    _, path = _recipe(tmp_path, "verifier_grpo_base")
    cfg = _compose(path)
    with pytest.raises(ValueError, match="score for every attempted trial"):
        validate_pi_evaluation_results(cfg, {"eval/all/num_attempted": 8.0, "eval/all/num_scored": 0.0})
    validate_pi_evaluation_results(cfg, {"eval/all/num_attempted": 8.0, "eval/all/num_scored": 8.0})


def test_teacher_preset_selects_ungraded_sft_and_independent_source(tmp_path):
    raw, path = _recipe(tmp_path, "teacher_demo_pi_base")
    del raw["generator"]["n_samples_per_prompt"]
    del raw["trainer"]["algorithm"]["advantage_estimator"]
    raw["trainer"]["eval_interval"] = -1
    raw["data"]["val_data"] = []
    raw["terminal_bench"]["installed_agent"]["teacher"] = {
        "source": "skyrl_managed",
        "model": "teacher",
        "model_path": "Qwen/independent-teacher",
        "revision": "pinned-revision",
        "student_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_context_tokens": 4096,
        "resources": {"backend": "vllm", "num_engines": 1, "tensor_parallel_size": 1},
    }
    path.write_text(yaml.safe_dump(raw))
    cfg = _compose(path)
    assert cfg.scenario.preset == "teacher_demo_pi_base"
    assert cfg.trainer.algorithm.policy_loss_type == "sft"
    assert cfg.terminal_bench_config.harbor.verifier_disable is True
    assert cfg.trainer.eval_interval == -1
    assert cfg.generator.n_samples_per_prompt == 1
    assert cfg.trainer.algorithm.dynamic_sampling.type is None
    assert cfg.terminal_bench_config.installed_agent.teacher.source == "skyrl_managed"
    plan = derive_role_plan(OmegaConf.to_container(cfg, resolve=True))
    teacher = plan.claim("teacher:pi_trace")
    assert teacher.kind is ModelRoleKind.TEACHER
    assert teacher.execution is RoleExecution.LOCAL
    assert teacher.colocation_group == "teacher:pi_trace"
    assert teacher.num_nodes == 1
    assert derive_num_nodes(plan) == 2


def test_distribution_preset_uses_shared_topk_scoring_and_teacher_role(tmp_path):
    raw, path = _recipe(tmp_path, "teacher_distribution_pi_base")
    del raw["generator"]["n_samples_per_prompt"]
    del raw["trainer"]["algorithm"]["advantage_estimator"]
    raw["teachers"] = {
        "independent": {
            "source": "local_inference",
            "placement": "pinned",
            "model": {"path": "Qwen/independent-teacher", "revision": "pinned-revision"},
            "backend": "vllm",
            "evidence": "topk_distribution",
            "top_k": 2,
            "resources": {
                "num_nodes": 1,
                "gpus_per_node": 8,
                "tensor_parallel_size": 8,
                "data_parallel_size": 1,
                "expert_parallel_size": 1,
                "colocation_group": "teacher:independent",
            },
        }
    }
    raw["teacher_routing"] = {
        "pi": {"revision": "r1", "routes": {"default": {"teacher": "independent", "weight": 1.0}}}
    }
    raw["trainer"]["algorithm"]["distillation"] = {
        "objective": "sparse_forward_kl",
        "reward_mode": "replace",
        "routing_plan": "pi",
        "coefficient": 1.0,
    }
    path.write_text(yaml.safe_dump(raw))

    cfg = _compose(path)

    assert cfg.scenario.preset == "teacher_distribution_pi_base"
    assert cfg.terminal_bench_config.installed_agent.agent.pi.mode == "literal"
    assert cfg.terminal_bench_config.installed_agent.get("teacher") is None
    assert cfg.trainer.algorithm.distillation.objective == "sparse_forward_kl"
    assert cfg.trainer.algorithm.distillation.reward_mode == "replace"
    assert cfg.generator.sampling_params.logprobs == 0
    assert cfg.generator.n_samples_per_prompt == 1
    assert cfg.trainer.step_wise_training is True
    plan = derive_role_plan(OmegaConf.to_container(cfg, resolve=True))
    assert plan.claim("teacher:independent").execution is RoleExecution.LOCAL
    assert derive_num_nodes(plan) == 2


def test_pi_preset_rejects_missing_served_thinking_format(tmp_path):
    raw, path = _recipe(tmp_path, "verifier_grpo_base")
    del raw["terminal_bench"]["installed_agent"]["agent"]["kwargs"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="thinking_format"):
        _compose(path)


def test_external_teacher_uses_remote_role_without_extra_gpu_node(tmp_path):
    raw, path = _recipe(tmp_path, "teacher_demo_pi_verified_base")
    del raw["generator"]["n_samples_per_prompt"]
    del raw["trainer"]["algorithm"]["advantage_estimator"]
    raw["trainer"]["eval_interval"] = -1
    raw["data"]["val_data"] = []
    raw["terminal_bench"]["installed_agent"]["teacher"] = {
        "source": "external_vllm",
        "model": "teacher",
        "model_path": "Qwen/independent-teacher",
        "revision": "pinned-revision",
        "student_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_context_tokens": 4096,
        "endpoint": "https://teacher.example/v1",
        "auth": "env:PI_TEACHER_TOKEN",
    }
    path.write_text(yaml.safe_dump(raw))
    cfg = _compose(path)
    plan = derive_role_plan(OmegaConf.to_container(cfg, resolve=True))
    assert cfg.trainer.algorithm.dynamic_sampling.type is None
    assert plan.claim("teacher:pi_trace").execution is RoleExecution.REMOTE
    assert derive_num_nodes(plan) == 1


@pytest.mark.parametrize(
    "scenario",
    [
        {"preset": "unreviewed"},
        {"kind": "verifier_grpo"},
        {"preset": "verifier_grpo_base", "plugin_path": "arbitrary"},
    ],
)
def test_unreviewed_or_ambiguous_scenario_fails_before_composition(tmp_path, scenario):
    raw, path = _recipe(tmp_path, "verifier_grpo_base")
    raw["scenario"] = scenario
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="scenario|Scenario"):
        parse_rl_config(str(path))


def test_scenario_conflicting_with_algorithm_recipe_fails(tmp_path):
    raw, path = _recipe(tmp_path, "verifier_grpo_base")
    raw["config_groups"] = {"algorithm_recipe": "grpo"}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="do not combine"):
        parse_rl_config(str(path))
