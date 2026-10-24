from hydra import compose, initialize_config_dir
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import validate_megatron_cfg
from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config
from cloud.iris.tests.test_nemotron_ultra_grading_config import _HPCStub, _skipped_grading_config
from skyrl_train.objective.losses import PolicyLossInputs
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.config.objective_spec import topk_loss_params
from skyrl_train.distillation import StudentTopKInput, TeacherTopKInput
from skyrl_train.objective.teacher import topk_teacher_loss
from cloud.iris.launch_config import load_launch_config
from cloud.iris.tests.test_launch_config import _raw_config


def replace_mode_config() -> DictConfig:
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(
        cfg,
        {
            "trainer": {
                "algorithm": {
                    "distillation": {
                        "objective": "sampled_reverse_kl",
                        "routing_plan": "opd",
                        "coefficient": 1.0,
                        "reward_mode": "replace",
                    }
                }
            },
            "teachers": {
                "primary": {
                    "source": "openai_compatible",
                    "placement": "external",
                    "model": {"path": "Qwen/teacher", "revision": "teacher-revision"},
                    "endpoints": [{"url": "https://teacher.example/v1", "max_concurrency": 8}],
                    "tokenizer_fingerprint": f"sha256:{'a' * 64}",
                    "max_sequence_length": 32768,
                    "request_timeout_seconds": 120,
                    "evidence": "chosen_token",
                }
            },
            "teacher_routing": {
                "opd": {
                    "revision": "route-revision",
                    "routes": {"default": {"teacher": "primary", "weight": 1.0}},
                }
            },
        },
    )
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    return cfg


def test_packaged_entrypoints_accept_distillation_only_replace_mode():
    cfg = replace_mode_config()
    validate_cfg(cfg)

    current = torch.tensor([[-1.0]], requires_grad=True)
    teacher_advantage = torch.tensor([[0.5]])
    values = PolicyLossRegistry.get(cfg.trainer.algorithm.policy_loss_type)(
        PolicyLossInputs(current, current.detach(), None, teacher_advantage, torch.ones_like(current)),
        cfg.trainer.algorithm,
    ).values
    values.sum().backward()
    torch.testing.assert_close(values, torch.tensor([[-0.5]]))
    torch.testing.assert_close(current.grad, torch.tensor([[-0.5]]))


def test_sampled_teacher_add_rejects_sft_that_ignores_teacher_credit():
    cfg = replace_mode_config()
    cfg.trainer.algorithm.policy_loss_type = "sft"
    cfg.trainer.algorithm.distillation.reward_mode = "add"

    with pytest.raises(ValueError, match="sampled_reverse_kl requires a policy loss that consumes advantages"):
        validate_cfg(cfg)


def skipped_grading_config() -> DictConfig:
    cfg = replace_mode_config()
    cfg.environment.task_sessions.nemotron_ultra.grading = "skip"
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    cfg.trainer.eval_before_train = False
    cfg.trainer.eval_interval = -1
    return cfg


def test_skipped_grading_is_accepted_for_pure_distillation(tmp_path):
    parsed = parse_rl_config(str(_skipped_grading_config(tmp_path, eval_interval=-1)))
    cfg = compose_skyrl_config(parsed, {"job_name": "grading-test", "num_nodes": 1}, _HPCStub()).config
    cfg.trainer.flash_attn = False
    # Successful validation is the contract for training with teacher credit and no environment grading.
    validate_cfg(cfg)
    validate_megatron_cfg(cfg)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("trainer.algorithm.distillation.reward_mode", "add", "reward_mode=replace"),
        ("trainer.algorithm.advantage_estimator", "grpo", "advantage_estimator=uniform"),
        ("trainer.algorithm.dynamic_sampling.type", "filter", "dynamic_sampling.type=null"),
        ("trainer.eval_interval", 5, "eval_interval<=0"),
        ("trainer.eval_before_train", True, "eval_before_train=false"),
    ],
)
def test_skipped_grading_rejects_configs_that_read_the_reward(path, value, message):
    cfg = skipped_grading_config()
    OmegaConf.update(cfg, path, value)

    with pytest.raises(ValueError, match=message):
        validate_cfg(cfg)


def selected_topk_config() -> DictConfig:
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(
        cfg,
        {
            "trainer": {
                "algorithm": {
                    "distillation": {
                        "objective": "student_topk_policy_surrogate",
                        "routing_plan": "opd",
                        "coefficient": 1.0,
                        "reward_mode": "replace",
                    }
                }
            },
            "teachers": {
                "primary": {
                    "source": "local_inference",
                    "placement": "pinned",
                    "model": {"path": "Qwen/teacher", "revision": "teacher-revision"},
                    "backend": "vllm",
                    "evidence": "student_selected_topk",
                    "top_k": 16,
                    "resources": {
                        "num_nodes": 1,
                        "gpus_per_node": 1,
                        "tensor_parallel_size": 1,
                        "colocation_group": "teacher",
                    },
                }
            },
            "teacher_routing": {
                "opd": {"revision": "route-revision", "routes": {"default": {"teacher": "primary", "weight": 1.0}}}
            },
        },
    )
    cfg.trainer.logger = "console"
    cfg.generator.sampling_params.logprobs = 16
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    cfg.trainer.use_sample_packing = False
    return cfg


def test_selected_topk_rollouts_require_matching_teacher_width():
    cfg = selected_topk_config()

    validate_cfg(cfg)

    transported = OmegaConf.create(OmegaConf.to_yaml(cfg))
    current = torch.full((1, 1, 16), 1 / 32).log().requires_grad_()
    evidence = StudentTopKInput(
        torch.arange(16).reshape(1, 1, 16),
        current.detach(),
        torch.full_like(current, 1 / 16).log(),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1),
    )
    result = topk_teacher_loss(
        evidence,
        current,
        topk_loss_params(transported.trainer.algorithm),
        vocabulary_size=32,
    )
    result.values.sum().backward()
    expected = -torch.tensor(2.0).log()
    torch.testing.assert_close(result.values, expected.reshape(1, 1))
    torch.testing.assert_close(current.grad, torch.full_like(current, expected / 16))

    cfg.generator.sampling_params.logprobs = 8
    with pytest.raises(ValueError, match="matching teacher top_k"):
        validate_cfg(cfg)


def test_teacher_only_topk_training_accepts_pbs_with_sequence_loss():
    cfg = selected_topk_config()
    cfg.generator.sampling_params.logprobs = 0
    cfg.trainer.algorithm.distillation.objective = "sparse_forward_kl"
    cfg.teachers.primary.evidence = "topk_distribution"
    cfg.trainer.algorithm.policy_loss_type = "gspo"
    cfg.trainer.algorithm.loss_reduction = "sequence_mean"
    cfg.trainer.algorithm.enable_token_reward_channel = True
    OmegaConf.update(
        cfg,
        "terminal_bench_config.harbor",
        {"enable_token_reward_channel": True, "enable_pbs_shaping": True},
        force_add=True,
    )
    validate_cfg(cfg)

    transported = OmegaConf.create(OmegaConf.to_yaml(cfg))
    current = torch.full((1, 1, 16), 1 / 32).log().requires_grad_()
    evidence = TeacherTopKInput(
        torch.arange(16).reshape(1, 1, 16),
        torch.full_like(current, 1 / 16).log(),
        torch.ones(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1),
    )
    result = topk_teacher_loss(
        evidence,
        current,
        topk_loss_params(transported.trainer.algorithm),
        vocabulary_size=32,
    )
    result.values.sum().backward()
    torch.testing.assert_close(result.values, torch.tensor(2.0).log().reshape(1, 1))
    torch.testing.assert_close(current.grad, torch.full_like(current, -1 / 16))


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("policy.megatron_config.tensor_model_parallel_size", 2, "top-K teacher objectives require"),
        ("algorithm.policy_loss_type", "kl_cov", "requires an advantage-linear policy loss"),
        ("algorithm.loss_reduction", "seq_mean_token_sum_norm_global", "top-K teacher rows require"),
    ],
)
def test_teacher_objective_is_validated_before_launch(tmp_path, key, value, error):
    teacher = selected_topk_config()
    raw = OmegaConf.create(_raw_config())
    raw.skyrl.teachers = teacher.teachers
    raw.skyrl.teacher_routing = teacher.teacher_routing
    raw.skyrl.trainer.algorithm.distillation = teacher.trainer.algorithm.distillation
    raw.skyrl.trainer.algorithm.advantage_estimator = teacher.trainer.algorithm.advantage_estimator
    raw.skyrl.trainer.use_sample_packing = False
    raw.skyrl.generator.sampling_params = {"logprobs": 16}
    OmegaConf.update(raw.skyrl.trainer, key, value, force_add=True)
    path = tmp_path / "launch.yaml"
    OmegaConf.save(raw, path)

    with pytest.raises(ValueError, match=error):
        load_launch_config(path)
