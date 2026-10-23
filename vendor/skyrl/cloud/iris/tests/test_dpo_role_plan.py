import pytest

from cloud.iris.role_plan import ModelRoleKind, RoleExecution, derive_num_nodes, derive_role_plan


def skyrl_config(**overrides) -> dict:
    config = {
        "trainer": {
            "strategy": "megatron",
            "evaluation_runner": "training",
            "placement": {
                "colocate_all": False,
                "colocate_policy_ref": True,
                "policy_num_nodes": 1,
                "policy_num_gpus_per_node": 8,
            },
            "critic": {"model": {"path": None}},
            "algorithm": {
                "use_kl_loss": False,
                "use_kl_in_reward": False,
                "policy_loss_type": "dpo",
            },
            "train_batch_size": 64,
            "policy_mini_batch_size": 64,
            "micro_train_batch_size_per_gpu": 8,
        },
        "generator": {
            "backend": "vllm",
            "run_engines_locally": True,
            "num_inference_engines": 8,
            "inference_engine_tensor_parallel_size": 1,
            "n_samples_per_prompt": 2,
        },
        "environment": {"env_class": "preference_pair"},
    }
    for key, value in overrides.items():
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return config


def test_preference_pair_launch_provisions_no_rollout_role():
    plan = derive_role_plan(skyrl_config())
    kinds = [claim.kind for claim in plan.claims]
    assert ModelRoleKind.ROLLOUT not in kinds
    assert sorted(kind.value for kind in kinds) == ["policy", "reference"]
    assert derive_num_nodes(plan) == 1
    assert not plan.colocate_all
    roles = {bundle.name: bundle.role_ids for bundle in plan.bundles}
    assert set(roles) == {"policy"}


def test_generation_launch_keeps_its_rollout_role():
    plan = derive_role_plan(
        skyrl_config(**{"environment.env_class": "gsm8k", "trainer.algorithm.policy_loss_type": "regular"})
    )
    rollout = plan.claim(ModelRoleKind.ROLLOUT)
    assert rollout.execution is RoleExecution.LOCAL
    assert derive_num_nodes(plan) == 2


@pytest.mark.parametrize("runner", ["task", "harbor"])
def test_preference_pair_task_validation_reserves_generation_capacity(runner):
    plan = derive_role_plan(skyrl_config(**{"trainer.evaluation_runner": runner}))
    assert sorted(claim.kind.value for claim in plan.claims) == ["policy", "reference", "rollout"]
    assert derive_num_nodes(plan) == 2
    assert plan.claim(ModelRoleKind.ROLLOUT).execution is RoleExecution.LOCAL


def test_repair_teacher_adds_separate_capacity_without_changing_student_roles():
    config = skyrl_config(**{
        "environment.env_class": "prompt_only",
        "trainer.algorithm.policy_loss_type": "mixed_policy_grpo",
        "generator.n_samples_per_prompt": 4,
    })
    original = derive_role_plan(config)
    config["identity_repair"] = {"teacher": {
        "model_path": "fixture/teacher", "revision": "pinned",
        "engines": 2, "tensor_parallel_size": 4, "data_parallel_size": 1,
        "expert_parallel_size": 4, "max_model_len": 8192, "gpu_memory_utilization": 0.8,
    }}
    plan = derive_role_plan(config)
    assert derive_num_nodes(plan) == derive_num_nodes(original) + 1
    assert tuple(claim for claim in plan.claims if claim.role_id != "repair_teacher") == original.claims
    teacher = plan.claim("repair_teacher")
    assert teacher.kind is ModelRoleKind.TEACHER
    assert (teacher.replicas, teacher.tensor_parallel_size, teacher.num_nodes) == (2, 4, 1)
    assert next(bundle for bundle in plan.bundles if bundle.name == "repair_teacher").role_ids == ("repair_teacher",)
    assert all("repair_teacher" not in bundle.role_ids for bundle in original.bundles)
