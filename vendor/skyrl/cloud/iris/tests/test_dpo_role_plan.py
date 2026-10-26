import pytest

from cloud.iris.role_plan import ModelRoleKind, RoleExecution, derive_num_nodes, derive_role_plan
from cloud.iris.launch_config import validate_iris_allocation


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
    plan = derive_role_plan(skyrl_config(), generate_only=False)
    kinds = [claim.kind for claim in plan.claims]
    assert ModelRoleKind.ROLLOUT not in kinds
    assert sorted(kind.value for kind in kinds) == ["policy", "reference"]
    assert derive_num_nodes(plan) == 1
    assert not plan.colocate_all
    roles = {bundle.name: bundle.role_ids for bundle in plan.bundles}
    assert set(roles) == {"policy"}


def test_generation_launch_keeps_its_rollout_role():
    plan = derive_role_plan(
        skyrl_config(**{"environment.env_class": "gsm8k", "trainer.algorithm.policy_loss_type": "regular"}),
        generate_only=False,
    )
    rollout = plan.claim(ModelRoleKind.ROLLOUT)
    assert rollout.execution is RoleExecution.LOCAL
    assert derive_num_nodes(plan) == 2


@pytest.mark.parametrize("colocate_all", [False, True])
@pytest.mark.parametrize("run_engines_locally", [False, True])
def test_generate_launch_reserves_local_resources(colocate_all, run_engines_locally):
    config = skyrl_config(
        **{
            "trainer.placement.colocate_all": colocate_all,
            "trainer.placement.policy_num_nodes": 4,
            "trainer.algorithm.use_kl_loss": True,
            "environment.env_class": "gsm8k",
            "generator.num_inference_engines": 1,
            "generator.run_engines_locally": run_engines_locally,
        }
    )
    config["generator"].update(inference_engine_data_parallel_size=8, inference_engine_expert_parallel_size=8)
    plan = derive_role_plan(config, generate_only=True)
    expected_roles = (
        [ModelRoleKind.ROLLOUT]
        if run_engines_locally
        else [ModelRoleKind.POLICY, ModelRoleKind.REFERENCE, ModelRoleKind.ROLLOUT]
    )
    assert [claim.kind for claim in plan.claims] == expected_roles
    expected_nodes = 1 if run_engines_locally else 4
    assert derive_num_nodes(plan) == expected_nodes

    allocation = validate_iris_allocation(
        {
            "skyrl": config,
            "run": {"mode": "train"},
            "runtime": {"entrypoint": "skyrl_train.entrypoints.main_generate"},
            "iris": {
                "allocation": {
                    "num_nodes": expected_nodes,
                    "gpus_per_node": 8,
                    "gpu_variant": "H100",
                    "cpu": 48,
                    "memory": "700GB",
                    "disk": "4TB",
                }
            },
        }
    )
    assert (allocation.num_nodes, allocation.gpus_per_node) == (expected_nodes, 8)


@pytest.mark.parametrize("runner", ["task", "harbor"])
def test_preference_pair_task_validation_reserves_generation_capacity(runner):
    plan = derive_role_plan(skyrl_config(**{"trainer.evaluation_runner": runner}), generate_only=False)
    assert sorted(claim.kind.value for claim in plan.claims) == ["policy", "reference", "rollout"]
    assert derive_num_nodes(plan) == 2
    assert plan.claim(ModelRoleKind.ROLLOUT).execution is RoleExecution.LOCAL
