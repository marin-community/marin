"""Strategy validation keeps native mesh geometry separate from Megatron DP."""

from omegaconf import OmegaConf

from skyrlrecipe import RecipePatch, sections

from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


def levanter_config():
    config = get_default_config()
    config.trainer.strategy = "levanter"
    config.trainer.logger = "console"
    config.trainer.placement.colocate_all = False
    config.trainer.placement.policy_num_nodes = 4
    config.trainer.placement.policy_num_gpus_per_node = 8
    config.trainer.policy.levanter_config.mesh_axes.data = 4
    config.trainer.policy.levanter_config.mesh_axes.expert = 8
    config.trainer.policy.levanter_config.python_executable = "/tmp/native/bin/python"
    config.trainer.algorithm.use_kl_loss = False
    config.trainer.algorithm.use_kl_in_reward = False
    config.generator.sampling_params.temperature = 1.0
    config.trainer.policy.optimizer_config.num_warmup_steps = 0
    config.trainer.train_batch_size = 128
    config.trainer.policy_mini_batch_size = 128
    config.generator.n_samples_per_prompt = 4
    return config


def test_validates_full_config_with_global_gpu_mesh_and_four_host_data_axis():
    config = levanter_config()
    # These unused Megatron dimensions must not determine native DP geometry.
    config.trainer.policy.megatron_config.tensor_model_parallel_size = 3
    # Native forward scoring and update execution can use independent slices.
    config.trainer.micro_forward_batch_size_per_gpu = 3
    validate_cfg(config)
    assert config.trainer.policy.levanter_config.mesh_axes.replica_dcn == 1
    assert config.trainer.algorithm.resolved_group_advantage is not None


def test_authored_native_recipe_round_trip_preserves_runtime_precision_and_mesh():
    authored = RecipePatch(
        trainer=sections.Trainer(
            strategy="levanter",
            policy=sections.Policy(
                levanter_config=sections.PolicyLevanterConfig(
                    python_executable="/opt/native/bin/python",
                    projection_logits_dtype="bfloat16",
                    allocator="vmm",
                    numpy_madvise_hugepage=False,
                    preallocate=False,
                    validate_checkpoint_roundtrip=True,
                    deterministic_gpu_ops=True,
                    head_gate_fp32=True,
                    xsa_fp32=True,
                    fp32_activations=False,
                    mesh_axes=sections.LevanterMeshAxes(data=4, expert=8),
                )
            ),
        )
    )
    restored = RecipePatch.from_document(authored.to_skyrl())
    config = OmegaConf.merge(levanter_config(), restored.to_skyrl())
    validate_cfg(config)
    runtime = config.trainer.policy.levanter_config
    assert OmegaConf.to_container(runtime.mesh_axes) == {
        "replica_dcn": 1,
        "replica": 1,
        "data": 4,
        "expert": 8,
        "model": 1,
        "context": 1,
    }
    assert runtime.projection_logits_dtype == "bfloat16"
    assert runtime.validate_checkpoint_roundtrip
    assert runtime.head_gate_fp32 and runtime.xsa_fp32
    assert not runtime.fp32_activations
    assert runtime.python_executable == "/opt/native/bin/python"
    assert restored.to_skyrl() == authored.to_skyrl()


def test_existing_megatron_recipe_still_validates_without_native_runtime_environment():
    config = get_default_config()
    config.trainer.logger = "console"
    validate_cfg(config)
    assert config.trainer.strategy == "megatron"
    assert config.trainer.policy.levanter_config.python_executable is None
