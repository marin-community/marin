from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import ray
import torch
from megatron.core import parallel_state as mpu

from skyrl_train.distributed.megatron.megatron_utils import get_model_config
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.utils import prepare_runtime_environment, sync_registries, validate_cfg
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl_train.workers.worker import PPORayActorGroup
from tests.gpu.grug_exact_checkpoint import (
    exact_numerics_config,
    exact_training_batch,
    policy_logprobs,
    write_snowball_checkpoint,
)
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.utils import init_worker_with_type

POLICY_WORLD_SIZE = 2
# One sequence per micro-batch, from under half the 2,048-token sliding window to over twice it.
BODY_LENGTHS = (900, 1500, 2200, 2200, 3000, 4300)
RESPONSE_LENGTH = 256
ENGINE_RANKS = 2


class GradientProbeWorkerBase(MegatronPolicyWorkerBase):
    def set_recompute(self, recompute: bool) -> None:
        """Full activation recompute in the next ``ppo_train`` when ``recompute``, none otherwise."""
        get_model_config(self.actor_module[0]).recompute_granularity = "full" if recompute else None

    def save_gradients(self, output_dir: str) -> None:
        """Save this pipeline rank's main gradients under ``output_dir``. They hold the last ``ppo_train`` mini-batch's
        gradient until the next mini-batch zeroes them."""
        gradients = {
            name: param.main_grad.detach().clone().cpu()
            for chunk in self.actor_module
            for name, param in chunk.named_parameters()
            if param.requires_grad
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(gradients, Path(output_dir) / f"pp{mpu.get_pipeline_model_parallel_rank()}.pt")


GradientProbeWorker = ray.remote(num_gpus=1)(GradientProbeWorkerBase)


def _config(model_path: str):
    cfg = exact_numerics_config(
        model_path, policy_gpus=POLICY_WORLD_SIZE, pipeline_stages=2, expert_parallel=1, engine_ranks=ENGINE_RANKS
    )
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.train_batch_size = len(BODY_LENGTHS)
    cfg.trainer.policy_mini_batch_size = len(BODY_LENGTHS)
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    # The optimizer step leaves the weights unchanged, so both arms differentiate the same weights, and no clipping
    # scales the gradients.
    cfg.trainer.policy.optimizer_config.lr = 0.0
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.0
    validate_cfg(cfg)
    return cfg


def _batch(vocab_size: int) -> TrainingInputBatch:
    """One row per body length, served alternately by the two engine ranks."""
    generator = torch.Generator().manual_seed(5)
    rows = [torch.randint(10, vocab_size, (length,), generator=generator).tolist() for length in BODY_LENGTHS]
    batch = exact_training_batch(rows, [row % ENGINE_RANKS for row in range(len(rows))], RESPONSE_LENGTH)
    # Positive advantages, so the tokens' gradients add up instead of cancelling.
    batch["advantages"] = torch.linspace(0.5, 1.5, RESPONSE_LENGTH).unsqueeze(0).repeat(len(rows), 1)
    return batch


def _init_ray(cfg) -> None:
    env_vars = prepare_runtime_environment(cfg)
    # Transformer Engine then picks deterministic attention backward algorithms, so two runs can agree bit for bit.
    env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    ray.init(
        runtime_env={
            "env_vars": env_vars,
            "worker_process_setup_hook": "skyrl_train.worker_setup.configure_worker_process",
        }
    )
    sync_registries()


@contextmanager
def _policy(cfg) -> Iterator[PPORayActorGroup]:
    """The gradient probe policy of ``cfg`` on its own Ray instance, shut down on exit."""
    _init_ray(cfg)
    try:
        yield init_worker_with_type(
            "policy", num_gpus_per_node=POLICY_WORLD_SIZE, cfg=cfg, ray_actor_type=GradientProbeWorker
        )
    finally:
        ray.shutdown()


def _gradients(policy: PPORayActorGroup, batch: TrainingInputBatch, output_dir: Path, *, recompute: bool) -> None:
    """Train ``policy`` on the mini-batch through ``ppo_train`` and save its gradients in ``output_dir``."""
    ray.get(policy.async_run_ray_method("pass_through", "set_recompute", recompute))
    ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))
    ray.get(policy.async_run_ray_method("pass_through", "save_gradients", str(output_dir)))


def _load_gradients(output_dir: Path) -> dict[str, torch.Tensor]:
    gradients = {}
    for rank in range(POLICY_WORLD_SIZE):
        for name, grad in torch.load(output_dir / f"pp{rank}.pt", map_location="cpu", mmap=True).items():
            gradients[f"pp{rank}/{name}"] = grad
    return gradients


def test_exact_gradients_equal_with_and_without_recompute(tmp_path):
    """Under exact numerics on two pipeline stages, a ``ppo_train`` mini-batch gives the same gradients bit for bit
    with full activation recompute in one-layer units and without recompute."""
    require_hoppers(POLICY_WORLD_SIZE)
    model_path = tmp_path / "model"
    model_path.mkdir()
    # Zero router weights send every token to the four experts of highest bias.
    write_snowball_checkpoint(model_path, router_weight_scale=0.0)
    batch = _batch(vocab_size=1000)
    with _policy(_config(str(model_path))) as policy:
        batch["action_log_probs"] = policy_logprobs(policy, batch) * batch["response_mask"]
        _gradients(policy, batch, tmp_path / "a", recompute=True)
        _gradients(policy, batch, tmp_path / "b", recompute=False)

    a, b = (_load_gradients(tmp_path / arm) for arm in ("a", "b"))
    assert a.keys() == b.keys()
    # Main gradients the optimizer step had cleared would pass every comparison below.
    assert any(gradient.any() for gradient in a.values())
    recompute_differs = [name for name in a if not torch.equal(a[name], b[name])]
    assert not recompute_differs, f"{len(recompute_differs)} of {len(a)} gradients differ: {recompute_differs}"
