from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import ray
import torch
from megatron.core import parallel_state as mpu
from ray.util.placement_group import placement_group

from skyrl_train.config.numerics import NumericsMode
from skyrl_train.distributed.megatron.megatron_utils import get_model_config
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import ROLLOUT_ENGINE_DP_RANKS_KEY
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.utils.utils import prepare_runtime_environment, sync_registries, validate_cfg
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl_train.workers.worker import PPORayActorGroup
from tests.gpu.grug_exact_checkpoint import exact_numerics_config, policy_logprobs, write_snowball_checkpoint
from tests.gpu.grug_gpu_gates import require_hoppers

POLICY_WORLD_SIZE = 2
PAD_TOKEN_ID = 0
# One sequence per micro-batch, from under half the 2,048-token sliding window to over twice it.
BODY_LENGTHS = (900, 1500, 2200, 2200, 3000, 4300)
RESPONSE_LENGTH = 256
ENGINE_RANKS = 2
# With every token's experts fixed, exact and native gradients differ by their bf16 roundings alone; a gradient that is
# missing or computed from the wrong value differs by about its own size.
NATIVE_RELATIVE_TOLERANCE = 0.5


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


def _config(model_path: str, mode: NumericsMode):
    cfg = exact_numerics_config(
        model_path, policy_gpus=POLICY_WORLD_SIZE, pipeline_stages=2, expert_parallel=1, engine_ranks=ENGINE_RANKS
    )
    cfg.trainer.numerics.mode = mode.value
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.train_batch_size = len(BODY_LENGTHS)
    cfg.trainer.policy_mini_batch_size = len(BODY_LENGTHS)
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    # The optimizer step leaves the weights unchanged, so every arm differentiates the same weights, and no clipping
    # scales the gradients.
    cfg.trainer.policy.optimizer_config.lr = 0.0
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.0
    validate_cfg(cfg)
    return cfg


def _batch(pad_token_id: int, vocab_size: int) -> TrainingInputBatch:
    """One left-padded row per body length, served alternately by the two engine ranks."""
    generator = torch.Generator().manual_seed(5)
    total_length = max(BODY_LENGTHS)
    rows, masks = [], []
    for body_length in BODY_LENGTHS:
        body = torch.randint(10, vocab_size, (body_length,), generator=generator).tolist()
        before = total_length - body_length
        rows.append([pad_token_id] * before + body)
        masks.append([0] * before + [1] * body_length)
    sequences = torch.tensor(rows, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.long)
    response_mask = attention_mask[:, -RESPONSE_LENGTH:]
    zeros = torch.zeros(len(BODY_LENGTHS), RESPONSE_LENGTH, dtype=torch.float32)
    # Positive advantages, so the tokens' gradients add up instead of cancelling.
    advantages = torch.linspace(0.5, 1.5, RESPONSE_LENGTH).unsqueeze(0).repeat(len(BODY_LENGTHS), 1)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "rollout_logprobs": zeros.clone(),
            "values": zeros.clone(),
            "returns": zeros.clone(),
            "advantages": advantages,
            "loss_mask": response_mask.clone(),
            "response_mask": response_mask.clone(),
            ROLLOUT_ENGINE_DP_RANKS_KEY: torch.arange(len(BODY_LENGTHS)) % ENGINE_RANKS,
        }
    )
    batch.metadata = {"response_length": RESPONSE_LENGTH, "global_step": 0}
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
def _policy(cfg, model_path: str) -> Iterator[PPORayActorGroup]:
    """The gradient probe policy of ``cfg`` on its own Ray instance, shut down on exit."""
    _init_ray(cfg)
    try:
        pg = placement_group([{"GPU": POLICY_WORLD_SIZE, "CPU": POLICY_WORLD_SIZE}], strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)
        policy = PPORayActorGroup(
            cfg,
            num_nodes=1,
            num_gpus_per_node=POLICY_WORLD_SIZE,
            ray_actor_type=GradientProbeWorker,
            pg=pg,
            num_gpus_per_actor=0.75,
            colocate_all=False,
            sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            record_memory=cfg.trainer.policy.record_memory,
        )
        ray.get(policy.async_init_model(model_path))
        yield policy
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


def _relative_difference(value: torch.Tensor, reference: torch.Tensor) -> float:
    """The L2 norm of ``value - reference`` over that of ``reference``; 0 when both are zero."""
    difference = (value.double() - reference.double()).norm().item()
    return difference / max(reference.double().norm().item(), torch.finfo(torch.float64).tiny)


def test_exact_gradients_equal_with_and_without_recompute_and_follow_native_gradients(tmp_path):
    """Under exact numerics on two pipeline stages, a ``ppo_train`` mini-batch gives the same gradients bit for bit
    with full activation recompute in one-layer units and without recompute, and every gradient is within
    ``NATIVE_RELATIVE_TOLERANCE`` of native numerics' gradient on the same weights, batch and expert routes, each
    mode's ratios taken against its own scoring forward."""
    require_hoppers(POLICY_WORLD_SIZE)
    model_path = tmp_path / "model"
    model_path.mkdir()
    # Zero router weights send every token to the four experts of highest bias under both numerics.
    write_snowball_checkpoint(model_path, router_weight_scale=0.0)
    batch = _batch(PAD_TOKEN_ID, vocab_size=1000)
    with _policy(_config(str(model_path), NumericsMode.EXACT), str(model_path)) as policy:
        batch["action_log_probs"] = policy_logprobs(policy, batch) * batch["response_mask"]
        _gradients(policy, batch, tmp_path / "a", recompute=True)
        _gradients(policy, batch, tmp_path / "b", recompute=False)
    with _policy(_config(str(model_path), NumericsMode.NATIVE), str(model_path)) as policy:
        batch["action_log_probs"] = policy_logprobs(policy, batch) * batch["response_mask"]
        _gradients(policy, batch, tmp_path / "native", recompute=False)

    a, b, native = (_load_gradients(tmp_path / arm) for arm in ("a", "b", "native"))
    assert a.keys() == b.keys() == native.keys()
    # Main gradients the optimizer step had cleared would pass every comparison below.
    assert any(gradient.any() for gradient in a.values())
    recompute_differs = [name for name in a if not torch.equal(a[name], b[name])]
    assert not recompute_differs, f"{len(recompute_differs)} of {len(a)} gradients differ: {recompute_differs}"
    relative = {name: _relative_difference(a[name], native[name]) for name in a}
    for name in sorted(relative, key=relative.get, reverse=True)[:10]:
        print(f"{name}: exact vs native relative difference {relative[name]:.3e}")
    far = {name: value for name, value in relative.items() if value > NATIVE_RELATIVE_TOLERANCE}
    assert not far, far
