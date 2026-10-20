"""CatCount configuration for the production trainer's CPU harness."""

import json
import os
from pathlib import Path

from examples.cat_count.cpu_canary import HELD_OUT_N, PROMPT, TRAIN_N
from omegaconf import DictConfig, OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.algorithm_registry import AdvantageEstimatorRegistry

from tests.cpu.tiny_training.experiment import run_tiny_training

FAST_STEPS = 10


def flipped_grpo(**kwargs):
    advantages, returns = AdvantageEstimatorRegistry.get("grpo")(**kwargs)
    return -advantages, -returns


def write_rows(path: Path, ns: list[int], source: str, repeats: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for _ in range(repeats):
            for n in ns:
                handle.write(
                    json.dumps(
                        {
                            "prompt": [{"role": "user", "content": PROMPT.format(N=n)}],
                            "env_class": "cat_count",
                            "data_source": source,
                            "extra_info": {"n": n, "data_source": source},
                        }
                    )
                    + "\n"
                )
    return path


def cat_count_config(
    root: Path,
    model: Path,
    *,
    steps: int = FAST_STEPS,
    staleness: int = 0,
    resume: bool = False,
    eval_interval: int | None = None,
    seed: int = 0,
) -> DictConfig:
    """Run counting with the real rollout pool, objective, evaluation and checkpoint callbacks."""
    return OmegaConf.merge(
        get_default_config(),
        {
            "data": {
                "train_data": [str(write_rows(root / "data/train.jsonl", TRAIN_N, "train", 50))],
                "val_data": [
                    str(write_rows(root / "data/eval_train.jsonl", TRAIN_N, "train")),
                    str(write_rows(root / "data/eval_heldout.jsonl", HELD_OUT_N, "heldout")),
                ],
            },
            "trainer": {
                "seed": seed,
                "debug_mode": "off",
                "placement": {"colocate_all": False, "policy_num_gpus_per_node": 1},
                "policy": {"model": {"path": str(model)}, "optimizer_config": {"lr": 1e-4, "weight_decay": 0.0}},
                "algorithm": {
                    "use_kl_loss": False,
                    "policy_loss_type": "behavior_clip" if staleness else "regular",
                    "off_policy_correction": "none" if staleness else "tis",
                    "group_admission": {"stall_timeout": 120},
                },
                "rollout_buffer": {"max_staleness_steps": staleness, "max_in_flight": 8, "object_store_root": None},
                "train_batch_size": 8,
                "policy_mini_batch_size": 8,
                "update_epochs_per_batch": 2,
                "micro_train_batch_size_per_gpu": 8,
                "micro_forward_batch_size_per_gpu": 8,
                "use_sample_packing": False,
                "max_prompt_length": 64,
                "max_steps": steps,
                "eval_before_train": True,
                "eval_interval": min(steps, 10) if eval_interval is None else eval_interval,
                "ckpt_interval": steps,
                "hf_save_interval": -1,
                "resume_mode": "latest" if resume else "none",
                "ckpt_path": str(root / "ckpts"),
                "export_path": str(root / "exports"),
                "logger": "console",
                "training_metrics": True,
                "policy_train_spans": True,
                "rollout_spans": True,
            },
            "generator": {
                "num_inference_engines": 1,
                "inference_engine_tensor_parallel_size": 1,
                "n_samples_per_prompt": 4,
                "max_turns": 1,
                "inference_stats_interval": 0,
                "enable_http_endpoint": False,
                "sampling_params": {"max_generate_length": 64, "logprobs": 0},
                "eval_sampling_params": {"temperature": 0.0, "max_generate_length": 64},
                "trajectory_retention": {"enabled": False},
            },
            "trajectory_runner": {
                "rollout_workers": {"num_workers": 2, "cpus_per_worker": 1, "start_interval_seconds": 0}
            },
        },
    )


def run_cat_count(cfg: DictConfig, worker_env: dict[str, str]) -> None:
    """Run one counting experiment in a process with its telemetry environment."""
    os.environ.update(worker_env)
    AdvantageEstimatorRegistry.register(
        "cat_count_flipped_grpo", flipped_grpo, group_contract=AdvantageEstimatorRegistry.group_contract("grpo")
    )
    run_tiny_training(cfg)
