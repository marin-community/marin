"""CatCount on-policy distillation on GPU from a synthetic expert teacher.

The recipe is the async CatCount GPU canary's (Qwen2.5-0.5B-Instruct, two Megatron data-parallel ranks, two vLLM
engines, 64 prompts with eight samples, two update epochs, rollout staleness 2, seed 17), with the environment
reward replaced by the teacher's per-token signal. The teacher is ``synthetic_teacher.py`` served over HTTP on the
same host, so it needs no GPU. ``ci/marin_nightly/run_cat_count_opd_h100.sh`` runs this on one Iris H100x4 task.
"""

import argparse
import json
import random
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.main_base import run
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.tokenizer import create_tokenizer

HELDOUT_NS = (3, 5, 13, 16)
EXTRAPOLATION_NS = (24, 28)
TRAIN_NS = tuple(n for n in range(1, 21) if n not in HELDOUT_NS)
EVAL_NS = (*TRAIN_NS, *HELDOUT_NS, *EXTRAPOLATION_NS)
BATCH_SIZE = 64
GROUP_SIZE = 8
MICRO_BATCH_SIZE = 16
GPUS = 2
STALENESS = 2
SEED = 17
SAMPLED_EVAL_SAMPLES = 8
EVAL_INTERVAL = 5


def record(n: int, split: str, index: int) -> dict:
    return {
        "data_source": f"cat_count_n{n}",
        "prompt": [
            {
                "role": "user",
                "content": f"Reply with the word cat exactly {n} times, separated by single spaces. Nothing else.",
            }
        ],
        "env_class": "cat_count",
        "reward_spec": {"method": "rule", "ground_truth": n},
        "extra_info": {"n": n, "split": split, "index": index},
    }


def write_data(directory: Path, steps: int, seed: int) -> tuple[Path, Path]:
    """Write the canary's rows: shuffled cycles of the training counts, and one row per evaluation count."""
    import polars as pl  # noqa: PLC0415

    rng = random.Random(seed)
    rows = BATCH_SIZE * (steps + STALENESS)
    schedule: list[int] = []
    while len(schedule) < rows:
        cycle = list(TRAIN_NS)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    directory.mkdir(parents=True, exist_ok=True)
    train, validation = directory / "train.parquet", directory / "validation.parquet"
    pl.DataFrame([record(n, "train", i) for i, n in enumerate(schedule[:rows])]).write_parquet(train)
    pl.DataFrame([record(n, "validation", i) for i, n in enumerate(EVAL_NS)]).write_parquet(validation)
    return train, validation


def metric_groups() -> dict[str, list[str]]:
    groups = {}
    for profile in ("eval", "eval/sampled"):
        for split, counts in (("train", TRAIN_NS), ("heldout", HELDOUT_NS), ("extrapolation", EXTRAPOLATION_NS)):
            groups[f"{profile}/{split}/avg_score"] = [f"{profile}/cat_count_n{n}/avg_score" for n in counts]
            groups[f"{profile}/{split}/environment/exact"] = [
                f"{profile}/cat_count_n{n}/environment/cat_count/exact" for n in counts
            ]
    return groups


def opd_config(args: argparse.Namespace) -> DictConfig:
    train, validation = write_data(args.output / "data", args.steps, args.seed)
    fingerprint = tokenizer_vocabulary_fingerprint(create_tokenizer(args.model, disable_fast_tokenizer=False))
    geometry = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
    }
    cfg = get_default_config()
    OmegaConf.set_struct(cfg, False)
    return OmegaConf.merge(
        cfg,
        {
            "data": {"train_data": [str(train)], "val_data": [str(validation)], "shuffle": False},
            "environment": {"env_class": "cat_count"},
            "trainer": {
                "strategy": "megatron",
                "flash_attn": False,
                "use_sample_packing": False,
                "gradient_checkpointing": False,
                "placement": {"colocate_all": False, "policy_num_nodes": 1, "policy_num_gpus_per_node": GPUS},
                "epochs": 1,
                "max_steps": args.steps,
                "train_batch_size": BATCH_SIZE,
                "policy_mini_batch_size": BATCH_SIZE,
                "micro_train_batch_size_per_gpu": MICRO_BATCH_SIZE,
                "micro_forward_batch_size_per_gpu": MICRO_BATCH_SIZE,
                "update_epochs_per_batch": 2,
                "max_prompt_length": 64,
                "eval_batch_size": len(EVAL_NS),
                "eval_interval": EVAL_INTERVAL,
                "eval_before_train": True,
                "ckpt_interval": -1,
                "hf_save_interval": -1,
                "resume_mode": "none",
                "seed": args.seed,
                "logger": "console",
                "project_name": "marin-cat-count-opd",
                "run_name": args.output.name,
                "export_path": str(args.output / "exports"),
                "ckpt_path": str(args.output / "ckpts"),
                "training_metrics": True,
                "algorithm": {
                    "advantage_estimator": "uniform",
                    "policy_loss_type": "behavior_clip",
                    "eps_clip_low": 0.2,
                    "eps_clip_high": 0.2,
                    "use_kl_loss": False,
                    "use_kl_in_reward": False,
                    "off_policy_correction": "none",
                    "dynamic_sampling": {"type": None},
                    "distillation": {
                        "objective": "sampled_reverse_kl",
                        "routing_plan": "opd",
                        "coefficient": 1.0,
                        "reward_mode": "replace",
                        "advantage_clip": args.advantage_clip,
                    },
                },
                "policy": {
                    "model": {"path": args.model},
                    "optimizer_config": {
                        "optimizer": "AdamW",
                        "lr": args.lr,
                        "weight_decay": 0.01,
                        "max_grad_norm": 1.0,
                    },
                    "megatron_config": {**geometry, "check_dp_weight_consistency": True},
                },
                "ref": {"megatron_config": geometry},
                "rollout_buffer": {
                    "max_staleness_steps": STALENESS,
                    "batch_policy": "full_batch",
                    "max_in_flight": BATCH_SIZE,
                    "object_store_root": None,
                },
                "callbacks": [
                    {
                        "type": "evaluation",
                        "eval_steps": EVAL_INTERVAL,
                        "eval_before_train": True,
                        "additional_evaluations": {
                            "sampled": {
                                "sampling_params": {"temperature": 1.0},
                                "n_samples_per_prompt": SAMPLED_EVAL_SAMPLES,
                            }
                        },
                        "metric_groups": metric_groups(),
                    }
                ],
            },
            "generator": {
                "backend": "vllm",
                "model_dtype": "bfloat16",
                "vllm_attention_backend": "FLASH_ATTN",
                "run_engines_locally": True,
                "enable_http_endpoint": False,
                "weight_sync_backend": "nccl",
                "use_conversation_multi_turn": False,
                "require_exact_chat_transport": False,
                "num_inference_engines": GPUS,
                "inference_engine_tensor_parallel_size": 1,
                "n_samples_per_prompt": GROUP_SIZE,
                "max_turns": 1,
                "gpu_memory_utilization": 0.7,
                "chat_template": {"source": "name", "name_or_path": "qwen2_5_with_generation_tag_simplified"},
                "sampling_params": {"temperature": 1.0, "top_p": 1.0, "logprobs": 0, "max_generate_length": 64},
                "eval_sampling_params": {"temperature": 0.0, "max_generate_length": 64},
                "eval_n_samples_per_prompt": 1,
                "engine_init_kwargs": {"max_model_len": 128},
            },
            "teachers": {
                "cat": {
                    "source": "openai_compatible",
                    "placement": "external",
                    "model": {"path": "synthetic/cat-count", "revision": args.teacher_revision},
                    "endpoints": [{"url": args.teacher_url, "max_concurrency": 8}],
                    "tokenizer_fingerprint": fingerprint,
                    "max_sequence_length": 128,
                    "request_timeout_seconds": 120,
                    "evidence": "chosen_token",
                }
            },
            "teacher_routing": {
                "opd": {"revision": "cat-count-v1", "routes": {"default": {"teacher": "cat", "weight": 1.0}}}
            },
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local Qwen2.5-0.5B-Instruct directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-url", required=True, help="the synthetic teacher's OpenAI base URL")
    parser.add_argument("--teacher-revision", required=True, help="the teacher's noise settings, for run identity")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    # Every healthy run on seeds 17, 23, 31, 47 and 59 passed at 5e-7; 1e-6 sometimes peaked and then fell.
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument(
        "--advantage-clip",
        type=lambda value: None if value.lower() == "none" else float(value),
        default=5.0,
        help="bound on each token's teacher credit; none disables clipping",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = opd_config(args)
    print("CAT_COUNT_OPD_CONFIG " + json.dumps(OmegaConf.to_container(cfg.trainer.algorithm.distillation)))
    run(cfg)


if __name__ == "__main__":
    main()
