"""CatCount on-policy distillation on GPU from synthetic expert teachers.

The recipe is the async CatCount GPU canary's (Qwen2.5-0.5B-Instruct, two Megatron data-parallel ranks, two vLLM
engines, 64 prompts with eight samples, two update epochs, rollout staleness 2, seed 17), with the environment
reward replaced by the teachers' per-token signal. Each teacher is ``synthetic_teacher.py`` for one word, served
over HTTP on the same host, so it needs no GPU. ``ci/marin_nightly/run_cat_count_opd_h100.sh`` runs this on one
Iris H100x4 task.

One teacher (``--teacher cat=URL``) is single-teacher OPD. Several (MOPD) give every count one row per route, each
row's ``teacher_route`` is its route, and evaluation also reports every route separately. A route is a word the
student repeats N times (cat), or ``evens``, the first N positive even numbers, a different skill with the same
shape. ``--swap-routes`` sends each route's rows to the next route's expert, a control that a working router
must fail.
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


def data_source(word: str, n: int) -> str:
    return f"{word}_count_n{n}"


EVENS_ROUTE = "evens"


def exact_metric(route: str) -> str:
    """The per-row exact-match metric name each route's environment reports."""
    return "environment/sequence/exact" if route == EVENS_ROUTE else "environment/cat_count/exact"


def record(word: str, n: int, split: str, index: int, routed: bool) -> dict:
    if word == EVENS_ROUTE:
        items = [str(2 * i) for i in range(1, n + 1)]
        row = {
            "data_source": data_source(word, n),
            "prompt": [
                {
                    "role": "user",
                    "content": f"List the first {n} positive even numbers, separated by single spaces. Nothing else.",
                }
            ],
            "env_class": "sequence",
            "reward_spec": {"method": "rule", "ground_truth": " ".join(items)},
            "extra_info": {"n": n, "items": items, "split": split, "index": index},
        }
    else:
        row = {
            "data_source": data_source(word, n),
            "prompt": [
                {
                    "role": "user",
                    "content": f"Reply with the word {word} exactly {n} times, separated by single spaces. Nothing else.",
                }
            ],
            "env_class": "cat_count",
            "reward_spec": {"method": "rule", "ground_truth": n},
            "extra_info": {"n": n, "word": word, "split": split, "index": index},
        }
    if routed:
        row["teacher_route"] = word
    return row


def write_data(directory: Path, words: tuple[str, ...], steps: int, seed: int) -> tuple[Path, Path]:
    """Write shuffled cycles of the training counts with one row per word each, and every evaluation count.

    Each count's rows stay adjacent and training reads them in order, so every batch holds every word equally.
    """
    import polars as pl  # noqa: PLC0415

    rng = random.Random(seed)
    counts = BATCH_SIZE * (steps + STALENESS) // len(words)
    schedule: list[int] = []
    while len(schedule) < counts:
        cycle = list(TRAIN_NS)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    routed = len(words) > 1
    train_rows = [record(word, n, "train", i, routed) for i, n in enumerate(schedule[:counts]) for word in words]
    validation_rows = [record(word, n, "validation", i, routed) for i, n in enumerate(EVAL_NS) for word in words]
    directory.mkdir(parents=True, exist_ok=True)
    train, validation = directory / "train.parquet", directory / "validation.parquet"
    pl.DataFrame(train_rows).write_parquet(train)
    pl.DataFrame(validation_rows).write_parquet(validation)
    return train, validation


def metric_groups(words: tuple[str, ...]) -> dict[str, list[str]]:
    """Group per-count scores by split over all words and, for MOPD, by split for each word."""
    groups = {}
    scopes = [("", words)] + ([(f"{word}_", (word,)) for word in words] if len(words) > 1 else [])
    for profile in ("eval", "eval/sampled"):
        for prefix, scope in scopes:
            for split, counts in (("train", TRAIN_NS), ("heldout", HELDOUT_NS), ("extrapolation", EXTRAPOLATION_NS)):
                sources = [data_source(word, n) for word in scope for n in counts]
                groups[f"{profile}/{prefix}{split}/avg_score"] = [f"{profile}/{source}/avg_score" for source in sources]
                groups[f"{profile}/{prefix}{split}/environment/exact"] = [
                    f"{profile}/{data_source(word, n)}/{exact_metric(word)}" for word in scope for n in counts
                ]
    return groups


def opd_config(args: argparse.Namespace) -> DictConfig:
    teacher_urls = dict(teacher.split("=", 1) for teacher in args.teacher)
    words = tuple(teacher_urls)
    if len(words) == 1:
        if args.swap_routes:
            raise ValueError("swapping routes needs at least two teachers")
        routes = {"default": {"teacher": words[0], "weight": 1.0}}
    else:
        experts = words[1:] + words[:1] if args.swap_routes else words
        routes = {word: {"teacher": expert, "weight": 1.0} for word, expert in zip(words, experts, strict=True)}
    train, validation = write_data(args.output / "data", words, args.steps, args.seed)
    # Evens answers for the extrapolation counts (24 and 28) take 68 and 80 tokens with EOS; training counts fit 64.
    eval_generate = 96 if EVENS_ROUTE in words else 64
    max_model_len = 160 if EVENS_ROUTE in words else 128
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
                "eval_batch_size": len(EVAL_NS) * len(words),
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
                        "metric_groups": metric_groups(words),
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
                "eval_sampling_params": {"temperature": 0.0, "max_generate_length": eval_generate},
                "eval_n_samples_per_prompt": 1,
                "engine_init_kwargs": {"max_model_len": max_model_len},
            },
            "teachers": {
                word: {
                    "source": "openai_compatible",
                    "placement": "external",
                    "model": {
                        "path": f"synthetic/{word}" if word == EVENS_ROUTE else f"synthetic/{word}-count",
                        "revision": args.teacher_revision,
                    },
                    "endpoints": [{"url": url, "max_concurrency": 8}],
                    "tokenizer_fingerprint": fingerprint,
                    "max_sequence_length": 128,
                    "request_timeout_seconds": 120,
                    "evidence": "chosen_token",
                }
                for word, url in teacher_urls.items()
            },
            "teacher_routing": {"opd": {"revision": "cat-count-v1", "routes": routes}},
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local Qwen2.5-0.5B-Instruct directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--teacher",
        action="append",
        required=True,
        help="WORD=URL: an expert's word and OpenAI base URL; repeat for MOPD",
    )
    parser.add_argument("--swap-routes", action="store_true", help="send each word's rows to the next word's expert")
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
