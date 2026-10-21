# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Launch the mixed-domain RLVR pivot arms of #10214 on the MarinSkyRL runtime they trained with.

    uv run python -m experiments.post_training.pivotrl.mixed_rlvr_arms validate <arm> --runtime <checkout> --out <dir>
    uv run python -m experiments.post_training.pivotrl.mixed_rlvr_arms submit <arm> --runtime <checkout> --out <dir>

``validate`` writes ``<out>/<label>/launch.yaml`` and ``resolved.yaml`` and dry-runs the launch; ``submit`` sends the
validated launch, detached, and writes ``<out>/<label>/submission.json``. ``--seed`` sets the run, trainer and
data-sampling seeds together (42 is the original series; another value gets its own ``<arm>-seed<N>/`` label).
``--smoke`` runs two steps of arm B's composition at an eighth of the batch on 6 nodes.

The recipe is the #9735 reference/KL arm (frozen reference, KL 0.01, sequence-mean loss, entropy 0.001, router replay,
512 prompts x 16 samples, 4e-6) from the October 5 Datakit SFT, capped at 12 steps. The arms differ only in the dataset
and the per-batch route counts from :mod:`experiments.post_training.pivotrl.mixed_rlvr`.

The arms trained on MarinSkyRL ``dml/mixed-rlvr-runtime`` at :data:`RUNTIME_COMMIT`: seven commits over the fork's main
(verifyit pivot graders for the Nemotron pivot rows, per-agent ``pivot_reward``, the SWE pivot agent name, null message
fields in Parquet prompts, request tools passed as chat tools, and a retry for vLLM engine gangs that die during
startup). ``vendor/skyrl`` does not carry them yet, so this module drives that checkout's launcher directly rather than
:mod:`marin.rl.skyrl`; ``--runtime`` must point at a clean checkout of that commit.
"""

import argparse
import contextlib
import json
import netrc
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import dotenv_values
from marin.rl.nemotron_pivot import SWE_AGENT, TERMINAL_AGENT

from experiments.post_training.pivotrl.mixed_rlvr import ARMS, DATASET_VERSION, Route, domain_weights
from experiments.post_training.pivotrl.runs import GRUG_SFT

REPO = Path(__file__).resolve().parents[3]
RUNTIME_COMMIT = "bfef1308"
BUCKET = "s3://marin-us-east-02a/marin"
COREWEAVE_ENDPOINT = "https://cwobject.com"
COREWEAVE_REGION = "US-EAST-02A"
WANDB_ENTITY = "marin-community"
ORIGINAL_SEED = 42
# The smoke run keeps arm B's composition at an eighth of the batch.
SMOKE_WEIGHTS = {Route.RLVR: 51, Route.SWE: 7, Route.TERMINAL: 6}

JUDGE = {
    "base_url": "https://api.together.xyz/v1",
    "model": "openai/gpt-oss-120b",
    "api_key_env": "TOGETHER_API_KEY",
    "reasoning_effort": "low",
    "timeout_seconds": 600.0,
}


def megatron(pipeline: int, context: int) -> dict:
    return {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": pipeline,
        "context_parallel_size": context,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "logprob_chunk_size": 256,
        "moe_router_replay": True,
        "transformer_config_kwargs": {"cp_comm_type": "a2a"},
    }


def recipe(run_id: str, weights: dict[Route, int], *, smoke: bool, seed: int) -> dict:
    batch = sum(weights.values())
    policy_nodes, engines = (4, 4) if smoke else (10, 12)
    policy_megatron = {
        **megatron(pipeline=2, context=1 if smoke else 5),
        "ddp_config": {"overlap_grad_reduce": False, "overlap_param_gather": False, "grad_reduce_in_fp32": False},
        "optimizer_checkpoint_sharding_type": "dp_reshardable",
    }
    return {
        "entrypoint": "standard",
        "context_budget": {
            "request_window_tokens": 32768 if smoke else 102400,
            "max_new_tokens_per_turn": 8192 if smoke else 49152,
            "max_turns": 999999,
            "generated_budget_fraction": 0.48,
            "overlong_cache_fraction": 0.5,
        },
        "trainer": {
            "strategy": "megatron",
            "seed": seed,
            "flash_attn": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "offload_optimizer_during_rollouts": True,
            "algorithm": {
                "advantage_estimator": "rloo_n",
                "advantage_batch_normalize": False,
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.2,
                "use_entropy_loss": True,
                "entropy_loss_coef": 0.001,
                "use_kl_loss": True,
                "kl_loss_coef": 0.01,
                "use_kl_in_reward": False,
                "loss_reduction": "sequence_mean",
                "group_advantage_min_size": 4,
                "dynamic_sampling": {"type": None},
                "policy_loss_type": "regular",
                "off_policy_correction": "tis",
            },
            "epochs": 1,
            "max_steps": 2 if smoke else 12,
            "train_batch_size": batch,
            "policy_mini_batch_size": batch,
            "update_epochs_per_batch": 1,
            "eval_batch_size": 100,
            "micro_forward_batch_size_per_gpu": 1,
            "micro_train_batch_size_per_gpu": 1,
            "use_sample_packing": True,
            "eval_before_train": True,
            "eval_interval": 2 if smoke else 4,
            "ckpt_interval": 2 if smoke else 4,
            "max_ckpts_to_keep": 4,
            "hf_save_interval": 2 if smoke else 4,
            "resume_mode": "none",
            "logger": "wandb",
            "project_name": "pivot-mixed-rlvr",
            "run_name": run_id,
            "debug_mode": "light",
            "policy": {
                # Constrain policy NCCL residency while the reference actor shares each GPU.
                "nccl_buffer_size_bytes": 262144,
                "optimizer_config": {"optimizer": "AdamW", "lr": 4.0e-06, "weight_decay": 0.0, "max_grad_norm": 0.5},
                "megatron_config": policy_megatron,
            },
            "ref": {"nccl_buffer_size_bytes": 262144, "megatron_config": megatron(2, 1 if smoke else 5)},
            "placement": {
                "colocate_all": False,
                "colocate_policy_ref": True,
                "policy_num_nodes": policy_nodes,
                "policy_num_gpus_per_node": 8,
                "ref_num_nodes": policy_nodes,
                "ref_num_gpus_per_node": 8,
                "policy_strict_spread_pg": True,
            },
            "rollout_buffer": {
                "max_staleness_steps": 2,
                "batch_policy": "rolling",
                "max_in_flight": 128,
                "object_store_root": (
                    f"s3://marin-us-east-02a/tmp/ttl=14d/skyrl/marin/users/dml/rollout-payloads/{run_id}"
                ),
            },
        },
        "generator": {
            "trajectory_reward_shaping": {"enabled": False},
            "trajectory_retention": {"phases": ["train", "eval"], "sample_count_per_step": 100},
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "inference_engine_tensor_parallel_size": 1,
            "inference_engine_pipeline_parallel_size": 1,
            "inference_engine_data_parallel_size": 4,
            "inference_engine_expert_parallel_size": 4,
            "num_inference_engines": engines,
            "n_samples_per_prompt": 16,
            "eval_n_samples_per_prompt": 1,
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 16384,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "weight_sync_pause_timeout_seconds": 600.0,
            # #9933 measured the pivots' pass rates with thinking on.
            "chat_template_kwargs": {"enable_thinking": True},
            "enable_http_endpoint": True,
            "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": 0},
            "engine_init_kwargs": {
                "enable_auto_tool_choice": True,
                "tool_call_parser": "hermes",
                "served_model_name": GRUG_SFT.location.split("/")[-1],
                "enable_return_routed_experts": True,
            },
        },
        "environment": {
            "task_sessions": {
                "nemotron_ultra": {
                    "judges": {"general": JUDGE, "safety": JUDGE},
                    "genrm": {
                        "judge": {**JUDGE, "response_transport": "chat_completions"},
                        "max_output_tokens": 512,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_concurrent_comparisons": 2,
                        "group_reasoning_length_penalty_coeff": 0.0,
                        "group_answer_length_penalty_coeff": 0.0,
                    },
                    # #9933's graders, as #10000 uses them.
                    "pivot_reward": {SWE_AGENT: "nemo", TERMINAL_AGENT: "string_90"},
                },
            },
        },
        "data": {
            "kind": "parquet",
            "shuffle": False,
            "sampling": {
                "kind": "domain-weighted",
                "domain_weights": {str(route): count for route, count in weights.items()},
                "seed": seed,
            },
        },
        "extra_env": {
            # Allow object-store transfer stalls during parallel model initialization.
            "RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS": "60000",
            "LOGURU_LEVEL": "INFO",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_DEBUG": "WARN",
            "TORCH_CPP_LOG_LEVEL": "WARNING",
        },
        "trajectory_runner": {
            "rollout_workers": {
                # Evaluation reserves worker 0 while training continues on the other three.
                "num_workers": 4,
                "cpus_per_worker": 8,
                "executor_threads": 256,
                "progress_timeout_seconds": 3600,
            },
        },
    }


def data_input(dataset: str, filename: str) -> dict:
    """Locate one split of a dataset that :func:`experiments.post_training.pivotrl.mixed_rlvr.arm_datasets` built."""
    name = f"users/dml/pivotrl/mixed-rlvr/{dataset}"
    return {
        "kind": "directory",
        "identity": f"{name}@{DATASET_VERSION}",
        "local_path": f"/tmp/marinskyrl/data/mixed-rlvr-{dataset}-{DATASET_VERSION}",
        "relative_path": filename,
        "uri": f"{BUCKET}/{name}/{DATASET_VERSION}",
    }


def launch_config(
    arm: str, *, smoke: bool, run_id: str, target_cluster: str, seed: int, runtime: Path, cluster_config: Path
) -> dict:
    definition = ARMS[arm]
    weights = SMOKE_WEIGHTS if smoke else domain_weights(definition.pivot_share)
    root = f"{BUCKET}/users/dml/skyrl/pivot-mixed-rlvr/runs/{run_id}"
    nodes = 6 if smoke else 16
    return {
        "schema_version": 1,
        "run": {
            "id": run_id,
            "attempt_id": "a1",
            "seed": seed,
            "mode": "train",
            "submission": "detach",
            "export_hf": False,
        },
        "runtime": {
            "launcher_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runtime, text=True).strip(),
            "profile": "megatron",
            "experiments_dir": "/app/experiments",
            "task_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        },
        "iris": {
            "cluster": yaml.safe_load(cluster_config.read_text())["name"],
            "cluster_config": str(cluster_config),
            "job_name": run_id,
            "wandb_entity": WANDB_ENTITY,
            "allocation": {
                "num_nodes": nodes,
                "gpus_per_node": 8,
                "gpu_variant": "H100",
                "cpu": 48.0,
                "memory": "700GB",
                "disk": "2TB",
            },
            "priority": "interactive",
            "max_retries": 0,
            "timeout": 604800,
            "target_cluster": target_cluster,
            "parent_cluster_config": str(cluster_config),
        },
        "ray": {
            "spill_dir": "/tmp/skyrl-ray-spill",
            "rendezvous_dir": f"{root}/rendezvous",
            "log_dir": f"{root}/ray-logs",
        },
        "artifacts": {
            "checkpoint_root": f"{root}/checkpoint",
            "export_root": f"{root}/export",
            "attempts_root": f"{root}/attempts",
            "resolved_config_uri": f"{root}/resolved.yaml",
            "terminal_manifest_uri": f"{root}/terminal.json",
        },
        "inputs": {
            "model": {
                "uri": GRUG_SFT.location,
                "identity": GRUG_SFT.revision,
                "local_path": GRUG_SFT.location,
                "tokenizer_uri": GRUG_SFT.location,
                "tokenizer_revision": GRUG_SFT.revision,
                "chat_template": None,
            },
            "data_kind": "parquet",
            "train_data": [data_input(definition.dataset, "train.parquet")],
            "validation_data": [data_input(definition.dataset, "validation.parquet")],
        },
        "skyrl": recipe(run_id, weights, smoke=smoke, seed=seed),
    }


def credentials(repo: Path) -> dict[str, str]:
    """Environment for the launcher, the training job and the evaluation CLI: CoreWeave storage, Hub, judge and W&B.

    Reads ``<repo>/.env`` for the Hub, CoreWeave and Together keys and ``~/.netrc`` for the W&B key.
    """
    values = dotenv_values(repo / ".env")
    env = {key: values[key] for key in ("HF_TOKEN", "CW_KEY_ID", "CW_KEY_SECRET", "TOGETHER_API_KEY") if values.get(key)}
    env.update(
        WANDB_API_KEY=netrc.netrc(Path.home() / ".netrc").authenticators("api.wandb.ai")[2],
        WANDB_ENTITY=WANDB_ENTITY,
        AWS_ACCESS_KEY_ID=values["CW_KEY_ID"],
        AWS_SECRET_ACCESS_KEY=values["CW_KEY_SECRET"],
        AWS_ENDPOINT_URL=COREWEAVE_ENDPOINT,
        AWS_DEFAULT_REGION=COREWEAVE_REGION,
        FSSPEC_S3=json.dumps(
            {"endpoint_url": COREWEAVE_ENDPOINT, "config_kwargs": {"s3": {"addressing_style": "virtual"}}}
        ),
    )
    return env


def load_environment(runtime: Path) -> None:
    """Export the credentials the launcher and the training job read, and import the runtime's launcher package."""
    os.environ.update(credentials(REPO))
    for part in ("skyrl-train", "skyrl-gym", "."):
        sys.path.insert(0, str(runtime / part))


def checked_runtime(runtime: Path) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=runtime, text=True).strip():
        raise SystemExit(f"{runtime} has uncommitted changes")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runtime, text=True).strip()
    if not head.startswith(RUNTIME_COMMIT):
        raise SystemExit(f"{runtime} is at {head}, expected {RUNTIME_COMMIT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("validate", "submit"))
    parser.add_argument("arm", choices=sorted(ARMS))
    parser.add_argument("--runtime", type=Path, required=True, help=f"MarinSkyRL checkout at {RUNTIME_COMMIT}")
    parser.add_argument(
        "--out", type=Path, required=True, help="directory that receives one receipt directory per launch"
    )
    parser.add_argument("--smoke", action="store_true")
    # cw-rno2a is #9735's reserved fleet; its runs kept data and checkpoints in the us-east-02a bucket.
    parser.add_argument("--cluster", default="cw-us-east-02a", choices=("cw-us-east-02a", "cw-rno2a"))
    parser.add_argument(
        "--cluster-config",
        type=Path,
        default=REPO / "lib/iris/config/marin.yaml",
        help="Iris hub config whose cached login submits the job; the default is this checkout's marin hub config",
    )
    parser.add_argument("--seed", type=int, default=ORIGINAL_SEED, help="run, trainer and data-sampling seed")
    args = parser.parse_args()

    load_environment(args.runtime)
    checked_runtime(args.runtime)
    # The launcher lives in the runtime checkout, which is not a Marin dependency.
    from cloud.iris.iris_backend import _iris_submission_state, launch  # noqa: PLC0415  # runtime checkout
    from cloud.iris.launch import execute_launch  # noqa: PLC0415  # runtime checkout
    from cloud.iris.launch_config import load_launch_config  # noqa: PLC0415  # runtime checkout
    from omegaconf import OmegaConf  # noqa: PLC0415  # runtime checkout
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415  # runtime checkout

    configure_coreweave_s3()
    label = (
        args.arm.lower() + (f"-seed{args.seed}" if args.seed != ORIGINAL_SEED else "") + ("-smoke" if args.smoke else "")
    )
    directory = args.out / label
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "launch.yaml"
    if args.action == "validate":
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_id = f"pivot-mixed-rlvr-{label}-{stamp}"
        config = launch_config(
            args.arm,
            smoke=args.smoke,
            run_id=run_id,
            target_cluster=args.cluster,
            seed=args.seed,
            runtime=args.runtime,
            cluster_config=args.cluster_config.resolve(),
        )
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        resolved = load_launch_config(path)
        OmegaConf.save(resolved, directory / "resolved.yaml")
        state = _iris_submission_state(path, resolved)
        state.dry_run = True
        with (
            (directory / "dry-run.log").open("w") as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            launch(state, str(resolved.runtime.launcher_commit))
        print("VALIDATED", run_id, f"{resolved.iris.allocation.num_nodes} nodes")
        return

    if not os.environ.get("TOGETHER_API_KEY"):
        raise SystemExit("TOGETHER_API_KEY is required: RLVR1's judge-backed agents grade through Together")
    receipt = directory / "submission.json"
    run_id = yaml.safe_load(path.read_text())["run"]["id"]
    if receipt.exists() and json.loads(receipt.read_text())["run_id"] == run_id:
        raise SystemExit(f"{run_id} was already submitted")
    with (
        (directory / "submission.log").open("a") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        result = execute_launch(path)
    if result.state.value != "submitted":
        raise SystemExit(str(result))
    receipt.write_text(json.dumps({"run_id": result.run_id, "job_id": result.iris_job_id}, indent=2) + "\n")
    print("SUBMITTED", result.iris_job_id)


if __name__ == "__main__":
    main()
