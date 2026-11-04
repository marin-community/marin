"""Evaluate the agent protocol probes with the production trainer's CPU harness.

The probes are canonical TaskSpec rows, so they take the same path as a launch: the trainer lowers
them with ``task_spec`` settings, rollout workers run the shell probes on ShellSim and serve
``fetch_url`` from the task's declared pages, and evaluation reports each check's results. The
required-file probe is left out: its answer file needs the ``local`` grading backend, which this
harness does not enable, and the worker rejects rows whose backend is not configured.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import TASK_SPEC_COLUMN
from taskcompendium.examples.protocol_probes import (
    PROBE_DATASET,
    PROBE_TURNS,
    SHELLSIM_MACHINE,
    lock_grader_environment,
    probe_check,
    probe_tasks,
)
from taskcompendium.models import ENABLE_THINKING
from taskcompendium.protocol import ProtocolCheck

from tests.cpu.tiny_training.experiment import run_tiny_training

RUN_TIMEOUT_SECONDS = 900
UNBUILT_GRADER_LOCK = "/unavailable/verifyit.lock"
MACHINE_FREE_CHECKS = ("tool_execution", "thinking_disabled", "url_loading")
MAX_GENERATE_LENGTH = 128


def write_probe_rows(path: Path, variants: int) -> Path:
    tasks = [
        task
        for task in probe_tasks(grader_environment=lock_grader_environment(UNBUILT_GRADER_LOCK), variants=variants)
        if probe_check(task) != ProtocolCheck.REQUIRED_FILE
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({TASK_SPEC_COLUMN: [task.model_dump_json() for task in tasks]})
    pq.write_table(table, path)
    return path


def protocol_probes_config(root: Path, model: Path, *, variants: int) -> DictConfig:
    """One update on the probes, evaluating them before and after it."""
    rows = write_probe_rows(root / "data/probes.parquet", variants)
    prompts = len(MACHINE_FREE_CHECKS) * variants
    defaults = get_default_config()
    # The per-dataset maps are empty in the defaults, so struct mode would reject their first keys.
    OmegaConf.set_struct(defaults, False)
    cfg = OmegaConf.merge(
        defaults,
        {
            "data": {"train_data": [str(rows)], "val_data": [str(rows)]},
            "environment": {
                "task_sessions": {
                    "task_spec": {
                        "max_turns_by_dataset": {PROBE_DATASET: PROBE_TURNS},
                        "task_machine_by_dataset": {PROBE_DATASET: SHELLSIM_MACHINE},
                    }
                }
            },
            "trainer": {
                "seed": 0,
                "debug_mode": "off",
                "placement": {"colocate_all": False, "policy_num_gpus_per_node": 1},
                "policy": {"model": {"path": str(model)}, "optimizer_config": {"lr": 1e-5, "weight_decay": 0.0}},
                "algorithm": {"use_kl_loss": False, "group_admission": {"stall_timeout": 300}},
                "rollout_buffer": {"max_staleness_steps": 0, "max_in_flight": 8, "object_store_root": None},
                "train_batch_size": prompts,
                "policy_mini_batch_size": prompts,
                "update_epochs_per_batch": 1,
                "micro_train_batch_size_per_gpu": prompts,
                "micro_forward_batch_size_per_gpu": prompts,
                "use_sample_packing": False,
                "max_prompt_length": 2048,
                "max_steps": 1,
                "eval_before_train": True,
                "eval_interval": 1,
                "ckpt_interval": -1,
                "hf_save_interval": -1,
                "ckpt_path": str(root / "ckpts"),
                "export_path": str(root / "exports"),
                "logger": "console",
                "dump_eval_results": True,
            },
            "generator": {
                "num_inference_engines": 1,
                "inference_engine_tensor_parallel_size": 1,
                "n_samples_per_prompt": 2,
                "max_turns": PROBE_TURNS,
                "inference_stats_interval": 0,
                "enable_http_endpoint": False,
                # The probes measure protocol following; a reasoning model's thinking would spend the budget.
                "chat_template_kwargs": {ENABLE_THINKING: False},
                "sampling_params": {"max_generate_length": MAX_GENERATE_LENGTH, "logprobs": 0},
                "eval_sampling_params": {"temperature": 0.0, "max_generate_length": MAX_GENERATE_LENGTH},
                "trajectory_retention": {"enabled": False},
            },
            "trajectory_runner": {
                "rollout_workers": {"num_workers": 2, "cpus_per_worker": 1, "start_interval_seconds": 0}
            },
        },
    )
    OmegaConf.set_struct(cfg, True)
    return cfg


def run_protocol_probes(cfg: DictConfig) -> None:
    run_tiny_training(cfg)
