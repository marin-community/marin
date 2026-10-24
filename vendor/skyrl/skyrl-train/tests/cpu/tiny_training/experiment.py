"""Run the production training entrypoints end to end on CPU with a tiny GSM8K policy.

The experiments swap only the Megatron policy worker and the vLLM engines for the CPU backend.
Ray runs locally with logical GPUs so placement code runs unchanged. Synchronous training runs the standard
entrypoint at staleness 0 and asynchronous training at positive staleness; the asynchronous run also writes each
rollout payload to its own object. Either mode runs single-turn groups, or multi-turn groups trained step-wise.

Usage::

    uv run --frozen --no-sync python -m tests.cpu.tiny_training.experiment --mode async --shape step-wise --steps 20

Without ``--model``, the experiment first writes the tiny policy under its root.
"""

import argparse
import json
import os
from enum import StrEnum
from pathlib import Path

import ray
from omegaconf import DictConfig, OmegaConf
from skyrl_train.config.rollout_validation import validate_rollout_launch
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import SourceTaskDataset
from skyrl_train.entrypoints.main_base import BasePPOExp, EntrypointOperation
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine
from skyrl_train.utils import validate_cfg

from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine, CPUPolicyWorker
from tests.cpu.tiny_training.tiny_model import build_tiny_policy, write_gsm8k_dataset

# Eight logical CPUs leave room for rollout workers and a second run's policy actor in a shared session.
LOGICAL_CPUS = 8
LOGICAL_GPUS = 4
METRICS_FILE = "metrics.jsonl"
# Reports a stalled run with an admission error well inside the test's run timeout. Concurrent test workers can
# starve a healthy run for tens of seconds, so this bounds hangs rather than measuring speed.
STALL_TIMEOUT_SECONDS = 120
TRAIN_BATCH_SIZE = 4
# Workers import the CPU backend from this package whatever directory the run starts in.
WORKER_ENV_VARS = {
    "PYTHONPATH": os.pathsep.join((str(Path(__file__).parents[3]), str(Path(__file__).parents[4] / "skyrl-gym"))),
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
}


class TrainingMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


class RolloutShape(StrEnum):
    SINGLE_TURN = "single-turn"
    STEP_WISE = "step-wise"


MAX_STALENESS_STEPS = {TrainingMode.SYNC: 0, TrainingMode.ASYNC: 1}
# Positive staleness needs an off-policy correction.
POLICY_LOSS_TYPE = {TrainingMode.SYNC: "regular", TrainingMode.ASYNC: "behavior_clip"}
MAX_TURNS = {RolloutShape.SINGLE_TURN: 1, RolloutShape.STEP_WISE: 2}
N_SAMPLES_PER_PROMPT = 4
# The async single-turn run draws prompts from an adaptive curriculum, which step-wise training does not support;
# the other runs read the dataset in seeded passes.
SAMPLING_KIND = {TrainingMode.SYNC: None, TrainingMode.ASYNC: "thompson"}
# The async run's worker processes write each payload to its own object; the sync run keeps them in Ray.
OBJECT_STORE_PAYLOADS = {TrainingMode.SYNC: False, TrainingMode.ASYNC: True}


def sampling_kind(mode: TrainingMode, shape: RolloutShape) -> str | None:
    return SAMPLING_KIND[mode] if shape is RolloutShape.SINGLE_TURN else None


def tiny_training_config(
    root: Path,
    model_dir: Path,
    mode: TrainingMode,
    shape: RolloutShape,
    *,
    max_steps: int,
    checkpoint_interval: int,
    num_prompts: int = 64,
    dp_size: int = 1,
    micro_batch_size: int = 8,
    max_in_flight: int = 8,
    dump_data_batch: bool = False,
) -> DictConfig:
    """Build a complete training config for the policy in ``model_dir``, writing its data and outputs under ``root``.

    The run resumes from the latest checkpoint under ``root``, if an earlier run left one.
    """
    max_turns = MAX_TURNS[shape]
    cfg = get_default_config()
    overrides = {
        "data": {
            "train_data": [str(write_gsm8k_dataset(root / "data" / "train.jsonl", num_prompts, max_turns=max_turns))],
            "val_data": [str(write_gsm8k_dataset(root / "data" / "validation.jsonl", 8, max_turns=max_turns))],
            "sampling": {"kind": sampling_kind(mode, shape)},
        },
        "trainer": {
            "step_wise_training": shape is RolloutShape.STEP_WISE,
            "debug_mode": "off",
            "placement": {"colocate_all": False, "policy_num_gpus_per_node": dp_size},
            "policy": {"model": {"path": str(model_dir)}, "optimizer_config": {"lr": 1.0e-3}},
            # A stalled step fails with the buffer's state long before the test's subprocess timeout.
            "algorithm": {
                "use_kl_loss": False,
                "policy_loss_type": POLICY_LOSS_TYPE[mode],
                "off_policy_correction": "tis" if mode is TrainingMode.SYNC else "none",
                # The CPU inference backend supplies chosen-token scores, without vLLM candidate heads.
                "score_centering_probe": {"interval": 0},
                "group_admission": {"stall_timeout": STALL_TIMEOUT_SECONDS},
            },
            "rollout_buffer": {
                "max_staleness_steps": MAX_STALENESS_STEPS[mode],
                "max_in_flight": max_in_flight,
                "object_store_root": str(root / "rollouts") if OBJECT_STORE_PAYLOADS[mode] else None,
            },
            "train_batch_size": TRAIN_BATCH_SIZE,
            "policy_mini_batch_size": TRAIN_BATCH_SIZE,
            "micro_train_batch_size_per_gpu": micro_batch_size,
            "micro_forward_batch_size_per_gpu": micro_batch_size,
            "dump_data_batch": dump_data_batch,
            "training_metrics": mode is TrainingMode.ASYNC,
            "use_sample_packing": False,
            "max_steps": max_steps,
            "eval_before_train": False,
            "eval_interval": -1,
            "ckpt_interval": checkpoint_interval,
            "hf_save_interval": -1,
            "resume_mode": "latest",
            "ckpt_path": str(root / "ckpts"),
            "export_path": str(root / "exports"),
            # The experiments replace the tracker, so no logger service is contacted.
            "logger": "console",
        },
        "generator": {
            "num_inference_engines": 1,
            "inference_engine_tensor_parallel_size": 1,
            "n_samples_per_prompt": N_SAMPLES_PER_PROMPT,
            "max_turns": max_turns,
            "inference_stats_interval": 0,
            "sampling_params": {"max_generate_length": 16},
            # Each retention storage operation spawns a process that re-imports the entrypoint.
            "trajectory_retention": {"enabled": False},
        },
        # Local workers read a warm page cache, so they start together.
        "trajectory_runner": {"rollout_workers": {"num_workers": 2, "cpus_per_worker": 1, "start_interval_seconds": 0}},
    }
    return OmegaConf.merge(cfg, overrides)


class JsonlTracker:
    """Append every logged metrics record to a JSON Lines file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, data, step, commit=True):
        with self.path.open("a") as handle:
            handle.write(json.dumps({"step": step, **data}, default=float) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        pass


def read_metrics(root: Path) -> list[dict]:
    """Return the metrics records an experiment under ``root`` logged, in order."""
    with (root / "exports" / METRICS_FILE).open() as handle:
        return [json.loads(line) for line in handle]


def cpu_inference_engine_client(cfg: DictConfig, tokenizer) -> InferenceEngineClient:
    engine_actor = ray.remote(CPUInferenceEngine)
    engines = [
        RayWrappedInferenceEngine(
            engine_actor.options(num_cpus=1).remote(cfg.trainer.policy.model.path, cfg.trainer.seed + index)
        )
        for index in range(cfg.generator.num_inference_engines)
    ]
    return InferenceEngineClient(engines, tokenizer, cfg)


class TinyTrainingExp(BasePPOExp):
    """The standard entrypoint with CPU policy workers and CPU inference engines."""

    def get_train_dataset(self):
        # Filtering a few dozen prompts in one process beats spawning preprocessing workers.
        return SourceTaskDataset(
            datasets=self.cfg.data.train_data,
            environment_configs=OmegaConf.to_container(self.cfg.environment.task_sessions, resolve=True),
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            num_workers=1,
        )

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval <= 0 or not self.cfg.data.val_data:
            return None
        return SourceTaskDataset(
            datasets=self.cfg.data.val_data,
            environment_configs=OmegaConf.to_container(self.cfg.environment.task_sessions, resolve=True),
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            num_workers=1,
        )

    def get_tracker(self):
        return JsonlTracker(Path(self.cfg.trainer.export_path) / METRICS_FILE)

    def get_worker_classes(self):
        if self.cfg.trainer.critic.model.path or self.cfg.trainer.algorithm.use_kl_loss:
            raise ValueError("the CPU backend provides only a policy worker")
        return ray.remote(num_gpus=1)(CPUPolicyWorker), None, None

    def create_inference_engine_client(
        self, *, operation: EntrypointOperation = EntrypointOperation.TRAIN
    ) -> InferenceEngineClient:
        return cpu_inference_engine_client(self.cfg, self.tokenizer)


def run_tiny_training(cfg: DictConfig) -> None:
    """Validate the config as the production driver does, then run in a fresh local Ray session."""
    run_with_exp(cfg, TinyTrainingExp)


def run_with_exp(cfg: DictConfig, make_exp) -> None:
    """Run one validated experiment class in a fresh local Ray session."""
    validate_cfg(cfg)
    validate_rollout_launch(cfg, EntrypointOperation.TRAIN)
    ray.init(
        num_cpus=LOGICAL_CPUS,
        num_gpus=LOGICAL_GPUS,
        runtime_env={"env_vars": WORKER_ENV_VARS},
        # No test reads the dashboard; skipping it saves each concurrent run its start-up time and memory.
        include_dashboard=False,
    )
    try:
        make_exp(cfg).run()
    finally:
        ray.shutdown()


def run_experiment(
    root: Path,
    model_dir: Path,
    mode: TrainingMode,
    shape: RolloutShape,
    *,
    max_steps: int,
    checkpoint_interval: int,
    dump_data_batch: bool,
) -> None:
    """Train the policy in ``model_dir`` under ``root``, resuming from a checkpoint an earlier run left there."""
    cfg = tiny_training_config(
        root,
        model_dir,
        mode,
        shape,
        max_steps=max_steps,
        checkpoint_interval=checkpoint_interval,
        dump_data_batch=dump_data_batch,
    )
    run_tiny_training(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", type=TrainingMode, choices=list(TrainingMode), required=True)
    parser.add_argument("--shape", type=RolloutShape, choices=list(RolloutShape), required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=-1, help="-1 saves no checkpoints")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="a tiny policy directory from build_tiny_policy")
    args = parser.parse_args()
    model_dir = args.model or build_tiny_policy(args.root / "model")
    cfg = tiny_training_config(
        args.root,
        model_dir,
        args.mode,
        args.shape,
        max_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
    )
    run_tiny_training(cfg)


if __name__ == "__main__":
    main()
