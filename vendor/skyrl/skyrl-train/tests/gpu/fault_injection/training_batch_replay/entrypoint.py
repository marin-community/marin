"""Opt-in TerminalBench batch capture/replay diagnostic entrypoint."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from pathlib import Path
import re

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import ray

from skyrl_train.entrypoints.main_base import config_dir, run_ray_driver
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.checkpoint_listing import extract_step_from_path
from skyrl_train.utils.trainer_utils import ResumeMode
from tests.training_batch_replay import (
    BatchReplayProvenance,
    CapturingRayPPOTrainer,
    DIAGNOSTIC_CONFIG_KEY,
    config_fingerprint,
    load_training_batch_artifact,
    replay_policy_forward,
    validate_artifact_destination,
)
from tests.gpu.utils import import_worker


class Mode(StrEnum):
    CAPTURE = "capture"
    REPLAY = "replay"


_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _diagnostic_value(cfg: DictConfig, name: str):
    value = OmegaConf.select(cfg, f"{DIAGNOSTIC_CONFIG_KEY}.{name}")
    if value is None or value == "":
        raise ValueError(f"+{DIAGNOSTIC_CONFIG_KEY}.{name}=... is required")
    return value


def _mode(cfg: DictConfig) -> Mode:
    value = str(_diagnostic_value(cfg, "mode"))
    try:
        return Mode(value)
    except ValueError as error:
        raise ValueError(f"{DIAGNOSTIC_CONFIG_KEY}.mode must be 'capture' or 'replay', got {value!r}") from error


def _provenance(cfg: DictConfig) -> BatchReplayProvenance:
    if ResumeMode(cfg.trainer.resume_mode) is not ResumeMode.FROM_PATH:
        raise ValueError("batch replay requires trainer.resume_mode=from_path")
    checkpoint_path = str(cfg.trainer.resume_path).rstrip("/")
    checkpoint_step = extract_step_from_path(checkpoint_path)
    if checkpoint_step < 0:
        raise ValueError("trainer.resume_path must end in global_step_<N>")
    target_step = int(_diagnostic_value(cfg, "target_step"))
    if target_step != checkpoint_step + 1:
        raise ValueError(
            f"{DIAGNOSTIC_CONFIG_KEY}.target_step must be checkpoint step + 1; "
            f"got checkpoint={checkpoint_step}, target={target_step}"
        )
    source_revision = str(_diagnostic_value(cfg, "source_revision"))
    if _REVISION_PATTERN.fullmatch(source_revision) is None:
        raise ValueError(f"{DIAGNOSTIC_CONFIG_KEY}.source_revision must be a full 40-character lowercase Git commit")
    return BatchReplayProvenance(
        source_revision=source_revision,
        config_fingerprint=str(_diagnostic_value(cfg, "config_fingerprint")),
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        target_step=target_step,
    )


def _artifact_path(cfg: DictConfig) -> Path:
    return Path(str(_diagnostic_value(cfg, "artifact_path"))).expanduser().resolve()


def _terminal_bench_experiment_class():
    # Harbor is a production-image dependency, not a CPU-test dependency. Keep
    # the TerminalBench import lazy so --help and pre-Ray validation work in the
    # ordinary development environment.
    from skyrl_train.entrypoints.terminal_bench import TerminalBenchExp  # noqa: PLC0415

    return TerminalBenchExp


def _capture_experiment_class():
    TerminalBenchExp = _terminal_bench_experiment_class()

    class CaptureTerminalBenchExp(TerminalBenchExp):
        def get_trainer(
            self,
            cfg,
            tracker,
            tokenizer,
            train_dataset,
            eval_dataset,
            inference_engine_client,
            trajectory_runner,
            colocate_pg,
        ):
            return CapturingRayPPOTrainer(
                context=TrainingContext.from_config(cfg, train_dataset, trajectory_runner),
                cfg=cfg,
                tracker=tracker,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                inference_engine_client=inference_engine_client,
                trajectory_runner=trajectory_runner,
                colocate_pg=colocate_pg,
                capture_artifact_path=_artifact_path(cfg),
                capture_provenance=_provenance(cfg),
            )

    return CaptureTerminalBenchExp


def _replay_experiment_class():
    TerminalBenchExp = _terminal_bench_experiment_class()

    class ReplayTerminalBenchExp(TerminalBenchExp):
        """Build model actors and a training context for one saved policy forward."""

        def _setup_replay_trainer(self):
            client_cfg = OmegaConf.merge(self.cfg, {"generator": {"enable_http_endpoint": False}})
            client = InferenceEngineClient([], self.tokenizer, client_cfg)
            runner = self.get_trajectory_runner(self.cfg, self.tokenizer, client)
            trainer = self.get_trainer(
                cfg=self.cfg,
                tracker=None,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                inference_engine_client=client,
                trajectory_runner=runner,
                colocate_pg=self.colocate_pg,
            )
            strategy = str(self.cfg.trainer.strategy)
            trainer.build_models(
                import_worker(strategy, "policy"),
                import_worker(strategy, "critic"),
                import_worker(strategy, "ref"),
                policy_pg=self.policy_pg,
            )
            return trainer

        @staticmethod
        def _load_replay_checkpoint(trainer, provenance: BatchReplayProvenance) -> None:
            loaded_step, loaded_path = trainer.load_checkpoints()
            if loaded_step != provenance.checkpoint_step or str(loaded_path).rstrip("/") != provenance.checkpoint_path:
                raise RuntimeError(
                    "Loaded checkpoint differs from validated artifact provenance: "
                    f"step={loaded_step}, path={loaded_path}"
                )
            trainer.global_step = provenance.target_step

        def run_replay(self) -> None:
            provenance = _provenance(self.cfg)
            training_input = load_training_batch_artifact(_artifact_path(self.cfg), expected=provenance)
            trainer = self._setup_replay_trainer()
            try:
                self._load_replay_checkpoint(trainer, provenance)
                result = asyncio.run(replay_policy_forward(trainer, training_input))
                shapes = {key: list(value.shape) for key, value in result.items() if value is not None}
                logger.info(f"TRAINING_BATCH_REPLAY_OK target_step={provenance.target_step} tensors={shapes}")
            finally:
                asyncio.run(trainer.shutdown())

    return ReplayTerminalBenchExp


@ray.remote(num_cpus=1, max_retries=0)
def diagnostic_entrypoint(cfg: DictConfig):
    if _mode(cfg) is Mode.CAPTURE:
        _capture_experiment_class()(cfg).run()
    else:
        _replay_experiment_class()(cfg).run_replay()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    mode = _mode(cfg)
    # Resolve interpolations once in the launcher process. Ray workers may have
    # different environment variables, but they must validate the same config.
    OmegaConf.update(
        cfg,
        f"{DIAGNOSTIC_CONFIG_KEY}.config_fingerprint",
        config_fingerprint(cfg),
        force_add=True,
    )
    provenance = _provenance(cfg)
    artifact_path = _artifact_path(cfg)
    if mode is Mode.CAPTURE:
        validate_artifact_destination(artifact_path)
    if mode is Mode.REPLAY:
        # Fail on provenance or artifact corruption before Ray initialization and
        # before any model worker can be dispatched.
        load_training_batch_artifact(artifact_path, expected=provenance)

    run_ray_driver(cfg, diagnostic_entrypoint)


if __name__ == "__main__":
    main()
