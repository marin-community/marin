"""
Main entrypoint for training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable, Mapping

from rolloutengine.contracts import TaskSession
from shellbox.machine import Machine
from rolloutengine.spec import LoweredTaskSpec, TaskSessionSpec

from ray.util.placement_group import placement_group, PlacementGroup
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.remote_function import RemoteFunction

from transformers import PreTrainedTokenizerBase
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import ray

import os
import signal
import time
import hydra
from loguru import logger
import asyncio
import multiprocessing as mp

from skyrl_train.inference_engines.chat_template import get_custom_chat_template
from skyrl_train.config.rollout_validation import (
    EntrypointOperation,
    validate_rollout_launch,
)
from marinskyrl.speculative_decoding import (
    STANDARD_TRAINING_ENTRYPOINT,
    parse_speculative_decoding_config,
    runai_model_uri,
)
from marinskyrl.runtime_options import (
    EvaluationRunner,
    WeightSyncTransport,
    inference_engines_required,
    static_preference_pairs_requested,
)
from marinskyrl.resource_locator import RolloutModelLoading
from skyrl_train.dataset.preference_pairs import PreferencePairFormat

if TYPE_CHECKING:
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
    from skyrl_train.trajectory_runners.base import TrajectoryRunner

# NOTE (sumanthrh): We use ray heavily and thus disable `fork` start method.
# forking within ray leads to undefined behaviour and often causes hard to debug
# memory leaks.  See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
# A common culprit is Pytorch dataloaders which use `fork` by default.
mp.set_start_method("spawn", force=True)

config_dir = str(Path(__file__).parent.parent / "config")
__all__ = ["BasePPOExp", "config_dir"]


def resolve_entrypoint_node_id(node_ip: str) -> str:
    """Return the live Ray node ID for an explicitly selected node address."""
    matching_node_ids = [
        node["NodeID"] for node in ray.nodes() if node.get("Alive") and node.get("NodeManagerAddress") == node_ip
    ]
    if len(matching_node_ids) != 1:
        raise ValueError(
            f"Expected exactly one live Ray node with address {node_ip!r}, found {len(matching_node_ids)}. "
            "Set trainer.entrypoint_node_ip to a NodeManagerAddress reported by ray.nodes()."
        )
    return matching_node_ids[0]


class EntrypointSupervisor:
    """Wait for an entrypoint task and cooperatively cancel it on termination."""

    def __init__(self, *, shutdown_timeout_seconds: float) -> None:
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._termination_signal: int | None = None

    def request_termination(self, signum: int, _frame=None) -> None:
        """Record the first termination signal without calling Ray from a signal handler."""
        if self._termination_signal is None:
            self._termination_signal = signum

    def wait(self, entrypoint_ref: ray.ObjectRef) -> int | None:
        """Wait for the task, returning ``None`` normally or ``128 + signal`` after termination."""
        while self._termination_signal is None:
            ready, _ = ray.wait([entrypoint_ref], timeout=1)
            if ready:
                ray.get(entrypoint_ref)
                return None

        signum = self._termination_signal
        logger.warning(
            "Received {}, asking the training entrypoint to shut down within {}s",
            signal.Signals(signum).name,
            self._shutdown_timeout_seconds,
        )
        ray.cancel(entrypoint_ref, force=False, recursive=False)
        deadline = time.monotonic() + self._shutdown_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error("Training entrypoint did not stop before its shutdown deadline; forcing cancellation")
                ray.cancel(entrypoint_ref, force=True, recursive=False)
                break
            ready, _ = ray.wait([entrypoint_ref], timeout=min(1, remaining))
            if ready:
                try:
                    ray.get(entrypoint_ref)
                except Exception as e:
                    # Cooperative task cancellation is reported as a Ray task
                    # error after the entrypoint's Python finally blocks finish.
                    logger.info("Training entrypoint stopped after the termination request: {}", e)
                break
        return 128 + signum


def create_ray_wrapped_inference_engines_from_config(
    cfg: DictConfig,
    colocate_pg,
    tokenizer: PreTrainedTokenizerBase,
    *,
    entrypoint: str = STANDARD_TRAINING_ENTRYPOINT,
    operation: EntrypointOperation = EntrypointOperation.TRAIN,
):
    from skyrl_train.inference_engines.configuration import (
        InferenceEngineRoleConfig,
        inference_engine_kwargs_from_config,
    )
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import (
        MODEL_METADATA_PATH_KEY,
        create_ray_wrapped_inference_engines,
        create_ray_wrapped_inference_engines_with_retry,
    )

    raw_speculative_decoding = cfg.generator.get("speculative_decoding")
    speculative_decoding = parse_speculative_decoding_config(
        None if raw_speculative_decoding is None else OmegaConf.to_container(raw_speculative_decoding, resolve=True),
        backend=cfg.generator.backend,
        run_engines_locally=cfg.generator.run_engines_locally,
        entrypoint=entrypoint,
        colocate_all=cfg.trainer.placement.colocate_all,
        num_inference_engines=cfg.generator.num_inference_engines,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        pipeline_parallel_size=cfg.generator.inference_engine_pipeline_parallel_size,
        engine_init_kwargs=OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True),
    )
    engine_init_kwargs = {
        **OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True),
        "openai_sampling_params": OmegaConf.to_container(cfg.generator.sampling_params, resolve=True),
    }
    if cfg.generator.backend == "vllm":
        engine_init_kwargs["chat_template"] = get_custom_chat_template(cfg.generator.chat_template)
        engine_init_kwargs["default_chat_template_kwargs"] = OmegaConf.to_container(
            cfg.generator.chat_template_kwargs, resolve=True
        )
    engine_init_kwargs["tokenizer"] = cfg.trainer.policy.model.tokenizer_path
    tokenizer_revision = cfg.trainer.policy.model.get("tokenizer_revision")
    if tokenizer_revision is not None:
        engine_init_kwargs["tokenizer_revision"] = tokenizer_revision
    policy_source_uri = cfg.trainer.policy.model.get("source_uri")
    stream_policy_model = policy_source_uri is not None and cfg.generator.model_loading == RolloutModelLoading.STREAM
    rollout_model_path = runai_model_uri(policy_source_uri) if stream_policy_model else cfg.trainer.policy.model.path
    if stream_policy_model:
        engine_init_kwargs["load_format"] = "runai_streamer"
        model_loader_extra_config = engine_init_kwargs.setdefault("model_loader_extra_config", {})
        model_loader_extra_config.setdefault("distributed", True)
        engine_init_kwargs[MODEL_METADATA_PATH_KEY] = cfg.trainer.policy.model.path
    if speculative_decoding is not None:
        engine_init_kwargs["speculative_config"] = speculative_decoding.vllm_speculative_config()
        if speculative_decoding.training is not None:
            # vLLM enables its asynchronous scheduler by default, but online EAGLE
            # capture must reconcile each target forward before the next schedule.
            engine_init_kwargs["async_scheduling"] = False
            engine_init_kwargs["weight_transfer_config"] = {"backend": "runai_streamer"}

    requested_logprobs = [cfg.generator.sampling_params.logprobs, cfg.generator.eval_sampling_params.logprobs]
    for callback in cfg.trainer.get("callbacks") or []:
        if callback.get("type") == "evaluation":
            requested_logprobs.extend(
                (profile.get("sampling_params") or {}).get("logprobs")
                for profile in (callback.get("additional_evaluations") or {}).values()
            )
    requested_logprobs = [
        value + 1
        for value in requested_logprobs
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    # Chat capture requests K+1 candidates so the sampled token cannot displace
    # a token from the natural top K. Reserve the actual request width in vLLM.

    role = InferenceEngineRoleConfig(
        pretrain=rollout_model_path,
        backend=cfg.generator.backend,
        num_inference_engines=cfg.generator.num_inference_engines,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        pipeline_parallel_size=cfg.generator.inference_engine_pipeline_parallel_size,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
        # vLLM Decode Context Parallel (DCP). Default 1 (disabled) -> forwarded as the
        # signature default and (per ray_wrapped_inference_engine) NOT passed to the vLLM
        # engine, so flag-off engine init is byte-identical to today (G1). When > 1 it is
        # threaded into vllm.LLM / AsyncEngineArgs as a native EngineArgs kwarg. DCP rides
        # the TP GPUs and is NOT part of any GPU/placement math (G4). Reaches both the
        # standard and terminal_bench entrypoints via this shared config-assembly seam (G5).
        decode_context_parallel_size=cfg.generator.get("inference_engine_decode_context_parallel_size", 1),
        shared_pg=colocate_pg,
        inference_engine_enable_sleep=(cfg.trainer.placement.colocate_all and operation is EntrypointOperation.TRAIN),
        max_logprobs=max([1, *requested_logprobs]),
    )
    model_revision = cfg.trainer.policy.model.get("revision")
    if model_revision is not None and policy_source_uri is None:
        engine_init_kwargs["revision"] = model_revision
    engine_kwargs = inference_engine_kwargs_from_config(
        cfg,
        tokenizer,
        role,
        engine_init_kwargs=engine_init_kwargs,
    )
    if cfg.generator.weight_sync_transport == WeightSyncTransport.EXPERT_BLOCK:
        engine_kwargs["require_verified_placements"] = True
    if cfg.trainer.strategy == "megatron":
        engine_kwargs["nccl_buffer_size_bytes"] = cfg.trainer.policy.nccl_buffer_size_bytes

    # Conditionally add LoRA parameters if LoRA is enabled
    if cfg.trainer.policy.model.lora.rank > 0:
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = cfg.trainer.policy.model.lora.rank
        engine_kwargs["sleep_level"] = 1
        engine_kwargs["max_loras"] = 1
        engine_kwargs["fully_sharded_loras"] = cfg.generator.fully_sharded_loras

        # TODO(devpatel): Bandaid solution, replace this once we have a better solution for LoRA performance degradation on the vLLM side
        if cfg.generator.enforce_eager and cfg.generator.backend == "vllm":
            logger.warning(
                "LoRA is enabled but generator.enforce_eager=true. "
                "This combination causes significant performance degradation (2-3x slower generation). "
                "Automatically setting enforce_eager=false for better performance. "
            )
            engine_kwargs["enforce_eager"] = False

    if stream_policy_model and rollout_model_path.startswith("s3://") and cfg.generator.backend == "vllm":
        retry = cfg.trainer.model_load_retry

        def create_engines(remaining_timeout_seconds: float):
            attempt_kwargs = {**engine_kwargs, "engine_init_timeout_seconds": remaining_timeout_seconds}
            return create_ray_wrapped_inference_engines(**attempt_kwargs)

        return create_ray_wrapped_inference_engines_with_retry(
            create_engines,
            model_path=rollout_model_path,
            engine_init_timeout_seconds=float(engine_kwargs["engine_init_timeout_seconds"]),
            max_retries=int(retry.max_retries),
            backoff_base_seconds=float(retry.backoff_base_seconds),
            backoff_cap_seconds=float(retry.backoff_cap_seconds),
        )
    return create_ray_wrapped_inference_engines(**engine_kwargs)


def create_remote_inference_engines_from_config(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase):
    from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines  # noqa: PLC0415

    # TODO(tgriggs): We may want a separate config for the model name in case it's different from the name used in the OpenAI API
    return create_remote_inference_engines(
        urls=cfg.generator.remote_inference_engine_urls,
        model_name=cfg.trainer.policy.model.path,
        engine_backend=cfg.generator.backend,
        tokenizer=tokenizer,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        pipeline_parallel_size=cfg.generator.inference_engine_pipeline_parallel_size,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
        # DCP metadata only: for remote engines the operator must pass `-dcp <n>` on the
        # external `vllm serve` launch — SkyRL does not spawn remote servers. Carried here
        # for geometry/GPU-accounting consistency (DCP reuses the TP GPUs; no extra GPUs).
        decode_context_parallel_size=cfg.generator.get("inference_engine_decode_context_parallel_size", 1),
    )


class BasePPOExp:
    def __init__(
        self,
        cfg: DictConfig,
        *,
        sessions: Mapping[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]] | None = None,
    ):
        """
        Initializes a PPO experiment.

        The `cfg` passed here will be the final config from Hydra, including CLI overrides.
        """
        self.cfg = cfg
        self.sessions = {} if sessions is None else dict(sessions)
        self._configure_log_level()
        self.tokenizer = self.get_tokenizer()
        self.train_dataset = self.get_train_dataset()
        self.eval_dataset = self.get_eval_dataset()
        self.colocate_pg = self.get_colocate_pg()
        # Reserve the policy/training placement group BEFORE the inference
        # engines (which are created later, in `_setup_trainer`), so that in the
        # disaggregated case the policy (and optional colocated reference)
        # claims GPU slots before inference engines use the remainder.
        self.policy_pg = self.get_policy_pg()

    def create_inference_engine_client(
        self, *, operation: EntrypointOperation = EntrypointOperation.TRAIN
    ) -> InferenceEngineClient:
        """Create the configured local or remote inference-engine client."""
        from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient  # noqa: PLC0415

        if not inference_engines_required(self.cfg):
            logger.info("Skipping inference engines: preference-pair training reads static completions")
            return InferenceEngineClient([], self.tokenizer, self.cfg)

        engine_mode = "local" if self.cfg.generator.run_engines_locally else "remote"
        logger.info("Starting inference engines: mode={}", engine_mode)
        if self.cfg.generator.run_engines_locally:
            entrypoint = STANDARD_TRAINING_ENTRYPOINT if type(self) is BasePPOExp else type(self).__module__
            inference_engines = create_ray_wrapped_inference_engines_from_config(
                self.cfg,
                self.colocate_pg,
                self.tokenizer,
                entrypoint=entrypoint,
                operation=operation,
            )
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, self.tokenizer)
        logger.info("Inference engines ready: mode={} count={}", engine_mode, len(inference_engines))
        return InferenceEngineClient(inference_engines, self.tokenizer, self.cfg)

    def _configure_log_level(self):
        """Configure loguru log level from trainer config."""
        import sys

        log_level = getattr(self.cfg.trainer, "log_level", "INFO").upper()
        # Remove default handler and add one with configured level
        logger.remove()
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>",
            colorize=True,
        )
        logger.info(f"SkyRL log level set to: {log_level}")

    @staticmethod
    def get_cfg_as_str(dict_cfg: DictConfig) -> str:
        return OmegaConf.to_yaml(dict_cfg)

    def get_tokenizer(self, padding_side="left"):
        """Initializes a tokenizer for the given model."""
        from skyrl_train.tokenizer import tokenizer_from_config  # noqa: PLC0415

        return tokenizer_from_config(self.cfg, padding_side)

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            PromptDataset: The training dataset.
        """
        if static_preference_pairs_requested(self.cfg):
            from skyrl_train.dataset import PreferencePairDataset  # noqa: PLC0415

            return PreferencePairDataset(
                tokenizer=self.tokenizer,
                data_format=PreferencePairFormat(self.cfg.data.preference_pair_format),
                datasets=self.cfg.data.train_data,
                max_prompt_length=self.cfg.trainer.max_prompt_length,
                num_workers=8,
                max_completion_length=self.cfg.generator.sampling_params.max_generate_length,
            )
        prompts_dataset = self.task_dataset(self.cfg.data.train_data)
        # make sure the dataset is large enough to train on
        assert len(prompts_dataset) >= self.cfg.trainer.train_batch_size, (
            f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        )
        return prompts_dataset

    def task_dataset(self, data_files):
        from skyrl_train.dataset.nemotron_ultra import NemotronTaskDataset  # noqa: PLC0415
        from skyrl_train.dataset.tasks import SourceTaskDataset  # noqa: PLC0415
        from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings  # noqa: PLC0415 - optional training dependencies

        terminal_data = list(self.cfg.data.get("terminal_bench_data", []))
        dataset_type = NemotronTaskDataset if terminal_data else SourceTaskDataset
        terminal_options = {}
        if terminal_data:
            terminal_options = {
                "terminal_bench_data": terminal_data,
                "cache_dir": Path(self.cfg.data.task_cache_dir),
                "grader_override": HarborTaskSettings.from_config(self.cfg.terminal_bench_config).grader_override(),
            }
        return dataset_type(
            datasets=data_files,
            environment_configs=OmegaConf.to_container(self.cfg.environment.task_sessions, resolve=True),
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            num_workers=8,
            **terminal_options,
        )

    def get_eval_dataset(self):
        """Load validation prompts for evaluation or new mismatch-probe generation."""
        probe = self.cfg.trainer.mismatch_probe
        needs_probe_prompts = probe.enabled and probe.reuse_probe is None
        if (self.cfg.trainer.eval_interval > 0 or needs_probe_prompts) and self.cfg.data.val_data:
            if EvaluationRunner(self.cfg.trainer.evaluation_runner) is EvaluationRunner.HARBOR:
                return self.harbor_task_dataset(self.cfg.data.val_data)
            return self.task_dataset(self.cfg.data.val_data)
        return None

    def harbor_task_dataset(self, data_files):
        from skyrl_train.dataset.harbor import HarborTaskDataset  # noqa: PLC0415
        from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings  # noqa: PLC0415

        return HarborTaskDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            cache_dir=Path(self.cfg.data.task_cache_dir),
            session=TaskSessionSpec.model_validate(
                {**dict(self.cfg.environment.task_sessions.session), "task_session": "shellbox"}
            ),
            grader_override=HarborTaskSettings.from_config(self.cfg.terminal_bench_config).grader_override(),
        )

    def get_colocate_pg(self, timeout: int | None = None) -> PlacementGroup:
        """Initializes a placement group for colocated training.

        A single placement group that packs all the inference engines together is created.

        Args:
            timeout (int): The timeout for the placement group to be ready.

        Returns:
            PlacementGroup: The placement group for colocated training.
        """
        from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout  # noqa: PLC0415

        timeout = int(self.cfg.trainer.distributed.placement_group_timeout_seconds) if timeout is None else timeout
        if self.cfg.trainer.placement.colocate_all and inference_engines_required(self.cfg):
            pg = placement_group(
                [{"GPU": 1, "CPU": 1}]
                * self.cfg.generator.num_inference_engines
                * self.cfg.generator.inference_engine_tensor_parallel_size
                * self.cfg.generator.inference_engine_pipeline_parallel_size
                * self.cfg.generator.inference_engine_data_parallel_size,
                strategy="PACK",
            )
            get_ray_pg_ready_with_timeout(pg, timeout=timeout)
            return pg
        else:
            return None

    def get_policy_pg(self, timeout: int | None = None):
        """Reserve policy GPU slots before starting inference engines.

        Whole-node bundles use STRICT_SPREAD; per-GPU bundles use PACK. Both
        reserve the policy footprint before the inference placement group.
        Returns None when not eligible (see `policy_strict_spread_eligible`),
        leaving model actors to reserve their placement group lazily.

        A colocated reference model shares this group with the policy. Reserving
        it here prevents inference actors from fragmenting the required nodes.
        """
        from skyrl_train.utils.utils import (
            get_ray_pg_ready_with_timeout,
            policy_per_gpu_bundles_enabled,
            policy_spread_bundles,
            policy_strict_spread_eligible,
        )  # noqa: PLC0415

        timeout = int(self.cfg.trainer.distributed.placement_group_timeout_seconds) if timeout is None else timeout
        if not policy_strict_spread_eligible(self.cfg):
            return None

        from ray.util.placement_group import placement_group as _placement_group

        bundles = policy_spread_bundles(self.cfg)
        per_gpu = policy_per_gpu_bundles_enabled(self.cfg)
        # Per-GPU bundles ({GPU:1} x world_size) must NOT use STRICT_SPREAD
        # (that would force each single-GPU bundle onto a distinct node).
        # PACK packs them ~num_gpus_per_node-per-node; reserving all world_size
        # GPU slots up front still forces the (PACK) inference engines onto the
        # disjoint remaining nodes — the disjointness guarantee comes from the
        # reservation ordering, not from the spread strategy. Whole-node bundles
        # keep STRICT_SPREAD (one whole-node bundle per node) as before.
        strategy = "PACK" if per_gpu else "STRICT_SPREAD"
        pg = _placement_group(bundles, strategy=strategy)
        get_ray_pg_ready_with_timeout(pg, timeout=timeout)
        logger.info(
            f"Reserved dedicated policy placement group (strategy={strategy}, "
            f"{'per-GPU {GPU:1}' if per_gpu else 'whole-node {GPU:N}'} bundles): "
            f"{self.cfg.trainer.placement.policy_num_nodes} node(s) x "
            f"{self.cfg.trainer.placement.policy_num_gpus_per_node} GPU, "
            f"reserved before inference-engine placement to guarantee disjoint nodes."
        )
        return pg

    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        """Run all task families through the common rollout worker.

        Returns:
            TrajectoryRunner: The runner.
        """
        if static_preference_pairs_requested(cfg):
            from skyrl_train.trajectory_runners.preference_pairs import PreferencePairTrajectoryRunner  # noqa: PLC0415

            if list(cfg.data.get("terminal_bench_data", [])):
                raise ValueError("preference-pair training cannot mix terminal-bench rows")
            return PreferencePairTrajectoryRunner(
                tokenizer,
                data_format=PreferencePairFormat(cfg.data.preference_pair_format),
                max_generate_length=cfg.generator.sampling_params.max_generate_length,
                max_input_length=cfg.generator.max_input_length,
                generator_config=cfg.generator,
            )
        return self.get_task_runner(cfg, inference_engine_client)

    def get_task_runner(self, cfg, inference_engine_client, *, harbor_config=None):
        """Build the shared task worker pool for training or validation."""
        from skyrl_train.rollouts.workers import RolloutWorkerPool, RolloutWorkerResources  # noqa: PLC0415
        from skyrl_train.rollouts.task_worker import TaskRolloutWorkerSpec  # noqa: PLC0415

        resources = RolloutWorkerResources.from_config(cfg)
        terminal_data = list(cfg.data.get("terminal_bench_data", []))
        if terminal_data and not cfg.get("terminal_bench_config"):
            raise ValueError("data.terminal_bench_data requires terminal_bench_config")
        return RolloutWorkerPool(
            TaskRolloutWorkerSpec.from_config(
                cfg,
                inference_engine_client.engines,
                harbor_config=harbor_config if harbor_config is not None else cfg.get("terminal_bench_config"),
                sessions=self.sessions,
            ),
            resources,
        )

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        trajectory_runner: TrajectoryRunner,
        colocate_pg,
    ):
        """Initializes the trainer, with the trajectory runner as its rollout workers.

        Returns:
            RayPPOTrainer: The trainer.
        """
        from skyrl_train.rollouts.context import TrainingContext  # noqa: PLC0415
        from skyrl_train.trainer import RayPPOTrainer  # noqa: PLC0415

        return RayPPOTrainer(
            context=TrainingContext.from_config(cfg, train_dataset, trajectory_runner),
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            trajectory_runner=trajectory_runner,
            eval_trajectory_runner=(
                self.get_task_runner(
                    cfg,
                    inference_engine_client,
                    harbor_config=cfg.terminal_bench_config
                    if EvaluationRunner(cfg.trainer.evaluation_runner) is EvaluationRunner.HARBOR
                    else None,
                )
                if EvaluationRunner(cfg.trainer.evaluation_runner) is not EvaluationRunner.TRAINING
                else trajectory_runner
            ),
            colocate_pg=colocate_pg,
        )

    def get_worker_classes(self):
        """Return the policy, critic, and reference Ray actor classes for the configured strategy."""
        if self.cfg.trainer.strategy != "megatron":
            raise ValueError(f"Unknown strategy type: {self.cfg.trainer.strategy}")
        from skyrl_train.workers.megatron.megatron_worker import PolicyWorker, CriticWorker, RefWorker  # noqa: PLC0415

        return PolicyWorker, CriticWorker, RefWorker

    def get_tracker(self):
        """Initializes the tracker for experiment tracking.

        Returns:
            Tracking: The tracker.
        """
        from skyrl_train.utils.tracking import Tracking  # noqa: PLC0415

        return Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
        )

    def _setup_trainer(self, tracker):
        """Construct the trainer and models with the experiment's active tracker."""
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        PolicyWorker, CriticWorker, RefWorker = self.get_worker_classes()

        tokenizer = self.tokenizer
        from skyrl_train.teacher_runtime import prepare_distillation_runtime, start_distillation_runtime  # noqa: PLC0415

        prepared_distillation = prepare_distillation_runtime(self.cfg, tokenizer)
        inference_engine_client = self.create_inference_engine_client()

        trajectory_runner: TrajectoryRunner = self.get_trajectory_runner(self.cfg, tokenizer, inference_engine_client)

        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            trajectory_runner=trajectory_runner,
            colocate_pg=self.colocate_pg,
        )

        # Pass the policy placement group reserved before inference startup.
        logger.info("Starting policy workers: strategy={}", self.cfg.trainer.strategy)
        try:
            trainer.build_models(PolicyWorker, CriticWorker, RefWorker, policy_pg=self.policy_pg)
            logger.info(
                "Policy workers ready: strategy={} count={}",
                self.cfg.trainer.strategy,
                len(trainer.policy_model.actor_infos),
            )
            distillation_runtime = asyncio.run(start_distillation_runtime(self.cfg, prepared_distillation))
            if distillation_runtime is not None:
                trainer.configure_distillation(distillation_runtime)
        except BaseException:
            asyncio.run(trainer.shutdown())
            raise
        return trainer

    def run(self):
        from skyrl_train.telemetry import TRAINER_ROLE, process_telemetry
        from skyrl_train.utils.progress import configure_progress  # noqa: PLC0415 - keep launcher imports Torch-free

        configure_progress(self.cfg.trainer.progress)

        with process_telemetry(TRAINER_ROLE):
            return self._run()

    def _run(self):
        # Force the orchestrator onto CPython's stock asyncio event loop (epoll),
        # NOT uvloop. Ray installs uvloop globally in every worker by default
        # (RAY_USE_UVLOOP defaults True -> default_worker.py:221 try_install_uvloop).
        # libuv's epoll-ctl machinery SIGABRTs this orchestrator under Daytona
        # sandbox-teardown socket churn (uv__epoll_ctl_prep AND uv__io_poll asserts;
        # present across libuv 1.45-1.49+). Reset the policy in this shared
        # BasePPOExp._run() path, which every training entrypoint funnels through
        # (main_base.skyrl_entrypoint, skyrl_train.entrypoints.terminal_bench's
        # TerminalBenchExp(BasePPOExp) which does NOT override run(), etc.) -- and
        # it runs immediately before the asyncio.run() below creates the loop, so
        # both asyncio.run() calls build a stock SelectorEventLoop with no libuv
        # path. Placing it on the per-entrypoint skyrl_entrypoint wrapper is a
        # trap: there are 26+ such functions and terminal_bench uses its own, so
        # the fix must live on this shared _run() method. Orchestrator is
        # network-RTT-bound (vLLM/Daytona) so uvloop's throughput edge is moot.
        #
        # DEPRECATION NOTE: asyncio.set_event_loop_policy() emits a
        # DeprecationWarning on Python 3.12+ and the policy system is slated for
        # removal (~3.16). It works on our 3.12 runtime. To future-proof when the
        # policy API is removed, drop this line and instead pass an explicit loop
        # to the asyncio.run() calls below:
        #   asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)  # Py 3.12+ Runner
        # See agent_logs/2026-05-29_skyrl_uvloop_integration_and_robustness_research.md
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

        with self.get_tracker() as tracker:
            trainer = None
            try:
                trainer = self._setup_trainer(tracker)
                asyncio.run(trainer.train())
            finally:
                # Release GPU actors before a Ray retry can start another attempt.
                if trainer is not None:
                    try:
                        asyncio.run(trainer.shutdown())
                    except Exception as e:
                        logger.warning(f"Error shutting down trainer: {e}")


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # NOTE: the uvloop->stock-asyncio reset that prevents the libuv epoll SIGABRT
    # lives in BasePPOExp.run() (the shared chokepoint all entrypoints funnel
    # through), NOT here -- terminal_bench and other entrypoints use their own
    # skyrl_entrypoint wrappers, so the fix must be on run(). See run() above.
    # make sure that the training loop is not run on the head node.
    exp = BasePPOExp(cfg)
    exp.run()


def run_ray_driver(
    cfg: DictConfig,
    entrypoint: RemoteFunction,
    *,
    operation: EntrypointOperation = EntrypointOperation.TRAIN,
    failure_message: str = "Training failed",
) -> None:
    """Run one packaged experiment entrypoint with the shared Ray driver lifecycle."""
    from skyrl_train.entrypoints.ray_lifecycle import exit_without_ray_destructors, shutdown_ray  # noqa: PLC0415
    from marinskyrl.process_diagnostics import write_exception_receipt  # noqa: PLC0415
    from skyrl_train.telemetry import DRIVER_ROLE, process_telemetry  # noqa: PLC0415
    from skyrl_train import objective  # noqa: F401, PLC0415 - register losses when the training runtime loads
    from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings  # noqa: PLC0415 - optional training dependencies
    from skyrl_train.utils import validate_cfg  # noqa: PLC0415
    from skyrl_train.utils.logging_utils import log_exception_as_text  # noqa: PLC0415
    from skyrl_train.utils.progress import configure_progress  # noqa: PLC0415 - keep launcher imports Torch-free
    from skyrl_train.utils.utils import initialize_ray  # noqa: PLC0415

    validate_rollout_launch(cfg, operation)
    if cfg.get("terminal_bench_config") is not None:
        HarborTaskSettings.from_config(cfg.terminal_bench_config)
    validate_cfg(cfg)
    configure_progress(cfg.trainer.progress)

    initialize_ray(cfg)

    with process_telemetry(DRIVER_ROLE):
        supervisor = EntrypointSupervisor(
            shutdown_timeout_seconds=float(cfg.trainer.get("entrypoint_shutdown_timeout_seconds", 120))
        )
        signal.signal(signal.SIGTERM, supervisor.request_termination)

        entrypoint_node_ip = cfg.trainer.get("entrypoint_node_ip")
        if entrypoint_node_ip:
            node_id = resolve_entrypoint_node_id(str(entrypoint_node_ip))
            logger.info("Pinning training entrypoint to Ray node {} ({})", node_id, entrypoint_node_ip)
            entrypoint = entrypoint.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            )

        exit_code = None
        failure: Exception | None = None
        try:
            exit_code = supervisor.wait(entrypoint.remote(cfg))
            if exit_code in (None, 0) and operation is EntrypointOperation.TRAIN:
                logger.info("Training done!")
        except Exception as e:
            log_exception_as_text(failure_message, e)
            receipt = write_exception_receipt(
                "skyrl-entrypoint-driver",
                e,
                metadata={"failure_message": failure_message},
            )
            logger.error("Preserved original training-driver exception at {}", receipt)
            failure = e
        finally:
            logger.info("Shutting down Ray on head node...")
            shutdown_ray()

        if failure is not None:
            # An externally owned Ray cluster must bypass native destructors, which have
            # previously replaced the useful RayTaskError with a late SIGABRT.
            exit_without_ray_destructors(1)
            raise failure.with_traceback(failure.__traceback__)

        if exit_code is not None:
            exit_without_ray_destructors(exit_code)
            raise SystemExit(exit_code)

    exit_without_ray_destructors()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
