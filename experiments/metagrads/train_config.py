import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import (
    LMMixtureDatasetConfig,
    UrlSingleDatasetLMConfig,
    UrlDatasetSourceConfig,
    HfDatasetSourceConfig,
)
from levanter.main.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from levanter.callbacks.watch import WatchConfig
from levanter.eval_harness import LmEvalHarnessConfig
from experiments.evals.task_configs import convert_to_levanter_task_config, EvalTaskConfig

from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, this_output_path
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from experiments.metagrads.tokenize_common_pile import comma_main_mixture


@dataclass
class DatagradsConfig:
    # =====================
    # Defaults from _tpu_on_pod.yaml (baked in; no YAML needed at runtime)
    # =====================

    # GCP bucket
    gcs_bucket: str = "data-values-us-central-1"

    # Data
    train_urls: list[str] | None = None
    validation_urls: list[str] | None = None
    cache_dir: str | None = None
    use_comma_main_mixture: bool = False
    # Allow passing a simple HF dataset id for single-dataset training
    hf_dataset_id: str | None = None
    # Optional: Pass a pre-built data config (preferred for executor-step based tokenization)
    data_config: LMMixtureDatasetConfig | None = None
    # Optional: YAML-like mixture support (mirrors LMMixtureDatasetConfig)
    mixture_configs: dict[str, dict] | None = None
    mixture_train_weights: dict[str, float] | list[tuple[int, dict[str, float]]] | None = None
    mixture_cache_dir: str | None = None
    mixture_block_size: int | None = None
    zero_loss_datasets: list[str] | None = None
    stop_strategy: str | None = None
    exp_name: str = "datagrads"

    # Model (gpt2)
    hidden_dim: int = 768 # 2048 # 768
    num_heads: int = 12 # 16 # 12
    num_layers: int = 12 # 16 # 12
    seq_len: int = 128 # 128
    gradient_checkpointing: bool = True
    layer_norm_epsilon: float = 1e-2
    qk_norm: bool = True
    mp: jmp.Policy | None = None

    # External model/tokenizer (optional overrides)
    model_config: object | None = None
    tokenizer: str | None = None

    # HF initialization for pretrained models (e.g., LLaMA)
    # When set, we initialize weights from a HF repo id or a local HF snapshot path.
    # If provided and no explicit model_config is set, we will default to a LlamaConfig
    # to ensure Levanter uses the correct model type.
    initialize_from_hf: str | bool = False
    use_hf_model_config: bool = False

    # Trainer
    train_batch_size: int = 2048
    num_train_steps: int = 4
    per_device_parallelism: int = 4
    per_device_eval_parallelism: int = 4
    model_axis_size: int = 1
    fsdp_axis: str | list[str] | None = "embed"
    replica_ici_axis_size: int = 1
    replica_dcn_axis_size: int = 1
    max_eval_batches: int | None = 1
    # Commented out for bisecting on older commits
    # profiler_tpu_trace_mode: str | None = "TRACE_COMPUTE_AND_SYNC"
    profiler: bool = False
    profiler_start_step: int = 2
    profiler_num_steps: int = 2
    # Commented out for bisecting on older commits
    # backward_profiler: bool = False
    # backward_profiler_start_step: int = 4
    # backward_profiler_num_steps: int = 2
    log_dir: str = "jax_profile_traces"
    jax_compilation_cache_dir: str = "jax_compilation_cache"
    # Commented out for bisecting on older commits without metagrad support
    metagrad_checkpoint_frequency: int = 20
    metagrad_segment_size: int = 10
    steps_per_eval: int = 1000

    # Optimizer
    lr_schedule: str = "linear"
    # Commented out for bisecting on older commits
    # schedule_steps: int = 12000
    lr: float = 1e-4
    weight_decay: float = 0.1
    warmup: float = 0.1
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    epsilon_root: float = 1e-7
    beta1: float = 0.9
    beta2: float = 0.95
    # Tracker
    wandb_project_name: str = "levanter"
    run_name: str | None = None

    # Checkpointing
    checkpoint_base_path: str | None = None
    checkpoint_save_interval_minutes: int = 5
    checkpoint_keep_every: int = 1  # Must be 1 so checkpointer doesn't conflict with metagrad's forward_ckpt_steps gating
    append_run_id_to_base_path: bool = True

    # Hardware
    tpu_type: str = "v4-64"
    slice_count: int = 1

    # Misc
    seed: int = 0
    data_seed: int | None = None
    nametag: str = ""
    allow_out_of_region: tuple[str, ...] = ()
    # Commented out for bisecting on older commits
    # train_only: bool = False
    initialize_from: str | None = None
    initialize_from_checkpoint_path: str | None = None
    # Commented out for bisecting on older commits
    # cfx_seed: int | None = None

    # Eval harness (optional)
    eval_task_spec: list[str | dict] | None = None
    eval_max_examples: int | None = None
    eval_max_eval_length: int | None = None
    eval_log_samples: bool = False
    eval_bootstrap_iters: int = 0
    eval_apply_chat_template: bool = False
    eval_harness_steps: int = 10000
    eval_harness_tasks: list[EvalTaskConfig] | None = None

    # Debugging / reproducibility helpers
    save_input_ids: bool = False
    record_only: bool = False

    def __post_init__(self):
        if self.train_urls is None:
            self.train_urls = [
                f"gs://{self.gcs_bucket}/raw/sam/fineweb_edu_10bt_shuffled/fineweb_edu_10bt.chunk.00.jsonl",
            ]
        if self.validation_urls is None:
            self.validation_urls = [
                f"gs://{self.gcs_bucket}/raw/sam/fineweb_edu_10bt_shuffled/fineweb_edu_10bt.val.jsonl",
            ]
        if self.cache_dir is None:
            self.cache_dir = f"gs://{self.gcs_bucket}/raw/sam/fineweb_edu_10bt_shuffled/cache_new"
        if self.checkpoint_base_path is None:
            self.checkpoint_base_path = f"gs://{self.gcs_bucket}/checkpoints/sam/"

    def build_name(self) -> str:
        lr_str = f"{self.lr:.1e}"
        name = f"{self.exp_name}-bs{self.train_batch_size}-s{self.num_train_steps}-lr{lr_str}-wd{self.weight_decay:.2f}-gc{self.max_grad_norm}-er{self.epsilon_root:.2e}{self.nametag}"
        return name[:100]

    def build_data_config(self) -> UrlSingleDatasetLMConfig | LMMixtureDatasetConfig:
        # If a pre-built data config is provided, use it directly (preferred for executor-step based tokenization)
        if self.data_config is not None:
            return self.data_config

        if self.use_comma_main_mixture:
            # Allow overriding tokenizer when using the comma mixture
            if self.tokenizer is not None:
                return comma_main_mixture(tokenizer=self.tokenizer)
            return comma_main_mixture()
        # If a simple HF dataset id is provided, construct a single-entry mixture using HF source
        if self.hf_dataset_id is not None:
            # Use a unique cache directory per HF dataset id to avoid stale/colliding caches
            safe_id = self.hf_dataset_id.replace("/", "__")
            base_cache = self.mixture_cache_dir or f"gs://{self.gcs_bucket}/sam/data_cache"
            dataset_cache_dir = os.path.join(base_cache, "hf", safe_id)

            return UrlSingleDatasetLMConfig(
                tokenizer=self.tokenizer or "gpt2",
                id=self.hf_dataset_id,
                cache_dir=dataset_cache_dir,
                shuffle=False,
            )
            '''
            typed_configs = {
                self.hf_dataset_id: HfDatasetSourceConfig(
                    id=self.hf_dataset_id,
                    stream=True,
                    cache_dir=dataset_cache_dir,
                )
            }
            return LMMixtureDatasetConfig(
                tokenizer=self.tokenizer or "gpt2",
                cache_dir=base_cache,
                configs=typed_configs,
                train_weights={self.hf_dataset_id: 1.0},
                mixture_block_size=self.mixture_block_size or 2048,
                #shuffle=False
            )
            '''

        # If mixture_configs provided, construct an LMMixtureDatasetConfig that mirrors YAML structure
        if self.mixture_configs is not None:
            typed_configs: dict[str, UrlDatasetSourceConfig | HfDatasetSourceConfig] = {}
            for name, cfg in self.mixture_configs.items():
                if "id" in cfg:
                    typed_configs[name] = HfDatasetSourceConfig(
                        id=cfg["id"],
                        name=cfg.get("name"),
                        stream=cfg.get("stream", True),
                        cache_dir=cfg.get("cache_dir"),
                    )
                else:
                    typed_configs[name] = UrlDatasetSourceConfig(
                        train_urls=cfg.get("train_urls", []),
                        validation_urls=cfg.get("validation_urls", []),
                        cache_dir=cfg.get("cache_dir"),
                        format=cfg.get("format"),
                        # TODO: add type
                    )

            weights = self.mixture_train_weights
            if weights is None:
                # default to equal weights for provided components
                n = max(len(typed_configs), 1)
                eq_w = 1.0 / n
                weights = {k: eq_w for k in typed_configs.keys()}

            return LMMixtureDatasetConfig(
                tokenizer=self.tokenizer or "gpt2",
                cache_dir=self.mixture_cache_dir or self.cache_dir,
                configs=typed_configs,
                train_weights=weights,  # type: ignore[arg-type]
                mixture_block_size=self.mixture_block_size or 2048,
                zero_loss_datasets=self.zero_loss_datasets,
                stop_strategy=self.stop_strategy or LMMixtureDatasetConfig.stop_strategy,  # preserve default if None
            )
        print('URLSingleDatasetLMConfig')
        return UrlSingleDatasetLMConfig(
            tokenizer=self.tokenizer, # or "gpt2",
            train_urls=self.train_urls,
            validation_urls=self.validation_urls,
            cache_dir=self.cache_dir,
            shuffle=False,
        )

    def build_trainer_config(self) -> TrainerConfig:
        if self.fsdp_axis is None:
            param_mapping = {}
        elif isinstance(self.fsdp_axis, list):
            param_mapping = {axis: "data" for axis in self.fsdp_axis}
        else:
            param_mapping = {self.fsdp_axis: "data"}

        return TrainerConfig(
            seed=self.seed,
            mp=self.mp or jmp.get_policy("p=f32,c=f32"),
            # Partitioning and parallelism
            mesh=MeshConfig(
                axes={
                    "model": self.model_axis_size,
                    "replica": self.replica_ici_axis_size,
                },
                dcn_axes={"replica_dcn": self.replica_dcn_axis_size},
                param_mapping=param_mapping,
            ),

            # Batching
            train_batch_size=self.train_batch_size,
            per_device_parallelism=self.per_device_parallelism,
            per_device_eval_parallelism=self.per_device_eval_parallelism,

            # Duration & eval
            num_train_steps=self.num_train_steps,
            max_eval_batches=self.max_eval_batches,

            # Profiler & logs
            profiler=self.profiler,
            profiler_start_step=self.profiler_start_step,
            profiler_num_steps=self.profiler_num_steps,
            # Commented out for bisecting on older commits
            # profiler_tpu_trace_mode=self.profiler_tpu_trace_mode,
            # backward_profiler=self.backward_profiler,
            # backward_profiler_start_step=self.backward_profiler_start_step,
            # backward_profiler_num_steps=self.backward_profiler_num_steps,
            log_dir=Path(self.log_dir),

            # Tracker
            tracker=WandbConfig(project=self.wandb_project_name, name=self.run_name, save_xla_dumps=False),

            # Watch
            watch=WatchConfig(
                watch_targets=["grads", "params"], #, "adam_mu", "adam_nu"],
                include_norms=True,
                include_per_parameter_norms=True,
                include_histograms=False,
                split_scan_layers=True,
                interval=10,
            ),

            # Checkpointer
            checkpointer=CheckpointerConfig(
                base_path=self.checkpoint_base_path,
                save_interval=timedelta(minutes=self.checkpoint_save_interval_minutes),
                keep=[dict(every=self.checkpoint_keep_every)],
                append_run_id_to_base_path=self.append_run_id_to_base_path,
            ),

            # Misc
            # Commented out for bisecting on older commits without metagrad support
            #metagrad_checkpoint_frequency=self.metagrad_checkpoint_frequency,
            metagrad_segment_size=self.metagrad_segment_size,
            jax_compilation_cache_dir=self.jax_compilation_cache_dir,
            steps_per_eval=self.steps_per_eval,
            initialize_from=self.initialize_from,
        )

    def build_optimizer_config(self) -> AdamConfig:
        return AdamConfig(
            beta1=self.beta1,
            beta2=self.beta2,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            lr_schedule=self.lr_schedule,
            warmup=self.warmup,
            # Commented out for bisecting on older commits
            # schedule_steps=self.schedule_steps,
            min_lr_ratio=self.min_lr_ratio,
            #epsilon_root=self.epsilon_root,
        )

    def _build_model_config(self) -> Gpt2Config | LlamaConfig | object:
        # If provided, use the external model config directly (e.g., LlamaConfig)
        if self.model_config is not None:
            return self.model_config
        # If initializing from HF and the user did not provide a model_config,
        # default to a LlamaConfig so the Levanter model type matches HF LLaMA.
        if self.initialize_from_hf:
            return LlamaConfig()
        # Default to GPT-2 config built from DatagradsConfig fields
        return Gpt2Config(
            max_seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            gradient_checkpointing=self.gradient_checkpointing,
            layer_norm_epsilon=self.layer_norm_epsilon,
            # Commented out for bisecting on older commits
            #qk_norm=self.qk_norm,
        )

    def build_pod_config(self) -> ResourceConfig:
        return ResourceConfig.with_tpu(tpu_type=self.tpu_type, slice_count=self.slice_count)

    def build_train_lm_config(self) -> TrainLmConfig:
        eval_cfg: LmEvalHarnessConfig | None = None
        if self.eval_harness_tasks is not None:
            eval_cfg = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(self.eval_harness_tasks))
        elif self.eval_task_spec is not None:
            eval_cfg = LmEvalHarnessConfig(
                task_spec=self.eval_task_spec,  # type: ignore[arg-type]
                max_examples=self.eval_max_examples,
                max_length=self.eval_max_eval_length,
                log_samples=self.eval_log_samples,
                bootstrap_iters=self.eval_bootstrap_iters,
                apply_chat_template=self.eval_apply_chat_template,
            )

        return TrainLmConfig(
            data=self.build_data_config(),
            trainer=self.build_trainer_config(),
            model=self._build_model_config(),
            optimizer=self.build_optimizer_config(),
            z_loss_weight=0.0,
            data_seed=self.data_seed,
            out_dir=f"gs://{self.gcs_bucket}/sam/results/{self.exp_name}/{self.run_name}/",
            eval_harness=eval_cfg,
            eval_harness_steps=self.eval_harness_steps,
            # Commented out for bisecting on older commits
            # train_only=self.train_only,
            initialize_from_checkpoint_path=self.initialize_from_checkpoint_path,
            initialize_from_hf=self.initialize_from_hf,
            use_hf_model_config=self.use_hf_model_config,
            # Commented out for bisecting on older commits
            # cfx_seed=self.cfx_seed,
            #save_input_ids=self.save_input_ids,
            #record_only=self.record_only,
        )

    def build_train_lm_on_pod_config(self) -> TrainLmOnPodConfig:
        return TrainLmOnPodConfig(
            train_config=self.build_train_lm_config(),
            resources=self.build_pod_config(),
            output_path=this_output_path(),
            allow_out_of_region=self.allow_out_of_region,
        )

    def __hash__(self):
        return hash(self.build_name())

    def __eq__(self, other):
        if not isinstance(other, DatagradsConfig):
            return False
        return hash(self) == hash(other)


def datagrads_train_step(datagrads_config: DatagradsConfig) -> ExecutorStep:
    train_lm_on_pod_config = datagrads_config.build_train_lm_on_pod_config()

    executor_step_name = os.path.join("checkpoints", "data_grads", datagrads_config.build_name())

    return ExecutorStep(
        name=executor_step_name,
        override_output_path=executor_step_name,
        fn=run_levanter_train_lm,
        description=(
            f"Train LM for {datagrads_config.num_train_steps} steps, "
            f"bs={datagrads_config.train_batch_size}, lr={datagrads_config.lr}, wd={datagrads_config.weight_decay}."
        ),
        config=train_lm_on_pod_config,
    )
