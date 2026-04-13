# Tunix Deep-Dive Notes

This document is a code-referenced orientation guide for future agents working with `google/tunix`.

Upstream snapshot used for this summary:

- Repo: `https://github.com/google/tunix`
- Branch: `main`
- Commit: `b3de6ec3fdb9831cb7e5f8f1f6ff7f6da87dc819`
- Commit date: `2026-03-09 20:31:11 -0700`
- Commit subject: `Merge pull request #1214 from google:atwigg/simple_math_reward`
- Local clone used during analysis: `/Users/ahmed/code/.cache/tunix`

Package snapshot:

- Package name: `google-tunix`
- Version in `pyproject.toml`: `0.1.6`
- Status from README: "V2 Release", still under active development

## 1. What Tunix is

Tunix is a JAX/Flax NNX post-training library for LLMs. The center of gravity is:

- supervised fine-tuning (full finetune or LoRA/QLoRA),
- preference optimization (DPO/ORPO),
- reinforcement learning (GRPO, PPO, DAPO, Dr.GRPO),
- agentic RL with multi-turn tool use,
- rollout backends ranging from a native JAX sampler to vLLM and SGLang-JAX.

The best one-line mental model is:

> Tunix is a reusable training/runtime library first, with a thin CLI and example-script layer on top.

The code exports this library surface directly from `tunix/__init__.py` via `PeftTrainer`, `DPOTrainer`, `GRPOLearner`, `PPOLearner`, `RLCluster`, `Sampler`, `AutoModel`-adjacent pieces, checkpointing, and perf APIs (`tunix/__init__.py`).

Primary public package entrypoints:

- `tunix/sft/peft_trainer.py`
- `tunix/sft/dpo/dpo_trainer.py`
- `tunix/distillation/distillation_trainer.py`
- `tunix/rl/rl_cluster.py`
- `tunix/rl/rl_learner.py`
- `tunix/rl/grpo/grpo_learner.py`
- `tunix/rl/ppo/ppo_learner.py`
- `tunix/generate/sampler.py`
- `tunix/models/automodel.py`

## 2. Repository map

Top-level directories worth knowing:

| Path | What it is |
| --- | --- |
| `tunix/` | The actual library |
| `tunix/cli/` | CLI config system and Python entrypoints |
| `examples/` | User-facing scripts and notebooks |
| `docs/` | Architecture, rollout, performance, and usage docs |
| `tests/` | Best executable spec for intended behavior |
| `scripts/` | Larger demos and setup helpers |

Useful internal subpackages:

| Path | Role |
| --- | --- |
| `tunix/sft/` | Base trainer, checkpointing, metrics, profiling |
| `tunix/sft/dpo/` | DPO and ORPO |
| `tunix/distillation/` | Teacher-student distillation |
| `tunix/rl/` | RL cluster, learners, rollout, inference, rewards, resharding |
| `tunix/rl/agentic/` | Agentic RL building blocks |
| `tunix/rl/experimental/` | Experimental async/agentic learners |
| `tunix/generate/` | Tokenizer adapter, native sampler, backend adapters |
| `tunix/models/` | Model families, naming, loading, safetensors helpers |
| `tunix/processors/` | Image preprocessing for VLM workflows |
| `tunix/perf/` | Perf tracing and export |
| `tunix/utils/` | Environment and misc utilities |

Docs that are actually useful:

- `README.md`
- `docs/design.md`
- `docs/quickstart.md`
- `docs/launching.md`
- `docs/rollout.md`
- `docs/performance.md`
- `docs/reliability.md`
- `docs/models.md`
- `docs/agentic_rl.md`

Important caveat: the docs are good for orientation, but code is the source of truth. Some docs still describe pieces as WIP even where the codepath now exists.

## 3. Core stack and recurring design choices

Tunix is built around a few consistent choices:

- Models are Flax NNX modules, not PyTorch modules.
- Distribution is expressed via JAX meshes and sharding.
- Optimizers come from Optax.
- Checkpointing comes from Orbax.
- Metrics/logging uses `metrax`.
- LoRA/QLoRA uses `qwix`.
- Rollout can be native JAX, vLLM, or SGLang-JAX.

You see this pattern across the repo:

1. construct model and tokenizer,
2. construct mesh,
3. construct optimizer/training config,
4. wrap in a trainer or RL cluster,
5. feed datasets / prompts,
6. checkpoint and log everything.

## 4. Installation and runtime assumptions

The package metadata in `pyproject.toml` says:

- Python `>=3.11`
- base dependency set includes `flax>=0.11.1`, `grain`, `google-metrax`, `qwix`, `transformers<=4.57.1`, `sentencepiece`, `omegaconf`, `tenacity`, `kagglehub`, `huggingface_hub`
- optional `prod` dependency installs `jax[tpu]>=0.6.0,!=0.7.2`

Operational notes from code/docs:

- TPU is the primary target.
- GPU and CPU are supported, but several examples and docs are TPU-centric.
- `transformers==4.57.2` is explicitly avoided in `pyproject.toml`.
- vLLM TPU support and SGLang-JAX are optional manual installs (`docs/quickstart.md`, `docs/rollout.md`).

### 4.1 Actual invocation model

There is no packaged `tunix` console script in `pyproject.toml` in this snapshot.

The real OSS entrypoints are Python modules:

- `python3 -m tunix.cli.peft_main ...`
- `python3 -m tunix.cli.grpo_main ...`

That is exactly how the example shell scripts invoke Tunix:

- `examples/sft/mtnt/run_qwen2.5_0.5b.sh`
- `examples/rl/grpo/gsm8k/run_llama3.2_1b.sh`

### 4.2 Config loading details that matter operationally

`HyperParameters` calls `dotenv.load_dotenv()` on startup, so `.env`-based local secrets are part of the intended workflow (`tunix/cli/config.py:127`).

It also special-cases the literal first argument `base_config.yaml` by resolving it relative to `tunix/cli/` (`tunix/cli/config.py:136-141`). This means:

- example scripts that pass plain `base_config.yaml` work because of a compatibility hack,
- example scripts that pass `tunix/cli/base_config.yaml` work because they are explicit,
- repo-root execution is the safest default if you are following examples verbatim.

### 4.3 Source and tokenizer constraints are stricter than the marketing docs

The config validation layer enforces several runtime rules:

- Hugging Face tokenizers require `HF_TOKEN` (`tunix/cli/config.py:242-256`).
- `model_source in {"kaggle", "huggingface"}` requires `intermediate_ckpt_dir` (`tunix/cli/config.py:295-304`).
- supported checkpoint sources depend on the model family:
  - Gemma / Gemma2: `kaggle` or `internal`
  - Gemma3: `gcs` or `internal`
  - other OSS families default to `huggingface` or `internal`

See `tunix/cli/config.py:307-340`.

## 5. Model loading, naming, and tokenizer handling

### 5.1 Naming model families

The naming logic is centralized in `tunix/models/naming.py`.

Important outputs of `ModelNaming`:

- `model_family`
- `model_version`
- `model_config_category`
- `model_config_id`

This is how Tunix maps strings like:

- `meta-llama/Llama-3.1-8B`
- `gemma2_2b_it`
- `qwen3-14b`

into the correct Python module path and `ModelConfig` factory.

Key references:

- `tunix/models/naming.py:40`
- `tunix/models/naming.py:114`
- `tunix/models/naming.py:225`
- `tunix/models/naming.py:241`

### 5.2 AutoModel

`AutoModel.from_pretrained(...)` is the main unified model loader. It dynamically:

- infers the model family,
- imports the right `model.py` / `params.py` / `params_safetensors.py`,
- downloads or resolves the model source,
- returns `(model, model_path)`.

Key references:

- `tunix/models/automodel.py:49`
- `tunix/models/automodel.py:69`
- `tunix/models/automodel.py:273`
- `tunix/models/automodel.py:353`

Supported model sources:

- `huggingface`
- `kaggle`
- `gcs`
- `internal` (not OSS-supported)

Important source-specific behavior:

- Kaggle Gemma/Gemma2 may require NNX conversion with an intermediate Orbax checkpoint cache (`tunix/models/automodel.py:145`).
- Gemma3 has separate checkpoint loading logic (`tunix/models/automodel.py:252`).
- safetensors loading is dynamically delegated by family (`tunix/models/automodel.py:314`).

### 5.3 Tokenizers

Tokenizer normalization lives in `tunix/generate/tokenizer_adapter.py`.

`TokenizerAdapter` supports:

- SentencePiece
- HuggingFace tokenizers
- custom tokenizers implementing `encode`, `decode`, `bos_id`, `eos_id`, `pad_id`

Important behavior:

- HF tokenizers with missing `pad_token_id` get `eos_token` reused as padding.
- Chat templating is supported through HF `apply_chat_template`, with a Gemma-style fallback for SP or custom tokenizers.

Key references:

- `tunix/generate/tokenizer_adapter.py:33`
- `tunix/generate/tokenizer_adapter.py:89`
- `tunix/generate/tokenizer_adapter.py:138`
- `tunix/generate/tokenizer_adapter.py:212`

## 6. SFT is the base abstraction

The most important architectural fact in Tunix is:

> `PeftTrainer` is the base training loop that almost everything else builds on.

Key references:

- `tunix/sft/peft_trainer.py:52`
- `tunix/sft/peft_trainer.py:170`
- `tunix/sft/peft_trainer.py:593`

### 6.1 `TrainingConfig`

`TrainingConfig` controls:

- eval cadence,
- max steps,
- gradient accumulation,
- checkpointing,
- metrics logging,
- profiler options,
- perf metrics options,
- input sharding axis,
- max inflight scheduled computations.

See `tunix/sft/peft_trainer.py:52`.

### 6.2 What `PeftTrainer` really does

`PeftTrainer`:

- wraps a model plus Optax optimizer in `nnx.Optimizer`,
- automatically detects LoRA and restricts updates to `nnx.LoRAParam` if present,
- owns the default loss, checkpoint manager, metrics logger, and progress bar,
- restores from the latest checkpoint on startup,
- JITs train and eval step functions,
- buffers metrics and writes them one step behind to overlap compute and logging.

Key points in code:

- LoRA detection and optimizer wiring: `tunix/sft/peft_trainer.py:200-210`
- checkpoint restore on init: `tunix/sft/peft_trainer.py:239-248`
- JIT setup: `tunix/sft/peft_trainer.py:382-421`
- main train loop: `tunix/sft/peft_trainer.py:593-734`
- final shutdown: `tunix/sft/peft_trainer.py:760-829`
- default causal LM loss: `tunix/sft/peft_trainer.py:831-860`

### 6.3 Input contract

Base SFT expects `TrainingInput`:

- `input_tokens`
- `input_mask`
- optional `images`

See `tunix/sft/peft_trainer.py:94`.

The trainer then expects a `gen_model_input_fn` that turns those into model-call kwargs such as:

- `positions`
- `attention_mask`
- sometimes `images`

This is how the same trainer is reused across language-only and vision-language models.

### 6.4 Checkpointing and metrics

Checkpointing is centralized in `tunix/sft/checkpoint_manager.py`.

Important behavior:

- disabled if `checkpoint_root_directory` is `None`
- saves model params plus optimizer state
- can save or restore only LoRA params
- uses different Orbax handler settings for Pathways

Key references:

- `tunix/sft/checkpoint_manager.py:34`
- `tunix/sft/checkpoint_manager.py:83`
- `tunix/sft/checkpoint_manager.py:137`

Metrics logging is centralized in `tunix/sft/metrics_logger.py`.

Important behavior:

- default backends are TensorBoard and WandB outside internal envs
- internal env uses CLU if available
- logging goes through `jax.monitoring`

Key references:

- `tunix/sft/metrics_logger.py:23`
- `tunix/sft/metrics_logger.py:33`
- `tunix/sft/metrics_logger.py:81`

## 7. DPO and ORPO are a thin specialization over SFT

`DPOTrainer` subclasses `PeftTrainer` rather than inventing a new training framework.

Key references:

- `tunix/sft/dpo/dpo_trainer.py:90`
- `tunix/sft/dpo/dpo_trainer.py:131`

Important behavior:

- can accept raw string triplets via `DataInput(prompts, chosen_responses, rejected_responses)`
- or pre-tokenized `TrainingInput`
- concatenates chosen and rejected examples into one batch
- computes reference logprobs if a reference model is provided
- supports both `algorithm="dpo"` and `algorithm="orpo"`

Key internal steps:

- `compute_logps(...)` helper: `tunix/sft/dpo/dpo_trainer.py:106`
- input preprocessing and packing: `tunix/sft/dpo/dpo_trainer.py:238-318`
- aux metrics logging for reward/logprob diagnostics: `tunix/sft/dpo/dpo_trainer.py:226-237`, `320-340`

Mental model:

- SFT base loop stays the same.
- DPO swaps in a special loss and special input preparation.

## 8. Distillation is also built on `PeftTrainer`

`DistillationTrainer` is another `PeftTrainer` subclass.

Key references:

- `tunix/distillation/distillation_trainer.py:29`
- `tunix/distillation/distillation_trainer.py:41`

It delegates most behavior to a strategy object from `tunix/distillation/strategies/`.

Important design:

- the strategy can preprocess student/teacher models,
- produce teacher outputs during train mode,
- define train and eval loss behavior,
- postprocess models on close.

Representative strategy base and implementations:

- `tunix/distillation/strategies/base_strategy.py`
- `tunix/distillation/strategies/logit.py`
- `tunix/distillation/strategies/feature_pooling.py`
- `tunix/distillation/strategies/feature_projection.py`
- `tunix/distillation/strategies/attention.py`

Good executable spec:

- `tests/distillation/`

## 9. RL architecture: `RLCluster` + `RLLearner`

The RL subsystem has two major layers:

- `RLCluster`: owns models, meshes, rollout/inference/trainer plumbing
- `RLLearner`: owns the algorithm loop and how to convert prompts into `TrainExample`s

Key references:

- `tunix/rl/rl_cluster.py:82`
- `tunix/rl/rl_cluster.py:142`
- `tunix/rl/rl_cluster.py:185`
- `tunix/rl/rl_learner.py:51`

### 9.1 `RLTrainingConfig`

`RLTrainingConfig` extends SFT training config with:

- actor and critic optimizers,
- mini-batch size,
- train micro-batch size,
- rollout micro-batch size,
- logprob micro-batch size

It also derives gradient accumulation automatically from `mini_batch_size // train_micro_batch_size`.

See `tunix/rl/rl_cluster.py:82-140`.

### 9.2 `ClusterConfig`

`ClusterConfig` is the runtime topology object. It specifies:

- `role_to_mesh`
- optional logical axis rules
- rollout engine
- CPU offloading
- RL training config
- rollout config

See `tunix/rl/rl_cluster.py:142-183`.

This is where collocated vs disaggregated RL is expressed.

### 9.3 What `RLCluster` owns

`RLCluster` is the central execution object for RL.

It owns:

- actor model
- rollout model
- optional critic/reference/reward models
- rollout engine instance
- inference worker for reference/critic/reward models
- actor trainer
- optional critic trainer
- perf tracers
- metric buffering for RL

The cluster lifecycle is implemented in `_init_cluster()`:

- initialize rollout engine,
- initialize perf tracers,
- initialize inference worker,
- initialize actor/critic trainers,
- optionally offload inactive models to CPU.

Key references:

- rollout engine init: `tunix/rl/rl_cluster.py:354-466`
- perf tracer init: `tunix/rl/rl_cluster.py:474-497`
- inference worker init: `tunix/rl/rl_cluster.py:498-509`
- actor/critic trainer init: `tunix/rl/rl_cluster.py:510-557`

### 9.4 Backbone sharing and offloading

One of the more important Tunix performance ideas is careful weight sharing.

`RLCluster` can detect and exploit:

- actor <-> rollout sharing if they are on the same mesh,
- actor <-> reference backbone sharing when LoRA is used and meshes line up.

See:

- `tunix/rl/rl_cluster.py:261-291`
- `tunix/rl/utils.py:94`
- `tunix/rl/utils.py:107`

CPU offloading logic:

- load model from pinned host to device only when needed,
- run workload,
- move it back to pinned host afterward.

See:

- `tunix/rl/rl_cluster.py:607-619`
- `docs/performance.md`

### 9.5 Generation and auxiliary inference

`RLCluster` exposes the reusable runtime operations:

- `generate(...)`
- `get_ref_per_token_logps(...)`
- `get_old_per_token_logps(...)`
- `get_values(...)`
- `get_rewards(...)`
- `sync_weights()`

Key references:

- `tunix/rl/rl_cluster.py:819`
- `tunix/rl/rl_cluster.py:910`
- `tunix/rl/rl_cluster.py:962`
- `tunix/rl/rl_cluster.py:1000`
- `tunix/rl/rl_cluster.py:1020`
- `tunix/rl/rl_cluster.py:1037`

`InferenceWorker` is intentionally tiny:

- reward model calls `common.compute_score`
- reference model calls `common.compute_per_token_logps`
- critic model also uses `compute_score`

See `tunix/rl/inference/inference_worker.py`.

### 9.7 Standard RL runtime flow, end to end

The hottest RL path is easier to understand as a concrete call graph:

1. The launcher builds `RolloutConfig`, `RLTrainingConfig`, `ClusterConfig`, `RLCluster`, then a learner (`tunix/cli/grpo_main.py`, `scripts/grpo_demo_llama3_qwen2.py`).
2. `RLLearner._prepare_data(...)` accumulates training micro-batches up to a service target batch size, repeats prompts when the algorithm needs grouped generations, and injects `trajectory_ids` before calling `_generate_and_compute_advantage(...)` (`tunix/rl/rl_learner.py:250-420`).
3. `RLCluster.generate(...)` optionally chat-templates prompts, chunks them by rollout micro-batch size, calls the selected rollout backend, and concatenates a single `RolloutOutput` (`tunix/rl/rl_cluster.py:819+`).
4. GRPO pads completions to `max_tokens_to_generate`, builds `prompt_mask` and `completion_mask`, optionally queries reference and old-policy logprobs, computes sequence rewards, converts grouped rewards to advantages, and returns a `TrainExample` (`tunix/rl/grpo/grpo_learner.py:195-343`).
5. PPO follows the same rollout path but additionally queries values and rewards, applies KL shaping, runs GAE, and returns token-level advantages and returns (`tunix/rl/ppo/ppo_learner.py`, `tunix/rl/ppo/ppo_helpers.py`).
6. The train loop drains queued `TrainExample`s, updates actor and critic trainers, then syncs actor weights back into the rollout engine at the appropriate boundaries (`tunix/rl/rl_learner.py:663-746`, `tunix/rl/rl_cluster.py:1000+`).

Shape intuition that matters when debugging:

- standard GRPO advantages are grouped/sequence-level,
- PPO advantages and returns are token-level,
- `trajectory_ids` are explicit strings of the form `{row_offset}_{group_offset}` in GRPO (`tunix/rl/grpo/grpo_learner.py:344-369`).

### 9.6 RL metric buffering

RL metrics are buffered separately from SFT metric logging because rollouts and async pipelines can produce metrics before the corresponding global step closes.

Important functions:

- `buffer_metrics(...)`: synchronous buffering
- `buffer_metrics_async(...)`: step-aware async buffering

See:

- `tunix/rl/rl_cluster.py:710`
- `tunix/rl/rl_cluster.py:754`

## 10. `RLLearner` is the generic RL loop

`RLLearner` defines the common RL algorithm skeleton.

Key references:

- `tunix/rl/rl_learner.py:51`
- `tunix/rl/rl_learner.py:250`
- `tunix/rl/rl_learner.py:512`

Its responsibilities:

- construct a reward manager from the function registry,
- coordinate async or sync rollout/data loading,
- merge micro-batches for efficient service calls,
- enqueue processed `TrainExample`s,
- drive actor and critic trainer updates,
- trigger weight sync between actor and rollout engine.

The most important internal flow is:

1. create micro-batch iterator if needed,
2. accumulate micro-batches until reaching the least-common-multiple target batch size,
3. call `_generate_and_compute_advantage(...)` once on the merged batch,
4. split results back to training micro-batches,
5. enqueue into queues,
6. train actor/critic,
7. sync rollout weights,
8. increment global steps.

Code references for that flow:

- `_prepare_data`: `tunix/rl/rl_learner.py:250-469`
- `train`: `tunix/rl/rl_learner.py:512-613`
- `_run_all_micro_batch_steps`: `tunix/rl/rl_learner.py:663-746`

Two conceptual counters matter:

- `_iter_steps`: micro-batch progress
- `global_steps` / trainer step: parameter-update progress

## 11. RL algorithms

### 11.1 Shared registry mechanism

Tunix uses a function registry for:

- policy losses,
- advantage estimators,
- reward managers.

See `tunix/rl/function_registry.py`.

This matters because many algorithms are configured by string names rather than hard-coded branching.

### 11.2 Algorithm config base

`AlgorithmConfig` is the small common base:

- `algo_variant`
- `advantage_estimator`
- `policy_loss_fn`
- `reward_manager`

See `tunix/rl/algorithm_config.py`.

### 11.3 GRPO

`GRPOConfig` defines:

- `num_generations`
- `num_iterations`
- `beta`
- `epsilon`
- optional `loss_algo` (`grpo` vs `gspo-token`)

See `tunix/rl/grpo/grpo_learner.py:44`.

`GRPOLearner`:

- configures actor trainer with the GRPO policy loss,
- generates multiple completions per prompt,
- optionally computes reference logprobs and old actor logprobs,
- computes rewards and relative advantages,
- returns a GRPO `TrainExample`.

Key code:

- learner init: `tunix/rl/grpo/grpo_learner.py:122-193`
- rollout and advantage path: `tunix/rl/grpo/grpo_learner.py:195-343`

### 11.4 PPO

`PPOConfig` adds:

- `gamma`
- `gae_lambda`
- `beta`
- clip ranges for policy and value
- entropy coefficient
- KL method

See `tunix/rl/ppo/ppo_learner.py:47`.

`PPOLearner`:

- requires a critic model,
- requires exactly one reward source: reward functions or reward model,
- configures both actor and critic trainers,
- computes generation, reference logprobs, old policy logprobs, values, final rewards, and GAE-style returns/advantages.

Key code:

- validation and trainer wiring: `tunix/rl/ppo/ppo_learner.py:130-230`
- rollout/value/reward path: `tunix/rl/ppo/ppo_learner.py:232+`

### 11.5 DAPO and Dr.GRPO

These are implemented as GRPO variants:

- `tunix/rl/grpo/dapo_learner.py`
- `tunix/rl/grpo/drgrpo_learner.py`

They follow the same extension pattern described in `docs/algorithms.md`: config subclass plus learner subclass.

## 12. Rollout system

### 12.1 Interface

The rollout abstraction is `BaseRollout` in `tunix/rl/rollout/base_rollout.py`.

The contract is:

- `generate(...)`
- `get_per_token_logps(...)`
- `update_params(...)`
- `pad_id()`
- `eos_id()`
- `model()`

The shared result type is `RolloutOutput`.

Key references:

- `tunix/rl/rollout/base_rollout.py:41`
- `tunix/rl/rollout/base_rollout.py:67`
- `tunix/rl/rollout/base_rollout.py:234`

### 12.2 Native JAX rollout

`VanillaRollout` wraps the native `Sampler`.

Key references:

- `tunix/rl/rollout/vanilla_rollout.py`
- `tunix/generate/sampler.py`

The native sampler:

- wraps an NNX transformer,
- keeps graphdef and state separate to reduce compile size,
- owns prefill/decode JITs,
- supports greedy, top-p, and beam search,
- builds and maintains KV cache,
- can swap full params or just LoRA params.

Key references:

- sampler state: `tunix/generate/sampler.py:44`
- cache config: `tunix/generate/sampler.py:97`
- sampler init: `tunix/generate/sampler.py:176`
- state replacement logic: `tunix/generate/sampler.py:223-311`

### 12.3 vLLM rollout

`VllmRollout` builds a `VllmSampler` using a `MappingConfig`:

- `tunix/rl/rollout/vllm_rollout.py:27`
- `tunix/generate/vllm_sampler.py:45`
- `tunix/generate/vllm_sampler.py:106`

Key ideas:

- Tunix maps trainer weights into backend/HF names via `MappingConfig`.
- rollout weights are initialized and then synchronized in-memory from the trainer.
- rollout config carries a large number of vLLM-specific tuning knobs.

Good file pair to read together:

- `tunix/rl/rollout/vllm_rollout.py`
- `tunix/generate/vllm_sampler.py`

### 12.4 SGLang-JAX rollout

`SglangJaxRollout` is the parallel integration for SGLang-JAX:

- `tunix/rl/rollout/sglang_jax_rollout.py`
- `tunix/generate/sglang_jax_sampler.py`

Like vLLM it:

- builds a backend-specific mapping config,
- initializes backend runtime,
- loads weights in memory,
- updates weights from trainer state during RL.

### 12.5 Mapping config

`tunix/generate/mappings.py` is important for backend integrations.

It standardizes:

- `to_hf_mappings`
- `lora_to_hf_mappings`
- hook functions
- transpose keys

This is how Tunix can synchronize trainer weights into heterogeneous inference backends without hand-writing one-off logic everywhere.

Key reference:

- `tunix/generate/mappings.py:65`

### 12.6 Advanced rollout backends are more Python-first than CLI-first

The docs advertise `vllm` and `sglang_jax` as the higher-throughput backends, and the codepaths are real:

- `tunix/rl/rollout/vllm_rollout.py`
- `tunix/rl/rollout/sglang_jax_rollout.py`
- `tunix/generate/vllm_sampler.py`
- `tunix/generate/sglang_jax_sampler.py`

But the most complete usage examples are still Python scripts and notebooks, not the default CLI path:

- `scripts/grpo_demo_llama3_qwen2.py`
- `examples/deepscaler/train_deepscaler_nb.py`

`docs/rollout.md` still describes parts of the vLLM CLI path as WIP, which matches the fact that the richer backend knobs show up most clearly in script-level construction.

## 13. Resharding is a first-class optimization

A major performance/memory optimization lives in `tunix/rl/reshard.py`.

The core idea:

- avoid expensive replicated all-gathers when moving weights between different sharding layouts,
- optionally insert an intermediate sharding layout before the final target sharding.

Key code:

- intermediate sharding heuristic: `tunix/rl/reshard.py:69-220`
- experimental pre-reshard path: `tunix/rl/reshard.py:223+`

This is one of the places where Tunix is doing more than a naive JAX training wrapper.

## 14. Agentic RL

Agentic RL is real in the codebase, but much of it is still under `experimental/`.

High-level docs:

- `docs/agentic_rl.md`

Core pieces:

- agents: `tunix/rl/agentic/agents/`
- environments: `tunix/rl/agentic/environments/`
- tools: `tunix/rl/agentic/tools/`
- parsers: `tunix/rl/agentic/parser/`
- trajectory engine: `tunix/rl/agentic/trajectory/trajectory_collect_engine.py`
- async orchestrator: `tunix/rl/agentic/pipeline/rollout_orchestrator.py`
- grouping queue: `tunix/rl/agentic/queue_manager/group_queue_manager.py`
- learner base: `tunix/rl/experimental/agentic_rl_learner.py`
- GRPO-style learner: `tunix/rl/experimental/agentic_grpo_learner.py`

### 14.1 Agent/environment abstraction

The mental model is classical RL:

- agent maintains chat state and proposes an action,
- environment executes or evaluates the action,
- trajectory engine records the interaction,
- orchestrator parallelizes many episodes.

The docs summarize these pieces correctly; the code to read is:

- `tunix/rl/agentic/agents/model_agent.py`
- `tunix/rl/agentic/agents/tool_agent.py`
- `tunix/rl/agentic/environments/task_environment.py`
- `tunix/rl/agentic/environments/tool_environment.py`
- `tunix/rl/agentic/tools/tool_manager.py`

### 14.2 `TrajectoryCollectEngine`

This is the per-episode async engine.

It:

- resets env and agent,
- repeatedly calls `model_call`,
- lets the agent parse/update state,
- steps the environment,
- stops on `done`, timeout, max steps, or context limit,
- computes final reward and Monte Carlo returns,
- can emit several output formats (`Trajectory`, `Steps`, `Token`, `Conversation`).

Key reference:

- `tunix/rl/agentic/trajectory/trajectory_collect_engine.py:41`

The important training-oriented detail is its `"Token"` mode. In that mode the engine returns:

- `prompt_tokens`
- flattened `conversation_tokens`
- flattened `conversation_masks`
- `status`
- `trajectory_reward`
- `policy_version`
- `original_input`
- `group_id`

See `tunix/rl/agentic/trajectory/trajectory_collect_engine.py:222-248`.

### 14.3 `RolloutOrchestrator`

This is the concurrency layer for many agent/env pairs.

It:

- spins up many trajectory collectors,
- groups trajectories into ready batches via `GroupQueueManager`,
- exposes `yield_batches(...)`,
- uses `RolloutSyncLock` so weight sync does not race with rollout generation.

Key reference:

- `tunix/rl/agentic/pipeline/rollout_orchestrator.py:47`

Grouping is explicit rather than implicit. `GroupQueueManager` batches trajectories by `group_id` until a full group is ready, which is how grouped GRPO-style objectives are recovered in the agentic setting (`tunix/rl/agentic/queue_manager/group_queue_manager.py`).

### 14.4 `AgenticRLLearner`

This is the async learner base for agentic RL.

It differs from normal `RLLearner` in that rollout is no longer "batch prompt -> batch completion". Instead it:

- creates agent/environment pairs,
- runs async collection,
- tracks policy version,
- patches rollout config max response length from algorithm config,
- bridges async trajectory collection back into train examples.

Key reference:

- `tunix/rl/experimental/agentic_rl_learner.py:95`

Practical note: if you are working on agentic RL, read the tests. The behavior surface is much easier to understand through:

- `tests/rl/agentic/`
- `tests/rl/experimental/agentic_grpo_learner_test.py`

### 14.5 Token masking and policy-version semantics

Two agentic details are easy to miss and are important for correctness:

1. Conversation masking is role-based, not step-based. `agentic/utils.py` tokenizes each message and assigns mask `1` to assistant tokens and `0` to everything else, so the policy loss only trains on assistant spans even though the flattened conversation contains user, tool, and environment text (`tunix/rl/agentic/utils.py:133-192`).
2. Agentic training tracks `policy_version` explicitly. `AgenticRLLearner` stamps each environment/task with the current policy version before rollout, and the token-mode trajectory output carries that version forward into training examples (`tunix/rl/experimental/agentic_rl_learner.py:216-223`, `357-373`, `800+`; `tunix/rl/agentic/trajectory/trajectory_collect_engine.py:240-246`).

This is a notable difference from the non-agentic learner, where rollout staleness is mainly managed through rollout sync timing and optional old-policy logprob queries.

## 15. CLI system

The CLI is not a separate framework. It is a configuration and launch layer that instantiates the library objects above.

Key files:

- `tunix/cli/base_config.yaml`
- `tunix/cli/config.py`
- `tunix/cli/peft_main.py`
- `tunix/cli/grpo_main.py`

### 15.1 Config merge rules

`HyperParameters` in `tunix/cli/config.py` merges config from:

1. base YAML file,
2. optional `override_config_file=...`,
3. CLI `key=value` overrides.

Environment variables with prefix `T_` can also override top-level YAML keys.

Key references:

- `tunix/cli/config.py:108`
- `tunix/cli/config.py:684`

Important detail:

- the `T_` env mechanism applies to top-level YAML keys, not arbitrary nested keys.

### 15.2 What the CLI constructs

`config.py` provides reusable builders for:

- Optax optimizer creation, including schedules: `tunix/cli/config.py:457`
- JAX mesh creation from string config: `tunix/cli/config.py:533`
- conversion of nested config dicts into `TrainingConfig`/Orbax/metrics/profiler objects: `tunix/cli/config.py:624`
- reward function discovery: `tunix/cli/config.py:853`

Reward-function loading is noteworthy:

- a configured reward module path is imported,
- all functions defined in that module are collected as reward functions unless `verl_compatible` is enabled.

That behavior is easy to miss and matters when editing reward modules.

### 15.3 CLI entrypoints

`peft_main.py`:

- loads config,
- creates mesh/model/tokenizer,
- creates optimizer,
- builds a `PeftTrainer`,
- sets the standard causal LM input function,
- loads translation dataset,
- trains.

Key reference:

- `tunix/cli/peft_main.py:34-88`

`grpo_main.py`:

- loads config,
- builds rollout config and cluster config,
- builds actor/reference models,
- optionally applies LoRA to actor,
- creates `RLCluster`,
- loads dataset,
- runs `GrpoLearner.train(...)`.

Key reference:

- `tunix/cli/grpo_main.py:42-231`

Current CLI reality:

- SFT and GRPO are the main supported CLI workflows.
- DPO and distillation exist in-library, but there is no equivalent top-level CLI entrypoint in this repo snapshot.

## 16. Examples and how to run Tunix

### 16.1 The simplest "real" CLI examples

SFT example:

File:

- `examples/sft/mtnt/run_gemma2_2b.sh`

What it does:

- calls `python3 -m tunix.cli.peft_main`
- uses `tunix/cli/base_config.yaml`
- overrides with `examples/sft/mtnt/configs/gemma2_2b.yaml`
- sets local cache/checkpoint/tokenizer paths
- overrides `batch_size` and `training_config.max_steps`

GRPO example:

- `examples/rl/grpo/gsm8k/run_gemma2_2b.sh`

What it does:

- calls `python3 -m tunix.cli.grpo_main`
- uses the same config-merge scheme
- computes `max_steps` and warmup schedule values in shell
- points to a reward-function module

These two scripts are the best "intended CLI usage" examples in the repo.

If you want the most realistic Hugging Face-based OSS examples rather than the Gemma/Kaggle path, start here instead:

- SFT: `examples/sft/mtnt/run_qwen2.5_0.5b.sh`
- GRPO: `examples/rl/grpo/gsm8k/run_llama3.2_1b.sh`

### 16.2 Example configs

SFT example config:

- `examples/sft/mtnt/configs/gemma2_2b.yaml`

GRPO example config:

- `examples/rl/grpo/gsm8k/configs/gemma2_2b.yaml`

Read them together with `tunix/cli/base_config.yaml` to understand what is overridden per workflow.

### 16.3 Library-first demos

Good library usage examples:

- `examples/grpo_gemma.ipynb`
- `examples/dpo_gemma.ipynb`
- `examples/qlora_gemma.ipynb`
- `examples/logit_distillation.ipynb`
- `scripts/grpo_demo_llama3_qwen2.py`

The GRPO demo script is especially useful because it exposes:

- different rollout engines,
- different cluster setups,
- LoRA toggles,
- TPU topology decisions,
- explicit RL cluster creation without the CLI.

### 16.4 Agentic and specialized examples

Agentic:

- `examples/agentic/gemma_grpo_demo_nb.py`

DeepScaler / DeepSWE:

- `examples/deepscaler/`
- `examples/deepswe/`

The best multi-turn tool-use reference is `examples/deepswe/`, especially:

- `examples/deepswe/train_deepswe_nb.py`
- `examples/deepswe/swe_env.py`
- `examples/deepswe/swe_agent.py`

Vision-language SFT:

- `examples/sft/vlm_training.py`

The VLM example is the clearest example of how Tunix reuses the same base trainer for non-text-only models.

## 17. Datasets

There are two small data utility modules under `tunix/examples/data/`:

- `math_dataset.py`
- `translation_dataset.py`

The CLI additionally supports:

- external dataset modules via `data_module`,
- local data directories,
- TFDS-based datasets.

Key helper:

- `tunix/cli/utils/data.py`

Important behavior:

- external dataset modules can be addressed by module path or Python file path,
- prompt chat templating can be applied automatically,
- post-processing can filter by prompt length and batch into iter datasets.

## 18. Metrics, profiling, and performance tracing

There are two observability layers:

1. standard metric logging through `MetricsLogger`
2. performance tracing for RL workflows

Performance metrics APIs:

- `tunix/perf/metrics.py`
- `tunix/perf/export.py`

`PerfMetricsOptions` is part of training config. It can enable:

- perf metrics v1
- perf metrics v2
- optional trace writing

Key references:

- `tunix/perf/metrics.py:74`
- `tunix/perf/metrics.py:118`
- `tunix/perf/export.py:53`

`PerfMetricsExport.from_cluster_config(...)` derives an export function from RL mesh topology and emits different metrics depending on whether the setup is:

- colocated,
- rollout disaggregated from actor/reference,
- fully disaggregated.

That logic is one of the best places to understand how the authors think about RL execution topology.

## 19. Reliability and checkpoint semantics

The docs in `docs/reliability.md` match the code reasonably well.

Important behaviors to remember:

- no checkpoint dir means no checkpointing
- actor and critic RL checkpoints are stored under subdirectories if enabled
- restored RL global step is stored in trainer custom metadata
- `PeftTrainer.close()` always tries to flush metrics and save a final checkpoint

Key references:

- `tunix/sft/checkpoint_manager.py`
- `tunix/rl/trainer.py:36`
- `tunix/rl/rl_cluster.py:523-554`

## 20. Tests are a major source of truth

The tests are extensive and should be treated as the executable specification.

Best directories by topic:

| Path | What it covers |
| --- | --- |
| `tests/sft/` | base trainer, checkpointing, metrics, sharding, profiler |
| `tests/sft/dpo/` | DPO and ORPO |
| `tests/rl/` | learner flow, function registry, reward manager, cluster behavior |
| `tests/rl/grpo/` | GRPO, DAPO, Dr.GRPO |
| `tests/rl/ppo/` | PPO helpers and learner behavior |
| `tests/rl/agentic/` | tools, parsers, orchestrator, trajectory engine |
| `tests/generate/` | sampler, tokenization, vLLM, SGLang |
| `tests/models/` | naming, loading, params, safetensors |
| `tests/distillation/` | distillation trainer and strategies |
| `tests/cli/` | config and CLI helper behavior |

If you need to understand intended behavior quickly, start from the relevant test before editing implementation.

## 21. Current caveats and development notes

These are the most useful "keep this in your head" notes from this snapshot:

1. Tunix is library-first. The CLI is real, but not every library feature has a CLI wrapper.
2. `PeftTrainer` is the main abstraction. SFT, DPO, distillation, and RL trainers all build from it.
3. RL is split cleanly into runtime plumbing (`RLCluster`) and algorithm control (`RLLearner` subclasses).
4. Model naming is strict. If model strings do not follow expected family/version patterns, `AutoModel` routing breaks early.
5. Reward functions loaded by the CLI are module-level function discovery, not a single named entrypoint by default.
6. LoRA is a first-class path, not an add-on. It changes optimizer filtering, checkpointing, and RL weight sharing.
7. Rollout backend integration depends on mapping logic. If backend sync breaks, inspect `tunix/generate/mappings.py` and the backend sampler first.
8. Agentic RL exists, but much of the learner stack still lives under `tunix/rl/experimental/`.
9. `RLCluster._load_model(...)` currently supports NNX modules, not loading arbitrary RL-side models from path at runtime (`tunix/rl/rl_cluster.py:293-352` ends with `NotImplementedError` for path inputs).
10. Docs are helpful but can lag code. Always verify with implementation and tests.
11. The rollout engine group/router abstraction exists, but in this snapshot it is mostly interface-only. `rollout_engine_group.py` and `rollout_traffic_router.py` define shapes and abstract methods, not a production multi-engine router (`tunix/rl/rollout/rollout_engine_group.py`, `tunix/rl/rollout/rollout_traffic_router.py`).
12. The Gemini tool parser is effectively a stub right now: `parse()` returns `[]` and `get_tool_prompt()` is a placeholder string, so the more mature tool-calling path is the Qwen-style parser (`tunix/rl/agentic/parser/tool_parser/gemini_parser.py`, `tunix/rl/agentic/parser/tool_parser/qwen_parser.py`).

## 22. Best files to open first by task

If the task is "how do I train an SFT model?":

- `tunix/sft/peft_trainer.py`
- `tunix/cli/peft_main.py`
- `examples/sft/mtnt/run_gemma2_2b.sh`

If the task is "how does RL work end-to-end?":

- `tunix/rl/rl_cluster.py`
- `tunix/rl/rl_learner.py`
- `tunix/rl/grpo/grpo_learner.py`
- `examples/rl/grpo/gsm8k/run_gemma2_2b.sh`

If the task is "why does rollout/backend sync fail?":

- `tunix/rl/rollout/base_rollout.py`
- `tunix/rl/rollout/vanilla_rollout.py`
- `tunix/rl/rollout/vllm_rollout.py`
- `tunix/rl/rollout/sglang_jax_rollout.py`
- `tunix/generate/mappings.py`
- `tunix/rl/reshard.py`

If the task is "how are models loaded?":

- `tunix/models/naming.py`
- `tunix/models/automodel.py`
- `tunix/cli/utils/model.py`

If the task is "how does the CLI interpret configs?":

- `tunix/cli/base_config.yaml`
- `tunix/cli/config.py`
- `tests/cli/config_test.py`

If the task is "how does agentic RL work?":

- `tunix/rl/experimental/agentic_rl_learner.py`
- `tunix/rl/experimental/agentic_grpo_learner.py`
- `tunix/rl/agentic/pipeline/rollout_orchestrator.py`
- `tunix/rl/agentic/trajectory/trajectory_collect_engine.py`
- `tunix/rl/agentic/utils.py`
- `tests/rl/agentic/`

## 23. Minimal example commands worth remembering

SFT:

```bash
python3 -m tunix.cli.peft_main \
  tunix/cli/base_config.yaml \
  override_config_file=examples/sft/mtnt/configs/gemma2_2b.yaml \
  batch_size=16 \
  training_config.max_steps=10
```

GRPO:

```bash
python3 -m tunix.cli.grpo_main \
  tunix/cli/base_config.yaml \
  override_config_file=examples/rl/grpo/gsm8k/configs/gemma2_2b.yaml \
  batch_size=1 \
  rl_training_config.max_steps=3738
```

Library-first model load:

```python
from tunix.models.automodel import AutoModel, ModelSource

model, model_path = AutoModel.from_pretrained(
    model_id="meta-llama/Llama-3.1-8B",
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
)
```

## 24. Final mental model

If you only remember five things, remember these:

1. `PeftTrainer` is the reusable training core.
2. RL is `RLCluster` plus an algorithm-specific `RLLearner`.
3. Rollout is swappable, but weight sync and sharding are central to performance.
4. `AutoModel` + `ModelNaming` are the canonical path for model loading.
5. The example shell scripts are thin wrappers over a real, inspectable library API.
