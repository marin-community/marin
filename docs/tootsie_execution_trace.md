# Tootsie Experiment Execution Trace

This document traces the execution flow of a tootsie experiment, mapping all dependencies
from the experiment script through training, datasets, models, and evaluation.

## Executive Summary

The tootsie experiments (in `experiments/tootsie/`) demonstrate a multi-phase training
approach for 8B+ parameter models. The code organization reveals a **significant coupling**
between the `experiments/` package and the `lib/marin/` library that should be addressed
for proper library extraction.

---

## 1. Execution Trace Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT SCRIPT (exp600_tootsie.py)                            │
│                        ─────────────────────────────────────────                        │
│  Entry point: executor_main(steps=[...])                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         EXECUTOR FRAMEWORK (lib/marin)                                  │
│                         ──────────────────────────────                                  │
│  marin.execution.executor:                                                              │
│    - ExecutorStep: DAG node with name, fn, config                                       │
│    - executor_main(): Entry point for running pipelines                                 │
│    - Computes versions, output_paths, dependencies                                      │
│    - Uses Fray for cluster management (Ray/TPU)                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
            ▼                             ▼                             ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│    DATASET CONFIG     │   │    MODEL CONFIG       │   │   TRAINING CONFIG     │
│    (experiments/)     │   │    (experiments/)     │   │    (experiments/)     │
│    ───────────────    │   │    ────────────       │   │    ───────────────    │
│                       │   │                       │   │                       │
│ pretraining_datasets/ │   │ llama.py:             │   │ simple_train_config:  │
│   dclm.py             │   │   llama_8b            │   │   SimpleTrainConfig   │
│   dolma.py            │   │   llama_8b_old_rotary │   │   (resources, lr,     │
│   dolmino.py          │   │   compute_num_params  │   │    batch_size, etc.)  │
│   nemotron.py         │   │                       │   │                       │
│   simple.py           │   │ From Levanter:        │   │                       │
│                       │   │   LlamaConfig         │   │                       │
│ Provides:             │   │   Llama3RotaryConfig  │   │                       │
│   - DCLM_MIXTURE_WGTS │   │                       │   │                       │
│   - tokenized datasets│   │                       │   │                       │
│   - lm_mixture_config │   │                       │   │                       │
└───────────────────────┘   └───────────────────────┘   └───────────────────────┘
            │                             │                             │
            └─────────────────────────────┼─────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         DEFAULT_TRAIN (experiments/defaults.py)                         │
│                         ───────────────────────────────────────                         │
│  Assembles:                                                                             │
│    - LMMixtureDatasetConfig (from marin.processing.tokenize.data_configs)               │
│    - TrainLmConfig (from Levanter)                                                      │
│    - TrainLmOnPodConfig (from marin.training.training)                                  │
│    - Adds validation sets (Paloma, uncheatable evals)                                   │
│    - Adds eval harness tasks (from experiments/evals/task_configs.py)                   │
│  Returns: ExecutorStep with fn=run_levanter_train_lm                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING (lib/marin/training/training.py)                       │
│                         ─────────────────────────────────────────                       │
│  run_levanter_train_lm():                                                               │
│    - Validates paths are in same GCP region                                             │
│    - Configures WANDB, JAX compilation cache                                            │
│    - Creates Fray JobRequest                                                            │
│    - Calls levanter.main.train_lm.main()                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         LEVANTER (lib/levanter)                                         │
│                         ───────────────────────                                         │
│  levanter.main.train_lm:                                                                │
│    - TrainLmConfig                                                                      │
│    - Trainer, TrainerConfig                                                             │
│    - Model creation (LlamaConfig -> LlamaLMHeadModel)                                   │
│    - Data loading (LMMixtureDatasetConfig)                                              │
│    - Optimization (AdamConfig, lr schedules)                                            │
│    - Checkpointing (CheckpointerConfig)                                                 │
│    - Evaluation harness integration (LmEvalHarnessConfig)                               │
│                                                                                         │
│  levanter.data.text:                                                                    │
│    - LMMixtureDatasetConfig                                                             │
│    - TextLmDatasetFormat, ChatLmDatasetFormat                                           │
│    - LMDatasetSourceConfig                                                              │
│                                                                                         │
│  levanter.models.llama:                                                                 │
│    - LlamaConfig                                                                        │
│    - Llama3RotaryEmbeddingsConfig                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION (experiments/evals/)                                 │
│                         ───────────────────────────────                                 │
│  In-training evals (via Levanter):                                                      │
│    - LmEvalHarnessConfig with CORE_TASKS                                                │
│                                                                                         │
│  Post-training evals (experiments/evals/evals.py):                                      │
│    - default_base_eval()                                                                │
│    - evaluate_lm_evaluation_harness()                                                   │
│    - Uses marin.evaluation.run.evaluate()                                               │
│                                                                                         │
│  Task configs (experiments/evals/task_configs.py):                                      │
│    - CORE_TASKS, CORE_TASKS_PLUS_MMLU                                                   │
│    - MMLU_0_SHOT, MMLU_5_SHOT                                                           │
│    - KEY_GENERATION_TASKS, KEY_MULTIPLE_CHOICE_TASKS                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Module Dependency Map

### Experiment Script: `experiments/tootsie/exp600_tootsie.py`

```python
# === FROM experiments/ PACKAGE ===
from experiments.pretraining_datasets.dclm import (
    DCLM_MIXTURE_WEIGHTS,
    dclm_components_llama3,
    dclm_mixture_config_llama3_old,
)
from experiments.defaults import default_train
from experiments.pretraining_datasets import tokenize_dolma
from experiments.pretraining_datasets import tokenize_dolmino_math, tokenize_dolmino_subset
from experiments.evals.evals import default_base_eval
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.exp934_hq_vs_pt import pt_vs_hq_components
from experiments.llama import llama3_tokenizer, llama_8b, llama_8b_old_rotary
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.simple_train_config import SimpleTrainConfig

# === FROM marin/ LIBRARY ===
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

# === FROM levanter/ LIBRARY ===
from levanter.schedule import ScheduleStep

# === FROM fray/ LIBRARY ===
from fray.cluster import ResourceConfig
```

---

## 3. Imports from `experiments/` Package (Training Dependencies)

These are modules in `experiments/` that training code depends on:

| Module | Purpose | Used By |
|--------|---------|---------|
| `experiments/defaults.py` | `default_train()`, `default_tokenize()`, `default_validation_sets()` | All experiment scripts |
| `experiments/simple_train_config.py` | `SimpleTrainConfig` dataclass | All training experiments |
| `experiments/simple_sft_config.py` | `SimpleSFTConfig` dataclass | SFT experiments |
| `experiments/llama.py` | Model configs (`llama_8b`, etc.), tokenizer names, chat templates | All Llama experiments |
| `experiments/marin_models.py` | Marin-specific model configs | Some experiments |
| `experiments/qwen3.py` | Qwen3 model configs | Qwen experiments |
| `experiments/pretraining_datasets/*.py` | Dataset definitions, mixture weights, tokenization steps | Training experiments |
| `experiments/midtraining_datasets.py` | Mid-training datasets (finemath, etc.) | Tootsie, other experiments |
| `experiments/evals/evals.py` | `default_eval()`, `default_base_eval()`, `evaluate_lm_evaluation_harness()` | Post-training evaluation |
| `experiments/evals/task_configs.py` | `CORE_TASKS`, `MMLU_*`, `KEY_*_TASKS` | In-training & post-training eval |
| `experiments/evals/engine_configs.py` | `DEFAULT_LM_EVAL_MODEL_KWARGS` | Evaluation |
| `experiments/paloma.py` | `paloma_tokenized()` | Validation sets |

---

## 4. Imports from `marin/` Library (Training Dependencies)

These are the core library modules used by training:

| Module | Purpose | Used By |
|--------|---------|---------|
| `marin.execution.executor` | `ExecutorStep`, `executor_main()`, `InputName`, `this_output_path()`, `versioned()` | All experiment scripts |
| `marin.processing.tokenize` | `TokenizeConfig`, `tokenize()`, `lm_data_config()`, `lm_mixture_data_config()`, `lm_varying_mixture_data_config()` | Dataset preparation |
| `marin.processing.tokenize.data_configs` | `TokenizerStep`, `add_validation_sets_to_mixture()` | Dataset configs |
| `marin.training.training` | `TrainLmOnPodConfig`, `run_levanter_train_lm()` | Training execution |
| `marin.download.huggingface.download_hf` | `DownloadConfig`, `download_hf()` | Dataset downloads |
| `marin.evaluation.evaluation_config` | `EvalTaskConfig`, `EvaluationConfig` | Evaluation setup |
| `marin.evaluation.run` | `evaluate()` | Evaluation execution |

---

## 5. Control Flow: From Experiment to Training

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. EXPERIMENT DEFINITION (exp600_tootsie.py)                                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  # Define data mixture                                                                 │
│  phase_3_tokenized = {**dclm_components_llama3}                                        │
│  phase_3_tokenized.update(...)  # Add dolma, finemath, etc.                            │
│                                                                                        │
│  # Create mixture config with varying weights                                          │
│  phase_3_data_mixture = lm_varying_mixture_data_config(                                │
│      components=phase_3_tokenized,                                                     │
│      weights_list=[(0, DCLM_MIXTURE_WEIGHTS), (PHASE_3_START, cooldown_weights)]       │
│  )                                                                                     │
│                                                                                        │
│  # Define training config                                                              │
│  train_config = SimpleTrainConfig(                                                     │
│      resources=ResourceConfig.with_tpu("v4-2048"),                                     │
│      train_batch_size=3072,                                                            │
│      learning_rate=1.7e-3,                                                             │
│      ...                                                                               │
│  )                                                                                     │
│                                                                                        │
│  # Create training step                                                                │
│  train_step = default_train(                                                           │
│      name="llama-8b-tootsie-phase3",                                                   │
│      tokenized=phase_3_data_mixture,                                                   │
│      model_config=llama_8b,                                                            │
│      train_config=train_config,                                                        │
│  )                                                                                     │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. DEFAULT_TRAIN (experiments/defaults.py)                                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  def default_train(name, tokenized, model_config, train_config, ...):                  │
│      # Add validation sets (Paloma, uncheatable evals)                                 │
│      pretraining_data = _prepare_data_config(tokenized, use_default_validation)        │
│                                                                                        │
│      # Create Levanter's TrainLmConfig                                                 │
│      inner_config = TrainLmConfig(                                                     │
│          data=pretraining_data,                                                        │
│          trainer=TrainerConfig(...),                                                   │
│          model=model_config,                                                           │
│          optimizer=AdamConfig(...),                                                    │
│          eval_harness=LmEvalHarnessConfig(...),                                        │
│      )                                                                                 │
│                                                                                        │
│      # Wrap in marin's pod config                                                      │
│      config = TrainLmOnPodConfig(                                                      │
│          train_config=inner_config,                                                    │
│          resources=train_config.resources,                                             │
│          output_path=this_output_path(),                                               │
│      )                                                                                 │
│                                                                                        │
│      # Return ExecutorStep                                                             │
│      return ExecutorStep(                                                              │
│          name=os.path.join("checkpoints", name),                                       │
│          fn=run_levanter_train_lm,                                                     │
│          config=config,                                                                │
│      )                                                                                 │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. EXECUTOR_MAIN (marin.execution.executor)                                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  if __name__ == "__main__":                                                            │
│      executor_main(steps=[train_step, *default_base_eval(train_step)])                 │
│                                                                                        │
│  # Executor:                                                                           │
│  #   1. Computes version hashes for all steps                                          │
│  #   2. Determines output paths (gs://marin-*/checkpoints/name-hash/)                  │
│  #   3. Resolves dependencies (tokenized data -> training -> eval)                     │
│  #   4. Launches steps in topological order via Fray                                   │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. RUN_LEVANTER_TRAIN_LM (marin.training.training)                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  def run_levanter_train_lm(config: TrainLmOnPodConfig):                                │
│      # Set up output paths for checkpoints, HF exports                                 │
│      config = _update_config_to_use_out_path(config)                                   │
│                                                                                        │
│      # Add environment variables (WANDB, GIT_COMMIT, JAX cache)                        │
│      env = _add_default_env_variables(...)                                             │
│                                                                                        │
│      # Validate GCS paths are in same region                                           │
│      _doublecheck_paths(config)                                                        │
│                                                                                        │
│      # Launch via Fray                                                                 │
│      job_request = JobRequest(                                                         │
│          entrypoint=Entrypoint.from_callable(train_lm.main, args=[train_config]),      │
│          resources=config.resources,                                                   │
│      )                                                                                 │
│      cluster.launch(job_request)                                                       │
│      cluster.wait(job_id)                                                              │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. LEVANTER TRAINING (levanter.main.train_lm)                                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  # This is where the actual JAX/TPU training happens                                   │
│  def main(config: TrainLmConfig):                                                      │
│      # Initialize model                                                                │
│      model = config.model.build()                                                      │
│                                                                                        │
│      # Create data loaders                                                             │
│      train_loader = config.data.build_train_loader()                                   │
│      eval_loader = config.data.build_eval_loader()                                     │
│                                                                                        │
│      # Create trainer                                                                  │
│      trainer = Trainer(config.trainer, model, ...)                                     │
│                                                                                        │
│      # Training loop with:                                                             │
│      #   - Gradient accumulation                                                       │
│      #   - Checkpointing                                                               │
│      #   - WandB logging                                                               │
│      #   - Periodic evaluation (lm_eval_harness tasks)                                 │
│      #   - HF checkpoint export                                                        │
│      trainer.train()                                                                   │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Analysis: What Should Migrate to `marin/` Library

### Currently in `experiments/` but should be in `marin/`

| Module | Reason for Migration |
|--------|---------------------|
| `experiments/defaults.py` | Core training entry point (`default_train`, `default_tokenize`) - fundamental library API |
| `experiments/simple_train_config.py` | `SimpleTrainConfig` is the primary training config interface |
| `experiments/simple_sft_config.py` | `SimpleSFTConfig` is the primary SFT config interface |
| `experiments/evals/task_configs.py` | `EvalTaskConfig` wrapper and task lists are reusable |
| `experiments/evals/evals.py` | `default_eval`, `default_base_eval` are standard patterns |
| `experiments/paloma.py` | Standard validation dataset setup |

### Should remain in `experiments/`

| Module | Reason to Keep |
|--------|----------------|
| `experiments/llama.py` | Contains specific model configs and chat templates - experiment-specific choices |
| `experiments/qwen3.py` | Model-specific configs |
| `experiments/marin_models.py` | Project-specific model variants |
| `experiments/pretraining_datasets/*.py` | Specific dataset paths, revisions, mixture weights - organization-specific |
| `experiments/tootsie/*.py` | Actual experiment definitions |
| `experiments/midtraining_datasets.py` | Specific dataset choices |

### Problematic Coupling (Circular Dependency Risk)

The file `marin/processing/tokenize/data_configs.py` has this import:

```python
from experiments.llama import llama3_tokenizer
from experiments.marin_models import marin_tokenizer
```

This is a **library importing from experiments**, which breaks the intended dependency direction.
This should be fixed by:
1. Moving tokenizer names to a shared constants module in `marin/`
2. Or making the comparison function accept tokenizer names as parameters

---

## 7. Dependency Graph Summary

```
                                    ┌─────────────┐
                                    │   fray/     │
                                    │  (cluster)  │
                                    └──────┬──────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              lib/levanter/                                   │
│  (JAX training, models, data loading, checkpointing, eval harness)           │
└──────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              lib/marin/                                      │
│  (executor framework, tokenization, training orchestration, evaluation)      │
│                                                                              │
│  ISSUE: marin/processing/tokenize/data_configs.py imports from experiments/  │
└──────────────────────────────────────────────────────────────────────────────┘
                                           │
                           ┌───────────────┴───────────────┐
                           │                               │
                           ▼                               ▼
┌──────────────────────────────────────┐   ┌──────────────────────────────────┐
│         experiments/defaults.py      │   │      experiments/evals/          │
│         experiments/*_config.py      │   │      experiments/llama.py        │
│  (Should move to marin/)             │   │  (Organization-specific)         │
└──────────────────────────────────────┘   └──────────────────────────────────┘
                           │                               │
                           └───────────────┬───────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         experiments/tootsie/                                 │
│  (Actual experiment definitions - uses all of the above)                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Recommended Migration Plan

### Phase 1: Fix Circular Dependencies
1. Remove `experiments.llama` and `experiments.marin_models` imports from `marin/processing/tokenize/data_configs.py`
2. Create `marin/constants/tokenizers.py` with known tokenizer equivalence mappings

### Phase 2: Extract Core Training API to `marin/`
1. Move `SimpleTrainConfig` to `marin.training.config`
2. Move `SimpleSFTConfig` to `marin.training.sft_config`
3. Move `default_train()` to `marin.training.api`
4. Move `default_tokenize()` to `marin.processing.tokenize.api`

### Phase 3: Extract Evaluation API to `marin/`
1. Move `EvalTaskConfig` (already in marin)
2. Move `CORE_TASKS`, `MMLU_*` task lists to `marin.evaluation.tasks`
3. Move `default_eval()`, `default_base_eval()` to `marin.evaluation.api`

### Phase 4: Update Experiments
1. Update all experiments to import from `marin.*` instead of `experiments.*`
2. Keep model configs, dataset configs, and actual experiments in `experiments/`

---

## Appendix: Key File Locations

| Purpose | File Path |
|---------|-----------|
| Tootsie experiment | `/Users/power/code/marin-experiments/submodules/marin/experiments/tootsie/exp600_tootsie.py` |
| Training entry point | `/Users/power/code/marin-experiments/submodules/marin/experiments/defaults.py` |
| Training config | `/Users/power/code/marin-experiments/submodules/marin/experiments/simple_train_config.py` |
| Model configs | `/Users/power/code/marin-experiments/submodules/marin/experiments/llama.py` |
| Dataset configs | `/Users/power/code/marin-experiments/submodules/marin/experiments/pretraining_datasets/` |
| Eval tasks | `/Users/power/code/marin-experiments/submodules/marin/experiments/evals/task_configs.py` |
| Executor framework | `/Users/power/code/marin-experiments/submodules/marin/lib/marin/src/marin/execution/executor.py` |
| Training orchestration | `/Users/power/code/marin-experiments/submodules/marin/lib/marin/src/marin/training/training.py` |
| Tokenization | `/Users/power/code/marin-experiments/submodules/marin/lib/marin/src/marin/processing/tokenize/` |
| Evaluation | `/Users/power/code/marin-experiments/submodules/marin/lib/marin/src/marin/evaluation/` |
