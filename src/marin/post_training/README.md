# Example RL Training Instructions

This code is adopted from Charlie Snell's initial implementation of RL training in Jax on TPUs. The code uses the same worker for training and inference but uses continuous batching and parameter resharding between inference and training to optimize both steps.

## General Setup

This will change as we begin to use Marin's executor framework. For now, below are intructions for manually allocating a TPU node and launch experiments on that node.

Install dependencies:

```
uv pip install -r requirements.txt
```

Allocating a TPU node. Change $TPU_NAME and $ACCELERATOR_TYPE if needed:

```
./allocate_tpu.sh
```

Run launcher setup to run installation on TPUs and training script on all hosts in the pod at once.

```
python launcher.py setup --project=$TPU_NAME
```

Add the following in `training_run.sh` to specify `HF_TOKEN` and `WANDB_API_KEY`.

```
export HF_TOKEN=...
export WANDB_API_KEY=...
```

Then Launch the training run:

```
python launcher.py launch training_run.sh --project=$TPU_NAME
```

This will: 1) copy the latest version of `llama3_train` to the TPUs; 2) stop anything running on the TPUs; 3) run the training script on the TPUs.

To print the output of the training run, you can run:

```
python launcher.py check --project=your_tpu_name
```

To terminate an ongoing training run, you can run:

```
python launcher.py stop --project=your_tpu_name
```

## Code Flow Summary

### 1. Training Pipeline Flow

The main training loop follows this high-level flow:

```
Configuration → Environment Setup → Model Loading → Training Loop → Checkpoint Saving
```

**Entry Points:**
- `train.py`: Main RL training script with environment integration
- `training_worker.py`: Training worker that reads from rollout queues
- `launch_training_worker.py`: Launcher for training workers

**Training Flow (`train.py`):**
1. **Initialization** (`main()` function)
   - Parse configuration from `TrainingConfig`
   - Initialize JAX distributed training
   - Setup mesh sharding for TPU clusters
   - Load model configurations and build models (training, prefill, generate)
   - Load tokenizer and environments from configuration files
   - Initialize Weights & Biases logging

2. **Model Setup** (`Trainer.__init__()`)
   - Create training, prefill, and generate model variants
   - Setup parameter sharding rules for distributed training
   - Configure optimizers with gradient accumulation
   - Compile JAX functions for training and inference
   - Initialize samplers for rollout generation

3. **Training Loop** (`Trainer.train()`)
   ```python
   for step in range(num_train_steps):
       # 1. Select random environment
       environment = random.choice(train_environments)
       
       # 2. Generate rollouts using current policy
       inference_params = reshard_params(train_state.params)
       rl_dataset, metrics = create_dataset_from_environment(
           environment, sampler, inference_params, reference_params, ...)
       
       # 3. Train on generated data
       for batch in rl_dataset.iterate_batches():
           train_state, train_metrics = train_step(train_state, batch)
       
       # 4. Evaluation and logging
       if step % log_freq == 0:
           eval_metrics = evaluate_data_from_environment(params)
           logger.log({**train_metrics, **eval_metrics, **dataset_metrics})
       
       # 5. Checkpoint saving
       if step % save_freq == 0:
           save_checkpoint(train_state, step)
   ```

### 2. Dataset Creation Flow

**Key File:** `rl_dataset.py`

The dataset creation process transforms environment interactions into RL training data:

```
Environment Step → Rollout Generation → Reference Logprobs → Advantage Computation → Training Batches
```

**Detailed Flow:**
1. **Environment Interaction** (`create_dataset_from_environment()`)
   - Sample problems from environment
   - Generate multiple responses per problem using policy model
   - Compute rewards using environment-specific grading

2. **Reference Logprobs Computation** (`RLDataset.from_env_step()`)
   - Tokenize prompts and responses
   - Compute log probabilities using reference model in batches
   - Pad sequences to consistent lengths

3. **Advantage Computation** (`compute_rloo_advantages_for_group()`)
   - Apply RLOO (Rejection sampling Leave-one-Out) advantages
   - Normalize advantages: `(reward - mean_reward) / std_reward`

4. **Training Data Preparation** (`_prepare_rloo_examples()`)
   - Concatenate prompts and responses
   - Create attention masks and position IDs
   - Setup loss masks (only train on response tokens)
   - Apply advantages as loss weights for policy gradient

### 3. Inference System Flow

**Key File:** `inference.py`

The inference system provides high-performance text generation with continuous batching:

```
Prompts → Tokenization → Prefill → Generation Loop → Output Processing
```

**Detailed Components:**

1. **Sampler Class** (`Sampler`)
   - Manages prefill and generate models with different attention kernels
   - Handles KV cache management and parameter resharding
   - Implements continuous batching for efficient GPU utilization

2. **Generation Loop** (`_sampling_loop_impl()`)
   ```python
   while not done:
       # Prefill new prompts when slots available
       if can_prefill():
           batch = build_prefill_batch()
           generation_state = prefill(params, batch, generation_state)
       
       # Generate next tokens for active sequences
       generation_state = generate_step(params, generation_state, rng)
       
       # Process completed generations
       for completed in extract_completed():
           yield completed
   ```

3. **State Management** (`GenerationState`, `SamplingState`)
   - Tracks KV cache, tokens, and completion status
   - Manages multiple generations per prompt
   - Handles dynamic batching and memory efficiency

### 4. Environment System Flow

**Key Files:** `environments/`, `load_environments.py`

Environments provide task-specific problem sampling and reward computation:

```
Configuration → Environment Loading → Problem Sampling → Response Grading → Metrics
```

**Environment Interface** (`environments/marin_env.py`):
- `step()`: Main interaction method that samples problems, generates responses, computes rewards
- `get_eval_examples()`: Provides evaluation datasets
- `_compute_rewards()`: Task-specific grading logic

**Available Environments:**
- **Math**: `math_env.py`, `numina_math_env.py`, `olym_math_env.py`, `olympiad_bench_env.py`
- **Reasoning**: `aqua_rat_env.py`, `svamp_env.py`, `open_math_reasoning_env.py`
- **Coding**: `swe_bench_env.py`, `orz_env.py`

### 5. Model Architecture Flow

**Key Files:** `model_helpers.py`, `llama3.py`, `model_config.py`

The system uses specialized model variants optimized for different tasks:

```
Base LLaMA Config → Model Variants → Attention Kernels → Parameter Sharding
```

**Model Variants:**
1. **Training Model** (`build_training_model()`)
   - Full precision (fp32) for gradient computation
   - Splash attention for long sequences
   - Model parallelism with FSDP sharding

2. **Prefill Model** (`build_prefill_model()`)
   - Lower precision (bf16) for memory efficiency
   - Splash attention for batch prefill
   - Optimized for prompt processing

3. **Generate Model** (`build_generate_model()`)
   - Paged attention for KV cache management
   - Optimized for autoregressive generation
   - Memory-efficient inference

### 6. Distributed Training Flow

**Key Files:** `training_worker.py`, `rollout_queue.py`, `launcher.py`

The system supports distributed training with separate rollout and training workers:

```
Rollout Workers → Queue System → Training Workers → Checkpoint Sharing
```

**Worker Architecture:**
1. **Rollout Workers** (planned)
   - Generate rollouts using current policy
   - Write training data to shared queue
   - Use latest checkpoints from training workers

2. **Training Workers** (`training_worker.py`)
   - Read rollout batches from queue
   - Perform gradient updates using RLOO loss
   - Save checkpoints for rollout workers

3. **Queue System** (`rollout_queue.py`)
   - `GCSRolloutQueue`: GCS-based queue for cloud deployment
   - `InMemoryRolloutQueue`: Local queue for development
   - Handles batching and timeout management

### 7. Configuration System Flow

**Key Files:** `training_config.py`, `model_config.py`

Hierarchical configuration system using dataclasses:

```
TrainingConfig
├── ModelConfig (model paths, dtypes, attention kernels)
├── TrainingHyperparameters (batch sizes, learning rates, RL coefficients)
├── LoggingConfig (WandB, checkpointing, evaluation)
├── EnvironmentConfig (task specifications)
├── DistributedConfig (sharding, TPU setup)
└── GenerationConfig (sampling parameters)
```

## File Descriptions

### Core Training Files
- **`train.py`**: Main RL training script with environment integration and RLOO implementation
- **`training_worker.py`**: Worker process for distributed training that reads from rollout queues
- **`training_config.py`**: Hierarchical configuration system using dataclasses
- **`launch_training_worker.py`**: Launcher script for training workers

### Model and Inference
- **`inference.py`**: High-performance inference system with continuous batching and KV cache management
- **`model_helpers.py`**: Helper functions for building and configuring model variants
- **`model_config.py`**: Configuration dataclasses for model components
- **`llama3.py`**: LLaMA model implementation with attention kernel support
- **`optimizer.py`**: AdamW optimizer configuration with learning rate scheduling

### Data Processing
- **`rl_dataset.py`**: Dataset class for RL training with RLOO advantage computation
- **`rollout_queue.py`**: Queue system for distributed rollout collection (GCS and in-memory)

### Environment System
- **`load_environments.py`**: Environment loading and configuration system
- **`environments/marin_env.py`**: Base environment interface with `EnvStep` data structure
- **`environments/*.py`**: Task-specific environments for math, reasoning, and coding problems
- **`environments.json`**: Environment configuration file

### Utilities
- **`utils.py`**: Shared utilities for JAX, GCS, checkpointing, and logging
- **`tpu_attention.py`**: TPU-optimized attention implementations

### Infrastructure
- **`launcher.py`**: TPU cluster management and job orchestration
- **`allocate_tpu.sh`**: Script for TPU allocation
- **`training_run.sh`**: Training job execution script
- **`requirements.txt`**: Python dependencies

## Key Features

### 1. RLOO (Rejection sampling Leave-one-Out) Training
- Generates multiple responses per prompt
- Computes advantages using leave-one-out baseline
- Applies policy gradient loss with KL regularization

### 2. Distributed Architecture
- Separate rollout and training workers
- Queue-based communication using GCS
- Efficient parameter sharing via checkpoints

### 3. High-Performance Inference
- Continuous batching for GPU efficiency
- Multiple attention kernel implementations (Splash, Paged, Default)
- Dynamic memory management with KV cache optimization

### 4. Multi-Task Training
- Support for math, reasoning, and coding environments
- Environment-specific reward computation
- Configurable task mixing and evaluation

### 5. TPU Optimization
- JAX/Flax implementation for TPU clusters
- Model and data parallelism with sharding
- Memory-efficient mixed precision training

## Usage

### Basic Training
```bash
# Single-node training
uv run python train.py --config=config.yaml

# Distributed training on TPUs
python launcher.py setup --project=my-tpu-cluster
python launcher.py launch training_run.sh --project=my-tpu-cluster
```

### Configuration
Configure training through `TrainingConfig` dataclass or YAML files. Key parameters include:
- Model paths and architectures
- Training hyperparameters (batch sizes, learning rates)
- Environment specifications
- Distributed training settings

### Environment Development
Create custom environments by extending `MarinEnv` and implementing:
- `step()`: Problem sampling and reward computation
- `get_eval_examples()`: Evaluation dataset provision
- Environment-specific configuration

This system provides a complete framework for RL fine-tuning of language models on reasoning tasks, with support for distributed training, multiple task types, and high-performance inference.