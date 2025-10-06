# Getting Started

This document provides a guide on how to launch model training and configure it according to your specific needs.
For an even more detailed guide, please refer to the [Levanter Documentation](https://levanter.readthedocs.io/en/latest/),
in particular the [Training on Your Own Data](https://levanter.readthedocs.io/en/latest/Training-On-Your-Data/) page.

## Installation

Please see the [Installation Guide](Installation.md) for more information on how to install Levanter.

## Quick Start Examples

{%
   include-markdown "../README.md"
   start="<!--levanter-user-guide-start-->"
   end="<!--levanter-user-guide-end-->"
%}

## Launch Model Training

To launch the training of a GPT2 model, run the following command:
```bash
python src/levanter/main/train_lm.py --config_path config/llama_small_fast.yaml
```

This will execute the training pipeline pre-defined in the [train_lm.py](https://github.com/stanford-crfm/levanter/tree/main/src/levanter/main/train_lm.py) and set model and training configuration
set in [llama_small_fast.yaml](https://github.com/stanford-crfm/levanter/tree/main/config/llama_small_fast.yaml). You can find more template configurations in the [config](https://github.com/stanford-crfm/levanter/tree/main/config/) directory.

Configuration files are processed using [Draccus](https://github.com/dlwh/draccus). Draccus is yet-another yaml-to-dataclass library.
It should mostly work like you would expect. Arguments may be passed in via the command line using arg-parse style
flags like `--trainer.num_train_steps`, and will override the values in the config file.

## Set Custom Training Configuration
In machine learning experiments, it is common to adjust model hyperparameters. In this section, we will provide examples of different use cases
and explain the corresponding parameters that you can change.

### Change Model Parameters
To change the dimensions of your GPT2 model and increase the number of training steps to 10,000, I can use the following command:

```
python src/levanter/main/train_lm.py \
    --config_path config/llama_small_fast.yaml \
    --model.num_heads 20 \
    --model.num_layers 36 \
    --model.hidden_dim 1280 \
    --trainer.num_train_steps 10000
```

This will overwrite the default model and training configurations and set the following parameters:
- `model.num_heads`: The number of heads in the multi-head attention layers of the transformer encoder and decoder.
- `model.num_layers`: The number of layers in the transformer encoder and decoder.
- `model.hidden_dim`: The hidden dimension of the model. This is the dimension of the hidden states of the transformer encoder and decoder. Note
that the hidden dimension must be divisible by the number of heads.
- `trainer.num_train_steps`: The number of training steps to run.

You can find a complete list of parameters to change from the `TrainerConfig` in [trainer.py](https://github.com/stanford-crfm/levanter/tree/main/src/levanter/trainer.py) and `Gpt2Config` in
[gpt2.py](https://github.com/stanford-crfm/levanter/tree/main/src/levanter/models/gpt2.py).

### Change Checkpoint Settings
To change the frequency of saving checkpoints, you can use the following command:

```
python src/levanter/main/train_lm.py \
    --config_path config/llama_small_fast.yaml \
    --trainer.checkpointer.base_path checkpoints/gpt2/ \
    --trainer.checkpointer.save_interval 20m
```

This will overwrite the default checkpoint settings from the `TrainerConfig` and `CheckpointerConfig` in [checkpoint.py](https://github.com/stanford-crfm/levanter/tree/main/src/levanter/checkpoint.py) to
save checkpoints every 20 minutes. The checkpoint will be saved to the directory `checkpoints/gpt2/${run_id}`

Note that:
- `--trainer.checkpointer.base_path` supports local path and cloud storage path (e.g. S3, GCS, etc.), as
long as the path is accessible from the machine that you are running the training script on.
- The `--trainer.checkpointer.save_interval` argument supports the following units: `s` (seconds), `m` (minutes), `h` (hours), `d` (days).
- You can also specify `--trainer.load_checkpoint_path` to load a checkpoint from a specific path. If you do not specify it, the trainer will
see if the `{checkpointer.base_path}/{run_id}` directory exists and load the latest checkpoint from there. If the directory does not exist,
 the trainer will start training from scratch.
- You can also specify `--trainer.initialize_from` to initialize the model from a specific checkpoint, without loading the optimizer state.

### Change Evaluation Parameters
To change how often the model is evaluated during training, you can use the following command:

```
python src/levanter/main/train_lm.py \
    --config_path config/llama_small_fast.yaml \
    --trainer.steps_per_eval 500
```

This will overwrite the default eval frequency (every 1,000) from the `TrainerConfig` in [config.py](https://github.com/stanford-crfm/levanter/tree/main/src/levanter/config.py) to every 500 steps.

### Change Parallelism Settings
By default, Levanter will split the number of examples in `train_batch_size` equally across all available GPUs.
To set explicit number of examples to process on each device during training and evaluation, you can use the following command:

```
python src/levanter/main/train_lm.py \
    --config_path config/llama_small_fast.yaml \
    --trainer.batch_size 256 \
    --trainer.per_device_parallelism 64 \
    --trainer.eval_per_device_parallelism 64
```

Note that:
- `batch_size` must be divisible by `per_device_parallelism` and `eval_per_device_parallelism`.
- If `num_devices * per_device_parallelism` is smaller than `batch_size`, we use gradient accumulation to accumulate gradients from multiple
batches before performing a parameter update. This is useful when you have a large batch size but do not have enough memory to fit the
entire batch.

### Change WandB Configuration
Levanter supports extensive logging, metrics tracking, and artifact saving on WandB. Once you have created a WandB account and finished setting up
your WandB API key, all of your training runs will be automatically logged to WandB, without any additional configuration in your training commands.

Suppose you want to set more control on your WandB logging, you can use the following command:

```
python src/levanter/main/train_lm.py \
    --config_path config/llama_small_fast.yaml \
    --trainer.wandb.project my_project \
    --trainer.wandb.name my_run \
    --trainer.wandb.group my_new_exp_group
```

This will overwrite the default WandB configuration from the `TrainerConfig` in [config.py](https://github.com/stanford-crfm/levanter/tree/main/src/levanter/config.py).
We pass all these arguments to the `wandb.init()` function at the same verbatim.
For more information on the WandB configuration, please refer to the [WandB documentation](https://docs.wandb.ai/ref/python/init).

#### Automated Wandb Workspace Setup
Levanter logs many metrics, and the default WandB workspace layout can sometimes be unhelpful for navigating them. We provide a script to automatically configure a WandB workspace with a preferred layout, including specific plots and grouping settings.

To run the script:
- Script location: `scripts/setup_wandb_workspace.py`
- Command-line arguments:
    - `--entity` (optional): Your WandB entity (username or team name). If not provided, it defaults to the current logged-in WandB user's default entity.
    - `--project` (optional): The WandB project name for your Levanter runs. Defaults to `"levanter"` if not provided.
    - `--workspace` (optional): The desired name for the workspace. Defaults to `"levanter-default"` if not provided.
    - `--base_url` (optional): If you are using a self-hosted WandB instance, specify its base URL here.
- Example commands:
  ```bash
  # Example: Setup a workspace with default project ("levanter") and workspace name ("levanter-default")
  # using your default WandB entity.
  python scripts/setup_wandb_workspace.py

  # Example: Setup a specific workspace in a specific project, using your default entity.
  python scripts/setup_wandb_workspace.py --project my-levanter-experiments --workspace my-custom-view

  # Example: Full specification (e.g., for a different entity or self-hosted instance).
  python scripts/setup_wandb_workspace.py --entity <your-wandb-entity> --project <your-wandb-project> --workspace <your-desired-workspace-name> --base_url <your-wandb-instance-url>
  ```

The script applies the following configuration:
- Sets run grouping to `group_by_prefix="first"`.
- Creates a "main" section with the following plots:
    - `log(train/loss)` vs `log(tokens)`
    - `log(train/loss)` vs `log(steps)`
    - `log(eval/bpb)` vs `log(tokens)`
    - A bar chart for `max(throughput/mfu)` (titled "Max MFU").

The script will update the workspace if it already exists or create it if it doesn't.

### Resume Training Runs
When you resume a training run, you may like to restart from a previously saved checking and resume the corresponding WandB run, as well.
To do so, you can use the following command. The `trainer.wandb.resume true` is optional, but will make WandB error out if the run ID does not exist.

```
python src/levanter/main/train_lm.py \
    --config_path config/llama_small_fast.yaml \
    --trainer.wandb.resume true \
    --trainer.id asdf1234
```

There are two things to note here:
1. Resume from a checkpoint: as we stated in a section above, the `--trainer.load_checkpoint_path` argument is optional. If it is specified,
the trainer will recursively search for the latest checkpoint in the directory and load it. If you do not specify it, the trainer will
start training from scratch.
2. Resume from a WandB run: if you set `--trainer.wandb.resume`, it will resume the corresponding WandB run with the ID `asdf1234`. You can
find the WandB run ID in the URL of your WandB run page. For more information, please refer to the
[WandB documentation](https://docs.wandb.ai/guides/runs/resuming).
