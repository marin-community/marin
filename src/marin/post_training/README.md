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
