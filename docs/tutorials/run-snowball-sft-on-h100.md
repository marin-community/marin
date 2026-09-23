# Fine-tune Snowball with Levanter on H100s

The [Snowball SFT recipe](../../experiments/sft/configs/snowball_h100.py) trains Levanter's `SnowballLMHeadModel` directly from the pinned Snowball Hugging Face weights. It uses the shared chat SFT launcher to transform OpenThoughts Agent conversations, supervise assistant tokens, and save native Levanter checkpoints. There is no Grug trainer checkpoint or model-specific conversion job.

This is a 4,096-token, ten-step starting recipe. It requests 32 H100s because a prior [Snowball training run](https://github.com/marin-community/marin/pull/9144) demonstrated the 67B model on 32 learner H100s at this length. That run was GRPO, not this SFT recipe; this exact end-to-end SFT job has not been measured. The model supports longer positions, but the current Levanter training path does not shard context for the 262K setting.

The recipe shards experts across the eight H100s on each host and uses four hosts for data parallelism. It keeps conversations separate (`pack=False`). Snowball currently builds its own causal attention mask and does not honor the packed-document mask supplied by Levanter. Its scan layers are checkpointed during reverse mode to bound activation memory.

## Prepare

Install the workspace and CUDA-enabled JAX packages as described in [Setting up a Local GPU Environment](local-gpu.md):

```bash
uv sync --all-packages --extra=gpu
```

Provide a shared `MARIN_PREFIX` in the same region as the H100 workers, a Hugging Face token that can read the [Snowball base model](https://huggingface.co/open-athena/snowball-67b-a2b-base-262k-qk175-skew8) and [OpenThoughts Agent dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K), and the required storage credentials. The recipe pins the model weights and dataset revisions. It loads the tokenizer from the model repository name.

```bash
export MARIN_PREFIX=s3://your-bucket/snowball-sft
export HF_TOKEN=hf_example
```

## Launch

Submit a CPU coordinator through Iris. It transforms the data and dispatches the 32-H100 training job through the shared SFT launcher:

```bash
uv run iris --cluster=marin job run --job-name snowball-sft-coord \
  --region us-east5 --cpu 1 --memory 2G --extra cpu \
  --priority interactive --no-wait \
  -e MARIN_PREFIX "$MARIN_PREFIX" -e HF_TOKEN "$HF_TOKEN" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.sft.configs.snowball_h100 --accelerator 32xH100
```

The coordinator and workers need the same storage and Hugging Face credentials. The completed training artifact is recorded under `checkpoints/snowball-openthoughts-agent-sft` in `MARIN_PREFIX`. Change the `SFTSpec` fields in the recipe to choose a different step count, batch size, or context length; increasing context length requires a separate memory and attention validation.
