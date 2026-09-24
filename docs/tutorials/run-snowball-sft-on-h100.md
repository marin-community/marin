# Fine-tune Snowball with Levanter on H100s

The [Snowball SFT recipe](https://github.com/marin-community/marin/blob/main/experiments/sft/configs/snowball_h100.py) trains Levanter's `SnowballLMHeadModel` directly from the pinned Snowball Hugging Face weights. It uses the shared chat SFT launcher to transform OpenThoughts Agent conversations, supervise assistant tokens, and save native Levanter checkpoints. There is no Grug trainer checkpoint or model-specific conversion job.

This is a 32,768-token, ten-step starting recipe on 32 H100s. A prior [Snowball training run](https://github.com/marin-community/marin/pull/9144) demonstrated the 67B model on 32 learner H100s at 4,096 tokens. That run used GRPO; this longer SFT recipe has not completed an end-to-end accelerator run. Do not assume the model's 262K position limit is a validated training length.

The mesh splits each sequence over four H100s and shards experts across the other eight mesh ranks. Attention keeps local queries and gathers the full keys and values. Large weight and optimizer leaves are sharded over both axes. The recipe keeps conversations separate (`pack=False`) because Snowball does not yet consume Levanter's packed-document attention mask. Its scan layers are checkpointed during reverse mode to bound activation memory.

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
