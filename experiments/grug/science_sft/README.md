# Converted science-forward Snowball SFT

This run starts from the pinned Step38 Snowball Hugging Face export and trains
one pass over the MiniMax-converted science-forward chat corpus. The trainer
packs conversations at 32,768 tokens, applies loss only to assistant reasoning
and final answers, and overwrites the frozen router bias with the previous
step's QB thresholds. It uses eight H100x8 nodes, batch size 64, a 5e-6 peak
learning rate, 5% warmup, and W&B online logging.

The converted source must first pass the completeness audit documented in
`experiments/datakit/science_sft_conversion/README.md`. The conversion and
token-store code must be merged before launching from a clean Marin main
checkout. Build the token store on RNO2A with CoreWeave storage access:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-converted-store-20260927 --cpu 8 --memory 32GB \
  --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.grug.science_sft.prepare \
    --input-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v1/outputs/main \
    --output-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted-store/2026.09.27-v1 \
    --tokenizer s3://marin-us-east-02a/models/open-athena--Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38 \
    --num-shards 1024 --max-workers 32
```

Inspect the resulting artifact's source counts. Preparation fails if any
conversation exceeds the 32K context. Once the store is complete, submit the
training coordinator from merged main. Its task environment must contain
CoreWeave credentials and `WANDB_API_KEY`:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-converted-step38-20260927 --cpu 8 --memory 32GB \
  --disk 20GB --extra cpu --no-wait -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.science_sft.launch \
    --store-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted-store/2026.09.27-v1 \
    --version 2026.09.27-v1
```

The launcher checks the model mirror revision and refuses a store with a
different tokenizer, context, or source name. It uses only full 64-sequence
batches; fewer than 64 packed sequences may remain unused at the end.
