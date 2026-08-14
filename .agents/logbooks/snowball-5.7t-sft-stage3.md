# Snowball 5.7T SFT Stage 3

Issue: https://github.com/marin-community/marin/issues/8225

## OpenCode checkpoint

- Native checkpoint: `s3://marin-us-east-02a/marin/grug/snowball_step105149_sft_s3_agentic_eot_5ep/2026.08.13.1/checkpoints/step-1888/`
- Public export: https://huggingface.co/laion/snowball-67b-a2b-sft-s3-opencode-step1888
- Data: `tokenized/grug-a2b-agentic-sft-eot@2026.08.05`
- Matching evaluation: OpenCode on `DCAgent/dev_set_v2`, three attempts per task, 300 trials, concurrency 128

## Nemotron-Terminal retry

- DRI: `benjaminfeuer`
- Source branch: `agent/snowball-s3-nemotron-terminal`
- Initialization: `s3://marin-us-east-02a/marin/grug/snowball_step105149_sft_s2_thinking/2026.08.13.1/checkpoints/step-630/`
- Data: `tokenized/nemotron_terminal_full-chat-7adc64@2026.07.17`
- Output: `s3://marin-us-east-02a/marin/grug/snowball_step105149_sft_s3_nemotron_terminal_steps1888/2026.08.14.1/`
- Cluster and priority: `cw-rno2a`, `interactive`
- Geometry: 8 nodes, 8 H100s per node, expert parallel 8, batch 64, sequence length 32,768
- Schedule: 1,888 steps, AdamH/Adam at `5e-6`, cosine decay, 3% warmup
- Checkpoints: hourly temporary saves, permanent every 1,000 steps and at completion, two-hour distributed commit timeout
- Tracker: W&B disabled to match the #7743 control; Iris and Finelog are authoritative
- Stop conditions: non-finite loss, data or initialization drift, repeated deterministic failure, or user request

Launch:

```bash
uv run iris --cluster cw-rno2a job run \
  --job-name snowball-step105149-sft-s3-nemotron-terminal \
  --cpu 1 \
  --memory 2G \
  --extra cpu \
  --priority interactive \
  --max-retries 10 \
  --no-wait \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin \
  -- python -m experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage \
    --stage second-cooldown-nemotron-terminal \
    --version 2026.08.14.1 \
    --run
```
