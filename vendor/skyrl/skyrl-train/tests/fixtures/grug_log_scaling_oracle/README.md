# Grug log scaling oracle

A tiny Hero-shaped model in the artifact-schema-3 HF layout, with the JAX Hero model's next-token
log-probabilities for two documents of 56 and 40 tokens, with log scaling on and off. The floor is
16 keys and the slope is 1.0, so log scaling moves the log-probabilities beyond the floor by about
0.6 nats. At or below the floor the two runs agree to 4e-6. `tests/gpu/test_grug_log_scaling_parity.py`
checks Megatron against it without installing JAX.

Producer: `experiments/grug/moe_hero_ep/ops/skyrl_log_scaling_oracle.py` in the Marin repository,
run on the CPU JAX backend. It is deterministic; regenerate with:

```bash
JAX_PLATFORMS=cpu uv run python -m experiments.grug.moe_hero_ep.ops.skyrl_log_scaling_oracle \
  --output /tmp/grug-log-scaling-oracle
```

Expected SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| `config.json` | `2d18eb626772b09094beea5e059f1df3d5aade1f229393895004616ad311eb5a` |
| `model.safetensors` | `8c9e49974050436ea037a13048f0cf8560020f8f7695bbf70208dc1a690d2a52` |
| `observations.npz` | `60c4bddd265fe893f0cf0b54ba60167f9eed8d20c05b63bc0307eb1a6c9af6b9` |
