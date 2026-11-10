# Grug log scaling oracle

A tiny Hero-shaped model in the artifact-schema-3 HF layout, with the JAX Hero model's next-token
log-probabilities for two documents of 56 and 40 tokens, with log scaling on and off. The floor is
16 keys, the slope 1.0 and the power 2.0, and the init std is twice the Hero's. Log scaling moves the
log-probabilities beyond the floor by 0.12 to 0.19 nats, about 12 times the bf16 noise of a JAX bf16
forward; at or below the floor the two runs agree to 1e-6. `tests/gpu/test_grug_log_scaling_parity.py`
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
| `config.json` | `a0d6bd2a340c3ae9896cf97ad3aaa09ee0a553fec0a999adaa271ff42991090f` |
| `model.safetensors` | `7ee0643a54ebead340aa44704568eb65eaf2d8dc9b5dd48f5edb69a9c92f1ea8` |
| `observations.npz` | `f6b44ab39401a443a80253a5b2be260d85e9c25b4b3201cd07cc35d04db79ace` |
