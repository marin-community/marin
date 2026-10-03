# grug-snowball-synthetic

Single-node harness that trains Snowball on uniform-random tokens through Levanter's `Trainer` and reports step time, tokens/s, MFU, and peak device memory. It exists to produce the AMD baseline for [#9462](https://github.com/marin-community/marin/issues/9462) and to be rerun after each kernel change, so by default it uses the portable code paths: reference attention and `RAGGED_DOT_IMPL=xla`. `--attention` selects any Grug attention implementation, for example `xla_flash`, which also runs everywhere.

Unlike the neighbouring variants, this directory carries no model copy. [`train.py`](./train.py) drives `levanter.models.snowball.SnowballConfig` directly, because the point is to measure the production Snowball code path rather than a variant of it.

Presets: `tiny` (CPU check), `medium` (4 layers at full width), and `full` (26 layers, 67B parameters). The default batch is one sequence per device; a global batch of 1 fails in the next-token loss under explicit mesh axes, so single-device runs need `--batch-size 2`. Every step sees fresh random tokens, so the loss cannot fall below ln(vocab_size) and a lower value means something leaks targets. No checkpoint is written.

```bash
RAGGED_DOT_IMPL=xla python experiments/grug/snowball_synthetic/train.py \
    --size full --steps 20 --profile-steps 3 --compilation-cache-dir "$WORK/jax-cache"
```

The profile lands under `<log-dir>/<run-id>/profiler/`; summarize it with `lib/marin/tools/profile_summary.py summarize --xplane-file <trace>.xplane.pb`.
