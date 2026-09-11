# Marin infra panel

This internal Grafana panel restores the retired infrastructure status page inside
Grafana. The `status` view combines these sections:

- Seven UTC days of linked nightly status and durations.
- An equal-width main-branch CI history strip.
- Current worker capacity by region.
- W&B hero-training series against cumulative tokens.

The `cluster` view renders live cluster packing and observed resource usage. The
`sm` view renders the fleet's per-GPU SM activity as a time raster. The separate
`nightlies`, `commits`, and `wandb` views remain available. The panel receives
Grafana data frames. The Python bridge owns credentials, queries, and caches.

The hero charts use linear y-axes:

- MFU starts at 0%, with a soft maximum of 30%.
- Train loss uses soft limits of 1.20–1.60 nats/token.
- Paloma loss defaults to **After initialization**. This view excludes samples
  below 1% of the maximum cumulative token count across all report samples.
  The axis includes all remaining values, with 10% of the loss span as padding at each end.
  For constant loss, the padding is 10% of the absolute loss, or 0.1 at minimum.
  Select **Full run** to include every evaluation and recalculate the range.

Soft limits expand to include extreme values. The charts show the original loss
values in nats/token and keep each run's color when the evaluation view changes.
The initialization cutoff advances as the latest token count increases.

```bash
npm ci
npm run typecheck
npm run lint
npm run test:ci
npm run build
```

The parent Dockerfile builds `dist/` and copies only that output into Grafana.
The private plugin is image-reviewed and allowlisted by its exact ID;
`marin-infra-panel` is the only unsigned plugin Grafana accepts.
