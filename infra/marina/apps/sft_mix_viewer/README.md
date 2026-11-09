# SFT Mix Viewer

Reads sampled documents from the data behind
[`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05):
216 SFT token stores (79.65% of training positions) and 391 long-context
pretraining replay caches (20.35%). Each source lists its share of training
positions and a few whole documents decoded with the run's tokenizer. "Draw by
mix share" picks a source in proportion to that share, then one of its sampled
documents.

`mix/` pins the allocation and replay definition of training run
`grug-67b-sft-20260929-special-token-lr-full-ab-4750-grouped`. Replay sources
are named `replay/c<cluster>/q<quality>/<length stratum>`; the `gt_64k` caches
were scheduled eight times as often as their `lte_64k` siblings.

The app serves a snapshot from its Postgres schema and never reads the caches
itself. Documents longer than 64,000 characters keep their first 48,000 and
last 16,000 characters.

## Refresh the snapshot

Sample next to the caches in `us-central2`, writing to a bucket the loading
machine can read:

```bash
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
  --enable-extra-resources --region us-central2 --cpu 8 --memory 16GB --disk 20GB \
  --job-name sft-mix-sample \
  -- python -m infra.marina.apps.sft_mix_viewer.sample \
  --output "s3://marin-us-east-02a/tmp/ttl=30d/sft-mix-viewer/2026.10.05-seed0"
```

`--sft-per-source`, `--replay-per-source` and `--seed` change the draw. Then
stage the snapshot and select it:

```bash
uv run python -m infra.marina.apps.sft_mix_viewer.load \
  --input "s3://marin-us-east-02a/tmp/ttl=30d/sft-mix-viewer/2026.10.05-seed0" \
  --app-id 378c3d80-275c-49f2-9913-0bd2c292c354
```

The loader inserts the rows under a new snapshot id, points `mix_current` at
it, and deletes the snapshots it replaced. Releasing the app does not load
data.
