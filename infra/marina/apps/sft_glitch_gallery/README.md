# SFT Glitch Gallery

Shows flawed documents from the 10.05 Grug SFT training mix, grouped by kind
of flaw, with the offending passage highlighted. The kinds and the rates shown
with them come from the data review in
[#11188](https://github.com/marin-community/marin/issues/11188).

The examples are chosen to be clear. They do not measure how common a flaw is:
each kind holds at most 40 examples, spread across sources.

`categories.py` defines the kinds. Examples come from two places:

- `detect.py` runs one narrow pattern per kind over a sampled snapshot written
  by the [SFT Mix Viewer](../sft_mix_viewer/README.md) sampler.
- Loop and re-emission examples come from a token-level repetition scan of the
  caches. They are loaded from a second Parquet file with the same columns.

Each example is cut to a window around the flaw. The app serves one load from
its Postgres schema and never reads the caches.

## Refresh the examples

```bash
uv run python -m infra.marina.apps.sft_glitch_gallery.detect \
  --snapshot "s3://marin-us-east-02a/tmp/ttl=30d/sft-mix-viewer/review-2026.11.10-seed1" \
  --output "s3://marin-us-east-02a/tmp/ttl=30d/sft-mix-viewer/glitches-2026.11.12"

uv run python -m infra.marina.apps.sft_glitch_gallery.load \
  --input "s3://marin-us-east-02a/tmp/ttl=30d/sft-mix-viewer/glitches-2026.11.12/examples.parquet" \
  --input "s3://marin-us-east-02a/tmp/ttl=30d/sft-mix-viewer/glitches-2026.11.12/loop_examples.parquet" \
  --app-id 52f10c19-26a0-42d4-b221-808824763bae
```

The loader stages the rows under a new load id, points `gallery_current` at
it, and deletes the loads it replaced. Releasing the app does not load data.

To add a kind of flaw, add a `Category` and a detector with the same key.
