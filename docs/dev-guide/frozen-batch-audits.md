# Audit a frozen training prefix

`marin.training.frozen_batch_audit` checks the actual global token and binary
loss-weight arrays yielded by a loader before the first optimizer update. It
buffers those batch objects and returns them ahead of the original iterator's
remaining suffix. It does not build another sampler or change the data stream.

The caller loads and verifies its source freeze files and constructs a
`FrozenBatchAudit` with:

- The fixed batch size, sequence length and data seed, plus `draccus.encode(config.data)`.
- Materialized `RowAccounting` entries from `marin.training.chat_token_accounting`.
- A native fingerprint map `(token_sha256, loss_weight_sha256) -> row_id`. Hash each
  full padded row as little-endian int32 bytes; loss weights must be binary.
- The ordered occurrence list, with each entry's `id` and `supervised_tokens`.
- Expected per-update summaries in `freeze["updates"]`, and cumulative summaries
  in `freeze["realized_prefixes"]`, keyed by occurrence count as strings.
- Selected accounting classes, the accounting module's SHA256 and source pins.

No batch geometry, source hash, horizon or identity label is hard-coded. The
source pins are recorded as caller declarations; this API does not read or verify
those external files. It separately checks the actual accounting-module bytes.
The tests in `tests/training/test_frozen_batch_audit.py` provide a complete small
example with two different batches and explicit independent expected counts.

Call `audit_training_iterator` on every rank in the same order:

```python
batches, prefetch_report = audit_training_iterator(
    batches,
    config=config,
    completed_updates=0,
    frozen=frozen,
    receipt_path=receipt_path,
)
```

The audit requires the frozen config and a single effective global microbatch;
`microbatch_size=None` means no microbatching and therefore the full global batch.
Ranks compare a fixed-size plan digest before entering horizon-dependent loops.
A read/check failure propagates across ranks before training can consume the
buffered prefix. The separate `levanter.utils.multihost` helpers preserve 64-bit
host values even with JAX x64 disabled and gather non-addressable global arrays
with a tiled collective.

Immediately before each optimizer step inside the frozen horizon, call
`checked_update(batch, update, frozen)` and retain that consumption receipt. Pass
the same batch object to the optimizer. A prefetch receipt proves what was
buffered, not how many updates training ultimately consumed; the caller controls
and records its training milestone separately.

Reports include per-family and per-class supervised counts, selected-class token
fractions and mean per-update CE coefficient mass. The pooled fraction and the
mean update mass can differ when batches have different supervision totals.
Neither quantity measures gradient share. A report is written only after all
planned checks pass, and an existing report cannot be silently replaced.
