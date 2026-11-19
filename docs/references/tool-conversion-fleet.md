# GLM tool conversion fleet

The two fleet entry points convert retained SFT shards through the shared GLM
relay. Run them inside an Iris task with CoreWeave storage credentials,
`GLM_BULK_TOKEN`, and Hugging Face access to the private
`open-athena/glm-tool-conversion-production-data` dataset:

```bash
uv run python -m experiments.glm_tool_serving.scale_production \
  --replica 1 --replicas 8 --slots 4 --catalog scale-eight-catalog.json \
  --destination <run-prefix>
uv run python -m experiments.glm_tool_serving.ortet_production \
  --replica 1 --replicas 24 --slots 4 --destination <tail-run-prefix>
```

`production_data` below means `experiments/glm_tool_serving/production_data`
in the task checkout. Startup downloads missing catalogs from the private dataset
using its committed reference index.

Both commands use the same batch loop. Each replica must have a single writer
and a fresh local work directory (`/tmp/tool-scale-r<N>` or
`/tmp/tool-ortet-r<N>`). The committed `production_data/config.json` is a template; the loop
fills in runtime paths, credentials, relay URL, generation and deadline, and
writes the effective config inside that work directory.

Before loading either tokenizer, startup fetches missing assets and checks their
SHA-256 values. The SFT tokenizer is pinned in `production_data/reference-index.json`;
the serving tokenizer's repository revision and file hashes are pinned in
`production_data/serving_tokenizer/pins.json`. Existing files must match those
pins. Tokenizer payloads and catalogs remain outside Git.

The scale fleet stops at the Ortet catalog's `migration.boundary_catalog_index`
when using its pinned parent, `scale-eight-catalog.json`. For the legacy
`scale-catalog.json`, it stops at the
first shard owned by Ortet. The catalog bytes and per-replica shard indices stay
unchanged, so existing cursors keep their identities.

Each batch writes a claim before acquisition, uploads its archive, and then
advances the cursor. `--predecessor` reads prior progress when the destination
has no cursor. A claim with an archive beside it requires explicit recovery,
whether it belongs to the destination or predecessor. An archive-less claim
can be replayed after its evidence is saved under the destination's
`reconciliations/` and `reconciled-claims/`; predecessor objects remain read-only.
Recovery JSON binds the saved cursor and archive SHA-256, lists preserved row
statuses in a `preserved` map from row ID to final status. The
`eligible_unsubmitted_ids` list permits only rows confirmed to have no submitted
request, POST marker or admission-ledger entry. Scale replicas read
`recovery-<N>.json`; Ortet takes `--recovery-file` relative to `production_data`.

Model requests make at most four attempts. Output-length exhaustion becomes
`length_deferred` immediately. Transport failures trigger relay discovery before
another attempt. Each attempt retains its raw stream and endpoint in the receipt;
the final stream remains `response.sse` for export verification. A failure without
a terminal response leaves an uncertain admission lease and holds cursor progress
for reconciliation.

Touch `/tmp/tool-scale-r<N>/STOP` or `/tmp/tool-ortet-r<N>/STOP` to stop
dispatch and active retries for that replica. The
allocation deadline also stops dispatch, retry waits and streaming reads.
Network reads have a maximum 30-second inactivity timeout; a stopped request
with an uncertain outcome keeps its lease. The stopped batch is archived before
the driver reports that reconciliation is required.

Deterministic mappings are accepted only when the pinned rule approval covers
every applied rule and matches the rule version and implementation hashes.
Changed approval artifacts or implementations fail verification.

`rule-approval-v4.json` carries the original sample audit forward after removal
of unused declarations from `native_seed.py` and `candidate.py`. Its
`implementation_equivalence` records the approved and cleaned hashes, source
commits, removed symbols and matching retained-module AST hashes. The original
`rule-approval-v3.json` remains available for historical provenance.
