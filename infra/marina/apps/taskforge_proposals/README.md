# Taskforge proposals

Browse the Taskforge `curriculum-v3-n25-20261102` proposal run: 1,999 catalog capabilities, each
planned into 25 slots, about 50,000 proposals with their preflight verdicts.

App: <https://applets.marina.oa.dev/a/05b2dac2-284c-4029-a4c0-629f5b2c9b23/> (private).

- **Proposals**: every slot, paged and sorted in Postgres, with full-text search over the plan title,
  workflow, challenge, summary and the proposal body, and filters for domain, verification kind,
  environment, verdict, slot status, cut-off and token ranges. The filters live in the URL.
- **Proposal**: the body's sections in order, the verdict and its structural checks, the plan's slot
  assignment, the header (build items, research, resources), token usage, and the stored
  `proposal.json` and `verdict.json` with copy and download. Links to the capability and its other slots.
- **Capability**: the 25 slots as cards, the plan's coverage rationale, excluded combinations and
  research priorities, the result and its provenance, the catalog entry, and the raw plan, result
  and failure JSON.
- **Stats**: counts and accept rates by kind, environment, verdict, status and domain, token
  histograms and a kind-by-domain table, all over the current filters.

The layout and colours follow the Taskforge viewer (`capability_env_gen/taskforge_viewer`).

## Data

`proposals` has one row per planned slot, including slots that were cut off or have not finished;
`capabilities` has one row per catalog capability; `run_stats` records the last ingest and the run's
provenance. Request and completions files are never loaded. Rows keep their paths, and the token
counts come from per-file usage summaries.

Those summaries come from the completions files (160 GB with stream events), so they are computed
in the storage region by an Iris job. Each job lists the run, skips files a previous part already
covers (same path, size and ETag) and writes `derived/slot-usage/part-<stamp>-<shard>-<attempt>.jsonl`
next to the run. Each shard owns a fixed stride of the sorted listing, so a retried shard reads only the
files its earlier attempt missed. From the repository root, with your own `CW_KEY_ID`/`CW_KEY_SECRET`
exported (the Iris client forwards them as job secrets, so they stay out of the job spec):

```bash
N=32; STAMP=$(date -u +%Y%m%d%H%M)
for i in $(seq 0 $((N-1))); do
  uv run iris --cluster=marin job run --target-cluster cw-us-east-02a --no-wait \
    --job-name tfviewer-usage-$STAMP-s$(printf %02d $i)of$N --cpu 4 --memory 3GB --disk 8GB --max-retries 2 \
    --sync-package marin-rigging \
    -- python infra/marina/apps/taskforge_proposals/slot_usage.py --shard $i --shards $N --stamp $STAMP
done
```

Then load the tables (a rerun uploads only rows whose source files or usage changed):

```bash
uv run python -m infra.marina.apps.taskforge_proposals.ingest --app-id 05b2dac2-284c-4029-a4c0-629f5b2c9b23
```

The ingest prints the stored counts and any problems: completions files without a usage summary,
or a capability whose summed usage differs from its `result.json`.

## Release

```bash
uv run marina validate infra/marina/apps/taskforge_proposals
uv run marina release infra/marina/apps/taskforge_proposals --local --json   # disposable preview
uv run marina apps versions 05b2dac2-284c-4029-a4c0-629f5b2c9b23 --json
uv run marina release infra/marina/apps/taskforge_proposals --base-version CURRENT_VERSION --json
```

For a preview, load a few capabilities into it with
`uv run python -m infra.marina.apps.taskforge_proposals.ingest --app-id PRINTED_ID --url PRINTED_ORIGIN --limit 50`. Tests:
`uv run pytest infra/marina/apps/taskforge_proposals`.
