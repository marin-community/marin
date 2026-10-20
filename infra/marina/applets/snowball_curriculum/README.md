# Snowball v0 curriculum browser

Private Marina applet for every environment in a frozen selection. The browser
supports source, family and difficulty-prior filters, literal text search, paging,
model-visible task details and contract/source lineage. It displays the actual
count, target and shortfall separately.

The database stores a public TaskSpec projection. Reference answers, grader
payloads, verifier resources and environment variables are excluded. Keep the
full training artifact in private storage; the optional artifact URI appears
separately in the provenance panel. Do not release this applet in public mode.

Historical native structural tasks show the supplied assistant prefix,
historical token IDs, format/control metadata and tokenizer pin as source audit
information. These values do not establish compatibility with a current model
or runtime. Current TaskSpecs display their public conversation, including an
unfinished assistant message when present. The browser does not compile tasks
or execute graders.

## Validate and release

```bash
uv run marina validate infra/marina/applets/snowball_curriculum
uv run marina release infra/marina/applets/snowball_curriculum --dry-run --json
uv run marina applets versions e2bd5c9d-0f93-4ce9-b7af-3c78ca55e5d1 --json
uv run marina release infra/marina/applets/snowball_curriculum --base-version CURRENT_VERSION --json
```

Use the inspected current version for the declared UUID,
`e2bd5c9d-0f93-4ce9-b7af-3c78ca55e5d1`. The build packages the maintained HTML,
JavaScript and stylesheet into `dist/`. Application release does not load data.
Use `--local` for a disposable preview; see the [Marina reference](../../../../docs/references/marina.md)
for branch previews and release provenance.

## Load the complete pool

The loader accepts a local directory containing the selection output:

- `artifact-manifest.json`: complete payload inventory; its expected SHA256 is a required loader input.
- `tasks.jsonl`: full selected wrapper records with `task_json`, identity and source fields.
- `selection.json`: `selected_count`, `target`, `shortfall` and `source_quotas`.
- `recipe.json` (optional): resolved source families, policy and provenance pins.
- `selected-aliases.jsonl` (optional): contract IDs and source/task aliases.
- `readiness.json` (optional): measured loader, grading and latest-stack smoke status.

Missing readiness evidence appears as `not performed`. Difficulty values are
displayed as provisional priors. Unknown values remain visible.

```bash
uv run python -m infra.marina.applets.snowball_curriculum.load \
  --pool /absolute/path/to/frozen-pool --manifest-sha256 MANIFEST_SHA256 \
  --applet-id e2bd5c9d-0f93-4ce9-b7af-3c78ca55e5d1 \
  --artifact-uri s3://private-bucket/frozen-pool/tasks.parquet
```

The source-only loader reuses the maintained curriculum bundle verifier. It
requires the data pipeline source checkout. It checks the pinned manifest and
every listed payload before and after reading, then verifies counts, quotas and
unique contracts before staging. It
checks stored counts before atomically selecting the complete snapshot. A failed
or concurrent load preserves the previously selected pool. Prior staged loads
remain available for inspection; loading does not delete them.
One authenticated HTTP session and expiring-token provider are reused throughout
the CLI load. Token renewal stays with the existing provider.

For a disposable preview, use `marina release ... --local` and pass its printed
origin to the loader with `--url`. Omitting `--url` targets production.

```bash
uv run pytest infra/marina/applets/snowball_curriculum/tests
```
