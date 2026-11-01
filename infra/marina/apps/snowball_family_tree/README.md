# Add a Snowball model

Add an entry to `nodes` in [graph-data.json](graph-data.json); the tree, timeline
and Read all views derive from it. Copy a nearby node and preserve its fields.

## Propose a graph change

`graph-data.json` on Marin's `main` branch is the source of truth. Every model
addition or lineage change goes through a reviewed Marin PR, including outside
contributions submitted from a fork or opened on the contributor's behalf.
Submit source changes in the PR; do not pass graph files or ZIPs to a publisher
for a separate edit to the live tree.

The tree is curated. Explain why the checkpoint matters to the model family and
link the experiment or model author's assessment. Include releases, distinct
training or merge stages, and negative results that explain a lineage decision;
individual trial checkpoints can remain aliases in `retained`. Automated discovery
may propose a PR, but must preserve review of model selection and parent evidence.
It must not add every discovered post-train directly to the live graph.

Each graph PR must identify its model IDs, parent evidence, uncertain or inferred
links, validation results, and whether it changes the public tree. A merge updates
the source; once activated, the nightly publisher releases changed main source
at 04:17 UTC.
The maintenance guide covers manual dispatch, release records and legacy bootstrap.

## Record the model and its parent

- Give `id` a unique, stable value; public model nodes normally use the full
  Hugging Face `owner/repo` ID. Set `model` to that ID, or `null` for a run without
  published weights. Set `title` and `summary` from the experiment record.
  An optional `short` string provides the compact map label; without it the
  frontend derives one from `title`.
  `stage` is a display label such as `SFT` or `RL`; `algorithms` is an array of
  strings. `status` is free text; include `failed` or `negative result` for the
  failure badge when applicable.
- Set `parents` to a one-element array containing the immediate weight parent's
  existing `id`; model merges list every pinned parent checkpoint instead.
  Follow the actual checkpoint and long-context extension variant.
  Architecture metadata and evaluation comparisons do not establish weight
  ancestry. Set `edge_kind` to `initialization` or
  `retained checkpoint` for progression within the same training-state chain.
- Use `lane: "2.7T"`, `"5.7T"` or `"10T"` according to cooldown ancestry.
  Set `org` to `open-athena` or `community` for experiment ownership. All existing
  2.7T checkpoints belong to open-athena, including exports in other namespaces.
- Set `source` to the canonical issue, model card or run record. Add supporting
  `evidence` entries with `url` and `text`. Use `confidence: "documented"` for
  established ancestry or `"inferred"` for a qualified link, which renders dashed.
  If no parent can be supported, add an entry to the top-level `unplaced` array
  with `model`, `url` and `reason` until its ancestry is established.
- Set `timeline` to an object with `date` (`YYYY-MM-DD` or `null`), `basis` and
  `source`. Explain whether the date comes from a checkpoint label, training run,
  report or repository creation. Undated nodes appear last. Keep provenance dates
  separate from the 10T playback opening date, September 7, 2026.
- Set `flops` to a scoped estimate or `FLOPs unknown`; explain assumptions in
  `notes`. Training tokens, rollout tokens and inherited compute have different
  scopes. Put same-run aliases or additional exports in `retained` as objects with
  `label`, `url` and optional `note`; create a node when a checkpoint needs its own
  children or training-stage description.

The app is public. Use public source links and small numeric extracts under
`evidence/` when needed. Keep credentials, private logs and local paths out of the
graph and bundle. Refresh the snapshot date in `graph-data.json` and the displayed
dates in `index.html` when rechecking evidence. Update the node count in the
[maintenance guide](../../../../docs/references/snowball-family-tree.md).
View counts are computed automatically.

## Check and submit

The build rejects duplicate IDs, missing parents, cycles and cross-cooldown
parent links. From the repository root, run the layout harness to check that
every visible node is placed once and every parent arrow ends at its child,
then validate the package and preview it:

```bash
node infra/marina/apps/snowball_family_tree/verify_layout.js
uv run marina validate infra/marina/apps/snowball_family_tree
uv run marina release infra/marina/apps/snowball_family_tree --dry-run --json
uv run marina release infra/marina/apps/snowball_family_tree --local --json
```

Open the printed local URL. Search for the new model, expand it, check its selected
ancestry and parent's Direct children list, then reveal it in its family timeline
and open the evidence links. Ctrl-C stops Marina and its disposable Postgres;
verify both stop. Each build recreates `dist/`; commit source files only.

Open a focused Marin PR with the information listed above.
For public updates, follow the existing UUID and current-version procedure in the
[maintenance guide](../../../../docs/references/snowball-family-tree.md#maintain-the-app)
and the `marina-app` skill. Opening a PR does not publish a public revision.
