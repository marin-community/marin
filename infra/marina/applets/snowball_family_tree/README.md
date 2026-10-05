# Add a Snowball model

Add an entry to `nodes` in [graph-data.json](graph-data.json); the tree, timeline
and Read all views derive from it. Copy a nearby node and preserve its fields.

## Record the model and its parent

- Give `id` a unique, stable value; public model nodes normally use the full
  Hugging Face `owner/repo` ID. Set `model` to that ID, or `null` for a run without
  published weights. Set `title` and `summary` from the experiment record.
  `stage` is a display label such as `SFT` or `RL`; `algorithms` is an array of
  strings. `status` is free text; include `failed` or `negative result` for the
  failure badge when applicable.
- Set `parents` to a one-element array containing the immediate weight parent's
  existing `id`. Follow the actual checkpoint and long-context extension variant.
  Architecture metadata and evaluation comparisons do not establish weight
  ancestry. Set `edge_kind` to `initialization` or
  `retained checkpoint` for progression within the same run.
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

The applet is public. Use public source links and small numeric extracts under
`evidence/` when needed. Keep credentials, private logs and local paths out of the
graph and bundle. Refresh the snapshot date in `graph-data.json` and the displayed
dates in `index.html` when rechecking evidence. Update the node count in the
[maintenance guide](../../../../docs/references/snowball-family-tree.md).
View counts are computed automatically.

## Check and submit

From the repository root, check that every new ID is unique, its parent exists,
the parent chain has no cycle and its cooldown agrees with `lane`. Then run:

```bash
uv run marina validate infra/marina/applets/snowball_family_tree
uv run marina publish infra/marina/applets/snowball_family_tree --local --mode public
```

Open the printed local URL. Search for the new model, expand it, check its selected
ancestry and parent's Direct children list, then reveal it in its family timeline
and open the evidence links. Ctrl-C stops Marina and its disposable Postgres;
verify both stop. Each build recreates `dist/`; commit source files only.

Open a focused Marin PR with the model IDs, parent evidence and validation results.
For public updates, follow the existing UUID and current-version procedure in the
[maintenance guide](../../../../docs/references/snowball-family-tree.md#maintain-the-applet)
and the `marina-applet` skill. Updating the source in a PR does not automatically
publish a new public revision.
