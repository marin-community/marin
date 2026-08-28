# Agentic-lint benchmark

`corpus.jsonl` is the labeled source of truth for changes to the `infra/lint/`
catalog. Each row contains a minimal diff, its lane, the rule codes expected to
fire, and label provenance. Cases with no expected rules are hard negatives.

The corpus has one positive case for each of the 63 catalog rules and three
hard negatives for each lane. The review-corpus exporter rejects missing rule
coverage, unknown or cross-lane labels, duplicate case IDs, and fewer than
three hard negatives per lane.

The exporter does not copy `corpus.jsonl` into a refinement archive. It writes
two files under `benchmark/`:

- `cases.jsonl` contains `alias`, `lane`, `diff`, and `changed_lines`. Prediction
  workers read this file and the catalog.
- `labels.jsonl` maps each alias to the source ID, description, expected rules,
  provenance, and optional source metadata. Evaluation reads this file after
  predictions are complete.

Aliases use `case-NNN`. The exporter hashes the model-visible case content,
sorts by that hash, and assigns aliases in order. The split and
`manifest.json:benchmark_sha` therefore remain stable when source rows or JSON
keys are reordered. `benchmark_sha` covers normalized cases and labels. Corpus
validation checks both split files, alias ordering, catalog coverage, and the
normalized hash.

## Adding cases

- Give every case a stable, label-neutral ID. Do not put `positive`, `negative`,
  or an expected rule code in the ID.
- Keep the diff small enough to audit while preserving the code shape the rule
  needs. Meta-lane cases must retain the relevant cross-file relationship and
  set `changed_lines` above the production lane's 100-line gate.
- Add hard negatives alongside positives, especially accepted suppressors and
  near misses.
- Record `source_url`, `source_pr`, and `source_author` for human-review cases.
  The runner excludes those fields from model input.
- Use `catalog-example` only for a case derived from the checked-in rule text,
  `human-review` for a labeled review example, and
  `synthetic-hard-negative` for a constructed near miss.

Candidate discovery evidence is development data. Candidate adoption requires
independently labeled fixed cases in this corpus and human approval of the
catalog change. Discovery requires evidence from three distinct PRs. A report
scores the fixed benchmark once for a catalog/corpus identity; its 7-day and
30-day evidence views do not produce separate benchmark baselines.
