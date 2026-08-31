# Agentic-lint benchmark

`corpus.jsonl` is the labeled source of truth for changes to the `infra/lint/`
catalog. Each row contains a minimal diff, its lane, the rule codes expected to
fire, and label provenance. Cases with no expected rules are hard negatives.

The corpus has one positive case for each of the 63 catalog rules and three
hard negatives for each lane. It is not loaded into the weekly PostgreSQL
workbench and does not gate the agent's ordinary catalog pull request. It is a
starting point for the separate agent-workbench evaluation: that harness should
hide `expected_rules` and provenance from the agent, score the returned codes,
and report its catalog and corpus identities.

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

Candidate discovery evidence is development data. Human review remains the
catalog merge gate. Add automated probe and benchmark guardrails only after the
weekly workflow has produced enough real catalog pull requests to establish
which checks reject useful changes or catch overbroad ones.
