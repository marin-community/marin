# Decontamination precision project (marin#6852)

Goal: fix decon FP mechanisms A/B/D + build a decon-specific web app to look at
the data, run precision/recall tests, and judge whether the algorithm is right.
Be data-driven: sample → hypothesize → fix → re-run → verify no data loss →
regression test → comment on #6852. Iterate on 100M (fast) / 100B, validate on 1T.

## Algorithm (as-is)
`decon.py`: per corpus doc, split on `\n` into paragraphs; per paragraph take
contiguous 13-grams (`_extract_ngrams`: `text.split()`); flag doc if any
paragraph's fraction of bloom-hit 13-grams ≥ 0.5. Bloom built from eval `text`
(`_build_filter` → `_extract_features`, same 13-gram-per-paragraph). Paragraphs
< 13 whitespace tokens contribute nothing (no fallback; #5656 removed it).
`eval_hash_index.parquet` (hash→eval_id) attributes matches.

## Data findings (looked at real eval records, R2 evals)
- Eval records are flat `{id, text}`. **Reading-comprehension passages are baked
  into `text`**: DROP text[0] = 334-word game recap; BoolQ = 458-word passage;
  SQuAD = `Title:…\n\nBackground:[passage]`. gsm8k is clean Q+A.
- `prepare_eval_corpus.py`: **AA** benchmarks use curated `text_fields`
  (e.g. HLE `("question","answer")`); **LMH** tasks use `_concat_strings` (dumps
  every string field → passages leak in). ⇒ cluster B is an LMH-text problem.
- Cluster A already partly merged: `DECON_EXCLUDED_EVAL_TASKS` excludes
  code2text_*, jsonschema_bench_*, swde, realtoxicityprompts.

## Clusters (fix A, B, D — not C per user)
- **A** answerless corpus-material evals → extend/verify the eval exclusion set
  (`DECON_EXCLUDED_EVAL_TASKS`). Data-check which excluded/remaining evals are
  "the eval doc IS ordinary public material".
- **B** shared public passages (squad/drop/boolq/coqa/qnli/race/history-MMLU) →
  index only Q+A for LMH reading-comprehension tasks, not the passage.
- **D** coincidental short/numeric overlaps → drop pure-numeric / pure-punct
  13-grams (and require a min. non-boilerplate token content) in `_extract_ngrams`.

## Web app (my exploration tool — extend as needed)
- Data prep (reno/CW job): per decon run export per-source flag rates + sampled
  flagged docs joined to their sample text + matched eval families
  (matched_hashes → eval_hash_index → eval_id/family). Compact JSON per run.
- App (local Flask): single-run view (per-source table → doc drill-down w/ matched
  spans + eval attribution) + two-run compare (per-source Δ, newly/no-longer flagged).

## Precision / recall
- Precision: sample flagged docs per source, classify TP/FP by reading text +
  matched eval (LLM judge = `experiments/datakit/decontam/ops/precision_analysis.py`).
- Recall: injection test — inject known eval items (verbatim, short-line-wrapped,
  embedded) into corpus docs, confirm flagged. Track before/after each fix.

## Status log
- (init) understood algorithm + eval data; launching 100M baseline decon.

## codex review findings (2026-07-09) — issues to address
- **CRITICAL (cache):** changing `_extract_ngrams` (D) needs a `FEATURE_FILTER_VERSION`
  hash-attr on BOTH `build_eval_bloom_step` and `decon_step` mark, else stale
  cached blooms/marks are silently reused. Also record in `EvalBloom`.
- **B placement:** do it in `prepare_eval_corpus.py` (decon sees only flat `{id,text}`).
  LMH rows append `doc_to_text`/`doc_to_target` THEN `_concat_strings(doc)` (every raw
  field) at prepare_eval_corpus.py:444/459 → passage indexed twice. Add
  `_extract_lmh_text(child_name, task, doc)`; for RC/QA build from question/answer/options,
  skip `doc_to_text` when it embeds the passage. Requires regenerating LMH parquet (new root).
- **D placement:** filter in `_extract_ngrams` (build+mark share candidate set → denominator
  consistent). Rule: skip ngrams with no alphabetic char. Accepts loss of numeric-only recall.
- **A:** `DECON_EXCLUDED_EVAL_TASKS` + `exclude_eval_dirs` is the clean path (hashed, read-time).
  Pitfall: inline `decon_step(eval_data_sources=)` has no exclusion param (testbed uses prebuilt bloom, OK).
- Algorithm nits: `matched_hashes` deduped to a set on mark but scoring counts dup ngrams
  (attribution != score); `_build_filter` n_records counts zero-feature records;
  `_discover_eval_files` excludes by immediate parent dir only.
- exporter (fixed): unused import, resolved-path in `_read_parquet`, reservoir sampling,
  bloom-by-name + sample-by-prefix, `only_sources` validation.

## Status log (cont.)
- 100M baseline: too sparse (21 flags, 104/115 at 0). Flagged-doc read confirms
  cluster B dominant (anli_r3 AP news, race 1.0, coqa 0.97, triviaqa). Posted to #6852.
- Viewer simplified to single-page report.py (+codex report fixes: head order, script-safe, escape).
- Local CW read wired (virtual-host addressing, endpoint cwobject.com).
- **Cluster D DONE**: `_extract_ngrams` drops no-alpha 13-grams; `FEATURE_FILTER_VERSION=2`
  on bloom+mark hash_attrs (cache). Tests: test_extract_ngrams_drops_letterless_ngrams,
  test_decon_skips_numeric_only_contamination. `pytest tests/datakit/test_decon.py` = 26 passed, 5 xfailed.
- TODO: A (verify/extend exclusion from 100B data), B (LMH Q+A-only, hardest),
  100B before/after validation w/ per-family cluster attribution, 1T final, precision+recall harness.

## Cluster B design (grounded in RC eval structure, 2026-07-09)
RC eval flattened text = doc_to_text(prompt+PASSAGE) + doc_to_target(answer) + _concat_strings(raw fields incl passage).
Real RC raw fields: race{article,question,answer,options}, boolq{passage,question,...}, anli{premise,hypothesis,label},
coqa{story,...}, squad{context,question,answers}, drop{passage,...}. triviaqa is pure Q+A (no passage).
FIX (surgical, in prepare_eval_corpus.py rows()): detect passage-bearing task = doc has a field in
_PASSAGE_FIELDS={passage,context,ctx,article,story,premise,background,document,paragraph,support}. For those,
SKIP doc_to_text (renders passage) + EXCLUDE passage field from raw concat; keep doc_to_target + other raw fields
(question/options/hypothesis). Non-passage tasks unchanged. Keeps genuine-leakage (Q+A) detection, drops public passage.
COST: requires regenerating the affected LMH eval parquets (delete + re-run prepare_eval_corpus for RC tasks) → new bloom → re-decon.
VALIDATE on 100M (RC-passage FPs on web sources: anli_r3/race/coqa/triviaqa → should drop; TP sources unchanged) + recall injection.
Corner (documented): passage field not in denylist, or question mis-named as passage → mis-handled (rare).
- 100B baseline succeeded; exporting (export_run job 091244).

## Status log (cont. 2)
- **Cluster B DONE (code+unit)**: prepare_eval_corpus._lmh_doc_text drops passage (raw field + doc_to_text)
  for passage-bearing docs (_PASSAGE_FIELDS), keeps Q+A. tests/datakit/decontam/test_prepare_eval_corpus.py = 13 passed.
  DATA-VALIDATION PENDING: needs regenerating RC eval parquets via lm-eval-harness (OUTPUT_ROOT=gs://marin-eu-west4),
  rebuild bloom, re-decon 100M. Heavy — scope to passage-bearing RC tasks.
- **GOTCHA**: my decon.py FEATURE_FILTER_VERSION edit changed decon step hashes → the 100B baseline (082814, old code)
  output is at OLD-hash paths; export_run with new code reads NEW-hash paths → 0/0. Killed export 091244.
  => before/after must both use consistent code. baseline_100m.json (exported pre-D-edit) is a valid BEFORE.
- **RECALL harness DONE + RUN** (ops/recall_test.py, local over R2 evals, 80 tasks/300 items):
  verbatim 100%, verbatim-in-doc 100%, short-line-wrapped 0%, embedded-1x 29.7%.
  Confirms precision-favoring/verbatim algorithm with short-line+embedded recall gaps (matches #6852).
- TODO: precision measure (systematic), B data-validation (regen), cluster A (from scaled data), 1T final.

## Status log (cont. 3) — MILESTONE
- **Cluster A DONE**: added wikitext + lambada_* to DECON_EXCLUDED_EVAL_TASKS (read-time, no regen). 16 excluded.
- **Cluster B DATA-VALIDATED** (ops/b_validate.py, real HF RC datasets, local): passage-only FP 300/300→~0/300
  (anli 1, race/boolq/squad 0); Q+A recall preserved exactly (base==fixed: 203/203, 94/94, 1/1, 57/57). NO DATA LOSS.
- **All 39 tests pass** (A/B/D + existing). Recall harness run (verbatim 100/short-line 0/embedded 30).
- REMAINING: lint + assemble PR; optional 1T final; algorithm write-up (precision-favoring + recall gaps).
