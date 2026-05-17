# Decontam Journal — Datakit All-Sources Decon

End-to-end log of the all-sources decontamination experiment under
`experiments/datakit/decontam/`. Updated as the pipeline runs.

## Goal

Build a single eval-overlap bloom filter from **every** AA Intelligence
Index v4.0 core eval + every lm-eval-harness task referenced by
`experiments/evals/task_configs.py` and use it to scan all 104 normalized
source corpora exposed via `marin.datakit.sources.all_sources`. For each
corpus, emit a parquet copy that flags every doc whose 13-gram word
content overlaps an eval at ≥0.5 ratio.

The headline question this experiment must answer: **does the decon
pipeline produce a corpus that is both (a) recall-rich (no eval text
leaks through into training) and (b) precision-clean (it doesn't flag
unrelated docs that just happen to share ngrams)?**

## Pipeline shape

```
experiments/evals/task_configs.py
       │  bundles → list[EvalTaskConfig]
       ▼
prepare_eval_corpus.py  ───► gs://marin-eu-west4/datakit/decontam/evals/
   ├─ aa/<eval>/<split>.parquet           (8 evals, AA core-8)
   └─ lmh/<task>/eval.parquet             (~850 leaves after group expansion)
                                          schema: {id: string, text: string}
       │
       ▼
count_docs.py                ───► bloom-sizing recommendation
                                  (eval unique-ngram count, ~2× headroom)
       │
       ▼
all_sources_decon.py
   ├─ build_eval_bloom_step over evals → combined filter.bin + hash-index
   └─ decon_step per source     ───► gs://.../datakit/decon/<source>/...
                                     parquet with eval-overlap flags
```

## Coverage decision (2026-05-17)

LMH coverage on iris fresh state: **850/944 leaves (90%)** after applying
a `trust_remote_code` monkey-patch with fallback. The remaining 94 fails
split into two structural buckets we explicitly decided NOT to address:

* 40 "Dataset scripts no longer supported" — `datasets` 4.x dropped
  python loader support; affects arithmetic_*, crows_pairs_english_*,
  ethics_*, logiqa*, mathqa, mc_taco, mutual*, prost, pubmedqa,
  qasper_*, social_iqa, wsc273, cmmlu, moral_stories.
* 53 `include_base_44_*` KeyErrors — task names not registered in our
  pinned lm-eval-harness commit `d5e3391f`.

Options considered: pin `datasets<4`, bump lm-eval commit, or accept
90%. Chose accept-90% — the missing evals are small, and any future
contamination incident in those families can drive a targeted fix.

## Decisions log

* **Format**: parquet ({id: string, text: string}) instead of jsonl.gz.
  Decon code reads via zephyr (which handles both), but parquet avoids
  gzip decompression on every bloom build and is the marin standard.
* **Bloom architecture** (TBD this run): the original
  `all_sources_decon.py` builds one bloom per eval and merges. With 850
  leaves that's 850 bloom build steps — wasteful. Likely collapse into
  one combined bloom over the whole `evals/` tree.
* **API key for Claude judge**: `.marin.yaml` → `ANTHROPIC_API_KEY`.

## Run log

### 2026-05-17 — prepare_eval_corpus.py (parquet)

Submitted iris job `iris-run-prepare_eval_corpus-20260517-072359` with
`--memory 16GB --cpu 2 --enable-extra-resources --extra=eval`. Started
07:23:59 UTC.

* 8 AA evals written by 07:24:49 (`hle`, `aa_omniscience`, `ifbench`,
  `gpqa_diamond`, `mmlu_pro`, `scicode`, `gdpval`, `livecodebench`).
* LMH phase entered: `_lmh_task_names()` reports 355 unique task names
  (vs 180 found by verify's AST regex — the bundles include
  multi-line `EvalTaskConfig(...)` definitions the regex missed).
* Status as of journal write: in progress.

### Earlier runs (this PR)

* Run 1 (jsonl.gz, --memory default): OOM at AA livecodebench.
* Run 2 (jsonl.gz, --memory 16G, --extra=cpu only): `ModuleNotFoundError: lm_eval`.
* Run 3 (jsonl.gz, --extra=cpu + --extra=eval): wrote 8 AA + 0 LMH; the
  initial `trust_remote_code=True`-always patch broke csv/parquet
  builders that don't accept the kwarg.
* Run 4 (jsonl.gz, surgical fallback): wrote 8 AA + 849 LMH (90%).

### TBD — count_docs.py + all_sources_decon.py

After parquet run lands, re-size the bloom, restructure the DAG,
submit on iris (eu-west4). 104 normalized sources, in-region scan.

## Analysis plan (placeholder; filled in as runs complete)

1. **Per-source flag rates**: how many docs in each of the 104 sources
   got flagged? Are rates plausible (small fraction) or alarming
   (a corpus that's full of eval contamination)?
2. **Precision via Claude judge**: sample ~200 flagged corpus docs,
   pull the candidate eval text via the hash-index sidecar, ask Claude
   whether the corpus doc substantively contains eval content. Report
   confusion matrix.
3. **Recall via synthetic injection**: take ~50 eval snippets, plant
   into a tiny held-out corpus, run targeted decon, see how many fire.
4. **Failure-mode catalogue**: cluster Claude-judged false positives
   and false negatives to find systematic gaps (eval-text formatting
   quirks? bloom FPR effects? ngram-length cliff?).

Will populate as we go.
