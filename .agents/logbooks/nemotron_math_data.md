# Nemotron Math Data — Midtraining, Dedup, Tokenization

Logbook for the nemotron_cc_math_v1 data path used by Delphi midtraining:
how the runs consume it (CPT vs cooldown), how token budgets scale per base,
why the math val set is byte-identical across scales, how the upstream
dedup pipeline works, and how the tokenizer is wired (with a determinism
analysis).

---

## 2026-06-03 — Codebase analysis: CPT and cooldown on nemotron math

### Two modes: CPT vs cooldown (`lib/marin/src/marin/midtraining/modes.py`)

The system has exactly two midtraining modes, modeled as a closed union
`TrainingMode = CptMode | CooldownMode`.

**CPT (`CptMode`)** — "continued pretraining":

- Loads **model weights only** (`checkpoint_init_mode: model_only`), fresh
  optimizer state. The Delphi sweep streams HF weights directly:
  `initialize_from_hf: <repo>@<pinned_revision>`
  (`experiments/midtrain_specs/delphi_small_cpt_k020.py:241`).
- LR schedule is a **triangular** restart: 10% linear warmup → linear decay
  to ~0 (`CPT_DEFAULT_WARMUP_FRACTION = 0.10`, `CPT_DEFAULT_DECAY = None`,
  `modes.py:74`). `decay=None` is deliberate: a float `0.9` leaves a 1-step
  stable plateau when `num_train_steps` isn't divisible by 10 (Levanter
  truncates fractional stages independently).
- Peak LR = base's pretrain LR × `lr_factor` ∈ {0.33, 0.5, 0.67, 0.83}.

**Cooldown (`CooldownMode`)** — "true midtraining":

- Stages a pretrain checkpoint mid-run (e.g. step 30411 of 38014) into the
  new run's output path, then resumes with **full optimizer state**
  (`checkpoint_init_mode: full_state`). Preserving optimizer/scheduler/
  state-step is hard-required (`CooldownResume.__post_init__`).
- Critically: `num_train_steps` stays = `base.num_train_steps`
  (`modes.py:362`). The run continues the original WSD schedule to its
  natural end — only the data is swapped to a math-heavy mix. A "cooldown
  ratio 0.20" run resumes at the 80% checkpoint and trains the final 20% of
  the original schedule.

### Token scaling per model scale

- Each base (3e18 → 1e22 FLOPs) registers
  `tokens = num_train_steps × batch_size × 4096` in
  `experiments/delphi_models.py:185`. Batch size and steps grow with scale
  (B=8/37k steps at 3e18 → B=1024/38k at 1e22), so token budgets scale
  automatically.
- **CPT**: `BudgetPolicy.pretrain_fraction(0.20)` — every base gets K=0.20
  of its own pretrain tokens. Resolved as
  `steps = round(0.2 × base_tokens / (batch × 4096))`
  (`lib/marin/src/marin/midtraining/budget.py:96-127`). Fixed-token and
  fixed-step (probe) policies also exist.
- **Cooldown**: budget is implied by checkpoint choice, not a policy —
  `experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/checkpoint_candidates.yaml`
  enumerates targets per base × cooldown ratio (30/20/10%):
  `target_step = (1−K) × base.num_train_steps`, then "suggested" = nearest
  available checkpoint by absolute step delta; K=0.20 rows are exact
  materialized prefixes. Rows require `review_status: approved` — the
  launcher refuses to pick checkpoints dynamically.

### Math val set identity — the byte-identical contract

The strictest invariant in the codebase.
`experiments/midtrain_specs/data_sections/{p33m67,p50m50,p67m33}.json` hold
**verbatim copies of the `data:` block from the canonical 1e21 K=0.20
runs**. Every smaller-scale launch (CPT and cooldown alike) passes these
via `data_section_override`, so the rendered Levanter YAML is bit-identical
to the 1e21/1e22 reference:

- Same math cache
  (`gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-2c5519`), same
  `num_validation_sequences: 12500`, same
  `shuffle_before_trainval_split: true`, same Feistel shuffle params
  (`io_block_size: 256`, `window_blocks: 512`) → identical 12,500-sequence
  val carve-out at every scale.
- Only `train_weights` may differ across mixes (33/50/67% math).
- Enforced three ways:
  1. Spec validators refuse launches without provenance, val carve-out, or
     `shuffle_before_trainval_split`
     (`lib/marin/src/marin/midtraining/spec.py:310-328`).
  2. `tests/midtraining/test_val_set_equivalence.py` asserts the rendered
     `data:` block equals the reference JSON for every (base × mix × ratio)
     cell — "the load-bearing test for cross-scale loss comparability".
  3. W&B math-val losses are therefore directly comparable across scales.

Provenance per mix (only `train_weights` differs):

| mix    | source run |
|--------|------------|
| p33m67 | `gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.5-efbc63` |
| p50m50 | `gs://marin-us-east5/checkpoints/delphi-1e21-p50m50-9p25b-lr0.5-973c46` |
| p67m33 | `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.5-114e49` |

---

## 2026-06-03 — Dedup pipeline (XenonMolecule/marin, branch `michael-distill-dedup`)

Source: `experiments/baseline_collection/dedup/README.md` +
`experiments/baseline_collection/dedup_extracted.py` /
`tokenize_deduped_extracted.py` on that fork.

> **Superseded note:** an earlier revision of this branch shipped per-source
> drivers (`dedup_llm_curated.py` / `dedup_resiliparse.py`) with a
> prep → exact + fuzzy sidecars → apply DAG writing to 14-day-TTL temp
> buckets. That generation is gone — the branch now has one generic driver.

One script, `experiments/baseline_collection/dedup_extracted.py`, dedupes
the first-N WARCs of **any** extraction spec: `--spec <name> --n <N>`. Dedup
runs on **extracted text**; tokenization is a separate downstream step.

### Why fuzzy dedup

LLM extraction is **stochastic** — two extractions of "the same page" come
out as different strings, so exact-match dedup catches almost nothing. The
MinHash-LSH fuzzy pass (~0.75 Jaccard over 5-char n-grams) is what actually
collapses near-duplicate pages into one canonical record. The fuzzy pass is
the point of the pipeline; don't skip it.

Dedup is **N-dependent**: deduping the first-N WARCs only collapses
duplicates *within* those N. The N=100 deduped tree is NOT a prefix of the
N=500 tree — each (spec, N) is its own run and its own output dir.

### Pipeline stages (six StepSpecs, StepRunner DAG)

| Step | What it does |
|---|---|
| **reshape** | First-N WARC hashes from manifest (default `experiments/distill/baseline_warcs_3000.txt`) → pull done batches from consolidated archive → flat 200-shard `data-XXXXX-of-00200.jsonl.gz` tree (text-only schema). Round-robin pre-bucketing per output shard + `skip_existing` so each shard checkpoints independently (survives iris-coord preemption, replaces `.reshard(n)` barrier). |
| **normalize** | `datakit normalize_to_parquet`: jsonl.gz → `NormalizedData` parquet with `xxh3_128` content-hash ids and `DedupMode.EXACT` — exact dedup is folded in here for free. 64 MB partitions (vs marin's 256 MB) → ~4× more shards → 4× parallelism width downstream; pure parallelism knob (`--target-partition-bytes`), identical outputs. |
| **minhash** | Per-shard MinHash attrs: 286 perms, 26 bands, 5-char n-grams, seed 42. Co-partitioned 1:1 with normalize. |
| **fuzzy** | Global LSH + connected components across all minhash shards → per-doc cluster markers `{id, attributes: {dup_cluster_id, is_cluster_canonical}}`. Exactly one row per cluster is canonical; singletons get no row. `cc_resume=True` resumes from the last complete CC iteration on disk (survives preemption). |
| **deduped** | Join normalize parquet ⨝ fuzzy attrs by shard basename, drop `is_cluster_canonical=False`, keep canonicals + singletons → deduped jsonl.gz — what the tokenizer reads. |
| **stats** | `dedup_stats.json`: all step paths + fuzzy params. |

### Parameters

286 perms / 26 bands / 5-char n-grams / seed 42 / approx Jaccard ≈ 0.75 on
`text`. Document-level only — no paragraph dedup. More aggressive than DCLM
BFF (word-13-gram, 0.8 overlap) and Nemotron-CC (260 perms / 20 bands), so
dedup rates read slightly higher than published comparisons.

### Output layout — permanent, regional, no TTL

```text
gs://marin-{region}/documents/baseline_{spec}_deduped/{n}warcs/
    reshape/data-XXXXX-of-00200.jsonl.gz
    normalize/outputs/main/part-*.parquet
    minhash/outputs/<basename>.parquet
    fuzzy/outputs/source_000/<basename>.parquet
    deduped/data-XXXXX-of-YYYYY.jsonl.gz   <-- tokenize reads here
    stats/dedup_stats.json
```

Fixed `override_output_path` per step — no content-hashed cache dirs;
reruns only change with `(spec, n)`, which already changes the bucket.
Changing `--region` without copying cached state reruns from scratch.

### Running

Region-portable: reads raw input AND writes outputs in the same region (no
cross-region reads). Supported: `us-central1` (default), `us-east5`,
`us-west4` (driver maps eu/east1 buckets too). Ray retired — `iris job run`.

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
    --cpu 4 --memory 16GB --disk 20GB \
    --priority interactive --extra cpu --enable-extra-resources \
    --region us-central1 \
    --job-name dedup-<spec>-<N>warcs \
    -e WANDB_API_KEY <KEY> -e HF_TOKEN <TOKEN> \
    -- python experiments/baseline_collection/dedup_extracted.py \
       --spec <spec> --n <N> [--region us-central1]
```

Flags: `--spec`, `--n`, `--manifest`, `--region`,
`--target-partition-bytes` (parallelism width knob). Heavy-stage sizing
lives in `build_steps()` (fuzzy: `max_parallelism=1024`, 64 GB
**preemptible** workers, `cc_resume=True`) — bump there, not CLI.

Hard-won settings in code comments:
- Fuzzy 32g → 64g RAM after stage1-Reduce OOM; 96g reverted — slots too
  rare cluster-wide; small slots schedule far easier and converge faster.
- Preemptible widens schedulable host pool ~10×; `cc_resume` is the safety
  net — CC iterations checkpoint to `{output}/metadata/cc/it_*/`.

### Tokenize the survivors (separate, in-region)

```bash
... -- python experiments/baseline_collection/tokenize_deduped_extracted.py \
       --spec <spec> --n <N> [--region us-east5]
```

Reads `…/baseline_{spec}_deduped/{n}warcs/deduped/data-*.jsonl.gz`, writes
a Llama-3.1-8B cache (`default_tokenize`, `llama3_tokenizer`) to
`gs://marin-{region}/tokenized/{spec}_{n}warcs-<cache_hash>/`. Canonical
name `{spec}_{n}warcs` is the entry pasted into `_D_OBS_DEFAULTS`.
Note: tokenize script's `--region` default is **us-east5** while dedup
defaults to us-central1 — pass regions explicitly.

### Why dedup before tokenize

Fuzzy works on 5-char n-grams of raw `text`; exact pass hashes that text.
Tokenizing first would be both wrong (wrong granularity for MinHash) and
wasteful (tokenizer spends compute on docs about to be dropped).

### Local availability

- Our repo has the dedup library: `deduplication/exact.py`,
  `deduplication/fuzzy_minhash.py` (`compute_minhash_attrs_step`),
  `deduplication/fuzzy_dups.py` (`compute_fuzzy_dups_attrs_step`),
  `connected_components.py` — same fuzzy defaults (286/26/5/42).
- Missing locally: the fork's `experiments/baseline_collection/` drivers —
  `dedup_extracted.py`, `tokenize_deduped_extracted.py`, the WARC manifest
  `experiments/distill/baseline_warcs_3000.txt`, and the consolidated
  archive layout (`gs://marin-us-central1/documents/baseline_llm_extraction_consolidated`).
- To adapt to nemotron math: skip reshape (math is already normalized
  parquet at `gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main/*.parquet`)
  → minhash → fuzzy → apply → tokenize with `llama3_tokenizer` in us-east5.

---

## 2026-06-03 — How tokenization works (and its determinism)

### The tokenization code

The chain, from experiment wiring down to the core:

1. `experiments/pretraining_datasets/nemotron_v2.py:47` —
   `tokenize_nemotron_v2_family("nemotron_cc_math_v1")` builds one
   `ExecutorStep` per subset with `fn=tokenize`, reading
   `normalized_step / "outputs/main/*.parquet"`, tokenizer =
   `llama3_tokenizer` (`meta-llama/Meta-Llama-3.1-8B`,
   `experiments/llama.py:16`), wrapped in `versioned(...)` so it
   participates in the output hash.
2. `lib/marin/src/marin/processing/tokenize/tokenize.py:360` —
   `tokenize()`: glob inputs → sort → group into size-balanced shards →
   run a Zephyr pipeline per split.
3. `lib/marin/src/marin/processing/tokenize/_core.py` — the actual work:
   `tokenize_pipeline()` attaches a content-hash id (`attach_id`, xxh3_128
   of the text), windows records, and `tokenize_batches_with_id()` runs
   Levanter's `BatchTokenizer` (HF Rust tokenizers backend) on each batch,
   preserving 1:1 id alignment.
4. `lib/marin/src/marin/processing/tokenize/store_builder.py:96` —
   `build_from_datasets()`: each Zephyr shard writes a Levanter cache at
   `part-NNNNN-of-MMMMM/`, then `consolidate_shard_cache_ledgers`
   (`lib/levanter/src/levanter/store/cache.py:1104`) merges per-shard
   ledgers into a top-level `shard_ledger.json`.

### Is it deterministic?

**Per-document token content: yes.**

- HF Rust tokenization is deterministic for a fixed tokenizer artifact.
  The tokenizer is staged via `mirror://tokenizers/` (a GCS mirror
  snapshot) before falling back to HF Hub (`levanter/tokenizers.py:576`),
  so in practice the artifact is frozen — though it's pinned by mirror
  snapshot, not by HF `revision=`.
- The long-string workaround (split >10K-char texts at whitespace before
  encoding) is deterministic.
- `attach_id` is a deterministic content hash.

**Document order in the cache: yes, given the same inputs.**

- Files are explicitly sorted by path before grouping (`tokenize.py:234`).
- `bundle_files_by_size` is a greedy, order-preserving bundling — groups
  are consecutive runs of sorted files.
- Within a shard, records stream in file order; tokenization is
  order-preserving (`map_shard` over sequential windows).
- Zephyr returns results keyed by shard index, not completion order
  (`_regroup_result_refs` — "each worker's ListShard maps to its own
  index"), so the consolidated ledger lists `part-00000`, `part-00001`, …
  in sorted-file order regardless of which worker finished first. Retries
  with `skip_existing` don't reorder anything.
- Even if `max_workers`/`num_shards` changed shard boundaries, the global
  concatenated document order would be unchanged (groups are consecutive
  runs of the same sorted file list).

**Caveats — what determinism depends on:**

- The input file set: the normalized parquet
  (`4plus_b05688a8/outputs/main/`). The normalize step upstream must
  itself be stable; the executor hash pins this.
- Tokenizer pinned by name, not content hash — the GCS mirror makes it
  stable in practice, but it's an operational guarantee, not a
  cryptographic one.
- Region-dependent executor hashes: paths feed the version hash, which is
  why the same logical step produced `4plus-0bd79d` (us-central2),
  `4plus-212a2d` (us-central1), and `4plus-2c5519` (us-east5) — different
  cache identities across regions even though contents would match.

### The important design point (tokenization)

The worktree doesn't rely on re-tokenization determinism at all. The data
sections pin the exact cache artifact
`gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-2c5519` by path,
and the byte-identical math val split comes from Levanter's training-time
carve-out on that fixed cache: deterministic feistel shuffle
(`io_block_size: 256`, `window_blocks: 512`) +
`shuffle_before_trainval_split: True` +
`num_validation_sequences: 12500`, all frozen in the JSON. So val-set
identity is guaranteed by artifact pinning, with tokenization determinism
only mattering if the cache ever had to be rebuilt.

---

## 2026-06-03 — Val-split contamination investigation (1e22 "too good" concern)

**Hypothesis:** the 12,500-sequence math val split contains near-duplicates
of training documents, inflating apparent val performance at large scale.

### Reconstructing the val split (no detokenization needed)

The val carve-out is fully deterministic
(`lib/levanter/src/levanter/data/text/datasets.py:549`):

1. Math component dataset = consecutive 4096-token windows over
   `4plus-2c5519` cache: 51,482,955,946 tokens // 4096 = **12,569,081 windows**.
2. Levanter shuffles with feistel permutation under fixed `PRNGKey(0)`.
3. Val = last 12,500 indices of permuted order. Same key in
   `train_set()`/`validation_sets()` guarantees disjointness with train.

Replayed permutation with levanter's own `Permutation.make("feistel", N, PRNGKey(0))`:
12,500 unique window indices → `scratch/nemotron_math_val_window_indices.npy`.

### Mapping windows → documents → parquet rows

- Cache `train/input_ids/offsets` (zarr3 tensorstore) gives doc boundaries:
  45,096,087 docs. Window i = tokens [4096i, 4096(i+1)) → searchsorted →
  **57,243 val docs** (~125.7M tokens incl. window-boundary spill).
- Doc index → (shard, row): cache parts and normalized parquet shards are
  1:1 by name (231 parts; per-shard row counts match exactly).
- Verified alignment: re-tokenized 5 sampled docs (Llama-3.1-8B) → cache
  doc lengths match (+1 EOS).
- Normalized parquet already carries `id = xxh3_128(text)` (32-hex), and
  files are **globally sorted by id**. Same hash family as dedup pipeline.

### Contamination checks running

1. **Exact dup scan** (local): ids sorted → exact dup ⟺ adjacent ids equal.
2. **Fuzzy scan** (iris job `/ahmed/math-val-contamination`, us-east5):
   - extracted 57,243 val docs → `scratch/ahmed/midtrain_dedup/val_docs/`
   - MinHash over full 45.1M docs (286 perms / 26 bands / 5-char shingles,
     seed 42, ≈ 0.75 Jaccard)
   - LSH bucket join restricted to buckets containing ≥1 val doc → candidate
     pairs (val_id, other_id) → contamination stats.

### Progress log — 2026-06-03 (PT)

| time | event |
|---|---|
| 20:22 | `/ahmed/extract-math-val-docs` succeeded: 57,243 val docs (231 shards) at `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/val_docs/` |
| 20:28 | Submitted `/ahmed/math-val-contamination` (us-east5, preemptible CPU) |
| 20:30-20:40 | Stage `minhash-attrs` over 45.1M docs / 231 shards completed |
| 20:41 | Collected 1,487,655 val buckets (57,243 x 26 bands = 1,488,318 max ⟹ only ~663 collide within val) |
| 20:42 | `val-bucket-join` started: corpus-wide LSH filter against val buckets, then group-by-bucket → candidate pairs |
| 20:43 | One worker OOM (16 GB); zephyr auto-retried — watching for recurrence |

**Exact-duplicate scan (local):** all 231 normalized parquet shards are sorted
by id internally, but NOT range-partitioned: every shard spans the full
xxh3_128 range, so cross-shard duplicates can't be ruled out by adjacency.
Within-shard adjacent dup ids = 0. Global uniqueness check across all 45M
ids running locally (`scratch/nemotron_math_all_ids.npy`).

Verification stage ready: `scripts/analysis/verify_math_val_contamination.py`
- 5-char shingle Jaccard ≥ 0.75 on candidate pairs.
- id → (shard,row) via local global index (no global parquet sort).
- Output: scratch/val_window_contamination.json (per-window contamination).
| 20:44 | `val-bucket-join` coordinator OOM (exit 137, 16 GB) — `/ahmed/math-val-contamination` stopped |
| 20:46 | Relaunched as `/ahmed/math-val-contamination-v2`: coordinator 8 cpu / 64 GB, join workers 64 GB (was 16 GB). MinHash outputs reused (skip_existing=True), only collect+join rerun |
| 20:50 | v2 OOM again — collect-val-buckets coordinator died collecting 1.5M rows; rewrote pass 1 to write parquet + driver reads from GCS |
| 20:53 | Relaunched `/ahmed/math-val-contamination-v3` |
| 20:54 | **Exact dedup verdict: 0 duplicate xxh3 ids across all 45,096,087 docs.** Exact contamination ruled out; fuzzy is the open question |
| 20:55 | v3 minhash+collect succeeded (parquet-write fix held; coordinator no longer OOMs). 1.5M val buckets at scratch/ahmed/midtrain_dedup/val_buckets/ |
| 21:01 | v3 `val-bucket-join` coordinator OOM ×3 — zephyr coordinator default is 2 GB and the val-bucket frozenset (1.5M strings) is pickled into every task closure |
| 21:03 | v4: `coordinator_resources` 32 GB (join) / 16 GB (collect), workers 64 GB. Minhash + collect reuse cached output |
| 21:11 | **v4 succeeded.** 2,447,225 (bucket,id) records in val buckets; 169,956 buckets with ≥1 val + ≥1 other doc; **1,025,264 candidate pairs** → val_candidate_pairs/ |
| 21:15 | Submitted `/ahmed/verify-math-val-pairs`: per-shard 5-char-shingle Jaccard on candidates; report all ≥0.5 → verified_pairs/ |
| 21:18 | `/ahmed/verify-math-val-pairs` succeeded: 374,929 pairs ≥0.5 Jaccard; verified pairs at scratch/ahmed/midtrain_dedup/verified_pairs/ |
| 21:20 | **Verdict — see contamination summary below** |

### Contamination verdict — 2026-06-03

Exact contamination: **none** (all 45.1M xxh3 ids unique).
Fuzzy near-duplicates: **substantial**.

| Jaccard cutoff | val docs ≥1 train dup | share of 57,243 |
|---|---|---|
| 0.50 | 20,850 | 36.4% |
| 0.70 | 12,703 | 22.2% |
| **0.75 (canonical)** | **9,757** | **17.0%** |
| 0.80 | 6,586 | 11.5% |
| 0.90 | 1,011 | 1.8% |
| 0.95 | 97 | 0.2% |

Window-level impact (12,500 val windows × 4096 tokens):
- 6,839 / 12,500 windows = **54.7%** contain ≥1 contaminated doc.
- 9.53 M / 51.2 M val tokens = **18.6%** of val tokens fall inside docs
  with a J≥0.75 near-duplicate in train.

Interpretation: the val carve-out is a random shuffle over an
internally-redundant corpus — no fuzzy dedup was performed within the
math cache. Any model with capacity to memorize templated math pages
gets disproportionate credit on ~19% of val tokens. The 1e22 model's
"too good" val loss is consistent with elevated memorization of
near-duplicates, not measurement error.

Artifacts:
- `scratch/nemotron_math_val_window_indices.npy` — 12,500 val window indices.
- `scratch/nemotron_math_val_doc_indices.npy` — 57,243 val doc indices.
- `scratch/contaminated_val_ids.json` — 9,757 contaminated val doc ids.
- `scratch/val_window_contamination.json` — window/token summary.
- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/` — val_docs/, minhash/,
  val_buckets/, val_candidate_pairs/, verified_pairs/.

Suggested follow-ups:
1. Re-evaluate checkpoints on decontaminated val (drop 9,757 docs / mask windows).
2. Re-score scaling law fits to see if 1e22 improvement persists.
3. Fuzzy-dedup the corpus before any future math midtrain (val carve-out from canonical docs only).

### Example contaminated pairs — same page, different extraction (2026-06-03 21:30)

Side-by-side excerpts saved at `scratch/contam_examples.txt` (5 examples,
J = 1.0 → 0.75):

| Jaccard | val id → train id | What differs |
|---|---|---|
| 1.000 | `abca6d83…` → `d7448a37…` | only whitespace: single vs double space after headers. xxh3 differs ⟹ exact dedup blind |
| 0.950 | `eb5c2994…` → `783d761c…` | `#` vs `##` headings, "the multiplication of" vs "multiplication of" |
| 0.900 | `465f3cc7…` → `09ac6794…` | same Q&A; page header "Distributive Property" vs "Basic Algebra" |
| 0.800 | `b87f6a0a…` → `deba95aa…` | same StackExchange question; `\frac1x` vs `\frac{1}{x}` |
| 0.750 | `655ef31a…` → `28cc6bde…` | same forum thread; "Discussion" vs "Comments and Analysis", plain vs LaTeX |

Mechanism: stochastic extraction/normalization — exactly the failure mode the
`michael-distill-dedup` README describes ("two extractions of 'the same page'
come out as different strings — exact-match dedup catches almost nothing").
nemotron_cc_math_v1/4plus had exact dedup only, no MinHash pass, so near-dup
crawl snapshots cross the val/train boundary freely.

### Method (full chain)

1. **Val reconstruction** — `Permutation.make("feistel", 12_569_081, PRNGKey(0))`;
   val = last 12,500 permuted windows (`scratch/nemotron_math_val_window_indices.npy`).
2. **Docs** — cache offsets → searchsorted → 57,243 doc indices; doc↔parquet
   1:1 verified by row counts and 5-doc re-tokenization (cache len = retok + EOS).
3. **Candidates** — full-corpus MinHash (286/26/5-char, seed 42) → 1.49M val
   buckets → bucket join (171.8k buckets ≥1 val + ≥1 other) → 1,025,264 pairs.
4. **Verification** — exact 5-char shingle Jaccard ≥ 0.75 (dupekit-style
   lowercase + whitespace collapse) → 374,929 pairs ≥0.5, 72,174 train pairs ≥0.75.
5. **Mapping back** — window-level: contaminated tokens = window∩doc overlap.

### Operational notes (for future jobs)

- Zephyr coordinators default to **2 GB**; broadcast closures (1.5M strings)
  OOM both pipeline coordinators. Fix: write intermediates to GCS parquet +
  `coordinator_resources=ResourceConfig(ram="16g"/"32g")`.
- 231-shard map jobs (val extract, minhash, join, verify) total runtime ≈ 30 min
  on us-east5 preemptible CPU.

---

## 2026-06-04 — Codex audit: per-scale exposure, not just full-corpus contamination

Claude's code-path analysis is mostly right, but the earlier contamination
verdict was a full-corpus property: it asked whether each val doc has *some*
train near-duplicate anywhere in the math train split. It did not ask whether a
given K=0.20 model scale actually sampled that train document.

Local replay generated:

- `scratch/nemotron_math_isoflop_contamination_exposure.json`
- `scratch/nemotron_math_isoflop_contamination_exposure.csv`

Method:

1. Replayed the Levanter val split:
   `Permutation.make("feistel", 12_569_081, PRNGKey(0))`; sorted val windows
   match `scratch/nemotron_math_val_window_indices.npy`.
2. Replayed the train stream:
   `train_lm` seed 0 -> `data_key` -> `mix_key`/`shuffle_key`; no `data_seed`
   override found in the large or small launchers. The data loader starts from
   step 0 because CPT uses model-only init and resets data-loader state.
3. Accounted for `MixtureDataset` block rounding. For 2048-example mixture
   blocks, effective math examples per block are:
   - `p33m67`: 1376 / 2048 = 0.671875, not exactly 0.67.
   - `p50m50`: 1028 / 2048 = 0.501953125, not exactly 0.50.
   - `p67m33`: 681 / 2048 = 0.33251953125, not exactly 0.33.
4. Replayed the math component's `BlockShufflingDataset` and the fixed
   train/val permutation to get actual original 4096-token math windows seen
   by each scale x mix.
5. Mapped sampled windows to document intervals via
   `scratch/nemotron_math_doc_offsets.npy`.
6. Mapped verified Jaccard >= 0.75 pairs back from ids to doc indices using
   `scratch/nemotron_math_all_ids.npy`. Sanity check matched the previous
   full-corpus number: 72,174 train pairs, 9,757 val docs.

Key corrections / shortcomings in the earlier note:

- The split is sequence-level, not document-level. A val window is excluded
  from train, but the same source document can contribute other windows to
  training. This is a separate contamination mode from fuzzy near-dupes.
- The fuzzy verification normalization is stricter than the dedup pipeline's
  `CleanText` normalization, so J>=0.75 counts are likely lower bounds for
  dedup-style near-duplicate exposure.
- MinHash-LSH candidate generation is not exhaustive at the exact cutoff; the
  verified pair count is precise for candidates found, but still a recall-limited
  lower bound for all true J>=0.75 pairs.
- The old statement that normalized shards are "globally sorted by id" is
  wrong. Shards are sorted internally but not globally range-partitioned.
- The redesign note's `1e22` K=0.20 step count of 7,635 is stale/inconsistent
  with the current registry and legacy large-run code. With
  `DELPHI_1E22.num_train_steps=38,235` and B=1024, K=0.20 resolves to 7,647
  steps = 32,073,842,688 scheduled tokens.

For the 67% math mix (`p33m67`), per-scale exposure:

| scale | scheduled tokens | math tokens | unique math docs | near-dup val tokens exposed | same-doc val tokens exposed | combined val tokens exposed |
|---|---:|---:|---:|---:|---:|---:|
| 3e18 | 0.245B | 0.164B | 0.184M | 0.364M | 0.272M | 0.635M |
| 9e18 | 0.581B | 0.390B | 0.437M | 0.717M | 0.608M | 1.323M |
| 2e19 | 0.723B | 0.485B | 0.543M | 0.904M | 0.702M | 1.603M |
| 3e19 | 0.997B | 0.670B | 0.748M | 1.159M | 0.915M | 2.064M |
| 9e19 | 2.112B | 1.419B | 1.577M | 2.037M | 1.810M | 3.787M |
| 2e20 | 2.961B | 1.989B | 2.204M | 2.568M | 2.470M | 4.949M |
| 3e20 | 3.723B | 2.502B | 2.764M | 2.984M | 2.904M | 5.755M |
| 1e21 | 9.251B | 6.215B | 6.731M | 4.985M | 5.721M | 10.282M |
| 1e22 | 32.074B | 21.550B | 21.698M | 7.988M | 14.200M | 20.165M |

For `p33m67`, the 1e22 run saw about 3.47x as many math tokens as 1e21
(21.55B vs 6.22B), but it exposed about 1.96x as many contaminated val tokens
under the combined same-document + J>=0.75 near-dup definition (20.17M vs
10.28M). Compared with the 3e20 anchor, 1e22 exposed about 3.50x as many
combined contaminated val tokens (20.17M vs 5.76M).

For 1e22 across mixes:

| mix | math tokens | unique math docs | near-dup val tokens | same-doc val tokens | combined val tokens | combined train contaminating tokens |
|---|---:|---:|---:|---:|---:|---:|
| p33m67 | 21.550B | 21.698M | 7.988M | 14.200M | 20.165M | 91.011M |
| p50m50 | 16.100B | 16.623M | 7.299M | 11.430M | 17.281M | 67.960M |
| p67m33 | 10.665B | 11.300M | 6.322M | 8.567M | 14.015M | 45.296M |

Interpretation: the original full-corpus fuzzy contamination number
(9.53M val tokens at J>=0.75) remains useful as an upper envelope for
near-dup exposure, but actual model exposure is scale-dependent. The bigger
1e22 issue is not only that it can memorize more of the 9.53M fuzzy-contam
val-token envelope; it also samples much more of the same source documents as
the validation windows because the split is over fixed token windows, not
source documents.

### Threshold view — fuzzy contamination only

Generated:

- `scratch/nemotron_math_isoflop_jaccard_threshold_percentages.json`
- `scratch/nemotron_math_isoflop_jaccard_threshold_percentages.csv`

Denominator:

- Math val set = 12,500 fixed 4096-token validation sequences.
- Total val tokens = 51,200,000.
- These windows touch 57,243 source documents.

Definition for the table below: percentage of validation tokens whose source
validation document has at least one actually-sampled train document with
verified 5-char-shingle Jaccard >= threshold. This excludes the same-source
document/window-split issue and measures fuzzy near-duplicate exposure only.

For the 67% math mix (`p33m67`):

| Jaccard cutoff | 3e18 | 9e18 | 2e19 | 3e19 | 9e19 | 2e20 | 3e20 | 1e21 | 1e22 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| >=0.50 | 2.90% | 5.36% | 6.24% | 7.61% | 12.33% | 14.95% | 16.69% | 24.67% | 34.26% |
| >=0.70 | 1.25% | 2.43% | 2.95% | 3.73% | 6.37% | 7.93% | 9.00% | 13.93% | 20.80% |
| >=0.75 | 0.71% | 1.40% | 1.77% | 2.26% | 3.98% | 5.02% | 5.83% | 9.74% | 15.60% |
| >=0.80 | 0.35% | 0.69% | 0.84% | 1.06% | 1.98% | 2.65% | 3.09% | 5.79% | 10.23% |
| >=0.90 | 0.02% | 0.03% | 0.03% | 0.05% | 0.09% | 0.13% | 0.15% | 0.37% | 1.15% |
| >=0.95 | 0.00% | 0.00% | 0.00% | 0.00% | 0.01% | 0.01% | 0.02% | 0.03% | 0.09% |

Full-corpus upper envelope, independent of scale sampling:

| Jaccard cutoff | val docs | val windows | val tokens |
|---|---:|---:|---:|
| >=0.50 | 20,850 / 57,243 = 36.42% | 10,226 / 12,500 = 81.81% | 20.05M / 51.20M = 39.15% |
| >=0.70 | 12,703 / 57,243 = 22.19% | 8,058 / 12,500 = 64.46% | 12.29M / 51.20M = 24.01% |
| >=0.75 | 9,757 / 57,243 = 17.04% | 6,839 / 12,500 = 54.71% | 9.53M / 51.20M = 18.61% |
| >=0.80 | 6,586 / 57,243 = 11.51% | 5,193 / 12,500 = 41.54% | 6.51M / 51.20M = 12.72% |
| >=0.90 | 1,011 / 57,243 = 1.77% | 978 / 12,500 = 7.82% | 0.99M / 51.20M = 1.93% |
| >=0.95 | 97 / 57,243 = 0.17% | 97 / 12,500 = 0.78% | 0.09M / 51.20M = 0.18% |

### Concrete Jaccard 0.900 example

Downloaded one representative pair locally:

- Directory: `scratch/nemotron_math_jaccard_0p90_example/`
- Validation doc: `val_465f3cc78d3392f0c1a1e7a20e9eb181.md`
- Train analogue: `train_09ac679436c9b5dd3a34d877a62bc74e.md`
- Preview/comparison: `side_by_side.md`
- Metadata: `metadata.json`
- Paragraph diff: `paragraph_diff.patch`

Verified Jaccard = 0.9000 from
`scratch/verified_pairs/verified-00004-of-00231.parquet`.

The pair is a same-topic/same-body extraction for a Basic Algebra /
Distributive Property page. Differences are mainly wrapper title, heading
levels, punctuation, and compacting multi-line statements into one paragraph.
The body content and ordering are mostly shared, which is what Jaccard 0.9
means in this scan: not byte-identical, but substantively the same extracted
page/list.

## 2026-06-06 — High-recall (r=4) val scan, all math subsets

Driver `scripts/analysis/nemotron_math_val_full_scan.py` (284 perms / 71
bands, ~99% recall at J=0.5). 4plus rescan, us-east5 (`/ahmed/val-scan-4plus-v2`):
3.74M val buckets, 2.99B candidate pairs (vs 1.0M at 26 bands), 973M
len-pruned, 2.24M verified J>=0.5. Val docs with train near-dup: 32,619 (0.5)
/ 13,673 (0.7) / 10,030 (0.75) / 1,011 (0.9). Old 0.9 count confirmed
complete; 0.5 was 56% under. Failures: verify broadcast OOM (3.3B pairs);
4-cpu drivers unschedulable us-central1 (2-cpu pool); 1-cpu v3 ok.
Pending: `3`, `4plus_mind` scans us-central1. Verified pairs:
`gs://marin-us-east5/scratch/ahmed/midtrain_dedup/4plus_284x71/verified_pairs`.

### math `3` scan done (2026-06-06, /ahmed/val-scan-3-e5b)

3.40B candidates, 1.31B len-pruned, 439,618 verified J>=0.5. Val docs with
near-dup in math/3: 13,238 (0.5) / 4,155 (0.7) / 2,702 (0.75) / 147 (0.9).
4plus_mind scan still running. Note: cross-subset scans needed val-doc minhash
(val ids absent from corpus) — fixed in nemotron_math_val_full_scan.py.

### 2026-06-07 00:23 UTC — CODEX-20260607T002316Z-05567eea01 — stopped stray scan and hardened reducer path

Context: Claude had relaunched the 4plus_mind high-recall scan after a dedup
stage failure. The earlier stop command used `--no-include-children`, and a
still-running local Claude CLI process later submitted `/ahmed/val-scan-4plus-mind-e5e`.

Operational cleanup:

- Confirmed `/ahmed/val-scan-4plus-mind-e5d` was killed.
- Found `/ahmed/val-scan-4plus-mind-e5e` running and stopped it with children:
  `/ahmed/val-scan-4plus-mind-e5e/zephyr-minhash-attrs-44fe81e5-p0-a0`.
- Killed the local `claude --worktree nemotron_contam --dangerously-skip-permissions`
  process that was still polling/submitting from this worktree.
- Rechecked Iris running jobs: no `val-scan-4plus-mind*` or `decon-val-summary`
  jobs remained running.

Code status:

- Worktree: `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam`
- Branch: `nemotron-math-contamination`
- Current head: `0afdb4f90662` (`[analysis] Explicit parquet schemas + tuple pair key in scan pipeline`)
- Script: `scripts/analysis/nemotron_math_val_full_scan.py`

Reducer diagnosis:

- The failed dedup reducer returned `[dict]` from a normal function. Zephyr
  dispatches group-by reducers via `inspect.isgeneratorfunction`; a normal
  function returning a list emits that list as one record, causing the Parquet
  writer to infer schema from a non-dict record and fail.
- The fixed path keeps the first-pair reducer as a real generator function,
  uses a tuple `(val_id, other_id)` key instead of a delimited string, and
  writes explicit schemas for candidate, deduped, and verified-pair Parquet
  outputs so empty shards remain readable by column name.

Verification:

- `uv run --with pytest --with pytest-timeout pytest tests/analysis/test_nemotron_math_val_full_scan.py -q`
  -> 1 passed.
- `./infra/pre-commit.py --fix scripts/analysis/nemotron_math_val_full_scan.py tests/analysis/test_nemotron_math_val_full_scan.py docs/debug-log-nemotron-math-val-scan.md`
  -> OK.

Uncommitted local artifacts:

- `tests/analysis/test_nemotron_math_val_full_scan.py` — local Zephyr regression
  test for the pair-dedup reducer writing reloadable Parquet rows.
- `docs/debug-log-nemotron-math-val-scan.md` — debugging notes from this incident.

Next action: do not relaunch 4plus_mind until the human confirms scope and
whether to keep the committed script hardening plus the untracked regression
test/debug log.

### 2026-06-07 00:37 UTC — CODEX-20260607T003738Z-4plusmind — relaunch high-recall 4plus_mind scan

Human confirmed scope: run the pending `4plus_mind` high-recall contamination
scan now and request enough resources to make it fast.

Planned command:

```bash
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
    --cpu 4 --memory 64GB --disk 50GB --priority interactive --extra cpu \
    --enable-extra-resources --preemptible --region us-east5 \
    --job-name val-scan-4plus-mind-fast-0037 \
    -- python scripts/analysis/nemotron_math_val_full_scan.py --subset 4plus_mind
```

Resource shape inside the driver: 336-way corpus stages for `4plus_mind`;
minhash workers `4 cpu / 24g`, join workers `2 cpu / 64g`, dedup workers
`2 cpu / 32g`, verify workers `2 cpu / 48g`, with 32g coordinators.

Submitted:

- Parent job: `/ahmed/val-scan-4plus-mind-fast-0037`
- Scratch prefix: `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/4plus_mind_284x71`
- Val-doc MinHash reused existing outputs and succeeded.
- Corpus `4plus_mind` MinHash reused existing outputs and succeeded.
- Bucket join started:
  `/ahmed/val-scan-4plus-mind-fast-0037/zephyr-val-bucket-join-4plus_mind-853ca8f7-p0-a0`
  with 567 map/scatter tasks. Early workers registered and began startup.

### 2026-06-07 — 4plus_mind high-recall scan completion and key-join incident

Status check after the fast relaunch: the `4plus_mind` high-recall scan is
complete via the shard-scan path. Final artifacts:

- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/4plus_mind_284x71/scan_stats.json`
  updated 2026-06-07 08:12:43 UTC.
- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/4plus_mind_284x71/verified_pairs/`.
- Duplicate non-preemptible verification output:
  `scan_stats_np0040.json` updated 2026-06-07 11:46:08 UTC, with
  `verified_pairs_np0040/`.

Final counters, same in both stats files:

- 57,243 validation ids.
- 3,738,907 validation buckets.
- 7,541,007,033 verification shard pairs.
- 5,327,272,303 exact matches/pruned duplicate comparisons.
- 2,213,734,730 length-pruned comparisons.
- 2,472 reported verified pairs.

Completed jobs:

- `/ahmed/val-scan-4plus-mind-fast-0037` succeeded and wrote
  `scan_stats.json`.
- `/ahmed/val-scan-4plus-mind-np-0040` succeeded and wrote
  `scan_stats_np0040.json`.

Key-join follow-up attempts are not needed for final counts. They failed with
`AttributeError: 'generator' object has no attribute 'keys'` during
`verify-val-pairs`; the likely cause is the key-join reducer wrapper:
`reducer=lambda key, items, v=val_docs: _verify_join_group(key, items, v)`.
That lambda is not itself a generator function, so Zephyr treats the returned
generator object as a single reducer output record and the Parquet writer then
looks for `.keys()`.

Observed failed key-join attempts include 0041, 0042, 0043, 0044, 0045, 0047,
0051, 0052, and 0053. Attempt 0050 failed for the separate cross-region
temporary-shuffle issue (`TransferBudgetExceeded` serialization surfaced as a
`TypeError`). Stale key-join parent attempts 0048/0049 were still present in
Iris at the status check; do not stop them without explicit human approval.

Next action: do not relaunch the scan for data purposes. If the key-join fast
path is kept, fix the reducer wrapper as a real top-level generator function
and add a regression test before running it again.

### 2026-06-07 13:44 UTC — CODEX-20260607T134438Z-decon-summary — rerun union decontamination summary

Question: whether we now have a decontaminated validation set, or at least a
way to decontaminate the documents/windows used for validation, against all
three scan subsets (`3`, `4plus`, and `4plus_mind`).

Finding before rerun: `decon_val_summary.json` exists but is stale/incomplete.
It reports 631 pair files, matching only `3_284x71` + `4plus_284x71`; the
newly completed `4plus_mind_284x71` verified-pair directory has 336 more pair
files and is not included in that summary.

Rerun command, on Iris in `us-east5` to keep the 45M-id replay arrays near the
data:

```bash
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
    --cpu 8 --memory 64GB --disk 20GB --priority interactive --extra cpu \
    --enable-extra-resources --preemptible --region us-east5 \
    --job-name decon-val-summary-0054 \
    -- python scripts/analysis/decon_val_summary.py
```

Expected output:

- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_summary.json`

Interpretation note: `scripts/analysis/decon_val_summary.py` unions verified
validation-doc near-duplicates across `4plus_284x71`, `3_284x71`, and
`4plus_mind_284x71` at several Jaccard thresholds. It produces a drop/count
summary for documents and validation tokens. It does not materialize a new
filtered validation dataset by itself.

Follow-up while monitoring: job 0054 reached `verified pair files: 967` and
`mapped 34565/34565 drop ids to doc indices`, but the cutoff token loop was too
slow because it recomputed the same window/doc overlap separately for every
cutoff. Patched `scripts/analysis/decon_val_summary.py` to precompute
per-document validation-token coverage once and then sum by cutoff. Replace
0054 with optimized rerun:

```bash
uv run iris --config lib/iris/config/marin.yaml job stop /ahmed/decon-val-summary-0054
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
    --cpu 8 --memory 64GB --disk 20GB --priority interactive --extra cpu \
    --enable-extra-resources --preemptible --region us-east5 \
    --job-name decon-val-summary-0055 \
    -- python scripts/analysis/decon_val_summary.py
```

Second optimization while monitoring: 0055 also reached `verified pair files:
967` and `mapped 34565/34565`, but the one-pass all-window/doc overlap was
still too slow. Patched the script to compute exact dropped-token overlap by
iterating only contaminated docs and binary-searching sorted validation
windows. Replace 0055 with optimized rerun 0056:

```bash
uv run iris --config lib/iris/config/marin.yaml job stop /ahmed/decon-val-summary-0055
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
    --cpu 8 --memory 64GB --disk 20GB --priority interactive --extra cpu \
    --enable-extra-resources --preemptible --region us-east5 \
    --job-name decon-val-summary-0056 \
    -- python scripts/analysis/decon_val_summary.py
```

Final status at 2026-06-07 13:55 UTC:

- `/ahmed/decon-val-summary-0056` succeeded.
- `/ahmed/decon-val-summary-0054` and `/ahmed/decon-val-summary-0055` were
  killed after they hit the all-three pair-file/mapping checkpoints but before
  finishing the slower token-count loop.
- No `decon-val-summary*` job is running.
- The local sequential parquet reader used for per-dataset counts was killed;
  it was only reading existing `verified_pairs` files and did not write
  artifacts.

Fresh union output, now including all three scan subsets:

- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_summary.json`
  updated 2026-06-07 13:52 UTC.
- `pair_files`: 967 = `3_284x71` 400 shards + `4plus_284x71` 231 shards +
  `4plus_mind_284x71` 336 shards.
- Union doc/token drop counts across `3`, `4plus`, and `4plus_mind`:
  - J >= 0.50: drop 34,565 docs, keep 22,678 docs, clean tokens 30,080,012.
  - J >= 0.70: drop 14,474 docs, keep 42,769 docs, clean tokens 42,451,797.
  - J >= 0.75: drop 10,636 docs, keep 46,607 docs, clean tokens 44,816,904.
  - J >= 0.80: drop 7,077 docs, keep 50,166 docs, clean tokens 46,999,393.
  - J >= 0.90: drop 1,120 docs, keep 56,123 docs, clean tokens 50,527,994.

Artifact inventory / answer to current question:

- We do not yet have a materialized replacement/decontaminated validation
  dataset artifact.
- We do have the evidence needed to decontaminate by validation document id:
  every scan subset has a `verified_pairs/*.parquet` directory containing
  `val_id` and `jaccard` for verified training/validation near-duplicate pairs.
  To get IDs for a threshold, group by `val_id`, take max `jaccard`, and filter
  at the chosen cutoff.
- Existing per-subset verified-pair sources:
  - `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/3_284x71/verified_pairs/`
    (400 shards).
  - `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/4plus_284x71/verified_pairs/`
    (231 shards).
  - `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/4plus_mind_284x71/verified_pairs/`
    (336 shards; `scan_stats.json` reports 2,472 verified pair rows).
- Per-subset unique validation-doc counts already known:
  - `4plus`: 32,619 (J >= 0.50), 13,673 (0.70), 10,030 (0.75),
    1,011 (0.90).
  - `3`: 13,238 (J >= 0.50), 4,155 (0.70), 2,702 (0.75), 147 (0.90).
  - `4plus_mind`: 1,510 (J >= 0.50), 135 (0.70), 69 (0.75), 39 (0.80),
    15 (0.90), computed from 2,472 verified pair rows.
- Separate, ready-to-download per-dataset/per-threshold ID-list JSON/CSV files
  have not yet been written. The IDs are present in the pair parquet files; the
  list materialization step is still open.

### 2026-06-07 13:59 UTC — CODEX-20260607T135945Z-decon-status-answer

User asked for the direct state of the decontaminated validation artifacts.

Current answer:

- We do not yet have a materialized replacement validation dataset.
- We do have the contamination evidence needed to decontaminate the validation
  documents/windows by validation document id.
- The evidence covers all three requested training subsets: `3`, `4plus`, and
  `4plus_mind`.
- For each subset, the doc ids are already present as `val_id` values in
  `verified_pairs/*.parquet`, with each row carrying the measured `jaccard`.
  Per-threshold IDs are produced by grouping by `val_id`, taking the maximum
  `jaccard`, and filtering at the selected cutoff.
- The all-three union summary is fresh:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_summary.json`.

Known counts:

- Union across all three: 34,565 docs at J >= 0.50; 14,474 at 0.70; 10,636 at
  0.75; 7,077 at 0.80; 1,120 at 0.90.
- `4plus`: 32,619 docs at J >= 0.50; 13,673 at 0.70; 10,030 at 0.75; 1,011 at
  0.90.
- `3`: 13,238 docs at J >= 0.50; 4,155 at 0.70; 2,702 at 0.75; 147 at 0.90.
- `4plus_mind`: 1,510 docs at J >= 0.50; 135 at 0.70; 69 at 0.75; 39 at
  0.80; 15 at 0.90.

Open next step: materialize ready-to-use artifacts, e.g. per-subset and union
`drop_val_ids_jaccard_ge_*.jsonl` plus a filtered/masked validation dataset or
validation-window manifest.

### 2026-06-07 — "Paranoid" val set: drop all train-leaking docs ∩ fuzzy-clean

Question: how many val tokens survive the maximally strict filter — drop every
val doc with *any* verbatim train exposure (window-split leakage) AND every doc
with a fuzzy near-dup in any of the three subsets?

Key structural fact (established first): the val split is window-level over a
packed token stream, so the 57,243 "val docs" are a superset of the val token
content. They total ~125.7M tokens of text but only 51.2M sit inside val
windows; the other ~74.5M tokens of those same documents fall in train windows
and were trained on. Re-tokenizing kept docs in full would therefore inject
verbatim train text into any "clean" val set built from documents.

Step 1 — docs fully contained in val windows (zero verbatim leakage), computed
locally from `nemotron_math_doc_offsets.npy` + val window indices:

- 33,790 / 57,243 docs, 25,956,642 tokens = 50.7% of the 51.2M val-window
  tokens. Only 9 of these span two (consecutive val) windows, so in practice
  "fully contained" = "fits in one 4096-token window": max len 3,989, mean
  768, median 642. **Length bias caveat**: long docs are excluded by
  construction — relative cross-scale comparison stays valid, absolute losses
  and content mix shift.
- Indices: `scratch/nemotron_math_val_fully_contained_doc_indices.npy`.

Step 2 — intersect with the three-subset union fuzzy drop list. Pulled all 967
`verified_pairs` parquet shards locally (~94 MB, val_id+jaccard columns, 19 s
with 48 threads), built val_id → max-Jaccard, mapped all 34,565 ids to doc
indices (0 unmapped; count matches `decon_val_summary.json` exactly).

Paranoid val set (fully-contained ∧ max-J < cutoff):

| cutoff | dropped (fuzzy, within fully-contained) | keep docs | keep tokens | % of 51.2M |
|---|---:|---:|---:|---:|
| 0.50 | 20,213 | 13,577 | 10,446,342 | 20.4% |
| 0.70 | 8,299 | 25,491 | 19,568,485 | 38.2% |
| 0.75 | 6,048 | 27,742 | 21,293,593 | 41.6% |
| 0.80 | 4,001 | 29,789 | 22,888,290 | 44.7% |
| 0.90 | 629 | 33,161 | 25,472,832 | 49.8% |

So the fully-paranoid set at the canonical J≥0.75 cutoff keeps 27,742 short
docs / 21.3M tokens (41.6% of the original val token budget) — large enough
for a low-variance val loss, but a shorter-doc distribution than the original.

Artifacts (local):

- `scratch/nemotron_math_val_fully_contained_doc_indices.npy` — 33,790 doc
  indices with zero train-window overlap.
- `scratch/nemotron_math_val_doc_max_jaccard_union.npy` — structured array
  (doc index, max Jaccard) for all 34,565 union-contaminated val docs; any
  cutoff's drop list is a filter over this.
- `scratch/nemotron_math_paranoid_val_matrix.json` — the table above.

Open: materialize an actual tokenized val cache from a chosen cutoff (filter
the already-extracted `gs://…/midtrain_dedup/val_docs/` by kept doc index →
tokenize with `llama3_tokenizer` in us-east5), then re-eval checkpoints.

### 2026-06-07 17:40 UTC — Building paranoid short-doc val sets (j050/j075/j090)

Plan approved (with Codex critique folded in): three separate tokenized
validation caches, one per Jaccard cutoff, paranoid filter = fully contained
in val windows ∧ max train Jaccard < cutoff. Branch `deconamint`.

Codex critique applied:

- **No `skip_existing` as overwrite policy** — driver hard-fails if any target
  exists unless `--resume` is set AND the existing `build_intent.json` matches
  this run exactly (keep-id xxh3 hashes, sources, expected counts). Intent is
  written before any other target; submit with `--resume` so iris preemption
  retries continue their own build but a different build can never silently
  reuse partials.
- One zephyr pass filters all 231 `val_docs` shards into the three cutoff
  dirs (no triple read); per-shard checkpointing via the stats sink written
  after the three docs files finalize.
- Filtered docs keep provenance columns: `id, text, shard, row, doc_index,
  max_jaccard` (explicit parquet schema).
- Validation-only tokenize confirmed supported (`TokenizeConfig.__post_init__`,
  `tokenize.py:152`): `train_paths=[]`, `validation_paths=[docs glob]`. Cache
  root is `…/j{cut}`; data lands under `<root>/validation`.

Step 1 (local) — `scripts/analysis/build_paranoid_val_keep_ids.py`:
generated keep-id JSONs from the replay artifacts; asserts passed
(13,577 / 27,742 / 33,161 docs; 10,446,342 / 21,293,593 / 25,472,832 tokens;
strict nesting j050 ⊂ j075 ⊂ j090; no duplicate ids). Uploaded to
`gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/keep_ids/`.
Verified both target prefixes (`decon_val_sets/`, `tokenized/nemotron_math_val_decon/`)
were empty before any write.

Step 2 — `scripts/analysis/build_decon_val_sets.py` (driver) +
`tests/analysis/test_build_decon_val_sets.py` (3 tests: keep-set
nesting/counts, resume-intent mismatch rejection, filter-shard routing with
provenance columns) — all pass; pre-commit OK.

Output layout (all distinct per cutoff, no shared writes):

- docs: `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/j{050,075,090}/docs/`
- caches: `gs://marin-us-east5/tokenized/nemotron_math_val_decon/j{050,075,090}/validation/`
- manifests: `…/decon_val_sets/j{050,075,090}/manifest.json`

Submitted 17:40 UTC: `/ahmed/decon-val-build-all` (us-east5, preemptible,
4 cpu / 32 GB driver). Driver asserts filtered doc counts and cache
`.stats.json` doc/token counts against expectations before writing each
manifest — token equality is exact (expectations come from the original cache
offsets incl. EOS; tokenization is deterministic). Babysitting.

Stale-job cleanup (user approved): stopped
`/ahmed/val-scan-4plus-mind-keyjoin-p-0048` and `-0049` **with children**
(both parents were `running` with `verify-val-pairs` children pending on CPU).
Confirmed zero active `val-scan-4plus-mind-keyjoin*` jobs afterwards.

### 2026-06-07 17:44 UTC — Off-by-one in doc offsets caught by the exact-count gate

`/ahmed/decon-val-build-all` **failed by design**: the driver's doc-count
assert fired — filter kept 13,568 / 27,733 / 33,152 docs vs expected
13,577 / 27,742 / 33,161 (−9 at every cutoff).

Root cause: `scratch/nemotron_math_doc_offsets.npy` (and the byte-identical
GCS copy at `…/midtrain_dedup/replay/nemotron_math_doc_offsets.npy`) is a
**doc ENDS array**, not a boundaries array: `len == 45,096,087 == num docs`,
`offsets[0] == 2740` (doc 0's end), `offsets[-1] == 51,482,955,946` (total
tokens). Doc d spans `[offsets[d-1], offsets[d])` with start 0 for d=0. The
2026-06-07 "paranoid" computation treated it as boundaries
(`starts = offsets[:-1]`, `ends = offsets[1:]`), so every span belonged to
doc d+1 — the fully-contained doc *indices* (and therefore ids) were shifted
by one, while aggregate counts looked plausible. Symmetric difference vs the
retokenization-verified original val-doc set was 12,483 docs each way.

Validation of the fix: with `starts = [0] + ends[:-1]`, the recomputed
touch-set **exactly equals** the original 57,243
`nemotron_math_val_doc_indices.npy`. Corrected paranoid matrix:

| cutoff | keep docs | keep tokens | % of 51.2M |
|---|---:|---:|---:|
| 0.50 | 13,947 | 10,282,799 | 20.1% |
| 0.70 | 25,868 | 19,064,069 | 37.2% |
| 0.75 | 28,089 | 20,782,728 | 40.6% |
| 0.80 | 30,038 | 22,382,501 | 43.7% |
| 0.90 | 33,196 | 25,346,090 | 49.5% |

(Fully-contained baseline unchanged at 33,790 docs / 25,956,642 tokens — the
old run described the same docs under shifted labels, so aggregates matched
while the id lists were wrong.) Corrected artifacts overwrite the old ones:
`scratch/nemotron_math_val_fully_contained_doc_indices.npy`,
`scratch/nemotron_math_paranoid_val_matrix.json` (carries a correction note).

**The earlier paranoid-matrix table in this logbook (13,577 / 27,742 / 33,161)
is wrong — use the corrected table above.**

Knock-on for Codex: `scripts/analysis/decon_val_summary.py:54-55` uses the
same `offsets[d] / offsets[d+1]` convention on the replay array, so the
dropped/clean **token** counts in `decon_val_summary.json` are computed over
each contaminated doc's *successor* span (doc counts are unaffected — they
come from id mapping). Should be rerun with the corrected convention if those
token numbers get used anywhere.

Hardening: `build_paranoid_val_keep_ids.py` now derives `starts` explicitly,
asserts fully-contained ⊆ verified val docs, and pins the corrected expected
counts. Cleanup + relaunch: deleted all failed-build artifacts
(928 objects under `decon_val_sets/`: intent, filter_stats, 693 docs shards,
old keep_ids; cache prefix confirmed never written), regenerated + re-uploaded
keep ids, resubmitted as `/ahmed/decon-val-build-all-v2` (17:50 UTC).

### 2026-06-07 18:55 UTC — Paranoid val sets BUILT and verified

`/ahmed/decon-val-build-all-v2` **succeeded** (17:50 → 18:41 UTC, ~51 min;
the three tokenize passes ran single-worker at ~49k tokens/s because the
filtered sets bundle into one file group — fine at this size).

All verification gates passed:

| gate | result |
|---|---|
| filter doc counts | exact: 13,947 / 28,089 / 33,196 |
| cache `.stats.json` docs+tokens vs expected | **exact**: 10,282,799 / 20,782,728 / 25,346,090 tokens |
| manifests (expected==actual, all fields) | ✓ all three |
| spot retokenization (3 docs) | ✓ — cache stores BOS + text + EOS (`append_bos: true`); span = HF-retok(with BOS) + 1 |
| eval-loadability (`LmDataConfig.validation_sets`, Pos=4096) | ✓ loads all three; sequences 2,510 / 5,073 / 6,188 == manifest `eval_sequences`; `tagged_eval_sets` yields 3 separately-tagged datasets |

The exact token equality is the strong end-to-end proof: expectations came
from the original cache's offsets, and independent re-tokenization of the
filtered text reproduced them bit-for-bit.

Final artifacts (ready for eval):

- `gs://marin-us-east5/tokenized/nemotron_math_val_decon/{j050,j075,j090}/validation/`
  — Levanter caches, llama3 tokenizer, validation split only.
- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/{tag}/manifest.json`
  + `{tag}/docs/` (filtered text with `id, text, shard, row, doc_index,
  max_jaccard`) + `keep_ids/` + `build_intent.json` + `filter_stats/`.

Definition reminder: **paranoid short-doc val sets** — val docs fully
contained in val windows (zero verbatim train spill) ∧ max train Jaccard <
cutoff (union of 3/4plus/4plus_mind high-recall scans). Short-doc biased by
construction (≤3,989 tokens/doc, median 642).

Next: eval-only sweep per `.agents/projects/decon_val_eval_plan.md` —
levanter `eval_lm` with the three caches as tagged validation-only components
plus the original 12,500-window carve-out as in-harness anchor; load midtrain
checkpoints from `hf/step-N/` (native checkpoints cleaned; arch is Qwen3);
`max_eval_length=4096` explicitly (levanter default 2048 is a trap); sanity
gate = reproduce 1e21's recorded math-val loss before trusting decon numbers.

### 2026-06-07 ~20:00 UTC — Decon eval sweep launched: p33m67 × lr0.33 × full ladder

User directive: evaluate the p33m67 lr0.33 models across the whole scaling
ladder; ONE job per model, with the filtered datasets logged as SEPARATE
datasets; interactive priority.

Run selection (registry-cross-checked, completion verified by final hf step ≈
round(0.2 × base `num_train_steps`); attempt suffixes are fresh-restart
counters, listed attempt = the one with a finished export):

| scale | run | final hf step |
|---|---|---:|
| 3e18 | delphi-3e18-p33m67-k0p20-lr33-a003 | 7,399 |
| 9e18 | delphi-9e18-p33m67-k0p20-lr33-a002 | 8,818 |
| 2e19 | delphi-2e19-p33m67-k0p20-lr33-a002 | 10,982 |
| 3e19 | delphi-3e19-p33m67-k0p20-lr33-a002 | 7,573 |
| 9e19 | delphi-9e19-p33m67-k0p20-lr33-a002 | 8,032 |
| 2e20 | delphi-2e20-p33m67-k0p20-lr33-a001 | 11,277 |
| 3e20 | delphi-3e20-p33m67-k0p20-lr33-a001 | 7,081 |
| 1e21 | delphi-1e21-p33m67-9p25b-lr0.33-58ebcb | 4,410 |
| 1e22 | delphi-1e22-p33m67-32p07b-lr0.33-e9132105 | 7,646 |

(`delphi-1e22-…-lr0.33-abdeba` is a dead attempt — zero hf exports.)

Driver: `scripts/analysis/eval_decon_val_sets.py` — per job, four datasets
evaluated as separately-tagged validation sets in ONE `eval_lm` invocation:
`decon_j050/j075/j090` (cache components, split=validation, weight 0) +
`nemotron_cc_math_v1/4plus` anchor (component copied verbatim from the
vendored `scripts/analysis/p33m67_data_section.json`: same cache
`4plus-2c5519`, `num_validation_sequences: 12500`, same feistel shuffle →
byte-identical original val split in-harness). Model config built per run
from `hf/step-N/config.json` via `Qwen3Config.from_hf_config` (never
hand-picked dims). `max_eval_length=4096` explicit; mp `p=f32,c=bfloat16`
to match training-time eval. Trackers: W&B (project marin, tag
`decon_val_eval`) + JSON file at
`gs://…/midtrain_dedup/decon_val_sets/evals/{run}/step-{N}/metrics.jsonl`
(refuses to overwrite without `--force`).

Incident: first batch (`decon-eval-p33m67-lr33-{scale}`) failed in ~1 min
with `Failed to open libtpu.so` — submitted with `--tpu` (hardware) but
without `--extra tpu` (dependency group), so jax had no TPU backend. All 9
stopped; resubmitted as `…-v2` with `--extra tpu`. TPU shapes: v6e-4 for
3e18→1e21, v6e-8 for 1e22 (9.7B); all preemptible interactive us-east5
(idle ready slices observed: 14× v6e-4, 5× v6e-8 in us-east5-b).

Sanity gate before trusting results: each run's anchor loss
(`eval/nemotron_cc_math_v1/4plus/loss`) must match the run's recorded final
math-val loss in its `tracker_metrics.jsonl` / W&B (~1e-3). Then the decon
deltas are meaningful.

Launch incidents (all infra, none in eval logic):

1. v1 batch: `Failed to open libtpu.so` — `--tpu` requests hardware but the
   TPU dependency group needs `--extra tpu`. ~1 min to fail; all stopped.
2. v2 batch: TPU backend OK, then `Container was OOM killed by the kernel` —
   no host resources requested (default is tiny; HF staging + data pipeline
   need real RAM). All stopped.
3. v3 batch: `--extra tpu --cpu 8 --memory 64GB` (96GB 1e21 / 120GB 1e22)
   `--disk 50GB` — worked. Per-job eval runtime ≈ 2-4 min on v6e-4
   (1,642 eval batches: 26,271 sequences = 12,500 anchor + 2,510 + 5,073 +
   6,188 decon, ~15 it/s at 3e18).
4. 1e22-v3 stuck pending — v6e-8 slices consumed + scale group in backoff
   ("Insufficient TPUs (need 8, available 0)"). Stopped; resubmitted as
   `…-1e22-v4` on **v6e-4** (9.7B bf16 = ~4.9 GB/chip of weights on 32 GB
   HBM chips; eval-only fits fine), `--per-device-parallelism 2`.

Output-path note: `JsonFileTrackerConfig(output_path=…/metrics.jsonl)`
treats the path as a directory — actual file is
`…/metrics.jsonl/eval_results.json`. Also weakens the driver's
exists-check (checks the dir path as a file); fix if the script is reused.

### 2026-06-07 ~21:00 UTC — RESULTS: p33m67 lr0.33 ladder, decon vs original val

**Sanity gates passed at both checked scales**: harness anchor loss vs
training-time recorded final math-val loss — 3e18: 1.4720 vs 1.4719;
1e21: 0.8104 vs 0.8102. The harness reproduces the byte-identical val split
to ~2e-4; decon deltas are trustworthy.

All losses (nats/token), final checkpoint per run:

| scale | anchor (orig val) | decon_j090 | decon_j075 | decon_j050 | anchor − j050 |
|---|---:|---:|---:|---:|---:|
| 3e18 | 1.4720 | 1.4266 | 1.3687 | 1.3597 | 0.1123 |
| 9e18 | 1.3034 | 1.2621 | 1.2070 | 1.1951 | 0.1083 |
| 2e19 | 1.2203 | 1.1813 | 1.1278 | 1.1150 | 0.1053 |
| 3e19 | 1.1640 | 1.1264 | 1.0741 | 1.0615 | 0.1025 |
| 9e19 | 1.0425 | 1.0090 | 0.9600 | 0.9490 | 0.0935 |
| 2e20 | 0.9737 | 0.9432 | 0.8970 | 0.8884 | 0.0853 |
| 3e20 | 0.9286 | 0.9001 | 0.8564 | 0.8503 | 0.0783 |
| 1e21 | 0.8104 | 0.7887 | 0.7547 | **0.7612** | 0.0492 |
| 1e22 | 0.5727 | 0.5630 | 0.5643 | **0.6126** | **−0.0399** |

Findings so far:

1. **Baseline offset**: at tiny scale the paranoid sets read ~0.11 nats
   *easier* than the anchor — short-doc distribution shift (paranoid sets
   exclude long docs by construction), not contamination. The signal is the
   *trend* in the gap, not its level.
2. **Monotone, accelerating gap shrink**: anchor−j050 falls 0.1123 → 0.0492.
   ~−0.003/rung through 3e19, ~−0.008/rung through 3e20, then **−0.029 in
   the single 3e20→1e21 step**. The anchor improves faster than the clean
   sets as capacity grows = memorization credit on contaminated val content,
   accelerating exactly where Codex's exposure replay said exposure ramps.
3. **Ordering inversion at 1e21**: j050 (0.7612) > j075 (0.7547) — at every
   smaller scale stricter filtering ⇒ lower loss; at 1e21 the j075 set
   (which still contains docs with J∈[0.5,0.75) train near-dups) drops
   *below* the strictest set. Memorization credit on moderate near-dups is
   now visible *within* the decon family, where doc distributions are nearly
   identical. This is the cleanest contamination signature in the data.

Artifacts: per-run `eval_results.json` under
`gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals/{run}/step-{N}/metrics.jsonl/`;
W&B runs `decon-eval-{run}` in marin-community/marin, tag `decon_val_eval`.

### 2026-06-07 ~22:00 UTC — 1e22 row landed: gap INVERTED. Verdict.

1e22 (`…-1e22-v4`, ran on a **v6e-4** — 9.7B bf16 eval is ~4.9 GB/chip
weights + ~5 GB activations/logits per chip, ~3× headroom on 32 GB HBM;
the earlier v6e-8 ask was training-shaped reflex, not arithmetic, and cost
~40 min of pending while that pool was in backoff. Eval sizing rule: bf16
eval fits v6e-4 up to ~40-50B params). Sanity gate: anchor 0.5727 vs
recorded 0.5725 ✓ (third gate, all passed).

Findings, final:

1. **The anchor−j050 gap fully inverts**: +0.1123 at 3e18 → −0.0399 at
   1e22, a 0.152-nat relative swing. The original val set starts 0.11 nats
   *harder* than the strictest clean set and ends 0.04 nats *easier* — the
   contaminated anchor's apparent progress increasingly comes from
   memorization credit, not capability.
2. **Complete ordering reversal within the decon family**: at 3e18,
   stricter filter ⇒ lower loss (j050 < j075 < j090 < anchor); at 1e22 the
   ordering is exactly reversed (j090 < j075 < anchor < j050) — the more
   near-dup content a set retains, the more memorization credit it
   collects. j050, the only set with no J≥0.5 train near-dup and no
   window-split leakage, is now the *hardest* — it is the honest signal.
3. **Quantifying the 1e22 "too good" effect** (the original question): the
   1e21→1e22 improvement is 0.2377 nats on the original val but only
   0.1486 nats on j050 — **~37% of the apparent 1e21→1e22 gain on the
   original math val is contamination-driven**. The 1e22 model is genuinely
   better, but the original val set overstates the jump by roughly a third
   at this rung; any scaling-law fit on the original math val loss bends
   optimistically at the top end.

Follow-ups suggested:

- Re-fit the math scaling laws on decon_j050 losses (this table is one
  lr/mix slice; sweep other lr factors/mixes as needed — driver takes any
  run, ~3 min/job on v6e-4).
- Adopt decon_j050 (or a doc-level-split, fuzzy-deduped successor) as the
  canonical math val for future midtrains; the byte-identical-contract
  machinery (data_sections JSON) can pin it the same way.
- Mind the short-doc bias when comparing absolute losses to the old anchor
  numbers; within-ladder comparisons on the same set are unaffected.

### 2026-06-07 22:15 UTC — Session close: state of record

Branch `deconamint`, commits this session:

- `e3176631ca` — paranoid val set build: `build_paranoid_val_keep_ids.py`,
  `build_decon_val_sets.py`, `tests/analysis/test_build_decon_val_sets.py`,
  eval plan (`.agents/projects/decon_val_eval_plan.md`).
- `bfd6331d3d` — eval sweep: `eval_decon_val_sets.py`, vendored
  `p33m67_data_section.json`, results in this logbook.

Durable artifacts:

- Caches: `gs://marin-us-east5/tokenized/nemotron_math_val_decon/{j050,j075,j090}/validation/`
  (13,947 / 28,089 / 33,196 docs; 10,282,799 / 20,782,728 / 25,346,090
  tokens — exact-verified; llama3 tokenizer).
- Build provenance: `gs://…/midtrain_dedup/decon_val_sets/`
  (keep_ids/, j*/docs/, j*/manifest.json, build_intent.json, filter_stats/).
- Eval results: `gs://…/midtrain_dedup/decon_val_sets/evals/{run}/step-{N}/metrics.jsonl/eval_results.json`
  ×9 runs; W&B tag `decon_val_eval` (marin-community/marin).

Cluster state: no jobs from this work left running (eval sweep all
succeeded; stale keyjoin 0048/0049 stopped with children earlier; build
jobs terminal). Total compute spent on the answer: ~9 preemptible v6e-4
eval jobs ≈ 35 min TPU time + two ~50-min CPU build jobs.

Open follow-ups (not started):

1. Re-fit math scaling laws on `decon_j050`; extend sweep to other lr
   factors / mixes (driver is ready: `eval_decon_val_sets.py --run <run>`).
2. Canonicalize a decontaminated math val for future midtrains
   (data-sections pinning, same byte-identical contract).
3. Rerun `decon_val_summary.py` with corrected offsets convention (its
   token counts treat doc-ends as starts; doc counts unaffected).
4. Key-join verify fast path in `nemotron_math_val_full_scan.py` still
   broken (non-generator reducer wrapper) — fix + regression test before
   any reuse; not needed for any current numbers.

### 2026-06-07 22:30 UTC — Download paths for the isoflop loss data (for future agents)

**One-stop consolidated files** (all 9 scales × 4 datasets, loss + bpb +
per-run source paths, with dataset definitions embedded):

- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals/summary_p33m67_lr33.json`
- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals/summary_p33m67_lr33.csv`
- Local copies: `scratch/nemotron_math_decon_eval_p33m67_lr33.{json,csv}`

JSON keys per row: `scale, run, step, source, anchor_orig_val, decon_j090,
decon_j075, decon_j050` (losses, nats/token) + `*_bpb`.

**Per-run raw tracker output** (full metric set incl. macro/micro/bpb), one
per scale — note `metrics.jsonl` is a *directory* (JsonFileTracker quirk):

```text
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals/<run>/step-<N>/metrics.jsonl/eval_results.json
```

| scale | run | step |
|---|---|---:|
| 3e18 | delphi-3e18-p33m67-k0p20-lr33-a003 | 7399 |
| 9e18 | delphi-9e18-p33m67-k0p20-lr33-a002 | 8818 |
| 2e19 | delphi-2e19-p33m67-k0p20-lr33-a002 | 10982 |
| 3e19 | delphi-3e19-p33m67-k0p20-lr33-a002 | 7573 |
| 9e19 | delphi-9e19-p33m67-k0p20-lr33-a002 | 8032 |
| 2e20 | delphi-2e20-p33m67-k0p20-lr33-a001 | 11277 |
| 3e20 | delphi-3e20-p33m67-k0p20-lr33-a001 | 7081 |
| 1e21 | delphi-1e21-p33m67-9p25b-lr0.33-58ebcb | 4410 |
| 1e22 | delphi-1e22-p33m67-32p07b-lr0.33-e9132105 | 7646 |

Loss keys inside `eval_results.json`:
`eval/nemotron_cc_math_v1/4plus/loss` (anchor = original val),
`eval/decon_j050/loss`, `eval/decon_j075/loss`, `eval/decon_j090/loss`,
plus matching `/bpb` and `eval/macro_loss` / `eval/loss` aggregates.
W&B mirror: project marin-community/marin, runs `decon-eval-<run>`, tag
`decon_val_eval`. This is one (mix, lr) slice — p33m67 × lr0.33 — of the
K=0.20 CPT sweep; other slices rerun via
`scripts/analysis/eval_decon_val_sets.py --run <run>` (~3 min/job, v6e-4).

### 2026-06-07 ~23:00 UTC — All four lr factors swept: contamination is scale-driven, lr-invariant

Extended the eval sweep to the full p33m67 K=0.20 grid: 9 scales × 4 lr
factors {0.33, 0.5, 0.67, 0.83} = 36 cells (lr0.33 from the earlier run +
27 new jobs). All on v6e-4 (incl. 1e22 — 9.7B bf16 eval is ~10 GB/chip
peak, ~3× headroom; no v6e-8 needed), interactive, preemptible, us-east5,
all reads+writes in-region. 27/27 succeeded. Run selection per (scale, lr)
verified complete against the registry (final hf step ≈ round(0.2 × base
num_train_steps)); excluded incomplete 1e22 attempts abdeba/91bcb9/089468.
Codex hardened the output-exists guard (`require_unused_output`, checks
JsonFileTracker dir + `eval_results.json` + dir contents; `--force` warns)
with regression tests `tests/analysis/test_eval_decon_val_sets.py`.

Sanity gates (harness anchor vs run's recorded final math-val loss):
3e18-lr33 1.4720/1.4719, 1e21-lr33 0.8104/0.8102, 1e22-lr33 0.5727/0.5725,
**1e21-lr0.5 (efbc63, the val-set provenance run) 0.7937/0.7935** — all ≤2e-4.

**anchor (original val) loss, nats/token:**

| scale | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---:|---:|---:|---:|
| 3e18 | 1.4720 | 1.4354 | 1.4157 | 1.4048 |
| 9e18 | 1.3034 | 1.2739 | 1.2585 | 1.2501 |
| 2e19 | 1.2203 | 1.1935 | 1.1799 | 1.1732 |
| 3e19 | 1.1640 | 1.1391 | 1.1268 | 1.1208 |
| 9e19 | 1.0425 | 1.0212 | 1.0115 | 1.0074 |
| 2e20 | 0.9737 | 0.9542 | 0.9458 | 0.9428 |
| 3e20 | 0.9286 | 0.9097 | 0.9023 | 0.9002 |
| 1e21 | 0.8104 | 0.7937 | 0.7875 | 0.7898 |
| 1e22 | 0.5727 | 0.5611 | 0.5597 | 0.5598 |

**anchor − decon_j050 (>0 = orig val harder/short-doc offset; <0 = orig val
easier = contamination credit):**

| scale | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---:|---:|---:|---:|
| 3e18 | +0.1123 | +0.1129 | +0.1133 | +0.1136 |
| 9e18 | +0.1083 | +0.1082 | +0.1082 | +0.1082 |
| 2e19 | +0.1053 | +0.1046 | +0.1043 | +0.1042 |
| 3e19 | +0.1026 | +0.1013 | +0.1007 | +0.1006 |
| 9e19 | +0.0935 | +0.0909 | +0.0898 | +0.0895 |
| 2e20 | +0.0853 | +0.0821 | +0.0809 | +0.0805 |
| 3e20 | +0.0783 | +0.0744 | +0.0730 | +0.0728 |
| 1e21 | +0.0491 | +0.0439 | +0.0423 | +0.0435 |
| 1e22 | −0.0399 | −0.0437 | −0.0432 | −0.0428 |

Findings:

1. **Contamination effect is lr-invariant.** At any scale the gap varies
   <0.005 nats across all four lr factors — far smaller than the
   0.15-nat scale-driven swing. The +0.11→−0.04 inversion is a property of
   scale/exposure, not optimization trajectory. Strong evidence it is genuine
   memorization of contaminated content, not an LR/schedule artifact.
2. **Inversion replicates in all four columns**: every lr column crosses from
   ~+0.11 at 3e18 to ~−0.04 at 1e22, sign-flipping between 1e21 and 1e22.
3. **j050>j075 ordering inversion at 1e21–1e22 holds across lr** (checked
   lr0.33 and lr0.5 explicitly): the strictest-clean set becomes the hardest,
   the contamination signature within the decon family.

Consolidated artifacts (all 36 cells: loss + bpb + per-run source paths +
dataset defs):

- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals/summary_p33m67_all_lr.json`
- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals/summary_p33m67_all_lr.csv`
- Local: `scratch/nemotron_math_decon_eval_p33m67_all_lr.{json,csv}`

Per-run raw: `…/evals/<run>/step-<N>/metrics.jsonl/eval_results.json`; W&B
`decon-eval-<run>` tag `decon_val_eval`. Run→step map in the json `rows`.
Next: re-fit math scaling laws on decon_j050 (best-over-lr per scale) vs the
original anchor to quantify how much the original isoflop fit's top-end
curvature is contamination.

### 2026-06-07 ~23:30 UTC — Pre-launch: extend decon eval to p50m50 + p67m33 (all scales × all lr)

Directive: run the same decon eval over the other two mixes across the whole
ladder, interactive priority. The eval data config is **mix-independent** —
the math anchor component (`nemotron_cc_math_v1/4plus`, cache `4plus-2c5519`,
`num_validation_sequences: 12500`, feistel params) is byte-identical across
`p33m67/p50m50/p67m33.json` (sha `f1360c86` all three; the byte-identical val
contract), and the decon caches are mix-independent. So `build_data_config()`
is reused unchanged; only the checkpoints differ. Anchor losses are therefore
directly comparable across mixes.

Run selection verified complete vs registry (final hf step ≈
round(0.2 × base steps)):

- **p67m33: 36/36** complete.
- **p50m50: 35/36** — `1e22 lr0.67` (`e78260`) only reached step 6112 (vs
  7646), excluded as incomplete; no other complete attempt exists.
- Excluded stray partial attempts throughout (e.g. p67m33 lr0.5 a003/a004/a005
  at step 19-29 — fresh restarts that never trained).

Extra sanity gates available: `delphi-1e21-p50m50-9p25b-lr0.5-973c46` and
`delphi-1e21-p67m33-9p25b-lr0.5-114e49` are the val-set provenance runs for
their mixes — anchor loss must reproduce their recorded final math-val loss.

Launching 71 jobs (36 p67m33 + 35 p50m50), all v6e-4 (1e22 incl.),
`--cpu 8 --memory 64/96/120GB --disk 50GB --extra tpu`, interactive,
preemptible, us-east5. Job names `decon-eval-<run>` (mix in run name → no
output collision; outputs keyed by run under `…/evals/<run>/`).

### 2026-06-07 ~22:15 UTC — All three mixes done: contamination is dose-dependent on math fraction

p50m50 (35/36) + p67m33 (36/36) eval sweeps complete; with the earlier
p33m67 (36/36) the full grid is **107 cells** (3 mixes × 9 scales × 4 lr).
All v6e-4, interactive, preemptible, us-east5.

Incidents:

- One real failure: `delphi-1e22-p50m50-32p07b-lr0.33-c43ada` — its final
  HF export `step-7646` is **truncated** (8 safetensors shards present but no
  `config.json`/index; export interrupted). Audited all seven 1e22
  p50m50/p67m33 finals: only c43ada is affected. Resubmitted against the last
  complete export `step-7640` (6 steps back, converged) → `…-s7640`. The
  driver's `final_hf_step` picks the max step blindly; it should skip exports
  missing config.json — minor hardening TODO.
- zsh word-splitting bit the launch loop twice (`for x in $RUNS` doesn't split
  in zsh); fixed with a `while IFS= read -r` loop over a runs file. Verified
  submission counts each time rather than trusting the loop.

Sanity gates: both new provenance runs pass — p50m50 `973c46` harness 0.8311
vs recorded 0.8310; p67m33 `114e49` harness 0.8801 vs recorded 0.8800. With
the three p33m67 gates that is six-for-six.

**anchor − decon_j050 (negative = original val artificially easier =
contamination credit), lr-averaged per mix:**

| scale | p33m67 (67% math) | p50m50 (50% math) | p67m33 (33% math) |
|---|---:|---:|---:|
| 3e18 | +0.1130 | +0.1122 | +0.1110 |
| 9e18 | +0.1082 | +0.1083 | +0.1076 |
| 2e19 | +0.1046 | +0.1054 | +0.1056 |
| 3e19 | +0.1013 | +0.1030 | +0.1041 |
| 9e19 | +0.0909 | +0.0949 | +0.0983 |
| 2e20 | +0.0822 | +0.0877 | +0.0931 |
| 3e20 | +0.0746 | +0.0818 | +0.0887 |
| 1e21 | +0.0447 | +0.0579 | +0.0720 |
| 1e22 | **−0.0420** | **−0.0216** | **+0.0068** |

**Dose-response — the decisive evidence.** The contamination effect scales
monotonically with the math fraction of the training mix. At small scale all
three mixes share the same ~+0.11 short-doc baseline offset (no memorization
yet); as scale grows the gap shrinks fastest for the most-math mix. At 1e22:
p33m67 (67% math) fully inverts to −0.042, p50m50 (50%) barely inverts to
−0.022, p67m33 (33%) only reaches ~0. More math training tokens ⇒ more
exposure to the contaminated near-duplicates ⇒ more memorization credit on the
original val. A distribution artifact would not track math fraction; this
ordering (p33m67 > p50m50 > p67m33 at every scale ≥9e19) is the signature of
genuine train/val contamination. Earlier within-mix findings (lr-invariance,
j050>j075 ordering inversion at large scale) hold across all three mixes.

Consolidated artifacts (all 107 cells, loss+bpb+source paths+dataset defs):

- `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals/summary_all_mixes_all_lr.json`
- `…/evals/summary_all_mixes_all_lr.csv`
- Local: `scratch/nemotron_math_decon_eval_all_mixes_all_lr.{json,csv}`
- Per-mix files still present: `summary_p33m67_all_lr.{json,csv}`.

Per-run raw + W&B as before (tag `decon_val_eval`). Next: re-fit the math
scaling law on decon_j050 per mix vs the anchor; the contamination correction
is largest for p33m67 and shrinks with math fraction.

## 2026-06-08 — Val loss vs Jaccard decontamination threshold (fine 0.05 sweep)

Filled in the 0.05-spaced Jaccard cutoff grid between the canonical
j050/j075/j090 paranoid caches — **j055, j060, j065, j070, j080, j085** — and
re-evaluated the **p33m67 × lr0.33** ladder on all nine decontaminated val sets
plus the original-val anchor, to plot **math val loss vs decontamination cutoff
τ, one curve per compute budget** (3e18 → 1e22). Question: how does the
contamination effect resolve when τ is swept finely rather than at three points?

Cutoff convention: `jXXX` = keep val docs whose max train Jaccard is `< 0.XX`
(paranoid: also fully contained in val windows). **Lower τ = more aggressive =
smaller, cleaner set.** j050 ⊂ j055 ⊂ … ⊂ j090.

### Method

- **Six new caches derived, not rescanned.** Every cutoff ≤0.90 keep-set is a
  subset of the existing **j090 universe** (`…/decon_val_sets/keep_ids/keep_ids_j090.json`:
  33,196 fully-contained docs + per-doc max train Jaccard). Derived all nine
  keep-lists by thresholding that universe + the doc-offsets replay array (for
  exact token counts) — no new MinHash scan. Self-verify reproduced all five
  published doc anchors (0.50→13,947 … 0.90→33,196) AND the three cache-exact
  token anchors (0.50/0.75/0.90) before building; the j070/j080 token totals
  (19,064,069 / 22,382,501) then landed exactly on the corrected-matrix values
  that were deliberately **not** asserted — independent confirmation of the
  token method.
- **Resumable + parallel build** (`scripts/analysis/build_decon_val_sweep.py`,
  test `tests/analysis/test_build_decon_val_sweep.py`). The first attempt was a
  single sequential job; it was preempted after finishing j055 and the
  entrypoint retry tripped a strict "caches exist" guard (the same fragility the
  canonical builder's `--resume` machinery exists to avoid). Rewrote to
  skip-if-complete / clean-partial / never-touch-existing and fanned the
  remaining five cutoffs into one job each (`--cutoffs <c> --skip-filter`; the
  shared filter ran once, all six docs dirs already written). 5 parallel
  us-east5 preemptible CPU jobs, **~18 min** wall (single-worker tokenize, ~22
  docs/s — small sets bundle into one file group).
- **Eval** (`scripts/analysis/eval_decon_val_sets.py`, extended 3→9 decon tags +
  new `--out-root`): one v6e-4 job per scale, each evaluating the anchor + 9
  decon sets as separately-tagged validation sets in one `eval_lm`, writing to a
  distinct `evals_sweep9` root so the canonical 3-tag per-run files are
  untouched. 8/9 succeeded first pass; **1e21** hit a transient GCS read failure
  during HF-checkpoint load (None-content JSON parse; step-4410 is complete on
  disk) and succeeded on a clean re-run (`decon-eval9-1e21-r2`).
- **Sanity gate**: the new run's anchor + j050/j075/j090 reproduced the prior
  3-tag values (logbook 2026-06-07 table) to **0.0000** at all nine scales — so
  the six interpolated cutoffs are measured by the same trusted harness, and the
  caches passed exact doc+token gates. The sweep structure below is signal, not
  a masking/mapping bug.

### Results — math val loss (nats/token)

| scale | anchor | j050 | j055 | j060 | j065 | j070 | j075 | j080 | j085 | j090 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3e18 | 1.4720 | 1.3597 | 1.3428 | **1.3389** | 1.3453 | 1.3537 | 1.3687 | 1.3868 | 1.4080 | 1.4266 |
| 9e18 | 1.3034 | 1.1951 | 1.1802 | **1.1772** | 1.1837 | 1.1922 | 1.2070 | 1.2244 | 1.2446 | 1.2621 |
| 2e19 | 1.2203 | 1.1150 | 1.1011 | **1.0985** | 1.1050 | 1.1134 | 1.1278 | 1.1446 | 1.1642 | 1.1813 |
| 3e19 | 1.1640 | 1.0615 | 1.0478 | **1.0453** | 1.0517 | 1.0599 | 1.0741 | 1.0904 | 1.1096 | 1.1264 |
| 9e19 | 1.0425 | 0.9490 | 0.9358 | **0.9331** | 0.9390 | 0.9467 | 0.9600 | 0.9751 | 0.9931 | 1.0090 |
| 2e20 | 0.9737 | 0.8884 | 0.8752 | **0.8721** | 0.8774 | 0.8845 | 0.8970 | 0.9112 | 0.9280 | 0.9432 |
| 3e20 | 0.9286 | 0.8503 | 0.8367 | **0.8334** | 0.8381 | 0.8445 | 0.8564 | 0.8697 | 0.8856 | 0.9001 |
| 1e21 | 0.8104 | 0.7612 | 0.7460 | **0.7403** | 0.7425 | 0.7462 | 0.7547 | 0.7645 | 0.7768 | 0.7887 |
| 1e22 | 0.5727 | 0.6126 | 0.5915 | 0.5785 | 0.5718 | 0.5669 | 0.5643 | 0.5628 | **0.5621** | 0.5630 |

(Bold = per-row loss minimum. Anchor = the original 12,500-window val carve-out,
a *different* short-doc-inclusive distribution; not on the paranoid curve.)

Valley minimum and within-decon slope per scale:

| scale | valley-min τ | j090 − j050 | anchor − j050 |
|---|---:|---:|---:|
| 3e18 | 0.60 | +0.0668 | +0.1123 |
| 9e18 | 0.60 | +0.0670 | +0.1083 |
| 2e19 | 0.60 | +0.0663 | +0.1053 |
| 3e19 | 0.60 | +0.0649 | +0.1026 |
| 9e19 | 0.60 | +0.0599 | +0.0935 |
| 2e20 | 0.60 | +0.0547 | +0.0853 |
| 3e20 | 0.60 | +0.0499 | +0.0783 |
| 1e21 | 0.60 | **+0.0275** | +0.0491 |
| 1e22 | **0.85** | **−0.0496** | −0.0399 |

### Findings

1. **The curves are non-monotonic — a valley the 3-point view hid.** The
   minimum-loss decontaminated set is **not** the strictest (j050) but **τ≈0.60**
   at every budget up to 1e21; dropping docs with max-J ∈ [0.5, 0.6) *raises*
   loss. The old j050/j075/j090 sampling saw only a clean monotone rise and
   missed this entirely. So "strictest = cleanest signal" is wrong — j050
   over-filters relative to the loss-minimizing j060.
2. **The contamination signature is the curve's scale-evolution, localized to
   the 1e21→1e22 rung.** The within-decon slope **j090 − j050 flips sign**
   between 1e21 (+0.027, strict end still easier) and 1e22 (−0.050, strict end
   now *hardest* — removing near-dups raises loss = memorization credit
   stripped). Simultaneously the **valley minimum jumps 0.60 → 0.85**: at 1e22
   the high-Jaccard docs (0.85–0.90) become the *easiest*, which only happens
   with the capacity to memorize them. Both pin the contamination "phase
   transition" to the same rung as the anchor-gap inversion (anchor − j050:
   +0.112 → −0.040).
3. **1e22 stays the lowest curve at every τ — no crossover.** The curves
   *converge* at the aggressive end as memorization credit is stripped, rather
   than 1e22 crossing above a smaller budget. (The crossing that does invert is
   1e22 vs its own anchor, not vs other budgets.)

**Caveat (do not over-read the small-scale valley).** The *absolute* valley
shape at small scale is a property of the doc sets, not contamination: at 3e18
there is no memorization, yet high-Jaccard docs are intrinsically a bit harder
(and they are *longer* on average — mean doc length rises 737→763 tokens from
j050→j090 — so it is content-type, not a short-doc length artifact). The
contamination conclusion rests on the **scale-evolution** (slope flip +
valley-min jump at 1e21→1e22), which is robust to whatever sets the baseline
shape.

### Artifacts and links

**Plot + consolidated data** (full precision, run→step, cache stats embedded):

- Plot: `plots/decon_val_loss_vs_cutoff_p33m67_lr0.33.png` (repo) ·
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_sweep9/decon_val_loss_vs_cutoff_p33m67_lr0.33.png`
- CSV: `plots/decon_val_loss_vs_cutoff_p33m67_lr0.33.csv` (+ GCS sibling)
- Summary JSON: `gs://…/evals_sweep9/summary_sweep9_p33m67_lr0.33.json` ·
  local `scratch/nemotron_math_decon_sweep_p33m67_lr0.33.json`

**Decon val caches** (`gs://marin-us-east5/tokenized/nemotron_math_val_decon/<tag>/validation`):

| tag | cutoff | docs | tokens | source |
|---|---:|---:|---:|---|
| j050 | 0.50 | 13,947 | 10,282,799 | canonical |
| j055 | 0.55 | 17,403 | 12,797,386 | new |
| j060 | 0.60 | 20,576 | 15,105,371 | new |
| j065 | 0.65 | 23,388 | 17,206,832 | new |
| j070 | 0.70 | 25,868 | 19,064,069 | new |
| j075 | 0.75 | 28,089 | 20,782,728 | canonical |
| j080 | 0.80 | 30,038 | 22,382,501 | new |
| j085 | 0.85 | 31,882 | 24,060,973 | new |
| j090 | 0.90 | 33,196 | 25,346,090 | canonical |

**Sweep build provenance**: `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sweep/`
(`keep_ids/keep_ids_j0{50..90}.json`, `<tag>/docs/`, `<tag>/manifest.json`,
`filter_stats/`).

**Per-run eval outputs** (anchor + 9 decon losses):
`gs://…/decon_val_sets/evals_sweep9/<run>/step-<N>/metrics.jsonl/eval_results.json`

| scale | run | step |
|---|---|---:|
| 3e18 | delphi-3e18-p33m67-k0p20-lr33-a003 | 7399 |
| 9e18 | delphi-9e18-p33m67-k0p20-lr33-a002 | 8818 |
| 2e19 | delphi-2e19-p33m67-k0p20-lr33-a002 | 10982 |
| 3e19 | delphi-3e19-p33m67-k0p20-lr33-a002 | 7573 |
| 9e19 | delphi-9e19-p33m67-k0p20-lr33-a002 | 8032 |
| 2e20 | delphi-2e20-p33m67-k0p20-lr33-a001 | 11277 |
| 3e20 | delphi-3e20-p33m67-k0p20-lr33-a001 | 7081 |
| 1e21 | delphi-1e21-p33m67-9p25b-lr0.33-58ebcb | 4410 |
| 1e22 | delphi-1e22-p33m67-32p07b-lr0.33-e9132105 | 7646 |

Loss keys inside each `eval_results.json`: `eval/nemotron_cc_math_v1/4plus/loss`
(anchor), `eval/decon_j0{50..90}/loss`. W&B mirror: project
`marin-community/marin`, runs `decon-eval-<run>`, tag `decon_val_eval`.

**Scripts** (tested, lint-clean): `scripts/analysis/build_decon_val_sweep.py`
(resumable per-cutoff sweep build), `scripts/analysis/eval_decon_val_sets.py`
(9 decon tags + `--out-root`), `scripts/analysis/plot_decon_val_loss_vs_cutoff.py`.

Next: same sweep over p50m50 / p67m33 — does the valley-min jump track math
fraction the way the anchor-gap inversion does (logbook 2026-06-07)?

## 2026-06-09 — Cross-corpus near-dup probe: subset `3` vs `4plus`

Setup for experiment #4 (decouple difficulty from contamination, per
`debug_midtrain.md`): use subset **`3`** (score-3 bucket, **never trained on** —
only `4plus` is in the midtrain math mix) as an untrained "difficulty" control,
and/or carve a fresh untrained val set from a clean `3` slice. Open question:
how much does `3` content overlap the trained `4plus`? If they share a lot of
near-dup content, `3` is a polluted control; if little, `3` is clean. This
needed a cross-corpus dedup we did NOT have (all prior scans are
val-docs-vs-subset, never subset-vs-subset).

### Subset `3` sizing

400 shards, **56.05M docs**, avg 5,086 chars/doc → **~1,488 tok/doc** (calibrated
on `4plus`'s exact 51.48B-token count via the doc-length ratio; `3` docs run
~30% longer than `4plus`'s 1,142). **~83.4B tokens total.** A 10B-token slice ≈
**6.72M docs ≈ 48 of 400 shards (12%)**; ~209M tokens/shard. datakit normalize
hash-orders + size-balances shards, so the first N shards are an unbiased random
N/400 sample — no sampling job needed.

### Probe method

25k-doc subsample of `3` shard 0 vs the first 24 `4plus` shards (10.4% of
`4plus`), **reusing the existing 284×71 MinHash for both sides** (the bucket keys
are deterministic from the signature, so the two minhash sets are directly
joinable — no re-signing). Bucket-join → exact 5-char-shingle Jaccard ≥0.3.
Driver `scripts/analysis/crossdedup_probe.py`.

- **Why not 1-shard-vs-1-shard** (the obvious quick test): shards are
  content-hash ordered, so a `3` doc's near-dup lands in a *random* `4plus`
  shard — one-vs-one catches ~1/231 of cross-dups (a handful). Must scan one
  `3` shard (subsampled) vs *many* `4plus` shards.
- **Operational saga (3 attempts):** (1) a non-preemptible 32 GB join
  coordinator (copied from the full-scale val scan) sat `pending` ~30 min on the
  congested cluster → small all-preemptible coordinators. (2) join workers hit
  `ModuleNotFoundError` — helpers imported from the sibling `nemotron_math_val_full_scan`
  module don't ship to Zephyr workers (only the installed `marin` package does);
  **functions must be defined in the entrypoint module to ship by value
  (cloudpickle).** Inlined them. (3) ran clean, but **pairwise verify is slow**
  (~21 min/shard at the 0.3 threshold, preemptible-retry-prone).

### Results (24/24 shards, 122,494 verified pairs, 5,021 distinct `3` docs ≥0.3)

Jaccard is **heavily weak-skewed** — median **0.36**, max **0.910**:

| cutoff | pairs | distinct `3` docs | % of 25k query |
|---|---:|---:|---:|
| ≥0.50 | 10,818 | 1,919 | 7.7% |
| ≥0.60 | 3,260 | 1,009 | 4.0% |
| ≥0.70 | 831 | 418 | 1.7% |
| ≥0.75 | 359 | 227 | 0.9% |
| ≥0.80 | 120 | 94 | 0.4% |
| ≥0.90 | 1 | 1 | 0.004% |

Near-identical re-crawls (J≥0.9) are **essentially absent** (1 in 122k). The
overlap is dominated by *weak* templated similarity (shared LaTeX/boilerplate
math phrasing at J≈0.3–0.4), not duplicate documents.

The high-J counts scaled **exactly linearly** with corpus coverage (≥0.75: 76
distinct docs at 8 shards → 227 at 24, ×2.99), i.e. no saturation — each strong
near-dup is specific. Extrapolating to the full 231 `4plus` shards (×9.6):
**~9% of `3` docs have a genuine (≥0.75) `4plus` near-dup corpus-wide**; ≥0.9 is
negligible.

### Verdict

`3` is a **usable untrained control** — it does not share near-identical content
with `4plus`. But ~9% of `3` docs DO have a real near-dup (≥0.75) somewhere in
`4plus`, so the dedup is worth doing: cut at **J≈0.7–0.75** to drop genuine
near-dups while keeping the weak-boilerplate tail (not actually shared
documents). A 10B-token `3` slice deduped at 0.75 loses ~9–12% → ~8.5–9B clean
untrained tokens to carve a fresh val set from.

**Operational rule:** the full `3`-slice-vs-`4plus` dedup must use the
connected-components primitive `compute_fuzzy_dups_attrs([three_mh, fourplus_mh])`
(clusters via the LSH graph, no per-pair text comparison) — **pairwise verify
does not scale** (~21 min/shard for just 24 shards here).

### Artifacts

- Driver: `scripts/analysis/crossdedup_probe.py` (self-contained; inlined
  bucket-join + verify helpers).
- Verified pairs: `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/crossdedup_probe/verified/`
  (122,494 pairs, `val_id`=`3` doc, `other_id`=`4plus` doc, `jaccard`).
- Reused MinHash (284×71, directly joinable, no re-signing):
  `…/3_284x71/minhash/outputs/`, `…/4plus_284x71/minhash/outputs/`.
- Corpora (both in us-east5): `3` = `…/normalized/nemotron_cc_math_v1/3_f8007d22`
  (400 shards), `4plus` = `…/4plus_b05688a8` (231 shards). Subsets are
  `nvidia/Nemotron-CC-Math-v1` rev `397a250` score folders (`3`/`4plus`/`4plus_MIND`).

## 2026-06-09 — Cutoff decision (eyeballed pairs) + clean-`3`-slice dedup launched

### Eyeballed 53 `3`↔`4plus` pairs across the Jaccard range

Extracted (driver `scripts/analysis/crossdedup_samples.py`, in-region — pulls the
`3` text from the probe query subsample and the `4plus` text by scanning the
scanned shards, so only a small JSON is downloaded) 53 verified pairs spanning
**J = 0.30 → 0.91**, both full texts. The scale maps cleanly to semantics:

- **J ≈ 0.30 — *different documents*.** e.g. a logic / conjunctive-normal-form
  problem (`3`) vs "Understanding the Fundamental Theorem of Calculus" (`4plus`).
  They score 0.30 purely from shared math-prose boilerplate (`\[ \]`, "Theorem",
  "continuous real-valued function"). **Not a duplicate.**
- **J ≈ 0.55 — *same auto-generated template, different content*.** e.g. "How Long
  is **86** Days in Minutes? = 123,840" vs "How Long is **56** Days = 80,640".
  Same converter page generator, different numbers — structurally similar but
  **not a memorizable duplicate** (different question/answer).
- **J ≈ 0.88 — *same document, reformatted*.** e.g. an identical "TWO SAMPLE
  Z-TESTS" lesson differing only in heading levels and "vs." vs "versus".
  **Genuine near-dup = the contamination that matters.**

So the boundary for "real duplicate worth removing" sits at **J ≈ 0.7**; ≤0.4 is
"both are math." Side-by-side at `scratch/crossdedup_samples.md`;
`gs://…/crossdedup_probe/samples.json`.

### Cutoff = conservative ~0.4, and the scalability reality

User chose a conservative ~0.4 (over their earlier 0.3 — which the J=0.30 example
showed would delete *unrelated* math). But an **exact** Jaccard cutoff needs
per-pair shingle verify, which does NOT scale: the probe's 25k × 24-shard verify
ran ~21 min/shard; the full 6.7M × 45M job is thousands× that. The scalable method
is the datakit LSH **connected-components** primitive
(`compute_fuzzy_dups_attrs`), which clusters at the 284×71 **band** threshold
(~0.31–0.35 soft: ~84% recall at J=0.4, ~99% at ≥0.5, ~44% at 0.3) — the
conservative reading of 0.4 (high recall on genuine ≥0.4 dups + ~half the weak
0.3–0.4 boilerplate, erring toward removing more). An exact 0.4 is unreachable at
scale; re-banding to lift the threshold would need re-signing (the stored minhash
keeps only the 71 band-buckets, not the raw 284-perm signatures).

### Stage 1 LAUNCHED — CC dedup of the 10B `3` slice vs all `4plus`

- **Fixed 10B sample** = first **48 of 400** `3` shards (~6.7M docs; hash-ordered
  head = unbiased random 12%).
- Driver `scripts/analysis/crossdedup_three_vs_fourplus.py`: server-side-copies
  the 48 `3` minhash shards into a slice dir, then
  `compute_fuzzy_dups_attrs([3-slice, 4plus])` over **279 shards** (48 + 231) —
  **reusing the existing 284×71 minhash for both, no re-signing.**
- **Coordinator-scheduling lesson (third time):** a non-preemptible 16 GB
  fuzzy-dups coordinator sat `pending` ~40 min on the congested cluster.
  Relaunched with a **preemptible** coordinator (`…-cc-v2`); `cc_resume`
  checkpoints each CC iteration so a preempted coordinator resumes rather than
  restarting. Standing rule for this cluster: **default coordinators to
  preemptible + small**; non-preemptible CPU slots are scarce.
- Job: `/ahmedah/three-vs-fourplus-cc-v2`. Output:
  `gs://…/three_vs_fourplus_cc/outputs/source_NNN/` (per-doc `dup_cluster_id`).
  Big job (52M-doc CC over dense math buckets) — hours, babysat.

### Stage 2 (pending, mechanical)

Read the per-source cluster markers; drop every `3`-slice doc whose
`dup_cluster_id` also appears among `4plus` docs (its cluster touches the trained
corpus); filter the 48 `3` shards' text; tokenize the clean **~8–9B-token** slice
(llama3) → a fresh untrained corpus to carve a val set from / use as the
experiment-#4 difficulty control.

Artifacts: `scripts/analysis/crossdedup_samples.py`,
`scripts/analysis/crossdedup_three_vs_fourplus.py`;
`gs://…/midtrain_dedup/three_slice_10b/` (slice minhash + `cc_manifest.json`),
`gs://…/midtrain_dedup/three_vs_fourplus_cc/` (cluster markers).

## 2026-06-10 — Clean `3` slice DONE (and the wrong-banding detour)

Built the clean untrained `3` slice by proper fuzzy dedup against `4plus`. The
first attempts were wrong; recording both the mistake and the fix.

### The mistake: dedup-with-the-scan-banding

I reused the existing **284×71** MinHash (the high-recall *contamination scan*
banding, r=4) for dedup and ran a 1-hop bucket semi-join with no verify. It
flagged **99.9%** of the slice (6.72M of 6.73M). Not a bug — at 284×71 every
math doc shares an LSH band with ~35k other math docs purely via shared
boilerplate n-grams (the 4plus val scan: **2.02B candidate comparisons →
2.24M real ≥0.5 pairs**, a ~900× over-collision). 284×71 is a *candidate
generator that requires a per-pair verify*; it is NOT a dedup banding. The
connected-components run on the same minhash ground for ~2.5 hr (mega-clusters
from boilerplate links) for the same reason.

**Root cause:** I conflated *measuring* contamination (high recall + verify,
the earlier work) with *removing* duplicates (standard dedup). When the goal
changed I should have switched tools.

### The fix: standard fuzzy dedup at 286×26

Proper marin/datakit fuzzy dedup uses the **default 286×26** banding (r=11,
~0.8 Jaccard) where a shared band-bucket already *means* a genuine
near-duplicate — so connected components alone gives clean clusters, **no
verify, no pair comparison**. The 286×26 `4plus` minhash already existed (the
original 2026-06-03 scan, 45.1M docs at `…/midtrain_dedup/minhash/outputs/`).
Driver `scripts/analysis/crossdedup_three_vs_fourplus.py` (rewritten): copy the
first 48 `3` shards → MinHash at 286×26 → `compute_fuzzy_dups_attrs([3-slice,
4plus])`. At r=11 the graph is sparse (boilerplate doesn't link everything), so
the CC ran cleanly (it_0…it_15, ~2 hr; hit the 15-iteration cap — fine, the
clusters are tight genuine near-dups). 51.8M docs → 31.7M singletons + 20.1M
cluster members across 2.97M clusters. `source_000`=`4plus`, `source_001`=`3`.

### Result

Stage 2 (`scripts/analysis/crossdedup_three_clean.py`): a `3` doc is a duplicate
of training iff its `dup_cluster_id` also appears among `4plus` docs.

| | |
|---|---:|
| slice (48 shards) | 6,728,439 docs (~10.0B tokens) |
| **duplicate of `4plus`** (cluster touches trained) | **920,643 docs = 13.7%** |
| **clean untrained slice** | **5,807,796 docs (~8.6B tokens)** |

~14% of the score-3 slice is genuinely near-duplicated (~0.8) in the trained
score-≥4 corpus — consistent with the probe's ~9% at ≥0.75 plus the CC's
transitive closure. (Caveat: CC ran to the 15-iter cap, so a few large
transitive clusters may not have fully merged — could nudge 13.7% slightly.)

### Stage 3 DONE — clean corpus materialized + tokenized

`scripts/analysis/crossdedup_three_materialize.py`: zephyr-filter the 48 slice
shards (drop the 920,643 ids) → clean docs → tokenize (llama3) → Levanter cache.
Job `/ahmedah/three-clean-materialize` succeeded (~15 min). Result:

- **5,807,796 clean docs / 7,961,234,840 tokens (~8.0B)** at
  `gs://marin-us-east5/tokenized/nemotron_math_3_clean_10b/train/` (llama3).
- Clean docs (text): `gs://…/three_slice_10b/clean_docs/` (48 shards);
  manifest `…/three_slice_10b/clean_cache_manifest.json`.

A fresh, untrained, deduplicated **~8.0B-token** math corpus — ready to carve a
clean val set from (contamination-free eval) / use as the experiment-#4
difficulty control. (Subset `3`, score-3 bucket, never in the midtrain math mix;
now also stripped of its ~0.8 near-dups with the trained `4plus`.)

### Lessons (operational)

1. **Use the dedup banding (286×26 default) for dedup, NOT the scan banding
   (284×71).** 284×71 is high-recall-for-measurement and needs verify; 286×26 is
   the duplication threshold where CC alone works.
2. The standard `compute_fuzzy_dups_attrs` over `[slice, corpus]` minhash is the
   right tool — reuses minhash, no text re-read, no per-pair verify.
3. **A hard ~0.4 Jaccard cutoff is not achievable at 6.7M×45M scale:** high
   recall at 0.4 needs the over-collisional banding → ~350B verify comparisons.
   ~0.8 (standard dedup) removes the genuine duplicates and keeps the
   non-memorizable templates (the "86 vs 56 days" pages) — which is also the
   *correct* threshold (removing 0.4–0.7 templates would bias the corpus).
4. **This cluster: default coordinators to preemptible + small** — non-preemptible
   CPU coordinators stalled `pending` repeatedly; `cc_resume` covers the
   preemption restart.

Artifacts: `crossdedup_three_vs_fourplus.py` (286×26 dedup),
`crossdedup_three_clean.py` (labeling), `crossdedup_three_materialize.py`
(filter+tokenize); `gs://…/three_slice_10b/` (docs, minhash_286x26,
contaminated_ids_286, clean_summary.json), `…/three_vs_fourplus_dedup/` (markers).

### How many tokens survive a *stricter* 0.5 cutoff? (estimate)

The clean 8.0B is deduped at ~0.8 — it still contains the [0.5, 0.8) "templated
but not identical" band. To estimate how much a 0.5-threshold dedup would keep,
scale the actual 286×26 drops by the doc-level max-Jaccard distribution from the
val↔`4plus` verified-pair scan (`plots/jaccard_histogram_summary.json`), the only
exact same-corpus near-dup distribution we have:

- docs with a match ≥ 0.50: 20,855; with a match ≥ 0.75: 9,760 → **2.14× more
  docs cross 0.5 than cross ~0.8**.
- Applied to the slice: ~920,643 drops at ~0.8 → **~1.97M drops at 0.5**, i.e.
  keep ~4.76M docs ≈ **~6.5B tokens** (range ~6.1–6.9B for a ±15% multiplier).
- So going 0.8 → 0.5 costs **~1.4B tokens (~18%)**; ~6.5B of the 8.0B is genuinely
  below 0.5 Jaccard to the trained corpus.

Caveat: the multiplier is from benchmark-val docs (short, curated) vs `4plus`;
score-3 web pages are longer, so treat 6.5B as ±0.4B. The **exact** number needs a
0.5-threshold pass: reuse the existing 284×71 slice+`4plus` minhash → candidate
pairs → exact-shingle verify → count slice docs with max-Jaccard ≥ 0.5 (the
val-scan pipeline pointed at 3-vs-`4plus`; ~couple hours, not yet run).

## 2026-06-11 — How the dedup/decon machinery works + val-set gotchas

Reference write-up of the MinHash → LSH → CC → verify stack, the exact val-decon
pipeline as actually run, and what we'll need to recompute as we do more evals.

### MinHash + LSH, from first principles

**Shingles.** Each doc → set of **character 5-grams** (`ngram_size=5`, char-level
over lowercased, whitespace-collapsed text — `fuzzy_minhash.py:141`). The verify
uses identical shingling so its exact Jaccard matches the scan's notion.

**MinHash — one hash is a coin whose bias *is* the Jaccard.** Hash every shingle,
take the global min. For two docs `P(min(A)=min(B)) = |A∩B|/|A∪B| = J`, *exactly*:
the union's lowest-hash shingle is uniform over the union, and the two minima
agree iff that winner is a **shared** shingle (event=intersection, space=union).
One hash = one Bernoulli(J) **outcome** (0/1), not the probability itself.

**Signature.** `num_perms=286` independent hashes → 286 numbers; the fraction of
matching slots is a sample mean of 286 Bernoulli(J) → an **unbiased** estimate of
J, std ≈ √(J(1−J)/286) ≈ ±0.03. Averaging kills **variance**, not bias (each slot
is already centered on J — a true bias would survive averaging).

**Ideal vs real hash.** The identity assumes a uniform random *permutation* of the
shingle universe (min-wise independence). A real 64-bit hash approximates it; its
only deviation is a tiny **systematic** bias from finite-range collisions
(negligible at 64-bit; bites at 32-bit on long docs). A true permutation is
infeasible (a rank for every possible shingle, ×286) — the hash is the only
implementable, reproducible, coordination-free stand-in.

**Banding = a second, different hash.** Split the 286-number signature into `b`
bands of `r` rows (`b·r=286`); **hash each band's r numbers into a bucket id** — a
plain *content* hash (NOT the band index 1..b, which is identical for every doc).
Two docs are **candidates** iff some band's r numbers are identical (shared
bucket). Code: dupekit `MinHash(num_perms)` → `MinHashLSH(num_bands)` → a `buckets`
list of `b` ids/doc (`fuzzy_minhash.py:86-91`). Carries no similarity logic — just
a cheap fixed-size group key (could group on the raw r-tuple instead).

**S-curve.** Per-slot match prob = J ⇒ `P(candidate) = 1 − (1 − J^r)^b`, with 50%
point at `J* ≈ (1/b)^(1/r)`. The `(r,b)` split is the **only** knob: more rows =
stricter AND/band = step slides right; more bands = more OR's = slides left. Plot:
`plots/lsh_scurve_bandings.png` (`scripts/analysis/plot_lsh_scurve.py`).

| banding | r×b | 50% J* | P(cand)@0.5 | @0.8 | role |
|---|---|---|---|---|---|
| **286×26** | r=11,b=26 | **0.72** | 0.013 | 0.90 | **DEDUP** — shared bucket ⇒ real dup, no verify |
| **284×71** | r=4,b=71 | **0.31** | 0.99 | ~1.0 | **SCAN** — high recall to ~0.3, *needs* verify |

**Connected components.** Candidate links → graph (edge = shared bucket) → CC
transitive closure → clusters (`dup_cluster_id`). At 0.8 edges are real → tight
families (safe); at 0.3 edges are mostly boilerplate → one giant blob swallows the
corpus (the 99.9% failure). CC is only trustworthy at the dedup banding.

**Verify.** Recompute exact shingle Jaccard on candidates
(`verify_math_val_pairs_zephyr.py:86`). Kills **false positives → 0**; only **false
negatives** (missed pairs) can leak, bounded by scan recall. Design rule: **set the
scan threshold below your decision cutoff, then verify** — 284×71 (0.31) gives
~99% recall at J=0.5.

### How the val decon sets were actually built

Per training source — **separately for `3`, `4plus`, `4plus_mind`** (each has
284×71 minhash + verified_pairs: 400 / 231 / 336 shards):

1. **284×71 scan** (val × source) → candidate (val, source) pairs.
2. **Verify** → exact 5-char Jaccard, keep ≥0.5 (`MIN_REPORT_JACCARD=0.5`).
3. **Max per val doc over all three sources** → `max_jaccard_by_doc` = a doc's
   worst-case similarity to *anything* in the math sources.
4. **Cutoff τ**: keep docs with union-max < τ. Sets **strictly nest** (j050 ⊂ … ⊂
   j090; 13,947 → 33,196 docs; 10.3M → 25.3M tok).

`plots/jaccard_histograms.png` is the **`4plus` slice only**; the cutoff folds all
three sources via the max. Construction is **exact in value** (verify) and
**~complete** (≥99% recall at 0.5, ~100% above 0.55) — no genuine contamination
removed, ~1% false-negative leak only in a thin band right at 0.5.

### ⚠️ Gotchas + what to recompute as we run more evals

**1. The decon source set must match the *trained* sources — and right now it does
not.** p33m67 trains **`4plus` only** (`nemotron_cc_math_v1/4plus` @ **0.67**; `3`
and `4plus_mind` are **absent** from the mix — verified in
`p33m67_data_section.json`). But the val sets threshold on the **union(3, 4plus,
4plus_mind)** max → they are **over-decontaminated**: they drop val docs similar to
the *untrained* `3` / `4plus_mind`, which aren't contaminated for this run. This
doesn't corrupt the 4plus-memorization signal (a 4plus-contaminated doc crosses τ
on its 4plus-Jaccard regardless) but adds a composition shift (extra clean docs
removed). **For the clean 4plus-only measurement: re-derive `max_jaccard_by_doc`
from the `4plus` verified_pairs only, rebuild cutoff sets, re-eval — no rescan
(per-source pairs already exist).** A val set is only "clean" *relative to a
specific training mix*.

**2. Decon gap — trained math/code sources NOT in the scan set.** The mix also
trains **`proofpile_2` @ 0.00165** (math: arxiv/textbooks/competition) and
**`starcoderdata` @ 0.0075** (code), neither of which the val set is
decontaminated against. Small weights, but proofpile is math-dense and may overlap
math-val docs → **undetected** contamination. Before trusting a "clean" number,
scan+verify val vs `proofpile_2` (and `starcoderdata` for code val).

**3. Per-mix recompute rule.** Any eval with a different math mix needs its val
cutoff re-derived over **exactly its trained sources**. The union over
{3,4plus,4plus_mind} is *superset-safe* for any subset of those three (never
under-cleans within the set) but matches no single-source run and covers nothing
outside the three. New trained source ⇒ new scan+verify of val vs that source.

**4. τ < 0.5 is unavailable from current data.** Verify kept only pairs ≥0.5, so a
stricter cutoff needs re-verifying at a lower `MIN_REPORT_JACCARD` (and the 0.31
scan floor caps recall below ~0.4 regardless).

**5. Population restriction.** The sweep is the **paranoid short-doc** slice (docs
fully contained in val windows). Absolute losses are "short-contained-doc math val
loss," not general math val. The ★ "none" point is a *different* population — don't
compare it on-axis to the curve.

**6. Composition confound (interpretation, not construction).** Exact cutoffs
still compare *different doc populations* across τ → loss-vs-τ entangles
memorization with intrinsic difficulty. The non-monotonic valley (min τ≈0.6) is
the fingerprint; the contamination signal lives in the **cross-compute slope flip**
(3e18: decon lowers loss; 1e22: decon raises it). The clean-untrained-`3` corpus
(8.0B) is the difficulty control that subtracts composition directly. → pair every
decon eval with the clean-3 control.

**7. Reusables (don't recompute).** 284×71 minhash exists for val + all three
sources **and** the 10B `3`-slice; 286×26 minhash exists for `4plus` (45M) + the
slice. Every recompute above (4plus-only cutoff, proofpile scan, exact-0.5 number)
reuses these — only re-verify or re-derive, never rescan from text. Keep shingling
identical (char 5-gram, lowercased, ws-collapsed) or Jaccards won't match.

## 2026-06-11 (later) — Corrections to the dedup/decon write-up above

Re-audited the 2026-06-11 reference write-up against the actual code and the
vendored data section. Three of its claims are wrong or incomplete; one gotcha is
mis-framed. The qualitative contamination findings survive (see "Net impact"),
but the τ labels and the "what's contaminated against what" picture were muddled.
Everything below was checked against source, not prose.

### Correction 1 — the scan and the verify do NOT use identical shingling

The write-up states (twice): *"The verify uses identical shingling so its exact
Jaccard matches the scan's notion"* and gotcha 7's *"keep shingling identical
(char 5-gram, lowercased, ws-collapsed) or Jaccards won't match."* **The pipeline
already violates this.** The two halves normalize text differently:

- **Candidate scan (MinHash, all bandings):** shingles the output of
  `dupekit.Transformation.CleanText` (`fuzzy_minhash.py:83`), which **lowercases,
  collapses whitespace, AND strips all punctuation / LaTeX**. Verified by running
  it: `\frac{1}{x^2}` → `frac1x2`, `-2/x^3` → `2x3`, `4.1` → `41`, `(p. 12)!` →
  `p 12`.
- **Verify (the reported Jaccard, and the τ cutoffs that built j050…j090):**
  `_shingles` in both `verify_math_val_pairs_zephyr.py:64` and
  `nemotron_math_val_full_scan.py:131` does **`_WS.sub(" ", text.lower()).strip()`
  only — it KEEPS every backslash, brace, slash, period, paren.**

So for LaTeX-heavy math the two produce different 5-char shingle sets. Recomputing
both Jaccards on the 53 local `crossdedup_samples` pairs (the verify recomputation
reproduces the stored value exactly, confirming the reported number is the
punctuation-keeping one):

| | CleanText_J − verify_J |
|---|---|
| mean | **+0.028** |
| median | +0.025 |
| range | −0.027 … **+0.120** |
| pairs with CleanText_J > verify_J | 46 / 53 |

The gap **grows with similarity**: for genuine near-dups (verify_J ≈ 0.7–0.9 — the
docs we actually want to drop) CleanText_J is **+0.05 to +0.12 higher**. So the
banding/candidate side sees each near-dup as markedly *more* similar than the
verified number that the cutoff is applied to.

**Consequences (concrete):**
- **Recall is fine — better than stated.** Candidates are generated at CleanText_J
  (higher), so a pair at verify_J = τ has CleanText_J ≈ τ + 0.03–0.05, comfortably
  above the 284×71 ~0.31 floor. No candidates are lost to this.
- **The τ labels are mis-calibrated, on the punctuation-keeping scale.** "j075 =
  keep docs with max train J < 0.75" is in verify units; those docs run
  ~0.78–0.82 on the dedup/CleanText scale. The "clean" sets therefore **retain
  genuine formatting-variant near-dups** (e.g. `\frac1x` vs `\frac{1}{x}`, which
  CleanText collapses to identical but the verify scores < 1). This is **mild
  UNDER-decontamination**, strongest right at the j050 boundary (compounded by
  gotcha 4: verify only kept pairs ≥ 0.5, so docs with CleanText_J ∈ [0.5, ~0.55]
  but verify_J < 0.5 are silently kept in the strictest set).

### Correction 2 — "p33m67 trains 4plus only" is wrong, and the gotchas scan the wrong leftovers

The actual `train_weights` (vendored `scripts/analysis/p33m67_data_section.json`,
33 components):

| weight | source |
|---:|---|
| 0.670 | nemotron_cc_math_v1/**4plus** (the only math subset) |
| **0.317 total** | **nemotron_cc/* TEXT family** (medium 0.101, hq_synth 0.082, medium_low 0.046, hq_actual 0.027, medium_high 0.025, low_actual 0.021, low_synth 0.019) |
| 0.0075 | starcoderdata |
| 0.00165 | proofpile_2 |
| 0.0 ×24 | paloma/* + uncheatable_eval/* (held-out eval components, weight 0) |

The model trains on **31.7% nemotron_cc TEXT**, by far the largest source the val
set is **not** decontaminated against. Gotcha 2 flagged only proofpile_2 (0.165%)
and starcoderdata (0.75%) as the un-scanned trained sources and **missed a channel
~190× larger than proofpile**. The correct decon-source set for these runs is
`{4plus} ∪ {nemotron_cc text, proofpile_2, starcoder}`; the scan actually used
`{3, 4plus, 4plus_mind}` — i.e. it got 4plus right, **added two untrained
sources** (`3`, `4plus_mind` — Correction 4), and **missed all three trained
non-4plus sources**.

**Provenance — checked against both papers (arXiv 2412.02595, 2508.15096):**
Nemotron-CC-Math is **NOT** a subset of, nor extracted from, Nemotron-CC text.
Both are **independent extractions of Common Crawl**:
- **Nemotron-CC** (text, 6.3T tok): CC-MAIN-2013-20 … 2024-30, **Justext** HTML
  extractor (general web; mangles/drops math — that limitation is the entire
  motivation of the math paper), + synthetic LLM rewrites (`hq_synth`).
- **Nemotron-CC-Math** (133B/3+, 52B/4+): raw CC (98 snapshots, same era),
  **math-URL-filtered** (seeded by MegaMath/FineMath/OWM/InfiMM-WebMath URL lists),
  **Lynx render + LLM→LaTeX** clean. `4plus` = scores 4-5 (45M docs, 52.3B tok);
  `3` = score-3-only (~56M docs, ~81B tok). Their own fuzzy dedup + decontam are
  **within the math corpus** / against downstream benchmarks — neither touches the
  marin train/val split nor the text corpus.

So the two are **siblings from the same archive**, not subset/superset, and the
67%-math / 31.7%-text mixes are genuinely different token streams. But overlapping
CC snapshots mean they **can contain the same underlying web pages, rendered into
very different strings** (Justext-general vs Lynx+LLM-LaTeX). Consequence for the
text channel: char-5-gram fuzzy Jaccard between a math val doc and any Justext twin
of the same page is **plausibly low** (radically different extraction + an LLM
rewrite on the math side), so a val-vs-text scan would likely flag **few** fuzzy
near-dups — the text channel is a **weak fuzzy contaminant** despite its size.
What fuzzy dedup *cannot* see is the **same-underlying-page semantic exposure**
(the model met the page's content during training, just formatted as prose instead
of LaTeX). Net: still worth a confirmatory scan as the largest unscanned source,
but the prior is "small fuzzy overlap," and the residual semantic overlap is
real and invisible to MinHash. (My earlier "parent corpus" phrasing here was wrong
— corrected.)

### Correction 3 — gotcha 1's over-decontamination is REAL but small (~6%)

Magnitude, from the per-subset counts already on record (no new compute): union
drop minus 4plus-only drop = docs dropped purely for resembling the *untrained*
`3` / `4plus_mind`:

| τ | 4plus-only drop | union drop | extra (over-decon) | % of union |
|---:|---:|---:|---:|---:|
| 0.50 | 32,619 | 34,565 | 1,946 | 5.6% |
| 0.70 | 13,673 | 14,474 | 801 | 5.5% |
| 0.75 | 10,030 | 10,636 | 606 | 5.7% |
| 0.90 | 1,011 | 1,120 | 109 | 9.7% |

So ~94% of `3`-contaminated val docs are *also* 4plus-contaminated (expected —
`3` and `4plus` share content). The 4plus-only re-derivation adds back only ~600
docs at τ=0.75 (≈2% of the 28,089-doc keep set). Worth doing (it's free from the
existing per-source `verified_pairs`), but a minor composition shift, not a
signal-changer.

### Correction 4 — the "per-mix" framing (gotcha 3) is moot: all three mixes are 4plus-only

p33m67 / p50m50 / p67m33 differ **only in `train_weights`** (the byte-identical
val contract; the `data:` block is verbatim-shared). The math source is
`nemotron_cc_math_v1/4plus` in all three — `3` and `4plus_mind` appear in **no**
mix. So "re-derive the cutoff over each run's trained sources" is **one and the
same 4plus-only re-derivation for all three mixes**, not a per-mix-varying task.

### Net impact on the headline finding (does the contamination verdict survive?)

**Yes, and the measured effect is if anything a slight under-estimate.** The two
construction errors push in opposite directions and are both small:
- Under-decon from the shingling offset leaves some memorization credit in j050 →
  j050's large-scale loss is artificially low → `anchor − j050` *less* negative →
  **understates** contamination.
- Over-decon from the union strips ~6% clean docs from j050 → small composition
  shift, partially offsetting.

Neither touches the load-bearing evidence, which is *scale-evolution*, not an
absolute level: the `anchor − j050` sign flip (+0.11 → −0.04), the within-decon
slope flip (`j090 − j050`: +0.027 → −0.050 at 1e21→1e22), the valley-min jump
(0.60 → 0.85), and the dose-response on math fraction. The "~37% of the 1e21→1e22
gain is contamination credit" number is robust to a few points and is a lower
bound under proper CleanText-scale decontamination.

### Concrete next actions (priority order)

1. **4plus-only re-derivation (free, no rescan).** Rebuild `max_jaccard_by_doc`
   from the `4plus_284x71/verified_pairs` *only*, regenerate the cutoff keep-lists,
   re-eval. Removes the over-decon (Correction 3) and gives the honest single-source
   number for all three mixes (Correction 4). Per-source pairs already exist.
2. **Decide the Jaccard scale and make it consistent.** Either (a) re-verify the
   existing candidates with CleanText shingling so the reported J and the τ cutoffs
   live on the same scale the banding uses (reuses candidates; re-verify only), or
   (b) explicitly relabel every τ as "verify-J / punctuation-keeping" and shift the
   intended dedup cutoff up ~0.03–0.05. Option (a) is the correct fix.
3. **Scan val vs the 31.7% nemotron_cc TEXT family** (Correction 2) before any
   "clean relative to the trained mix" claim — the largest unscanned trained
   source, though expect **few fuzzy hits** (different-extractor strings; the real
   residual is same-page semantic overlap MinHash can't see). Needs the text family
   normalized + minhashed in us-east5 (a job; not yet available). proofpile_2 /
   starcoderdata are a distant second.

Reusables note from the prior entry still holds (minhash exists for val + 3
sources + slices); the only correction is that "keep shingling identical" was
aspirational — the scan (CleanText) and verify (punctuation-keeping) were never
identical, and action 2 is what actually makes them so.

### 🔁 HANDOFF (2026-06-11) — discrepancies, unknowns, recompute menu (read first)

Surfaced while explaining the pipeline. Resolve before trusting cross-references
between artifacts. **Banding cheat-sheet:** the val SCAN = **284×71 (71 bands**,
r=4, thr ~0.31, +verify); the clean-corpus DEDUP = **286×26 (26 bands**, r=11,
thr ~0.72, no verify). The val-loss plot is the **71-band** scan.

**⚠️ WEIRD #1 — two `4plus` verified_pairs sets disagree ~6×.**
- `gs://…/midtrain_dedup/verified_pairs/` (OLD): **374,929** pairs, **20,855** val
  docs. This is what `plots/jaccard_histogram_summary.json` + the histograms + the
  6.5B estimate use (`parquet_glob: scratch/verified_pairs`).
- `gs://…/midtrain_dedup/4plus_284x71/verified_pairs/` (CANONICAL, 231 shards):
  **2,238,591** pairs, **32,619** val docs. This is what the **decon val sets** use
  (`build_decon_val_sweep.py` `SCAN_SUBSETS`); it reproduces the j090 universe's
  `KNOWN_DOCS` exactly, so it's the real one.
- **Unknown why 6× apart.** Likely the OLD set covered only the paranoid short-doc
  val subset (or earlier candidate-gen) and the canonical one a larger/full val
  set. Confirm which val population each scan covered before mixing their numbers.

**⚠️ WEIRD #2 — the 6.5B "≥0.5 clean tokens" estimate rests on the OLD scan.** It
used the 2.14× doc-max ratio (20,855 / 9,760) from the old histogram. The canonical
4plus scan implies a **steeper** ratio (in-universe 18,152 / 4,820 ≈ **3.8×**), so
**6.5B is likely an over-estimate** of clean tokens (steeper ⇒ more docs cross 0.5
⇒ fewer clean). Don't trust 6.5B; the exact-0.5 pass (#2 below) resolves it.

**⚠️ WEIRD #3 — `plots/jaccard_histograms.png` ≠ the data the decon sets use.** Same
root cause as #1 (old scan). Regenerate from `4plus_284x71/verified_pairs` before
citing histogram shape next to the decon curve.

**Quantified — union-decon over-removal (gotcha #1 magnitude).** Val sets threshold
on max over (3, 4plus, 4plus_mind) but p33m67 trains `4plus` only. Inside the j090
universe (33,196 docs), docs removed by **union but clean vs 4plus**:

| τ | over-removed | 4plus-keep vs union-keep |
|---|---|---|
| 0.50 | 1,097 | 15,044 vs 13,947 (+7.9%) |
| 0.60 | 654 | 21,230 vs 20,576 |
| 0.70 | 429 | 26,297 vs 25,868 |
| 0.80 | 203 | 30,241 vs 30,038 |

Modest, shrinks with τ. Per-id 4plus-only max already written to
`gs://…/decon_val_sweep/fourplus_only_max_by_id.json`
(`scripts/analysis/decon_val_fourplus_only.py`) → 4plus-only cutoff sets are a
threshold-and-tokenize away, no rescan.

**Unverified / unsure (check, don't assume):**
- Whether `3` and `4plus_mind` also have an old-vs-new duplicate scan (only `4plus`
  was checked). Their per-subset paths (400 / 336 shards) are what the build uses.
- Docs with union-max ≥ 0.90 are pre-excluded from the j090 universe, so the
  over-removal table can't see 4plus-clean docs killed by a ≥0.9 twin in 3 / mind
  (a separate, smaller leak at the top). Need the full val list to assess.
- The 13.7% `3`-vs-`4plus` contaminated figure: CC hit the 15-iter cap, so a few
  transitive clusters may not have fully merged → possibly a slight under-count.

**Recompute menu (all reuse existing minhash — never rescan from text):**
1. **4plus-only re-derivation + re-eval** — data ready (`fourplus_only_max_by_id.json`);
   tests whether the ~8% over-decon moves the loss curve.
2. **Regenerate histograms** from canonical `4plus_284x71/verified_pairs`.
3. **Exact-0.5 pass (#2)** — slice+4plus 284×71 minhash → verify → real ≥0.5 number
   (replaces the suspect 6.5B).
4. **Scan val vs `proofpile_2`** (trained @0.00165, math, NOT deconned) — gotcha #2.
5. **Reconcile the two verified_pairs sets** (WEIRD #1) — prerequisite for trusting
   any cross-artifact comparison.

New scripts this session (uncommitted, with the crossdedup_* set):
`plot_lsh_scurve.py` (S-curve), `decon_val_fourplus_only.py` (4plus-only contam).

## 2026-06-12 — 4plus-only decon val sets built + re-evaled: the over-decon fix changes no conclusion

Acted on Correction 3 (logbook 2026-06-11): the canonical decon caches threshold
on the **union(3, 4plus, 4plus_mind)** max Jaccard, but p33m67/p50m50/p67m33 train
**`4plus` only** as their math source, so the union sets are **over-decontaminated**
(drop val docs resembling the untrained `3`/`4plus_mind`). Rebuilt the paranoid
short-doc sets decontaminated against **`4plus` alone** and re-evaled the
p33m67 × lr0.33 ladder to check whether the contamination findings survive the
correct single-source decon.

### Build (`scripts/analysis/build_decon_val_4plus_only.py`, + test)

Recomputes each val doc's max train Jaccard from the **`4plus_284x71`
verified_pairs only** (no rescan — reuses the existing pairs + replay arrays).
Same paranoid filter (fully contained in val windows AND max 4plus J < cutoff),
same 0.05 grid, fresh roots (`…/decon_val_4plus`, `tokenized/nemotron_math_val_decon_4plus`).
Self-verifying: reproduced the fully-contained baseline exactly (33,790 docs /
25,956,642 tok), reproduced the published **union** j050/j075/j090 keep counts,
and confirmed `4plus_max <= union_max` for every union-j090 doc. One in-region
us-east5 job; the slow single-worker tokenize was parallelized into one
`--skip-filter` job per remaining cutoff (the fast path). All 9 caches token-exact.

**Over-decontamination magnitude (clean docs the union wrongly dropped, added back):**

| cutoff | union keep | 4plus-only keep | added back |
|---|---:|---:|---:|
| j050 | 13,947 | 15,053 | +1,106 (+7.9%) |
| j055 | 17,403 | 18,243 | +840 |
| j060 | 20,576 | 21,241 | +665 |
| j065 | 23,388 | 23,961 | +573 |
| j070 | 25,868 | 26,314 | +446 |
| j075 | 28,089 | 28,395 | +306 |
| j080 | 30,038 | 30,263 | +225 |
| j085 | 31,882 | 32,007 | +125 |
| j090 | 33,196 | 33,249 | +53 |

Largest at the strict end (~94% of `3`-contaminated val docs are *also*
4plus-contaminated, so they drop either way; the residual ~6% is the over-decon).

### Eval (p33m67 × lr0.33, 9 scales, v6e-4)

`eval_decon_val_sets.py` extended with `--decon-cache-root` / `--wandb-tag`
(W&B tag `decon_val_eval_4plus`, outputs under `…/decon_val_sets/evals_4plus`).
**Sanity gate 9/9:** the harness anchor (`eval/nemotron_cc_math_v1/4plus/loss`,
the byte-identical original val) reproduced the union sweep's anchor to **≤2e-4**
at every scale — the anchor component is identical across variants, so the
4plus-vs-union deltas are pure decon-set composition. (Infra: 1e22 looped on
preemption restarts at pdp=2; ran it non-preemptible to completion.)

**4plus-only math val loss (nats/token), final checkpoint per scale:**

| scale | anchor | j050 | j055 | j060 | j065 | j070 | j075 | j080 | j085 | j090 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3e18 | 1.4720 | 1.3724 | 1.3547 | **1.3512** | 1.3565 | 1.3622 | 1.3754 | 1.3917 | 1.4111 | 1.4276 |
| 9e18 | 1.3034 | 1.2076 | 1.1918 | **1.1891** | 1.1945 | 1.2004 | 1.2133 | 1.2289 | 1.2472 | 1.2630 |
| 2e19 | 1.2203 | 1.1273 | 1.1124 | **1.1102** | 1.1157 | 1.1213 | 1.1338 | 1.1491 | 1.1670 | 1.1822 |
| 3e19 | 1.1640 | 1.0736 | 1.0590 | **1.0567** | 1.0622 | 1.0677 | 1.0799 | 1.0947 | 1.1123 | 1.1273 |
| 9e19 | 1.0425 | 0.9605 | 0.9466 | **0.9440** | 0.9490 | 0.9541 | 0.9653 | 0.9793 | 0.9954 | 1.0098 |
| 2e20 | 0.9737 | 0.8996 | 0.8856 | **0.8826** | 0.8871 | 0.8916 | 0.9021 | 0.9152 | 0.9303 | 0.9439 |
| 3e20 | 0.9286 | 0.8610 | 0.8470 | **0.8436** | 0.8475 | 0.8515 | 0.8612 | 0.8735 | 0.8877 | 0.9008 |
| 1e21 | 0.8104 | 0.7718 | 0.7558 | **0.7500** | 0.7514 | 0.7525 | 0.7592 | 0.7682 | 0.7788 | 0.7895 |
| 1e22 | 0.5727 | 0.6227 | 0.6003 | 0.5868 | 0.5792 | 0.5717 | 0.5680 | 0.5657 | 0.5636 | **0.5635** |

### Comparison vs union sweep — every signature replicates

| scale | anchor−j050 (4plus / union) | j090−j050 (4plus / union) | valley-min τ (4plus / union) |
|---|---:|---:|---:|
| 3e18 | +0.0996 / +0.1123 | +0.0551 / +0.0668 | 0.60 / 0.60 |
| 9e18 | +0.0958 / +0.1083 | +0.0554 / +0.0670 | 0.60 / 0.60 |
| 2e19 | +0.0930 / +0.1053 | +0.0549 / +0.0663 | 0.60 / 0.60 |
| 3e19 | +0.0905 / +0.1026 | +0.0538 / +0.0649 | 0.60 / 0.60 |
| 9e19 | +0.0820 / +0.0935 | +0.0493 / +0.0599 | 0.60 / 0.60 |
| 2e20 | +0.0741 / +0.0853 | +0.0443 / +0.0547 | 0.60 / 0.60 |
| 3e20 | +0.0676 / +0.0783 | +0.0398 / +0.0499 | 0.60 / 0.60 |
| 1e21 | +0.0386 / +0.0491 | +0.0177 / +0.0275 | 0.60 / 0.60 |
| 1e22 | **−0.0500 / −0.0399** | **−0.0591 / −0.0496** | **0.90 / 0.85** |

### Verdict — fixing the over-decontamination changes NO conclusion

All three contamination signatures from the union sweep replicate under the
correct 4plus-only decon, and the signal is **marginally stronger**:

1. **anchor−j050 still inverts** +0.10 → −0.05 (sign-flips between 1e21 and 1e22),
   vs union +0.11 → −0.04. The contaminated original val ends up *easier* than
   the strictest clean set at 1e22.
2. **Within-decon slope j090−j050 still flips** + → −0.059 at the 1e21→1e22 rung
   (union → −0.050): removing near-dups *raises* loss = memorization credit
   stripped.
3. **Valley minimum still jumps** 0.60 → **0.90** at 1e22 (union 0.60 → 0.85): the
   high-Jaccard docs become the *easiest* only with capacity to memorize them.
4. **The non-monotonic valley at τ=0.60 holds at every scale ≤1e21** — strictest
   (j050) is not the loss-minimizing clean set, in both variants.

Mechanism of the small offset: the 4plus-only j050 adds back ~1,106 clean-but-
slightly-harder docs (they resembled the untrained `3`/`4plus_mind`), nudging j050
loss up ~0.012 and shrinking the positive gaps — but the strict set is now the
*honest* hardest set even more cleanly, so the 1e22 inversion is slightly larger.
The union sets were superset-safe; correcting to single-source neither created nor
removed the contamination signal. **The earlier "~37% of the 1e21→1e22 gain is
contamination credit" conclusion stands** (4plus-only 1e22 anchor−j050 is more
negative, so if anything a touch higher).

### Artifacts

- Caches: `gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus/j0{50..90}/validation`
  (15,053 → 33,249 docs; exact-verified; llama3).
- Build provenance: `gs://…/midtrain_dedup/decon_val_4plus/` (keep_ids, docs, manifests, filter_stats).
- Evals: `gs://…/decon_val_sets/evals_4plus/<run>/step-<N>/metrics.jsonl/eval_results.json` ×9;
  W&B tag `decon_val_eval_4plus`.
- Plot + CSV: `plots/decon_val_loss_vs_cutoff_p33m67_lr0.33_4plus.{png,csv}` (+ GCS sibling under `evals_4plus/`).
- Scripts (tested, lint-clean): `build_decon_val_4plus_only.py` (+ `tests/analysis/test_build_decon_val_4plus_only.py`),
  `eval_decon_val_sets.py` (`--decon-cache-root`/`--wandb-tag`), `plot_decon_val_loss_vs_cutoff.py` (`--eval-root`/`--slice`).
- Note: the same 4plus-only re-derivation applies unchanged to **all three mixes**
  (all train 4plus only); only p33m67 × lr0.33 was re-evaled here. Open: still does
  not decontaminate against the trained 31.7% nemotron_cc TEXT family (Correction 2).

### Operational lessons (running the 4plus-only build + eval)

Concrete, reusable for future build/eval goals on this cluster:

1. **Large-model eval + preemptible = restart loop.** The 1e22 eval (9.7B,
   ~54k eval sequences) takes ~43 min at `--per-device-parallelism 2` on a v6e-4,
   but preemptible slices here are reclaimed about every ~15 min. `eval_lm` does
   **not** checkpoint, so each preemption restarts it from iteration 0 — it never
   finishes (caught it by the elapsed counter resetting 16:51→16:16 and iters going
   *backward* 2.67k→2.56k across a 46-min wall gap). Fix: run the long eval
   **non-preemptible** (finishes uninterrupted), or shrink it below one preemption
   window. Small-model evals (<15 min) finish fine on preemptible. Rule: a single
   eval whose wall-clock exceeds the preemption interval must be non-preemptible.
2. **Killing a tokenize job mid-finalize leaves a partial cache.** Stopping the
   sequential builder right after the cache *parts* were written but before
   `.stats.json` was consolidated left j065 as a parts-only dir (no stats). The
   builder's `cache_status` correctly classified it "partial" and rebuilt it on
   `--skip-filter`. Takeaway: a cache is only complete when `.stats.json` exists;
   never count a "records written" log line as done.
3. **Fan-out beats sequential for the multi-cutoff tokenize.** The single job
   tokenizes 9 caches sequentially at ~36–54 docs/s single-worker (~11 min each →
   ~100 min). Since the filter writes all 9 cutoffs' `docs/` up front, the remaining
   tokenizes are independent: kill the sequential job, relaunch one
   `--cutoffs <c> --skip-filter` job per cutoff → ~15 min wall. The `--skip-filter`
   path reads keep_ids from GCS and never re-derives or re-filters.
4. **Transient `jax.distributed` port-bind crash on TPU startup** (`Failed to add
   port to server '[::]:8482'`) — pure infra flake; a plain relaunch fixed it (2e20).
5. **Sanity asserts must be calibrated to the data, not to intuition.** Two
   self-checks in the build driver were too strict and failed on correct data:
   (a) `max 4plus Jaccard >= 0.99` — short fully-contained docs top out at 0.9882;
   (b) a union-keep bound that defaulted out-of-universe docs (union_max≥0.90) to
   Jaccard 0.0 instead of "dropped." Both were guard bugs, not data bugs (the
   fully-contained count, the 32,619 4plus-near-dup count, and the union j050/j075/
   j090 reproductions all passed). Lesson: gate on values you can anchor exactly
   (counts that match published numbers), and make loose-bound guards loose.
6. **In-region discipline held.** Every heavy read (721MB `all_ids`, 231
   verified_pairs, val_docs, caches) and every write stayed in us-east5; the only
   cross-region traffic was the few-KB eval_results.json pulled to skampere3 for
   tabulation. iris coordinators kept preemptible+small.

### Takeaways (consolidated — the whole decon thread)

1. **The contamination verdict is robust to how the val set is decontaminated.**
   Union(3,4plus,4plus_mind) and 4plus-only decon give the *same* picture: the
   anchor−clean gap inverts at 1e21→1e22, the within-decon slope flips, the valley
   jumps to the high-J end. "~37% of the 1e21→1e22 gain on the original math val is
   contamination credit" holds under both. The earlier worry (over-decon corrupts
   the signal) is empirically a ~0.01-nat level shift with identical shape.
2. **A val set is only "clean" relative to a specific training mix** — but here all
   three mixes share the single math source (`4plus`), so one 4plus-only
   re-derivation serves all of them; `3`/`4plus_mind` were never trained and should
   not have been in the decon source set.
3. **The measured cutoff scale (verify, punctuation-keeping) ≠ the banding scale
   (CleanText, punctuation/LaTeX-stripped)** — Correction 1. τ labels sit ~0.03–0.05
   below the dedup-scale similarity, so the "clean" sets mildly *under*-decontaminate
   (retain formatting-variant near-dups). This makes the contamination estimate a
   lower bound, not an over-statement. The right fix is to re-verify candidates with
   CleanText shingling so reported-J and τ live on the banding's scale.
4. **Nemotron-CC-Math is NOT a subset of Nemotron-CC** — Correction 2. Both are
   independent Common Crawl extractions (Justext-general vs Lynx+LLM-LaTeX) over
   overlapping snapshots. The 31.7% nemotron_cc TEXT in the mix is the largest
   *unscanned* trained source; expect *few* fuzzy hits (different-extractor strings)
   but real same-page semantic overlap MinHash can't see. This is the next real gap
   to close before any absolute "clean relative to the trained mix" claim — proofpile
   (0.165%) and starcoder (0.75%) are a distant second.
5. **Net for the project:** the decontaminated math val and the contamination
   quantification are sound and now corrected for the over-decon. Remaining work is
   coverage (text-family scan, other mixes/lrs) and calibration (CleanText-scale
   verify), not a redo.

## 2026-06-12 — Why does 1e22 loss only *grow* as you decontaminate? It's verbatim memorization

Question: on the decon sweep, lowering τ (dropping higher-Jaccard docs) *raises*
1e22's loss but *lowers* every smaller model's. Investigated at the document and
token level. **Answer: at 1e22 the high-Jaccard documents are the model's
*easiest* — it has memorized their training near-duplicates — so removing them
strips the easy wins and the average rises. At small scale those same docs are the
*hardest*, so removing them lowers the average. The flip is the memorization
phase transition, localized to 1e21→1e22.**

### Free decomposition — per-Jaccard-band loss vs scale (no new compute)

The nested cutoff losses unpack into per-band marginal loss (token-weighted:
`band[τ] = (tok_τ·loss_τ − tok_{τ-.05}·loss_{τ-.05}) / (tok_τ − tok_{τ-.05})`,
clean band = j050 loss). 4plus-only sweep, p33m67 lr0.33:

| band (max-J) | 3e18 | 1e21 | 1e22 |
|---|---:|---:|---:|
| clean <0.5 | 1.372 | 0.772 | **0.623** |
| [0.50,0.55) | 1.269 | 0.679 | 0.492 |
| [0.70,0.75) | 1.533 | 0.839 | 0.524 |
| [0.85,0.90) | **1.754** | 1.001 | **0.563** |

The [0.85,0.90) band is the **hardest** band at 3e18 (+0.38 over clean) and stays
above clean through 1e21, then at 1e22 drops **below** clean (0.563 < 0.623). Every
contaminated band sits below the clean band at 1e22. That inversion *is* why
decon raises 1e22 loss.

### Document-level experiment — curate by band, score per-token across the ladder

Built a controlled set (`curate_perplexity_gap_docs.py`, in-region): 20 val docs
(5 each in clean / ~0.60 / ~0.75 / ~0.88 max-4plus-J, length-windowed 1.5–7 kB) +
the **actual `4plus` train near-duplicate** of each high-J doc (best verified-pair
match, text pulled from the normalized corpus by a Zephyr scan). Scored every doc
per-token with 6 Delphi checkpoints (3e18…1e22) via levanter `score_main`
(`score_perplexity_gap_docs.py`), then plotted (`analyze_perplexity_gap.py`).

**Band-mean loss (curated docs), val:**

| band (max-J) | 3e18 | 2e19 | 9e19 | 3e20 | 1e21 | 1e22 | Δ(1e21→1e22) |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean (0) | 1.209 | 0.996 | 0.833 | 0.726 | 0.647 | 0.536 | −0.110 |
| j050 (.50–.55) | 1.548 | 1.290 | 1.150 | 1.016 | 0.902 | 0.591 | −0.310 |
| j060 (.58–.63) | 1.563 | 1.345 | 1.195 | 1.071 | 0.964 | 0.670 | −0.294 |
| j075 (.73–.78) | 1.676 | 1.443 | 1.251 | 1.147 | 1.003 | 0.735 | −0.268 |
| j088 (.86–.90) | 2.218 | 1.953 | 1.742 | 1.620 | 1.460 | 0.728 | **−0.732** |

The 1e21→1e22 drop grows monotonically with Jaccard — clean −0.11, then −0.31 /
−0.29 / −0.27 / **−0.73** — i.e. the high-J band improves **6.7× more** than clean in
that one step. That is not capability scaling (which is ~uniform across bands) but
memorization switching on. Even the j050 band (J≈0.5) shows a ~3× excess drop over
clean. Through 1e21 the bands scale in parallel (j088 stays ~2× clean); 1e22
collapses the contaminated bands toward clean. (j050 band added 2026-06-13;
5 docs in [0.50,0.55) — Paper Bridge Project, Pressure Questions, Ellipses/Kepler,
Quadratic Inequality, Number Theory divisibility. τ<0.5 stays unavailable: the
verify floor is 0.5, so "clean" is "no recorded near-dup ≥0.5".)

**⚠️ Curated-sample vs full-population — read the band plot correctly (2026-06-15
correction).** This per-band table/plot is a **5-doc-per-band sample** scored with
`score_main` (fresh whole-doc tokenization). It reliably shows the **gradient**
(high-J band Δ −0.73 vs clean −0.11 at 1e21→1e22) and the per-token memorization,
but it does **NOT** reproduce the absolute *crossover* (high-J band dipping below
clean at 1e22): here curated j088 1e22 = 0.728 stays *above* clean 0.536, because
only 2 of the 5 j088 docs are strongly memorized (1e22 losses 0.27/0.41/0.83/1.01/
1.12 — huge n=5 variance) and `score_main` levels differ from `eval_lm`. The
crossover is a **full-population** effect, established by the free eval band
decomposition (all docs, `eval_lm` on the cache; 2026-06-12): at 1e22 the
`[0.85,0.90)` band = **0.563 < clean 0.623**. That decomposition + the loss-vs-τ
sweep are the authoritative evidence for the inversion; the curated study's job is
the document/token-level memorization, not the band crossover. The plot title
originally overclaimed "crosses below clean" and was corrected to "memorization
gradient; crossover is a full-population effect not resolved by n=5".

**Flagship doc** — `000bf3fd…`, a 2nd-grade "Unit 14: Time is Money" coin-counting
lesson; its train twin `1c16a53e…` (J=0.869) is the *same page* differing only in
capitalization, an added author line, and heading levels:

| | 3e18 | 1e21 | 1e22 |
|---|---:|---:|---:|
| val doc (J=0.869) | 2.695 | 1.539 | **0.265** |
| its train twin | 2.651 | 1.459 | **0.191** |
| a clean doc | 1.206 | 0.656 | 0.558 |

The single **hardest** doc at 3e18 (2.695, ~2× the clean doc) becomes one of the
**easiest** at 1e22 (0.265, <½ the clean doc), and the model does nearly as well on
the val near-dup as on the exact train copy (0.265 vs 0.191) — it generalizes
memorization across the formatting differences.

**Per-token (the smoking gun):** on the flagship val doc, median token loss goes
**1.99 → 0.003** from 3e18 to 1e22; **78% of its tokens have loss <0.05 nats**
(>95% prob) at 1e22 vs 7% at 3e18. The clean doc improves far less — median 0.31 →
0.021, 31% → 57% near-zero. The near-dup is **recited**; the clean doc is
**predicted**.

### Answer to the question

"Loss only grows for 1e22 as you decontaminate" because decontamination removes
exactly the documents 1e22 has memorized. Memorization requires capacity it only
reaches at 1e22 (the bands scale in parallel through 1e21, then the contaminated
ones collapse). This is the same phase transition as the anchor−j050 inversion and
the valley-min jump (both at 1e21→1e22) — now shown at the document and token level
on the actual train/val near-duplicate pairs. Confirms the contamination is real
verbatim memorization, not a distributional artifact.

### Artifacts

- Plots: `plots/ppl_gap_per_band_vs_scale.png`, `ppl_gap_per_doc_vs_scale.png`,
  `ppl_gap_per_token_j088-000bf3fd.png` (+ GCS siblings + `ppl_gap_report_data.json`
  under `…/midtrain_dedup/perplexity_gap/`).
- Curated docs + per-token scores:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/perplexity_gap/`
  (`curated_docs.jsonl`, `curated_meta.json`, `scores/<run>/step-<N>/scored_documents.parquet`).
- Scripts (lint-clean): `curate_perplexity_gap_docs.py`, `score_perplexity_gap_docs.py`
  (levanter `score_main` per Delphi checkpoint), `analyze_perplexity_gap.py`.
- Op note: the per-token scorer reuses levanter `score_main` (the engine behind
  `experiments/.../perplexity_gap`), which writes per-byte loss + token byte spans
  per doc — readable for any per-token analysis. `gs://` HF checkpoints load fine;
  the wandb `config.yaml` artifact-log error is a benign background-tracker drop.

## 2026-06-15 — HANDOFF to the next agent

### Where this stands (TL;DR)

The nemotron math val contamination story is **answered and internally consistent**.
The 12,500-window math val carve-out contains fuzzy near-duplicates of training
docs; large models memorize them; this inflates apparent large-scale val gains and
bends the scaling-law fit optimistically. Quantified three independent ways that
all agree:

1. **Decon eval (union + 4plus-only):** anchor−j050 inverts +0.11→−0.05 at
   1e21→1e22; ~37% of the 1e21→1e22 "gain" on the original val is contamination
   credit. Robust to the decon-source correction (4plus-only ≈ union; 2026-06-12).
2. **Per-Jaccard-band loss vs scale:** high-J bands are the *hardest* at small scale
   and drop *below* clean at 1e22 (the inversion, decomposed; 2026-06-12 / 06-13).
3. **Document + per-token:** on a real train/val near-dup pair (J=0.869), the val
   doc's median token loss goes 1.99→0.003 from 3e18→1e22, 78% of tokens memorized;
   a clean doc only reaches 0.021/57%. Verbatim memorization, not an artifact.

This is **analysis on a branch, not a shipped change.** No training/data-pipeline
code was modified; the deliverables are decon val caches + eval/score artifacts +
logbook. Nothing is committed (user's standing instruction).

### Durable artifacts (GCS, us-east5) — reuse, don't recompute

- Decon val caches (llama3, validation-only), exact-verified:
  - union sweep: `…/tokenized/nemotron_math_val_decon/j0{50..90}/validation`
  - **4plus-only** (matches what p33m67/p50m50/p67m33 actually train):
    `…/tokenized/nemotron_math_val_decon_4plus/j0{50..90}/validation`
- Eval results (anchor + 9 decon tags per scale):
  `…/midtrain_dedup/decon_val_sets/evals_sweep9/<run>/…` (union),
  `…/decon_val_sets/evals_4plus/<run>/…` (4plus-only). W&B tags `decon_val_eval`,
  `decon_val_eval_4plus`.
- Verified near-dup pairs (val × source), 284×71 high-recall + exact verify:
  `…/midtrain_dedup/{3,4plus,4plus_mind}_284x71/verified_pairs/` — the source of
  every cutoff. `4plus_284x71` is the one that matters for these runs.
- Perplexity-gap study: `…/midtrain_dedup/perplexity_gap/` (`curated_docs.jsonl`,
  per-token `scores/<run>/step-<N>/scored_documents.parquet`, plots).
- Replay arrays: `…/midtrain_dedup/replay/` (`nemotron_math_all_ids.npy` 45M ids,
  `nemotron_math_doc_offsets.npy` doc-END offsets, `…_val_window_indices.npy`).
- MinHash (reusable, no re-sign): 284×71 for val + all 3 sources + the 10B `3`
  slice; 286×26 for `4plus` (45M) + the slice.

### Scripts (repo, branch `deconamint`, all uncommitted + lint-clean)

- `build_decon_val_4plus_only.py` (+ `tests/…`) — **the** 4plus-only cache builder
  (recompute max-J from 4plus pairs only; self-verifying). NOTE: there is a separate
  older `decon_val_fourplus_only.py` that only *quantifies* (no cache build) — use
  the `build_` one to (re)build caches.
- `eval_decon_val_sets.py` — eval driver; `--decon-cache-root` / `--wandb-tag`
  select union vs 4plus-only.
- `build_decon_val_sweep.py` — the 9-cutoff union sweep builder.
- `plot_decon_val_loss_vs_cutoff.py` — loss-vs-τ plot; `--eval-root` / `--slice`.
- `curate_perplexity_gap_docs.py` / `score_perplexity_gap_docs.py` /
  `analyze_perplexity_gap.py` — the document/per-token memorization pipeline.
- `crossdedup_*` — the clean untrained `3`-slice (8.0B tok) build + the
  `3`-vs-`4plus` overlap probe.

### Open threads (prioritized) — each reuses existing artifacts

⚠️ **Read the 2026-06-15 "CONCEPTUAL MISS" entry first** — `max_jaccard ≥ τ` means
"near-dup somewhere in 4plus," NOT "1e22 trained on it" (1e22 sampled only ~48% of
4plus docs). The document/per-token study needs to be **exposure-conditioned**; the
aggregate scaling result is fine (Codex's exposure replay already conditions on it).

1. **Scan val vs the 31.7% nemotron_cc TEXT family** (Correction 2) — the largest
   *trained* source the val set is NOT decontaminated against. Expect few fuzzy hits
   (different-extractor strings) but it's the honest gap before any absolute "clean"
   claim. Needs the text family normalized+minhashed in us-east5 (a real job; not
   yet available). proofpile_2 (0.165%) / starcoder (0.75%) a distant second.
2. **Calibrate the Jaccard scale** (Correction 1) — the cutoffs τ are on the
   *verify* (punctuation-keeping) Jaccard, ~0.03–0.05 below the *CleanText* scale the
   banding uses → mild under-decon. Fix = re-verify the EXISTING candidates with
   CleanText shingling (reuses candidates; re-verify only, no rescan).
3. **τ<0.5 band** — the verify floor was 0.5, so "clean" = "no near-dup ≥0.5"; a
   J=0.4 twin is invisible. Re-verify candidates at a lower `MIN_REPORT_JACCARD` to
   open this band (the 284×71 ~0.31 scan floor still caps recall below ~0.4).
4. **Extend to p50m50 / p67m33** — the 4plus-only re-derivation + perplexity-gap apply
   unchanged (all three mixes train 4plus only); only p33m67 × lr0.33 was done at the
   document level.
5. **LaTeX-dense high-J docs** — the curated j088 docs were prose-and-light-notation;
   re-curate restricted to `\frac`/`\begin{...}`-dense pages to test whether
   memorization is even stronger where the surface form is more distinctive.
6. **Canonicalize a decontaminated math val** for future midtrains (data-sections
   pinning, same byte-identical contract) once #1–#2 are closed.

### Gotchas the next agent must not re-trip

- A val set is "clean" only relative to a **specific training mix**; decon source set
  must = trained sources. All three current mixes train `4plus` ONLY (not `3` /
  `4plus_mind`); the union sets over-decontaminate by ~6% (harmless to the signal).
- Scan (CleanText: lowercase + strip punctuation/LaTeX) and verify
  (`_WS.sub(" ", text.lower())`: keeps punctuation) use **different** shingling —
  recall is fine, τ labels are mis-calibrated low (thread #2).
- Nemotron-CC-Math is **not** a subset of Nemotron-CC — independent CC extractions
  (Lynx+LLM-LaTeX vs Justext); same crawl, different strings.
- `nemotron_math_doc_offsets.npy` is a doc-**END** array (`len==num docs`), not
  boundaries: doc d spans `[ends[d-1], ends[d])`, start 0 for d==0. The original
  off-by-one cost a rebuild.
- Cluster ops: **default Zephyr coordinators to preemptible+small** (non-preemptible
  sit PENDING for many minutes — bit this session). A large-model eval whose
  wall-clock exceeds the ~15-min preemption window (e.g. 1e22 at pdp=2, ~43 min) must
  be **non-preemptible** or it restart-loops. A cache is complete only when
  `.stats.json` exists. Multi-cutoff tokenize: fan out one `--skip-filter` job per
  cutoff (the filter writes all cutoffs' docs up front).
- iris from this worktree: `export PATH=/lfs/skampere3/0/ahmedah/.pyenv/versions/3.12.0/bin:$PATH`;
  `.marin.yaml` symlink injects WANDB+HF tokens.

### Branch / commit state

`deconamint`, head `bea9f1784`. Everything this session is **uncommitted**: modified
`eval_decon_val_sets.py`, `plot_decon_val_loss_vs_cutoff.py`, the logbook; new
`build_decon_val_4plus_only.py` (+test), `curate_/score_/analyze_perplexity_gap*.py`,
`crossdedup_*`, `plot_lsh_scurve.py`. The user has not asked to commit — confirm
scope before committing (the `crossdedup_*` + `decon_val_fourplus_only.py` predate
this session and may want separate handling).

## 2026-06-15 — Is the decon a "drop-all" filter that nukes within-val twins too?

Question: the decon drops every val doc with max-J ≥ τ (it is a *filter*, not a
keep-one *dedup*), and the scan corpus (all of `4plus`) contains the val docs
themselves — so in principle two within-val near-dup copies would *both* be
dropped (no canonical kept). Measured how often that actually happens by splitting
each dropped val doc's near-dup into **trained** (other_id ∉ val set) vs
**within-val** (other_id ∈ val set). All 57,243 val ids from `val_docs`; max-J per
doc from `4plus_284x71/verified_pairs` (one-shot read, no rescan).

| τ | val docs dropped | dropped via a **trained** near-dup | dropped **only** via a val-twin |
|---|---:|---:|---:|
| 0.50 | 32,619 | 32,615 | 4 |
| 0.60 | 22,288 | 22,284 | 4 |
| 0.70 | 13,673 | 13,666 | 7 |
| 0.75 | 10,030 | 10,028 | 2 |
| 0.80 | 6,625 | 6,621 | 4 |
| 0.90 | 1,011 | 1,011 | 0 |

**Verdict: the within-val "drop both copies" case is negligible — ≤7 docs at any
cutoff (>99.97% of drops are due to a trained near-dup).** Reason: the val set is a
tiny random sample (57,243) of a 45.1M-doc corpus, so the trained side is ~788×
larger; a val doc's near-dup is almost always one of the 45M trained docs, virtually
never another val doc. Two near-dup copies *both* landing in the held-out val
windows is lottery-odds. So although the paranoid filter is formally a drop-all
(not keep-one) and *would* over-drop within-val twins, in practice it is
effectively "drop val docs that resemble training data," and a keep-one dedup would
rescue ~0 docs. (other_id ∉ val ⟹ in the train SPLIT: every non-val 4plus doc has
all its tokens in train windows. ⚠️ "train split" ≠ "actually sampled by a given
run" — see the 2026-06-15 exposure correction below. Applies the same to the
fully-contained paranoid subset.)

## 2026-06-15 — CONCEPTUAL MISS: "near-dup in 4plus" ≠ "1e22 trained on it"

A real gap, flagged by the user. Throughout this session the decon scan and the
document/per-token memorization study treated **"a val doc has a near-duplicate
somewhere in `4plus`"** as if it meant **"the model trained on that near-dup."**
It does not, for two compounding reasons:

1. **The scan corpus is *all* of `4plus` (45.1M docs)** — it finds a near-dup
   anywhere in the corpus, including in the val carve-out itself.
2. **A CPT run only *samples* a fraction of the train split.** 1e22 p33m67 trained
   **21.55B math tokens ≈ 21.70M unique `4plus` docs out of 45.10M (~48%)** (Codex
   exposure replay, 2026-06-04). So a flagged train-twin can sit in the **untrained
   ~52%** and the model never saw it. "In the train split" (all non-val 4plus docs)
   is *eligibility*, not *exposure*.

So `max_jaccard ≥ τ` is a **leaky proxy** for "memorization-contaminated for run
X": at the document level the full-corpus scan over-counts ~2× relative to
"actually sampled by 1e22."

### What this does and does NOT break

- **Does NOT break the headline scaling result.** The "~37% of the 1e21→1e22 gain
  is contamination" / exposure tables come from Codex's **per-scale exposure
  replay**, which already reconstructs each scale's sampled stream and restricts to
  *actually-trained* near-dups. At 1e22 that replay says **~84% of the J≥0.75
  near-dup val-token envelope was exposed** (7.99M of 9.53M tokens) — most high-J
  near-dups *were* trained, just not all. That analysis is exposure-aware and
  stands.
- **Does NOT break the decon caches.** They threshold on full-corpus max-J on
  purpose: the decon sets are **scale-independent** (one set, all scales), and
  different scales sampled different subsets, so a scale-specific exposure-
  conditioned cutoff can't serve all scales. Full-corpus max-J is the conservative
  scale-independent choice (over-decontaminates, never under).
- **DOES weaken the per-band / per-document memorization study (this session).** It
  curated bands by max-J alone and implicitly read "high-J band = trained near-dup."
  The high-J band is therefore **diluted with docs whose twin 1e22 never sampled**,
  which: (a) inflates the band-mean loss, (b) is a likely chunk of the within-band
  variance (the j088 docs that *collapse* — flagship 0.27 — had their twin trained;
  the ones that *don't* — 0.83–1.12 — plausibly had it in the untrained ~52%), and
  (c) is part of why the n=5 curated band plot doesn't reproduce the full-population
  crossover. The per-token memorization on the docs that DO collapse is still valid
  (you can't recite what you never saw) — but "contaminated band" should have meant
  "exposure-confirmed," and it didn't.
- **Loose wording to retire:** earlier entries (incl. the within-val analysis above)
  used "trained" to mean "in the train split." Read "trained" as "eligible (in the
  train split)" unless it says "sampled/exposed."

### The fix (open)

Exposure-condition the document study: for each curated doc, use the per-scale
exposure replay (Codex 2026-06-04 method — replay `train_lm` seed → mixture
shuffle → block rounding → sampled windows → doc indices) to verify its train twin
was actually in the **1e22** stream, and build the bands from
**exposure-confirmed** docs only. Then every doc in the high band genuinely *was*
trained, the band memorizes uniformly, and the per-band plot should match the full
sweep. Folds into the pending "contiguous bands + 50–100 docs/band" rebuild — the
correct version is **exposure-conditioned** bands, not just bigger ones. (The
replay machinery is not in this worktree; Codex ran it locally — needs to be
reconstructed or located.)

## 2026-06-15 — Deep-dive Q&A: what the Jaccard scan actually measures (insights)

A long clarifying session. Capturing the conceptual results and the plot/axis
distinctions so they aren't re-derived.

### The big reframe — high Jaccard conflates THREE things, only one is contamination

A high val↔4plus Jaccard does **not** uniformly mean "contamination." It is a flag
for "this val doc resembles recurring corpus content," which decomposes into:

| case | what it is | model behavior at 1e22 | is it contamination? |
|---|---|---|---|
| **(a) trained duplicate** | a near-identical copy (J≳0.8, same page) **was sampled** in training | **verbatim recall** — loss collapses to ~0.2 | **YES** — the real effect |
| **(b) trained template** | same *template*, different content (J≈0.5–0.7: "86 days" vs "56 days"), an instance *was* trained | **format generalization** — predicts structure well, not verbatim | **NO** — legitimate learning |
| **(c) untrained template/dup** | a near-copy exists in 4plus but 1e22 never sampled it (the untrained ~52%) | predicted via templates learned from *other* instances; no recall | **NO** — just says the doc is boilerplate-y |

Only (a) is memorization contamination. The decon scan (max-J over all of 4plus)
lumps all three together. This is why the curated j088 band is heterogeneous: the
docs that collapse (flagship 0.27) are case (a); the ones that don't (0.83–1.12)
are (b)/(c). "max-J band" is a proxy for *all three*, not for (a).

### Val-set quality is a separate axis from contamination

Because the math val was carved from a corpus that was only exact-deduped and is
**stuffed with templated/auto-generated pages** (unit converters, "prime factors of
N", lesson-plan templates, StackExchange-format Q&As), a large share of val docs are
intrinsically boilerplate. High within-corpus Jaccard is symptomatic of this. So:
**even a perfectly train-decontaminated math val would still contain lots of
"predict the web boilerplate" docs** that any capable model nails via format
learning. That is a val-set-quality problem, *orthogonal* to train contamination —
worth keeping distinct when interpreting absolute losses. (A doc-level fuzzy dedup
of the corpus *before* the val carve-out is the real fix; logbook follow-up.)

### Two plots, two axes — do not conflate (recurring confusion this session)

| plot | x-axis | each line | population | file |
|---|---|---|---|---|
| **decon sweep** | cutoff τ | a compute budget | **full** decon caches (thousands of docs/cutoff) | `decon_val_loss_vs_cutoff_p33m67_lr0.33{,_4plus}.png` |
| **curated study** | **compute** | a fixed Jaccard band | **30 hand-picked docs** (5/band) | `ppl_gap_per_{band,doc}_vs_scale.png` |

- The curated study is **one fixed 30-doc set**, NOT 50-per-cutoff. Bands are
  **narrow, non-contiguous windows** (clean=0; j050=[.50,.55); j060=[.58,.63);
  j075=[.73,.78); j088=[.86,.90)) anchored near round Jaccard values — gaps in
  between, no tiling.
- The curated per-band plot **does not reproduce the full-population crossover**
  (high-J below clean at 1e22) — n=5 + narrow windows + un-exposed dilution (case c).
  Its plot title originally overclaimed "crosses below clean"; corrected. The
  authoritative crossover/inversion lives in the **full sweep + the all-docs eval
  band decomposition** (`[0.85,0.90)`=0.563 < clean 0.623 at 1e22), not the n=5 plot.
- The curated plot looking "intuitive" (high-J always hardest, dedup always lowers
  loss) is a *symptom* of it being unrepresentative — it misses the very crossover
  that makes the full sweep's 1e22 inversion happen.

### Within-val twins: measured, negligible (recap)

The filter is drop-all (not keep-one), so two within-val near-dup copies would both
be dropped — but that affects **≤7 docs at any τ** (>99.97% of drops are vs a
non-val/train-split doc), because val is a 788×-smaller sample than the corpus.

### Artifacts added this stretch

- `scripts/analysis/plot_decon_sweep_union_vs_4plus.py` — overlays the 4plus-only
  (solid) and union (dotted) sweeps on shared axes; output
  `plots/decon_val_loss_vs_cutoff_p33m67_lr0.33_4plus_vs_union.png` (+ GCS). Shows
  the two decon families are ~indistinguishable (over-decon is ~6%).
- j050 band added to the curated study (`curate_perplexity_gap_docs.py`,
  `analyze_perplexity_gap.py`); 30-doc set re-scored across the ladder.

### Net open item (supersedes the plain rebuild)

Rebuild the document study as **exposure-conditioned, contiguous bands**: (1) wire
up the per-scale exposure replay to label each curated doc's twin as
sampled-by-1e22 or not; (2) keep only exposure-confirmed docs in the "contaminated"
bands; (3) contiguous tiling bands ([.5,.6),[.6,.7),… ) with 50–100 docs each.
Then the high band is uniformly case (a) and should reproduce the full-sweep
inversion. Optionally separate case (b) (trained template) to show generalization
vs memorization explicitly.
