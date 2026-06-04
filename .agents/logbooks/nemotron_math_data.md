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
