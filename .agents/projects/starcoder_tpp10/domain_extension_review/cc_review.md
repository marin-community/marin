I have everything I need. Here is the review.

---

# Review: Wikipedia and FineMath-3+ repetition sweeps at TPP10

**Scope:** scientific setup and feasibility review of `.agents/projects/starcoder_tpp10/domain_extension_review/SPEC.md` (dated 11 Sep 2026) against the frozen StarCoder TPP10 implementation. Read-only; nothing was written, edited, or submitted.

---

## 1. Verdict

**Does it answer the stated narrow question?** Partly. It answers *"along the fixed path where focus fraction and focus epochs move together, which grid point minimizes the seven-component Uncheatable mean for each domain?"* It does **not** answer the question the motivation needs, because `p` controls mixture weight and epoch count in lockstep (`E_matched = 15.825·p`, SPEC:28). A difference in `argmin p` across domains is not attributable to repetition.

The repo already demonstrates this. In `pilot_metrics.csv`, the StarCoder **unmatched** curve (same weights, 18× fewer epochs) is monotone decreasing all the way to p=1 — 1.7807 → 1.0617 BPB, **no interior minimum**. The **matched** curve over the identical weights has an interior minimum at p=0.50. The interior minimum *is* the repetition effect, and it is only visible as a difference between arms. A matched-only stage measures the sum of two effects and reports the location of their crossing.

**Does it support the intended cap claim?** No — not even in the narrowed form at SPEC:11 ("distinct optima would show that a common preferred epoch count is inadequate"). Distinct `p*` across domains is consistent with domains having *identical* repetition tolerance and merely different marginal mixture value under a fixed heterogeneous objective. The SPEC's caution is correct as far as it goes but does not rescue the inference. The claim becomes supportable in a weak, proxy-scale, single-seed form once the repetition penalty Δ(p) is measured (§3.1). The SPEC's separation of *upper cap* from *shared preferred count* (SPEC:11) is right and should be preserved verbatim: Δ curves can refute "all buckets degrade alike at a given epoch level," but a sufficiently high cap remains harmless by construction, and refuting a cap requires the separately specified proposer/selection protocol.

**Ready for training submission?** No. Three scientific prespecification changes and eleven implementation gates are open. The builders and generalized launcher do not exist, and one prerequisite (re-scoring the completed pilot on the new primary metric) must produce a number before the budget is justified. Correctly, SPEC:85 does not claim submission-readiness.

**Numeric accounting: verified.** Every figure I could check is right — `15.825`, `15.82782369`, `0.87190083`, `18.15324`, `267.53×`, the 0.0504% allocator bound (`allocation_audit.json:4` = 5.037e-4), all eight cost-table rows, the "16 not 14" run count, the evidence.json extractions (FineMath 7.243→14.486 rise = 0.0036030; synth-math = 0.0013044; the 14.486 synth-math minimum is indeed OlmoBaseEval's, not Uncheatable's), and the pilot/proposed grid intersection (missing exactly 5% and 20%). Two accounting gaps in §5 below.

---

## 2. What is sound and should not be changed

- **Matched TPP and matched actual epochs (requirement 2): satisfied.** Max matched-vs-target epoch mismatch on the proposed grid is 5.037e-4 at p=5% (`allocation_audit.json:52`), and it transfers *exactly* to new domains rather than approximately — `sequence_allocation` (`starcoder_tpp10.py:339-351`) never touches tokens, and `MixtureDataset` indexes components by dict insertion order, not name (`mixture.py:89`, `:143-169`); `_compute_block_assignment` (`mixture.py:497-500`) folds only the fixed `DATA_SEED` key. Given identical weights, block size, and web-then-focus insertion order (`launch_starcoder_tpp10.py:110-111`), the allocation is bit-identical across focus-domain names.
- **Nonrepeating unmatched proxy (requirement 3): satisfied by construction.** Max focus draw is 81,024 sequences vs a 92,928-sequence parent (14.7% headroom) at p=1; enforced at `starcoder_tpp10.py:359-360`.
- **Frozen downsampling and target (requirement 5): satisfied at the mechanism level.** Pools are specified by exact output token quotas (`prepare_starcoder_tpp10.py:198-201`), not by converting legacy Llama3 counts — correct, since the 32k TinyLlama tokenizer yields ~1.15–1.3× more tokens per character for English. S_j nests in P_j via the pre-materialized parent permutation (`prepare_starcoder_tpp10.py:251-260`, `:263-287`). Target reuses P_j directly; web pools are already target-sized with 1.2× headroom (`starcoder_tpp10.py:149-151`) and shared across domains.
- **Shared p=0 control (requirement: fair reuse): correct and better than stated.** `MixtureDataset` drops zero-weight components entirely (`mixture.py:84-88`), so the p=0 token stream is independent of which focus cache is attached at weight 0. Displaying one shared control rather than replicates is right. It also helps more than the SPEC realizes: the two p=0 trainer seeds differ by 0.0283 BPB (`pilot_metrics.csv:3-4`), ~10× the noise elsewhere, but that noise is *perfectly common* to all three curves when the run is literally shared.
- **Range logic (SPEC:33):** correct. P_j ≥ D_proxy caps target repetition at 18.153; the chosen parent leaves margin at 15.828. Reaching ~30 epochs would force P_j = 100.4M < D_proxy and break the nonrepeating unmatched arm. Do not shrink the parent.
- **Objective discipline (SPEC:57, :61):** no reweighting, no post-hoc objective change, no fitted curve selecting the optimum, ties to smaller p, boundary minima marked, unfavorable curves preserved. Keep all of it.

---

## 3. Blocking scientific changes

### 3.1 Promote the unmatched arm from optional to required; make Δ(p) the primary cross-domain quantity

**Reason.** This is the only contrast in the package that varies repetition at fixed mixture weight. matched(p) and unmatched(p) share trainer seed, initialization, web stream, block allocation, interleaving, and total focus token count; they differ *only* in how many distinct focus sequences supply those tokens (5,120 vs 92,928). Δ(p) = BPB_matched(p) − BPB_unmatched(p) is therefore the repetition penalty at epoch level 15.825·p, and it differences out the mixture-weight effect that dominates the raw curves.

The existing pilot shows Δ is both large and monotone, while `argmin p` is fragile. From `pilot_metrics.csv` (Paloma-code metric, 6 matched replicates and 2 unmatched replicates per point):

| p | actual matched epochs | matched mean | unmatched mean | Δ(p) |
|---:|---:|---:|---:|---:|
| 0.10 | 1.5764 | 1.29996 | 1.30385 | −0.0039 |
| 0.30 | 4.7684 | 1.19902 | 1.19102 | +0.0080 |
| 0.50 | 7.9379 | 1.15874 | 1.12636 | +0.0324 |
| 0.70 | 11.0980 | 1.16711 | 1.09537 | +0.0717 |
| 0.90 | 14.2650 | 1.21640 | 1.07258 | +0.1438 |
| 1.00 | 15.8250 | 1.29876 | 1.06173 | +0.2370 |

Δ at p=0.50 is ~9× the single-run SD; at p=1.00 it is ~13×. By contrast the matched curve's minimum-vs-neighbour gap (p=0.50 vs p=0.70) is 0.0084 — see §3.2.

**Exact fix.** Add 16 unmatched proxy runs to the first stage (2 new domains × 7 nonzero p, plus StarCoder unmatched at 5% and 20%, which the pilot does not cover). Cost 3.98542e17 FLOPs — **0.83% of the Figure-5 pilot**. Total first stage becomes 32 runs / 7.9708e17 FLOPs (1.66% of Figure-5). Prespecify Δ(p) as the primary cross-domain quantity, with the matched-curve minimum reported as a secondary, explicitly non-causal descriptive statistic.

Note this deliberately contradicts requirement 3's "optional." The matched-only stage remains valid for the narrow mixture-response question; it is the repetition-preference and cap framing that fails without the unmatched arm. Scaling this down is the call of whoever owns the budget, but the claim must then be narrowed to match.

### 3.2 Derive the replicate SD from existing runs and prespecify a resolution rule before submitting

**Reason.** One trainer seed and one subset seed per point (SPEC:55) is not supported by the measured noise. Pooled within-cell SD across the six 6-replicate matched cells in `pilot_metrics.csv` is **0.00954 BPB** (30 df); excluding the p=1 cell it is **0.00640**. The matched curve's minimum-to-adjacent gap at p=0.50→0.70 is **0.0084**, i.e. 1.3 pooled SD. All six replicates happen to select p=0.50 on the coarse 20-point-spaced pilot grid, so the coarse selection is robust — but the refinement rule at SPEC:61 ("add the adjacent unmeasured 5% grid points") moves to p=0.45/0.55, where the true curve separation is far below 0.0064 and selection becomes noise. The refinement as written is guaranteed to produce an unreproducible minimum.

**Exact fix.** (a) Before submitting new training, compute the pooled within-cell SD **on the actual primary metric** from the existing 36 matched pilot checkpoints (§4, gate G0). (b) Prespecify: two grid minima whose BPB difference is below 1 σ̂ are reported as *unresolved*, not ranked; curves whose full range is below 2 σ̂ are reported as *flat*; the refinement stage is only triggered when the observed minimum is separated from both neighbours by ≥ 2 σ̂. (c) Prespecify a boundary rule: if a domain's minimum falls at p=0 or p=0.05, the cross-domain optimum comparison is not reported and the analysis falls back entirely to Δ(p) at common epoch levels.

### 3.3 Prespecify the in-domain-weight robustness analysis for the common objective

**Reason.** The fixed seven-component objective is 2/7 in-domain for StarCoder (`github_python`, `github_cpp`), 1/7 for Wikipedia (`wikipedia_english`), and ~0/7 for FineMath (`arxiv_physics`/`arxiv_computer_science` are only loosely related). Under a common objective, a domain with more in-domain evaluation weight will prefer a higher focus fraction for reasons that have nothing to do with repetition tolerance. This asymmetry alone can generate "distinct optima" and is a direct confound for the headline claim. Keeping the objective fixed (SPEC:57) is right; what is missing is a prespecified way to bound the confound.

**Exact fix.** Prespecify — before any results are viewed — a secondary analysis that recomputes each domain's curve on the leave-in-domain-components-out mean (StarCoder: drop the two GitHub components; Wikipedia: drop `wikipedia_english`; FineMath: drop nothing, or symmetrically drop both arXiv components). This costs **zero** additional compute: `evaluate_starcoder_tpp10_uncheatable.py:284-293` already persists every component BPB and `:327-338` already writes them to `measurements.csv`. State in advance that a cross-domain optimum ordering that reverses between the primary and leave-out aggregations is reported as confounded, not as a repetition result.

---

## 4. Implementation gates (must close before submission; none of these are scientific flaws)

**G0 — Prerequisite, not a gate on training but on the budget decision.** Run the already-implemented `evaluate_starcoder_tpp10_uncheatable.py` over the completed 57-run pilot. This is evaluation-only, costs no training, and is the single highest-value action in the package: it supplies σ̂ on the real primary metric (§3.2), the StarCoder Δ(p) curve on the real primary metric (§3.1), and — decisively — whether the proposed grid brackets an interior minimum under the new objective at all. SPEC:59 says the review should not wait for this work; that is fine for *reviewing the design*, but the first-stage budget cannot be justified without it. The concern is concrete: the frozen design's primary metric is `eval/paloma/dolma_100_programing_languages-tpp10/bpb` (`starcoder_tpp10.py:50`, `design.json:4`, `design.md:38`), i.e. fully in-domain for StarCoder. Re-weighting to 2/7 in-domain plus 5/7 text components moves the balance point sharply left, and the archived 60M evidence already shows FineMath *worse than p=0* at focal weight 0.0779 (`evidence.json:104-107`). The most likely outcome is a left-boundary minimum for all three domains, in which case the grid must be re-sited before spending anything.

**G1 — Six files and one directory are byte-frozen.** `starcoder_tpp10.py`, `prepare_starcoder_tpp10.py`, `launch_starcoder_tpp10.py`, `launch_starcoder_epoch_matching.py`, `starcoder_epoch_matching.py`, `uv.lock`, and **every file in `starcoder_tpp10_assets/`**. Any byte change to the first breaks `load_design` (`starcoder_tpp10.py:246-256`) because `build_design` hashes its own source (`:230`). Changing any of them breaks `verified_training` (`launch_starcoder_tpp10.py:85-86`) via `code_pins()` (`:44-68`), which enumerates all six plus `sorted(experiment.ASSETS.iterdir())` at `:67` — so **adding a `sources.json` for the new domains to that assets directory would invalidate the frozen StarCoder recipe identities**. Changing `prepare_starcoder_tpp10.py` or `starcoder_tpp10.py` additionally breaks `validate_spec` (`evaluate_starcoder_tpp10_uncheatable.py:142-143`) and forks the Uncheatable eval cache path, which is keyed on `file_sha256(preparation.__file__)` (`:195-198`). New module, new assets directory, same tokenizer path string.

**G2 — Parquet reader must not touch `uv.lock`.** `pyarrow` 23.0.1 is already locked (`uv.lock:7344`), as is `polars` (`:6954`). Build the FineMath reader on those only. It must mirror `remote_documents` (`prepare_starcoder_tpp10.py:100-119`): pin `generation`, verify `size_bytes` and `crc32c`, fail rather than fall back to another generation/partition/region, bound bytes read, produce deterministic row order, and enforce an exact token quota through `token_records` (`:122-137`).

**G3 — Do not submit the two missing StarCoder points through the new launcher, and do not run `--stage dense`.** `select_runs(design, "dense")` returns all 183 runs; `main()` then runs every pending one (`launch_starcoder_tpp10.py:316-318`) — that is 21 target runs and ≈1.44e20 FLOPs, roughly 3× the Figure-5 pilot (`design.md:34`). The 5%/20% matched recipes already have frozen identities in `design.json:134-141` and `:377-384`. Produce them with a filtered-submission wrapper over the frozen dense plan and assert the resulting `ArtifactStep` fingerprints equal those in `dense_plan.json`. Also note: `design.json` defines six matched replicates at each of 5% and 20%; running one each leaves those cells asymmetric with the rest of the StarCoder curve.

**G4 — Allocator parity, with exact expected values.** The new launcher's allocator must reproduce `allocation_audit.json` exactly at the shared coordinates. Proxy focus-sequence counts at p = {0, 5, 10, 20, 30, 50, 70, 100} must be **{0, 4034, 8071, 16189, 24414, 40642, 56822, 81024}** and target counts **{0, 73254, 146520, 293743, 443123, 737587, 1031308, 1470848}**. Any deviation means the port changed weights, `BLOCK_SIZE`, `DATA_SEED`, or component insertion order. This is a stronger check than "recheck the bound" and it is free.

**G5 — Extend the zero-weight/renaming audit to the new names.** `zero_weight_stream_audit.json` only exercises the name `"starcoder"`. Shuffle keys are name-derived (`launch_starcoder_tpp10.py:156-162`, applied at `datasets.py:1255-1261`), so re-run the audit with focus names `wikipedia` and `finemath` and assert the six web component stream hashes are unchanged at p ∈ {0, 0.05, 1} and that the block assignment is unchanged. Expected result: web streams identical, focus stream different (unavoidable and correct).

**G6 — No p=0 run in the new plan.** Reuse `tpp10_unmatched_p000_s20260910` for all three curves; justified by `mixture.py:84-88`. Assert the new plan contains zero p=0 runs.

**G7 — Evaluation-population parity receipt.** The Uncheatable upstream is an unpinned GitHub repo re-fetched on demand (`experiments/datasets/uncheatable.py:6-8`); the only protection is the frozen blob generation/size/crc32c triple (`evaluate_starcoder_tpp10_uncheatable.py:107-113`, enforced fail-closed at `prepare_starcoder_tpp10.py:107-108`). Freeze and record the seven generations and assert they equal those in the Figure-5 spec, and assert the resolved cache paths are byte-identical across all arms. Keep the Paloma control attached to training (`launch_starcoder_tpp10.py:135`) so the 5e-5 checkpoint-restoration check still applies (`evaluate...:178`, `:281-283`).

**G8 — New analysis module required.** `analyze_starcoder_tpp10.analyze` rejects any grid other than `PILOT_GRID`/`DENSE_GRID` (`analyze_starcoder_tpp10.py:46-50`) and requires all three arms present; the proposed 8-point grid is neither.

**G9 — Wikipedia compressed-byte budget.** `remote_documents` caps compressed bytes at `max(8 MiB, 4 × quota)` (`prepare_starcoder_tpp10.py:110`). Dolma v1.7 wiki has exactly two shards (`lib/marin/src/marin/datakit/download/dolma.py:30`), so the quota is 95,158,272 tokens per shard and the cap is 363 MiB. Run a bounded regional probe first and record observed compressed-bytes-per-token; the failure mode is a hard raise mid-preparation ("Compressed source-read budget exhausted before its token quota", `:96`).

**G10 — FineMath inventory freeze.** Confirm `gs://marin-us-central1/raw/finemath-7090a5/` contains the `finemath-3plus` partition at HF revision `8f233cf` (`experiments/datasets/midtraining.py:33-50`), that no `finemath-4plus` or FineWeb-Edu objects enter the inventory, and that the text column name at that revision is `text`. Record generations, sizes, and crc32c for all 16 selected objects.

**G11 — Legacy-cache exclusion, asserted not assumed.** `tokenized/finemath_3_plus-a26b0f/` (`midtraining.py:47`) and `tokenized/dolma/wiki-212315` (`experiments/datasets/dolma.py:58`, carrying "never recompute these caches" at `:42`) are Llama3/128k-vocab. Guards exist (`CacheMetadata` comparison at `prepare_starcoder_tpp10.py:140-144`; vocab check at `starcoder_tpp10.py:144`), but add an explicit assertion that no new cache path resolves to a legacy pin.

---

## 5. Corrections to the spec text and accounting

1. **Report actual, not nominal, epochs.** SPEC:53's epoch table is nominal. The allocator floors `p·2048` and assigns the block remainder to the largest component (`mixture.py:143-150`), which is `medium` for p ≤ 0.25 and the focus domain above it — so the sign of the deviation flips at p ≈ 0.255:

   | p | nominal | actual | deviation |
   |---:|---:|---:|---:|
   | 0.05 | 0.79125 | 0.787891 | −0.42% |
   | 0.10 | 1.5825 | 1.576367 | −0.39% |
   | 0.20 | 3.165 | 3.161914 | −0.10% |
   | 0.30 | 4.7475 | 4.768359 | **+0.44%** |
   | 0.50 | 7.9125 | 7.937891 | +0.32% |
   | 0.70 | 11.0775 | 11.098047 | +0.19% |
   | 1.00 | 15.825 | 15.825 | 0.00% |

   Max nominal-vs-actual deviation (0.44%) is ~9× the matched-vs-target mismatch (0.05%). Both are far below grid spacing, so identifiability is unaffected — but since the deviation is identical across domains, actual epochs should simply be reported.

2. **Cost table, three-domain unmatched row.** "Optional unmatched Wikipedia/FineMath curves … 14 … 3.48724e17" is correct as labelled, but a *three-domain* common-grid unmatched package also needs StarCoder unmatched at 5% and 20% (the pilot covers 0/10/30/50/70/90/100): 16 runs, **3.98542e17**.

3. **Missing line items.** The table omits (a) data preparation for two parents plus two subsets, (b) Uncheatable evaluation of the new checkpoints, and (c) the G0 re-scoring of the 57 existing checkpoints, which is a hard dependency of the primary figure even though SPEC:59 scopes it out.

4. **Provenance mismatch in the motivating evidence.** `evidence.json` rows are `dolma3_wikipedia` and `dolma3_finemath_3plus` — Dolma 3 — while the proposal builds from Dolma **v1.7** `wiki-0000/0001` and raw HF FineMath. Same domain labels, different pools. SPEC:67 already says the archived optima are not transportable; add that the corpora versions also differ.

5. **The archived optima are within their own curves' jitter.** FineMath's archived Uncheatable sequence is non-monotone (1.09419 → 1.09397 → 1.09404 → 1.09421 → 1.09360) with ~2–4e-4 BPB of bounce, and the claimed FineMath-vs-Wikipedia optimum separation rests on gaps of ~5–6e-4. Those data do not establish distinct optima even in their original setting — which is a reason to require §3.2's resolution rule, not a reason to abandon the domains.

6. **Epoch/weight coupling makes the archived optima unreachable at their original weights.** Wikipedia's archived optimum (3.621 epochs) occurred at focal weight 0.0021; here 3.621 epochs requires p ≈ 0.229 — a 110× higher mixture weight. FineMath's 7.243 epochs occurred at weight 0.0389; here it requires p ≈ 0.458, 12× higher. SPEC:53's claim that the grid "brackets the historical hints" is true in epochs and false in mixture weight.

7. Trivial: SPEC:55 reads "except 5% and20%".

---

## 6. Optional improvements, ranked by value per unit cost

| # | Improvement | Cost | Why it is worth it |
|---|---|---|---|
| **O1** | **Add a third, deeper nested focus support** (e.g. 1,280 sequences = 2.62M tokens, nested inside S_j) at p ∈ {0.05, 0.10, 0.20, 0.30}, giving epochs {3.2, 6.3, 12.6, 19.0} at *deployable* mixture weights | 12 runs, **2.99e17** FLOPs (0.62% of Figure-5) | This is the design that actually measures epoch preference. The current package can only reach high epochs at absurd weights (11.1 epochs requires 70% of pretraining be one domain); a third support size decouples the two axes and makes the cap question directly answerable at realistic weights. Highest scientific value in the list. |
| **O2** | **Intra-pool redundancy diagnostic per parent**: exact-duplicate rate over the 92,928 packed sequences plus a document-level near-duplicate rate | CPU-only, minutes | The three pools are built very differently — StarCoder from all 49 shards (~0.09% prefix each), Wikipedia from **only 2** shards (~4–5% prefix each), FineMath from 16 parquet objects. A pool with more internal redundancy behaves as if already repeated. Without this, "Wikipedia tolerates less repetition" is confounded with "our Wikipedia pool is more internally redundant." |
| **O3** | Report σ̂ error bands (from the 6-replicate cells) on all three curves, and show the 6-replicate StarCoder mean alongside the single-seed curve | Zero | SPEC:55 discards a √6 ≈ 2.4× precision advantage to equalize noise across domains. Equal *displayed* noise is achievable by showing the band instead of throwing away the replicates. |
| **O4** | Report the pilot's measured proxy→target bias as a stated prior | Zero | On the code metric the matched proxy minimum is p=0.50 (7.94 epochs) while the target minimum is p=0.70 (11.10 epochs) — one full grid step, target regret 0.0099 (`pilot_metrics.csv`). Proxy optima appear biased toward *fewer* epochs, which matters directly: a cap inferred from first-stage proxy curves would be set too low. |
| **O5** | Add one extra trainer seed only at the two or three coordinates adjacent to each domain's observed minimum | ~6–9 runs, ≤2.2e17 | Targeted variance reduction exactly where the selection decision is made, instead of uniformly. Decide the count from the G0-measured σ̂. |
| **O6** | Drop p=0.90 from consideration and consider replacing one high-p point with a second low-p point once G0 reveals where the minimum sits | Neutral | Six of seven nonzero grid points sit at ≥10% focus weight; if G0 shows a left-boundary minimum, the grid is spending 85% of its budget past the answer. |

---

## 7. What evidence would be needed for stronger claims

- **"Different buckets prefer different repetition levels."** Requires Δ_j(E) curves for all three domains that separate by more than the resolution threshold at a common epoch level E, plus §3.3's leave-in-domain-out robustness agreeing in sign, plus O2's redundancy diagnostic showing the pools are not differentially redundant. Achievable inside the first stage as amended (§3.1) — one seed, proxy scale, descriptive only.
- **"A fixed epoch cap is limiting."** Requires (a) at least one domain whose *target-scale* optimum exceeds the candidate cap, and (b) the cap-constrained proposer and selection protocol that SPEC:11 correctly defers. Proxy curves alone cannot supply this, and O4's measured proxy→target bias means proxy optima are a *lower* bound on target optima — the wrong direction for a cap argument.
- **"These proxy optima transfer to the target."** Requires target runs at the proxy-selected p per domain plus a target grid for regret (9.33e19 FLOPs for the full 14-point version). **Not required to approve the exploratory proxy stage**, and I am not asking for it. A cheaper interim: the existing StarCoder pilot already provides one measured proxy→target transfer (regret 0.0099 on the code metric), and G0 will extend that to the Uncheatable objective for free.
- **Any significance or domain-universal claim.** Requires independent parent-corpus replicates, not subset replicates within one finite parent — explicitly out of scope here, and SPEC:61 and `design.md:47` already say so. Keep that language.

---

## 8. Recommended path

1. Run G0 (evaluation-only re-scoring of the 57 completed pilot checkpoints). Publish σ̂ and the StarCoder Uncheatable Δ(p) curve.
2. If the StarCoder Uncheatable minimum is interior and σ̂ permits resolution, re-site the grid if needed, prespecify §3.2 and §3.3, and amend the first stage to 32 runs (matched + unmatched, 7.97e17 FLOPs) per §3.1. Strongly consider O1 (+2.99e17).
3. Close G1–G11, with G1, G3, and G4 as the highest-blast-radius items.
4. Then seek release approval. The target expansion stays deferred; nothing in the amended first stage changes P_j, S_j, the target geometry, or pool membership, so requirement 5 remains satisfied.

Total blocking training added: **3.99e17 FLOPs, 0.83% of the Figure-5 pilot.** The rest of the blocking work is prespecification and evaluation of checkpoints that already exist.