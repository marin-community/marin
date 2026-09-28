I read the design cold first, then checked it against the prior report and the historical launcher. Reading order and scope were as instructed: no edits, no commands, no delegation.

## Verdict on cold recoverability

**The scientific claim and the treatment are recoverable; the decision rule and several design-determining quantities are not.**

A cold reader can recover: the three arms, that StarCoder support is the treatment (`design.md:23`), the grid, the seed/subset crossing, and the estimand (sign of matched-minus-unmatched target regret). Every number I could check reproduces exactly — I rebuilt both parameter counts from the stated geometry (Qwen3, tied embeddings, MHA at head_dim 128 → 2 and 8 heads, Q/K norms counted) and got 16,587,008 and 301,241,344 to the digit; TPP, epochs, FLOPs (6ND plus the 12·L·d·T·D attention term), and the 57/183 run counts all reproduce. That is unusually solid and I want it on the record before the criticisms.

What a cold reader **cannot** recover: `p` is never defined (first used at `design.md:31`); "regret" is never defined; the promotion gate "whether the response is informative" (`design.md:39`) has no criterion; the prior experiment is characterized in one sentence with no link; and the one internal link is broken.

---

# Blockers

### B1. The only internal link is dead, and it holds the operative data spec

`design.md:7`: "[Research notes](research.md) identify the pinned tokenizer, regional raw sources, and reusable cache APIs."

`.agents/projects/starcoder_tpp10/research.md` **does not exist**. The directory contains only `design.md`, `source_inventory.json`, `tokenizer/`, and review logs. The only `research.md` in the tree is `starcoder_epoch_matching/research.md`, which is the *previous* project and does not describe this tokenizer or these sources.

**Fix:** either add the file, or inline the three pins (tokenizer revision hash, the `source_inventory.json` digest — `9e3b663d…` is already recorded — and the cache API entry points).

### B2. p=0 aliasing is asserted, and zero-weight handling has already broken a shuffle key in this codebase

`design.md:31`: "At p=0, each matched draw aliases the corresponding unmatched run."

This is load-bearing: it produces the 57 and 183 run counts and the FLOP totals. But `RESULTS.md:48` records that in the prior run, "The original p=1 run used a different StarCoder parent because **removing zero-weight web components changed its shuffle key**" — requiring a corrected re-run. Zero-weight component removal demonstrably perturbs data order here. Meanwhile the launcher threads `simulated_epoch_subset_seed` into the training config unconditionally (`launch_starcoder_wsd_80_20_surface.py:263,352`), so three subset seeds plausibly yield three distinct config fingerprints at p=0 even when training is identical.

Two consequences, both bad: the counts are wrong by 6 proxy runs per stage if aliasing fails, and — more seriously — if dropping the zero-weight StarCoder component shifts the web shuffle key, **p=0 is not on the same data trajectory as p>0**. p=0 is a candidate minimum for every curve and the shared anchor for the excess-loss transform.

**Fix:** demote the aliasing to a prespecified offline check (inspect zero-weight handling in the allocator and the config fingerprint). State the run counts for both outcomes (57/183 if aliased, 63/189 if not).

### B3. The one soft promotion gate is undefined, which is where outcome-based selection can leak in

`design.md:39`: "The pilot checks … and whether the response is informative."

Every other gate on that line is objective. This one is not, and it sits directly upstream of the dense-expansion release. The doc is otherwise disciplined about not selecting on outcome (`design.md:29,39`), so this is a gap, not an intent problem — but as written, "informative" can be read after seeing the pilot regret difference.

**Fix:** give it a numeric, outcome-neutral form before any pilot run, e.g.: *the target BPB range across the pilot grid must exceed k× the proxy trainer-seed standard deviation measured at the same coordinates.* Add explicitly that a boundary-located target minimum, or a flat response, is a **reportable result and not grounds for re-specifying the setting** — consistent with `design.md:39`'s existing "Flat or unfavorable curves remain reportable."

### B4. Batch is inherited unchanged while the model shrinks 12.7×, leaving the proxy 633 optimizer steps

`design.md:29`: "batch 128, 1% warmup and cosine decay during the final 20% of steps."

At 128×2048 = 262,144 tokens/step: proxy = 165,937,152/262,144 = **633 steps**; target = **11,491 steps**. (This matches the historical batch — the prior report's step 1059 for proxies and 28259 for the target reproduce exactly at the same 262,144.) Tokens-per-step per parameter is 0.0158 for the new proxy versus 0.00125 for the prior 210M proxy — **12.6× larger**, on a model that is already short at 633 steps with a 6-step warmup.

This attacks the experiment's own premise. If batch is supercritical at 16.6M parameters, a large share of each batch is wasted, the *effective* TPP is below the nominal 10 that the title claims to control, and the mixture response flattens toward seed noise — which is precisely the failure mode B3's gate is supposed to catch.

**Fix:** allow batch to differ *between* scales (`design.md:29` already only fixes the recipe "within each scale"), prespecify the proxy batch by a mixture-agnostic rule, and add a pilot check comparing proxy loss at two batch sizes at fixed token budget, decided on p=0 or a fixed non-selected coordinate.

### B5. The stated tokenizer rationale is only half-achieved, and the resulting nonembedding-TPP gap is 1.76×

`design.md:7`: "the existing Llama3 caches would make embeddings dominate the smallest model."

The 32k vocab takes the proxy's embedding share from ~80% (128,256×256 of 41.2M) to **49.4%** (8,192,000 of 16,587,008). Embeddings still are half the proxy. Consequently nonembedding TPP is **19.8 for the proxy versus 11.2 for the target** — a 1.76× mismatch under the alternative normalization.

`design.md:43` discloses this qualitatively ("Matching total TPP does not match nonembedding TPP") but gives no numbers, so a reader cannot judge the central design choice.

**Fix:** add a "Nonembedding parameters / nonembedding TPP" row to the table at `design.md:13`. The title's scoping is already honest; the numbers just need to be visible.

### B6. The design never states the mechanism by which matched TPP addresses the prior failure

`design.md:3-4`: "The earlier fixed-model experiment had total TPP 1.32 versus 35.27 and produced worse selection from the matched proxy… This experiment changes the scale control."

This is the redesign's entire scientific content and it is left implicit. Working it out from the prior report: in the old setting, matching *epochs* left **unique code tokens per parameter** at 10.486M/210.05M = 0.050 for the matched proxy versus 279.97M/210.05M = 1.333 for the target — a **26.7× mismatch**, which is a sufficient explanation for the matched proxy's 1.43 BPB blowup at p=1 (`RESULTS.md:34`) against the target's 0.032. Under the new design that quantity is 10.486/16.587 = **0.632** versus 190.317/301.241 = **0.632** — matched to 0.05%, because matching total TPP *and* epochs forces it.

Without this, a cold reader cannot distinguish this proposal from a rerun of a failed experiment.

**Fix:** one sentence in Design plus a "unique StarCoder tokens per parameter" table row. Note also that the same quantity is *not* matched under nonembedding normalization (1.249 vs 0.709), and that this residual runs in the direction of the proxy suffering *less* repetition damage than the target — the opposite of the prior failure.

### B7. "Replicates the small subset" is inaccurate and blocks cross-experiment attribution

`design.md:4`: "This experiment changes the scale control and replicates the small subset."

Relative to `starcoder_epoch_matching`, this changes at least six things: scale control (TPP 1.32/35.27 → 10/10); architecture (one fixed 210.05M model for all arms, `RESULTS.md:40` → 16.6M/301M); tokenizer (Llama3 → TinyLlama 32k); StarCoder parent (279.97M → 190.3M, new sources); epochs (26.5 → 15.8); and **subset construction — "the matched subset is its exact prefix" (`RESULTS.md:42`) → "uniform samples without replacement of 5,120 packed sequences" (`design.md:25`)**.

That last change matters more than the doc acknowledges. A 10.486M-token *prefix* of a shard-concatenated 279.97M parent is compositionally narrow — plausibly a couple of shards, i.e. a few languages or repo clusters. That is a live alternative explanation for the prior adverse result that `RESULTS.md:42` does not consider (it lists only "shorter horizon, tiny fixed subset and their interaction"). Switching to random sequence sampling is a genuine improvement, but it means a different outcome this time **cannot** be attributed to the TPP change.

**Fix:** replace the sentence with the explicit change list and state that no cross-experiment attribution is claimed. The phrase is also ambiguous on its face — "replicates the small subset" reads either as "reproduces the prior condition" or "adds replication over subsets"; only `design.md:25` disambiguates.

### B8. The unmatched arm's code composition varies with p; the matched arm's does not

`design.md:27` audits *wrapping* ("no web component wraps and … the unmatched proxy never wraps StarCoder"), but never specifies the **order** in which the unmatched arm consumes the parent.

The matched arm sees the identical 10.5M-token subset at every p, differing only in repetition. The unmatched arm consumes 0.436 epochs of the parent at p=0.5 and 0.872 at p=1. If the parent is laid out shard-by-shard — and `design.md:46` confirms the construction is "random shard selection followed by within-shard prefixes" — then the unmatched curve's *shape across p* is partly composition drift, not mixture response. That curve is one of the two terms in the headline comparison.

**Fix:** one sentence stating that the parent is consumed under a seeded global permutation over its 92,928 sequences (`design.md:50`'s "source-index shuffles" suggests this already holds), and add per-shard composition of the unmatched arm's realized draw at p=0.1 and p=1 to the allocator audit.

---

# Optional improvements

1. **Define `p` and `regret` on first use** (`design.md:31`, `design.md:35`), state that the proxy selects on the *same* Paloma programming-languages BPB used to score the target, and give a tie-break rule for the grid argmin (e.g. smallest p). Selection rules with undefined ties are a real degree of freedom on a 21-point grid.

2. **Justify the 190.3M-token parent.** `source_inventory.json` shows 49 shards at ~4.2 GB gzipped each, so 190.3M tokens is ~0.4% of a single shard's prefix taken 49 times — emphatically a choice, not an availability limit. That choice sets the entire repetition regime: parent size *P* gives target epochs 3.012e9/*P*, and 190.3M sits just 15% above the 165.9M floor imposed by "unmatched must not wrap," which maximizes the treatment contrast. That is a defensible choice for a first test; it should be stated as such, with the result scoped to a heavily-repeated target.

3. **Add a deliberately mis-matched proxy arm** (e.g. 4× or ¼× the target's epochs, via a differently sized subset). Without it, "epoch matching" and "reduced unique support" are the same manipulation and the causal claim in the title is not identified. Cost is one more proxy arm ≈ 120 runs × 2.491e16 ≈ **3e18 FLOPs, ~2% of the dense budget**. This is the cheapest large gain available in the design.

4. **The three paired differences are not three replicates.** `design.md:35` reports "all three paired regret differences and their mean," but all three share a *single* unmatched selection and a *single* target curve, so the comparison has effectively one degree of freedom in each of those terms. Report them individually and drop the mean, or label it as a descriptive average with no replication interpretation. Separately, report the **six** per-run argmins (3 subsets × 2 seeds), not just the three subset-averaged ones — that is the honest picture of subset-vs-seed sensitivity and costs nothing.

5. **Add a second held-out evaluation domain.** Selecting a *code* fraction by minimizing *code-only* BPB is nearly monotone by construction — the prior unmatched curve was strictly decreasing through p=1 (`RESULTS.md:15,34`) and the target's interior minimum existed only because repetition damage eventually overtook the benefit. A web-domain BPB column costs zero training and lets a reader see whether conclusions are metric-specific. Also report the target curve's local point-to-point variation so regret differences can be read against the target's own measurement noise.

6. **The cost model omits the term that will dominate.** `design.md:31` excludes "compilation and hardware inefficiency," but 162 of the 183 runs are 633-step tiny-model runs where per-run fixed overhead likely exceeds compute. Also note the pilot is 33% of the dense cost (4.79e19 of 1.44e20) purely because of its 7 target points; a proxies-first pilot with 3 target points would test most pilot criteria for a fraction of that. Re-state "cheap" in chip-hours.

7. **Specify the optimizer beyond LRs.** `design.md:29` gives Muon 0.02 / Adam 0.008 but not which parameter groups take Muon versus Adam — decisive when embeddings are 49.4% of the proxy — nor weight decay, gradient clipping, or LR floor. Weight decay in particular interacts with a 15.8-epoch regime.

8. **Note that the Nemotron proportions are inherited constants.** `design.md:27` says the six components keep "their fixed internal proportions"; those come from `NEMOTRON_TOKEN_COUNTS` (`launch_starcoder_wsd_80_20_surface.py:50-57`), which are **Llama3-tokenizer** counts. Reusing them under a 32k tokenizer preserves the *token* proportions while shifting the *content* proportions. State which invariant is intended.

9. **Spearman is weak on a U-shaped curve** (`design.md:37`) — it conflates the two limbs. Excess-curve RMSE is the more meaningful of the two secondary measures.

10. **Aspect ratio and head count are not scaled.** Proxy is 256/8 (ratio 32, **2 heads**); target is 1024/16 (ratio 64, 8 heads). Two attention heads is an unusual configuration and it is not a shape-preserving scale-down, which bears directly on Muon LR transfer.

---

# Canary versus offline

**Decidable offline, no accelerator:** parameter counts and all TPP/epoch/FLOP arithmetic (I verified these — they reproduce exactly); the B2 zero-weight/shuffle-key and config-fingerprint behavior; the full sequence-allocation audit at every coordinate including no-wrap and the B8 composition check (run the real allocator, no training); subset nesting and pairwise overlap; web-pool sufficiency from shard sizes; bounded-tokenization cache lengths on synthetic text; tokenizer and source-generation hashes; the B1 missing file; Muon/Adam group assignment; and the 633-vs-11,491 step counts.

**Requires a live canary:** optimizer stability at the 256/8, 2-head geometry under MuonH at 633 steps (spikes, grad norms, divergence); whether batch 128 is supercritical at 16.6M parameters (B4 — needs a two-point batch comparison at fixed tokens); trainer-seed noise magnitude, which is the *input* to B3's informativeness criterion and therefore must be measured before that gate can be evaluated; whether Paloma programming-languages BPB is non-degenerate at this scale; whether the target curve has an interior minimum (a few target points suffice, not 21); regional source/cache receipts and the child-runtime canary already required at `design.md:50`; and per-run wall-clock for the revised cost model.

Note the ordering constraint this creates: **B3's gate cannot be specified numerically until the canary returns seed noise.** Prespecify the *form* of the criterion now and the constant immediately after the canary, before any pilot regret is computed.

---

# On the three open questions

**Q1 — optimizer adequacy.** Adequate as a pilot *only* with a prespecified failure contingency, which the doc lacks. The specific risk the doc does not name: repetition and learning rate interact. The matched arm sees 15.8 epochs of a 10.5M-token pool; a too-high LR or the 6-step warmup can damage that arm more than the fresh-data arm, producing "matched selects worse" as an **optimizer artifact rather than a scientific result** — the same sign as the prior adverse finding. Add: a stability screen applied identically per arm, per-arm training diagnostics (grad norm, update RMS, spike counts) reported at every p, and a rule that if retuning is needed it happens once, on a mixture-agnostic criterion, applied identically to both proxy arms, followed by a pilot re-run. Given B4, I'd fold the batch decision into the same stage. A fully separate preregistered calibration stage is defensible but not required if that contingency is written down first.

**Q2 — single target curve.** Sufficient for the stated illustration, and `design.md:37` scopes it correctly. But the headline quantity reduces to a difference of the target curve at exactly two points, both single-seed. Consider a second target seed at only the selected coordinates: ~4 points × 6.664e18 ≈ **2.7e19, about 18% of the dense budget** — materially cheaper than the "separately reviewed release" framing implies, and it converts the headline from a point difference to one with an error bar.

**Q3 — subset versus seed.** Only qualitatively, and less well than the doc implies. Two seeds give 1 df per subset (3 df pooled) and three subsets give 2 df; the three subsets are drawn from one parent and may overlap (expected ~282 of 5,120 sequences shared per pair), so they are positively correlated; and per B2 the p=0 point has zero subset variance by construction, so the variance decomposition is not comparable across the grid. The design is honest about this at `design.md:25,44`. The cheap improvement is #4 above — report all six per-run argmins rather than three subset-averaged ones.

---

Two caveats on my own conclusions. I verified arithmetic and file existence directly, but the B2 aliasing claim, the B8 read-order claim, and the B4 supercritical-batch claim are inferences from the historical launcher and the prior report — B2 and B8 are settled by reading the allocator, B4 needs the canary. And I did not read `starcoder_epoch_matching/research.md` or the pilot results, so the prior experiment's construction details I cite come from its refinement `RESULTS.md` and the launcher only.