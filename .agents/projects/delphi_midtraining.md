# Delphi midtraining — model & run catalogue

> ## 🚨 CRITICAL TRAP #1 — THE "1e20" ENTRY IN THIS CATALOGUE IS NOT DELPHI 🚨
>
> **Discovered 2026-05-14.** Any 1e20 entry below that points at `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` is the **wrong checkpoint** — that's a deprecated v5 isoflop ablation point with a different architecture (d=2048, L=21) and different optimizer recipe than the v6 Delphi family. The catalogue's own §1 even says Delphi = "AdamH **v6** scaling-ladder suite" — that's the contradiction that should have caught this earlier.
>
> The canonical Delphi 3e20 base is:
>
>     isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6
>     gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/
>     Registered: experiments/exp1337_eval_suite.py:180
>
> Note that **Delphi proper only goes 1e21 → 1e23.** The 3e20 entry is one of the 7 ISOFlop-bucket winners used to fit the Delphi scaling law (see HF collection). There is no "1e20 Delphi" — closest is the 3e20 bucket winner above.
>
> **Full post-mortem:** [`.agents/ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md`](../ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md)
>
> **Rule (verbatim, do not deviate):** Base checkpoints for any Marin scaling-law / Delphi / isoflop experiment come from `exp1337_eval_suite.py` EVAL_BASES OR `MARIN_SCALING_SUITES["nemotron-completed-adamh"]` OR the HF Delphi collection. NEVER from a GCS `gsutil ls` grep — the bucket preserves every deprecated experiment generation.

> ## 🚨 CRITICAL TRAP #2 — "-v5-" IN 1e21 / 1e22 / 1e23 NAMES IS **NOT** THE BROKEN v1 RECIPE 🚨
>
> **Verified 2026-05-16 via hparam math against the published Delphi blog.** The canonical 1e21, 1e22, and 1e23 Delphi runs have `-v5-` in their directory names (`adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021`, `…-1e+22-v5-025b0e`, `…-1e+23-v5-27f2fb`). This **looks like** they should be the failed Cautious-AdamC attempt-1, and a future agent will be tempted to discard them. **Do not.** They are the AdamH attempt-2 held-out validation runs and are what the Delphi paper actually publishes.
>
> Per the Delphi blog (https://openathena.ai/blog/delphi/), there were two recipes:
>
> - **Attempt 1 (broken)** — Cautious AdamC, `experiments/scaling_law_sweeps/c_adamc.py`. Fixed `weight_decay=0.1`, `β1=0.95`, `β2=0.98^(B/B0)`, `ε=1e-15`, `max_grad_norm=1.0`, projection LR `η = 0.33·√B / H`. The fit diverged at 1e23 by **2.5% then more**. This recipe is *not* used by any Delphi run on GCS that we keep.
> - **Attempt 2 (Delphi)** — AdamH + Complete(d)P, `experiments/scaling_law_sweeps/completed_adamh.py`. Weight decay removed (AdamH bounds the Frobenius norm), `β1=0.9`, `β2=clip(0.9999^(B/B0), 0.9, 0.9999)`, `ε=ε0·√(B0·T / B·T0)`, `max_grad_norm=0.1`, projection LR `η = η0·√(B/B0)·(T0/T)^0.3`, separate Adam-scalar LR `η_Adam = η0_Adam·√(B·T0 / B0·T)`. Fit lands all held-out runs within 0.5%. **This is what every "Delphi" checkpoint we care about was trained with.**
>
> The Delphi scaling law was **fit on the 7 IsoFLOP optima at 3e18 → 3e20** (see §4.1 below — these are the "v6 isoflop" entries in the codebase). **1e21, 1e22, and 1e23 are held-out validation runs**, trained with hparams predicted by that fit, not refit on them. So directionally, smaller compute-optimal runs predicted the larger ones — and the larger "v5"-named runs are *downstream* of the fit, not predecessors of it.
>
> The `-v5-` suffix is a hardcoded experiment-iteration tag from `exp1337_delphi_suite.py:232` (`f"-v5{suffix}"`). The same file uses `LABEL = "adamh_scaling_v6"` (line 62) → these runs ARE canonical AdamH Delphi. The naming overlap is documented in the [[project_delphi_canonical_bases]] memory under "v5/v6 string overlaps three different things."
>
> **Empirical check — when in doubt, do this. Do not trust the directory name.**
>
> The v2/AdamH/Complete(d)P recipe's reference constants from the blog: `B0=64, T0=2.5e9 tokens, η0=0.00630, η0_Adam=0.000656, ε0=1.85e-8`. For any base with `(B, T, H)`, plug into both recipes and check which one the registry's stored `peak_lr` / `peak_adam_lr` / `beta2` match.
>
> Verified for the two suspect entries on 2026-05-16:
>
> | Base | Stored peak_lr | v1 / Cautious AdamC prediction | v2 / AdamH prediction | Stored beta2 | v1 prediction | v2 prediction |
> |---|---|---|---|---|---|---|
> | DELPHI_1E21 (B=512, T=46.3B, H=2560) | **7.425e-3** | 2.92e-3 ❌ | 7.42e-3 ✅ | **0.99920** | 0.922 ❌ | 0.9992 ✅ |
> | DELPHI_1E22 (B=1024, T=160B, H=3840) | **7.232e-3** | 2.75e-3 ❌ | 7.23e-3 ✅ | (clamped 0.9999) | 0.851 ❌ | 0.9999 ✅ |
>
> Both match the v2 AdamH/Complete(d)P recipe to within rounding; both miss the v1 Cautious-AdamC recipe by ~2.5×. **The "v5"-named 1e21 and 1e22 are AdamH runs. Use them. Do not relaunch them under a different name; do not "fix the typo".**
>
> **Rule:** if a Delphi base entry looks suspicious because of "v5" in its name, **do the hparam-math check above before discarding it.** If the v2 formula matches, the run is canonical Delphi regardless of the version string in the path.

---

**Goal of this doc:** give a future agent a single place to find every "Delphi" compute-optimal scaling-ladder model — where the runs live on W&B, where every Levanter / HF checkpoint lives on GCS, which budgets finished vs crashed, and how this suite connects to the broader Mantis-style midtraining program.

Author-agent context: captured 2026-04-21 from a live dump of `marin-community/marin` + `marin-community/marin-analysis` on W&B and `gs://marin-us-central2/` on GCS. If you are a future agent, **verify the W&B run states and GCS paths before acting** — anything marked "crashed" below may have been resumed or deleted since this was written.

---

## 1. What is "Delphi"?

"Delphi" is an internal Marin codename for the **AdamH v6 scaling-ladder suite on Nemotron-CC v1 HQ** — i.e. a small family of compute-optimal models trained at 1e21, 1e22, and 1e23 FLOPs using the `completed_adamh_heuristic` (sqrt-batch LR, no /H on adam_lr). It sits in the Marin lineage alongside its siblings:

```
Delphi  →  Starling  →  Bison  →  Mantis
```

All four share the same Nemotron-CC **v1** pretraining mix via `experiments/pretraining_datasets/nemotron.py` (not v2). Mantis is the 32B final cooldown; Delphi is the compute-optimal "scaling-ladder reference" model family that informs downstream heuristic choices.

The suite is defined by `experiments/exp1337_delphi_suite.py`:

- Label: `adamh_scaling_v6`
- Heuristic: `experiments.scaling_law_sweeps.completed_adamh.completed_adamh_heuristic`
- Seq-len: 4096
- Mixture: `nemotron_mix` (Nemotron-CC v1 HQ, no math/code specialty splits)
- Target budgets and TPUs (from the source file):
  - `1e21 → v4-128 / batch 512`
  - `1e22 → v4-512 / batch 1024`
  - `1e23 → v4-1024 / batch 2048`

The suite also produces an isoflop-analysis step that fits a scaling law from an earlier isoflop sweep — this fit is what picks the compute-optimal `(params, tokens)` at each target budget.

---

## 2. Isoflop analysis step

W&B (project `marin-community/marin-analysis`):

| run id | state | created | name |
|---|---|---|---|
| `un24w72g` | finished | 2025-12-26T18:41Z | `exp2166-scaling-ladder-nemotron-analysis` |
| `2gxvdky7` | finished | 2025-12-26T19:28Z | `exp2166-scaling-ladder-nemotron-analysis` |

GCS output (produced by the `run_isoflop_analysis_step` ExecutorStep):

- `gs://marin-us-central2/adamh-scaling-ladder-nemotron-analysis-9200ec/`
  - `isoflop_analysis_result.json` — 7 `(flops, optimal_tokens, optimal_params, loss_at_optimal)` records
  - `fit_curves.json` — raw fit curves
  - `.artifact`, `.executor_info`, `.executor_status`

**Scaling fit for `label=adamh_scaling_v6`:** α = 0.5800829529762268, A = 0.026719627901911736 (stored in `scaling_fits["adamh_scaling_v6"]`).

Representative minima from the fit (abbreviated):

| FLOPs | opt. params | opt. tokens | loss @ optimal |
|---:|---:|---:|---:|
| 2.9 e18 | 4.47 e8 | 1.39 e9 | 1.088 |
| 9.1 e18 | 5.50 e8 | 2.68 e9 | 1.028 |
| 1.8 e19 | 8.37 e8 | 3.91 e9 | 0.994 |
| 3.1 e19 | 9.98 e8 | 5.30 e9 | 0.972 |
| 8.9 e19 | 1.38 e9 | 9.61 e9 | 0.925 |
| 1.7 e20 | 1.93 e9 | 1.52 e10 | 0.898 |
| 3.1 e20 | 2.54 e9 | 2.08 e10 | 0.879 |

---

## 3. Architectures (fixed per budget)

All Delphi runs share seq-len 4096, tokenizer Llama3, `block_cross_document_attention=True`, `allow_nondivisible_batch_size=True`. Architecture is a deterministic function of the target budget (seed/rev do not change H, L):

| Budget | Hidden dim | Layers | N (params) | Batch size (v5 rev) | Num train steps (v5) |
|---:|---:|---:|---:|---:|---:|
| 1e21 | 2560 | 26 | ≈ 3.38 B | 512 | 22 057 |
| 1e22 | 3840 | 37 | ≈ 9.7 B | 1024 | 38 235 |
| 1e23 | 5376 | 51 | ≈ 25 B | 2048 | 74 884 |

Earlier revs (v1…v4) used smaller batch sizes — see §4.

---

## 4. All runs — state, checkpoints, metrics

All 24 optimal-training runs live in W&B project **`marin-community/marin`** (team-private). Every run has its GCS root at `gs://marin-us-central2/<run-name>/` with two subtrees:

- `checkpoints/step-*/` — Levanter TensorStore checkpoints (`manifest.ocdbt`, `metadata.json`, `d/`)
- `hf/step-*/` — periodic HuggingFace-format exports at `hf_save_steps=10000` (and a final export at the last training step)

### 4.1 Usable / fully-trained models (8 runs with HF exports)

| Budget | Run name | Final step | Last HF step | Paloma c4_en loss | c4_en bpb | macro bpb |
|---:|---|---:|---:|---:|---:|---:|
| 1e21 | `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021` | 22056 | 21979 | 2.7326 | 0.8410 | 0.9964 |
| 1e21 | `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-seed42-e251d0` | 22056 | 22007 | 2.7325 | 0.8410 | 0.9972 |
| 1e21 | `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-seed62746-659a1b` | 22056 | 22013 | 2.7318 | 0.8407 | 0.9968 |
| 1e21 | `adamh-scaling-ladder-nemotron-optimal-1e+21-v6-77f848` | 22056 | 21999 | 2.7322 | 0.8409 | 0.9960 |
| 1e22 | `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e` | 38234 | 38206 | 2.5182 | 0.7750 | 0.9164 |
| 1e22 | `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-seed42-deeff4` | 38234 | 38163 | 2.5192 | 0.7753 | 0.9177 |
| 1e22 | `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-seed62746-10f597` | 38234 | 38164 | 2.5178 | 0.7749 | 0.9165 |
| 1e23 | `adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb` | 74883 | 74878 | 2.3435 | 0.7213 | 0.8549 |

Three seeds × two budgets (1e21, 1e22) give small cross-seed error bars (Δc4_en_loss ≈ 0.001 at 1e21, ≈ 0.001 at 1e22). The 1e23 tier only has the one `v5-27f2fb` seed.

### 4.2 Partial / crashed runs (10 runs with at least one checkpoint)

Still useful for warm-starts, debugging, or partial-budget comparisons:

| Run name | State | Steps done / total | Last ckpt step | HF? |
|---|---|---:|---:|---|
| `adamh-v6-scaling-ladder-nemotron-optimal-1e+22-81073a` | crashed | 16320 / 152940 | 16261 | yes |
| `adamh-v6-scaling-ladder-nemotron-optimal-1e+23-a128a5` | crashed | 7486 / 599072 | 7438 | no |
| `adamh-scaling-ladder-nemotron-optimal-1e+21-v2-988231` | crashed | 317 / 44114 | 190 | no |
| `adamh-scaling-ladder-nemotron-optimal-1e+21-v3-28b815` | crashed | 1177 / 22057 | 1001 | no |
| `adamh-scaling-ladder-nemotron-optimal-1e+21-v7-8c50f0` | crashed | 11134 / 22057 | 11030 | yes |
| `adamh-scaling-ladder-nemotron-optimal-1e+22-v2-b51e9c` | crashed | 164 / 76470 | 99 | no |
| `adamh-scaling-ladder-nemotron-optimal-1e+22-v3-c77e58` | crashed | 394 / 38235 | 337 | no |
| `adamh-scaling-ladder-nemotron-optimal-1e+22-v6-500e71` | crashed | 27630 / 38235 | 27605 | yes |
| `adamh-scaling-ladder-nemotron-optimal-1e+22-v7-5f064e` | crashed | 7354 / 38235 | 7281 | no |
| `adamh-scaling-ladder-nemotron-optimal-1e+23-v3-845c5b` | crashed | —    / 149768 | — | no |

### 4.3 Runs with no useful checkpoint (6 runs — launched but never produced a ckpt)

`adamh-scaling-ladder-nemotron-optimal-1e+21-22c265`,
`adamh-scaling-ladder-nemotron-optimal-1e+22-10aaee`,
`adamh-scaling-ladder-nemotron-optimal-1e+23-479d5f`,
`adamh-scaling-ladder-nemotron-optimal-1e+21-v4-a880a2`,
`adamh-scaling-ladder-nemotron-optimal-1e+22-v4-a56015`,
`adamh-scaling-ladder-nemotron-optimal-1e+23-v4-d43cd0`.

The GCS dirs for these exist but `checkpoints/` is empty and `hf/` is missing.

### 4.4 Notable incident

The original "v6" launch attempt crashed almost immediately on 2026-03-03:

- `adamh-v6-scaling-ladder-nemotron-optimal-1e+23-a128a5` — the 25B-params 1e23-FLOPs run died at step 7486 / 599 072 = **1.25 % of its target budget**. This is the run flagged in `midtraining_math.md` §"Primary sources" (line 922 of that doc). Last ckpt: `step-7438`, no HF export.
- `adamh-v6-scaling-ladder-nemotron-optimal-1e+22-81073a` — 9.7B-params 1e22 run died at step 16 320 / 152 940 = ~10.7 %. Last ckpt: `step-16261`, HF export present.

After these two crashes the relaunch was renamed from `adamh-v6-scaling-ladder-…` to `adamh-scaling-ladder-…-v{2..7}` and eventually converged at rev `v5`.

---

## 5. W&B tag conventions

Not every run got the intended tag set. When tags *are* present, they are (from `exp1337_delphi_suite.py` and the earlier v6 launch):

- `optimal-training`
- `completed-adamh` (current code) or `completed-adamh-v6` (original v6 launch)
- `label=adamh_scaling_v6`
- `FLOPs=<budget>` e.g. `FLOPs=1.0e+21`
- `N=<params>` e.g. `N=3.4e+09`, `9.7e+09`, `2.5e+10`
- `seed=<int>` on the seeded reruns

Most of the 1e22 and 1e23 v5 runs have **empty tags** because they were launched without the tag block; the 1e21 v5 runs and both v6 crash-runs have the full tag set. **Do not rely on tag filtering alone** to enumerate the Delphi suite — always combine with a name-regex filter:

```
(?i)(adamh-v6-scaling-ladder|adamh-scaling-ladder-nemotron-optimal)
```

---

## 6. GCS paths — copy-paste

Every run stores under `gs://marin-us-central2/<run-name>/`. Fully-trained model paths:

```
# 1e21 FLOPs (~3.4 B params) — four seed replicates
gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/{checkpoints,hf}
gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-seed42-e251d0/{checkpoints,hf}
gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-seed62746-659a1b/{checkpoints,hf}
gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v6-77f848/{checkpoints,hf}

# 1e22 FLOPs (~9.7 B params) — three seed replicates
gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/{checkpoints,hf}
gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-seed42-deeff4/{checkpoints,hf}
gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-seed62746-10f597/{checkpoints,hf}

# 1e23 FLOPs (~25 B params) — single run
gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb/{checkpoints,hf}

# Isoflop analysis output (scaling fit JSON)
gs://marin-us-central2/adamh-scaling-ladder-nemotron-analysis-9200ec/
```

Bucket is `us-central2` — stay in-region for reads/writes; cross-region egress is a primary cost driver for Marin (see AGENTS.md).

---

## 7. How this connects to Mantis-style midtraining

Delphi is the scaling-ladder *reference* for Marin's compute-optimal pretraining heuristic. The heuristic is then reused by the downstream Mantis-style cooldown / midtraining recipes that are the subject of [midtraining_math.md](./midtraining_math.md).

### Summary of [midtraining_math.md](./midtraining_math.md)

`midtraining_math.md` is a ~922-line critical survey (author-agent: 2026-04-19) of **open-source math corpora** for a Mantis-style cooldown, under a hard constraint of no Qwen-generated or Qwen-rewritten text (Qwen-as-classifier is fine; Qwen-as-generator is not). It (1) enumerates what the current 32B Mantis cooldown actually mixes in — MegaMath + FineMath-3+ + Dolmino-clean + ProofPile-2 — and flags that Delphi / Starling / Bison / Mantis all use the older Nemotron-CC **v1** splits while the entire **v2 stack** (including `nemotron_cc_math_v1/{3, 4plus, 4plus_mind}` totaling ~260 B+ tokens) is tokenized in `gs://marin-us-central2/` but *unused* by any training mix; (2) does a paper-by-paper deep-dive on OpenWebMath, FineMath, MegaMath, Nemotron-CC-Math, InfiMM-WebMath, UltraData-Math, MathPile, Dolmino, MathCode-Pile, etc., with a seven-confounder critique of the "we beat everyone" ablation chain (base-model leakage, train-compute-not-matched, math-weight-not-matched, ≤8B scale, no 32B ablations anywhere, decon-claimed-not-verified, synthetic-vs-natural conflation); (3) documents the Bison→Mantis +14.4-point GSM8K recovery as the primary evidence that Dolmino `all_math` was contaminated in practice despite its decontamination claims; (4) cross-references NVIDIA's own Nemotron Nano 2 recipe (math is only 3-11% of their pretraining mix, AIME wins come from curriculum + reasoning post-train, not from flooding math); and (5) proposes three concrete recipes — Recipe A (strict no-Qwen, zero ingestion cost, swaps the `all_math`-equivalent weight onto the already-tokenized `nemotron_cc_math_v1/4plus` + `/3` + a trimmed MegaMath-Web + ProofPile-2 mirror), Recipe B (adds Phi-4-synthetic `4plus_mind` + Nemotron-MIND + MathCoder2), Recipe C (adds UltraData-Math L1/L2, InfiMM-WebMath, MathPile, MathCode-Pile — requires new tokenization). It also recommends a pre-cooldown phase-2 HQ-upweight phase before the linear LR decay as the single biggest structural change from Mantis. The doc itself lives outside this worktree at the absolute path below.

**Canonical location (untracked; not present in this worktree):**
- `/Users/ahmed/code/marin/.agents/projects/midtraining_math.md`
- relative link (only works if the file has been copied into the current worktree): [`midtraining_math.md`](./midtraining_math.md)

If you need to reference it, either `cat` the absolute path or copy it into your worktree's `.agents/projects/` first.

---

## 8. Midtraining LR + budget strategy (2026-05-01 lab discussion)

This section captures the design direction for *running midtraining experiments on top of the Delphi suite* — i.e. where we go after the catalogue in §1-§7. It is the working plan, not a finished result.

### 8.1 Working hypothesis

**Optimal midtraining LR is fixed per mixture, not per scale or per token budget.** If true, this factors the search: fit one LR per mixture on a small Delphi scale and transfer it across the ladder.

The hypothesis is fragile in two directions and we should test both:

1. **LR may depend on token count.** If midtrain LR drifts with the number of midtrain tokens, fixed-LR-per-mixture will break when budgets differ. Mitigation: hold the *pretrain : midtrain proportion* constant (e.g. 2/3 pretrain : 1/3 midtrain) and scale the midtrain budget linearly with the pretrain budget — this way LR is transferred between matched regimes, not across regime boundaries.
2. **LR may depend on model scale.** The current heuristic already scales LR across model sizes (sqrt-batch + the AdamH v6 rules in §1). Keeping the *ratio* of midtrain LR to pretrain LR fixed across scales is a strictly weaker claim than fixed-absolute-LR and is the safer fallback if (1) holds but the absolute LR still drifts.

### 8.2 Methodology

- **Fit regressions across (mixture × token budget × scale).** The output isn't a single optimal LR — it's a family of fits parameterized by mixture, with budget and scale as covariates. Predict not just the final loss but the *shape* of the midtrain loss curve.
- **Validate on a held-out subset of the midtrain mixture itself.** Carve out a small validation slice from the midtraining corpus before it enters any training mix, and use its loss as the optimization target. Not Paloma c4_en, not external evals — the eval has to reflect the mixture we're tuning for. It must be cheap enough to run frequently during the small-scale sweeps. Concretely: hold out a fixed-token slice of each midtrain mixture component pre-tokenization, store it next to the checkpoints, and ensure it never appears in any training mix.
- **Small-scale fit → large-scale extrapolation.** Run the LR / mixture sweep at the cheapest tier (1e21 — ~3.4 B params, batch 512, ~22 k steps) and use the fits to predict 1e22 and 1e23. Treat the larger tiers as confirmation runs, not search runs.

### 8.3 Reframe: midtraining budget is *not* fixed across the ladder

Each Delphi model has a *different pretrain token budget* (see §3 — 1e21 trains on a Chinchilla-optimal slice that is two orders of magnitude smaller than the 1e23 slice). Fixing an *absolute* midtrain token budget across scales is therefore incoherent — it would let midtrain dominate the small models while being a rounding error on the big ones.

**Each scale gets its own midtrain budget, set dynamically as a fraction of its pretrain budget.** The fraction is itself a knob (the 2/3 pretrain : 1/3 midtrain split is one candidate) but it must be held *constant* across scales when extrapolating. Performance is then modelled as

```
validation_loss = f(scale, mixture, budget_fraction, budget_point_along_decay)
```

not as `f(absolute_midtrain_tokens)`. The "budget_point_along_decay" axis matters because under linear LR decay the loss curve shape — not just its endpoint — is what scales between Delphi tiers.

### 8.4 Default LR schedule

Stick with **linear LR decay** as the baseline schedule. It's the tried-and-true setting in the Delphi pretraining recipe and Mantis cooldown; deviating from it adds a confound we don't need until the LR-vs-mixture story is settled.

### 8.5 Concrete next steps

1. Pick a midtrain mixture candidate (most likely one of the Recipe A / B / C variants from `midtraining_math.md` §"Three concrete recipes").
2. Carve out the held-out validation slice from that mixture and write its GCS path + token count into this doc.
3. Fix the pretrain : midtrain proportion (default proposal: 2/3 : 1/3) and derive the per-scale midtrain budgets from §3.
4. Run the LR sweep at 1e21 only, warm-starting from the 1e21 Delphi v5/v6 checkpoints (§4.1).
5. Fit `(scale, mixture, budget_fraction, decay_point) → validation_loss` and plot predicted 1e22 / 1e23 curves with uncertainty.
6. Spend larger compute only on confirmation runs at 1e22 and 1e23, sized to the dynamic budgets from step 3.

### 8.6 Generalized midtraining-mix framework

The concrete API design (`MidtrainMixSpec` / `MidtrainComponent` / `build_midtrain_lm_data_config` / `midtrain_token_budget`) plus four-layer safety assertions and a numbered corner-case catalogue (CC1-CC25) lives in the logbook section §"2026-05-01 21:00 UTC — generalized midtraining-mix framework + safety assertions" of `.agents/logbooks/midtraining_delphi.md`. That framework generalizes the §8.3 budget heuristic and the §8.2 held-out val slice to arbitrary single-component or multi-component midtraining mixtures, with runtime guarantees that val never leaks into training (sample-based hashing assertion + pinned val-partition fingerprint). Refactor `experiments/midtraining_mixes.py` per that section before launching the next sweep.

### 8.7 Visualization: marimo endpoint-scaling notebook

Use the marimo notebook for interactive final-loss scaling and within-run
prediction diagnostics:

```bash
uv run --with marimo marimo edit \
  --headless --no-token --host 127.0.0.1 --port 2718 \
  scripts/analysis/delphi_small_final_loss_scaling_notebook.py
```

The notebook reads cached outputs from:

- `midtrain_analysis_outputs/small_final_loss_scaling/endpoints.csv`
- `midtrain_analysis_outputs/small_final_loss_scaling/fit_summary.csv`
- `midtrain_analysis_outputs/small_final_loss_scaling/extrapolation_targets.csv`
- `midtrain_analysis_outputs/small_final_loss_scaling/extrapolation_predictions.csv`
- `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_points.csv`
- `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_prefix_predictions.csv`
- `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_prefix_summary.csv`
- `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_method_selection.csv`

Generate or refresh endpoint outputs first with:

```bash
uv run python scripts/analysis/delphi_small_final_loss_scaling.py
```

Then refresh within-run prefix-prediction outputs with:

```bash
uv run python scripts/analysis/delphi_within_run_prediction.py
```

Refresh without `--use-cache` after sweeps finish. The script queries W&B live
for both the small-ladder endpoints and the `1e21`/`1e22` extrapolation targets;
the older local trajectory dump is only a fallback when W&B is unavailable.

Notebook UX decisions that matter:

- Use **marimo controls**, not the Plotly legend, as the source of truth.
- Learning-rate filtering is four explicit checkboxes (`lr33`, `lr50`, `lr67`, `lr83`). This was intentional: Plotly legend toggles are ambiguous when each recipe has multiple traces (points, fit line, residual line, partial marker).
- The endpoint plot renders only checked LR recipes, so visible curves always correspond exactly to selected controls.
- The `1e21`/`1e22` points are held-out extrapolation targets. They are rendered past the `2e20` fit boundary and never enter the small-ladder fit.
- Target markers are diamonds for complete-like held-out runs and open x markers for best-prefix runs that did not reach the final planned step.
- The fit-quality readout should prioritize residuals and leave-one-scale-out error in raw loss units. `R^2` is secondary because these monotone endpoint curves can make bad extrapolations look visually plausible.
- The `floor + A * compute^-alpha` fit is diagnostic only until we have more scales; the default baseline is `log(loss) = a + b log(compute)`.
- The within-run section tunes prefix-prediction methods on completed small
  ladder runs through `2e20`, then evaluates `1e21`/`1e22` as held-out large
  targets. Use the prefix controls to inspect the accuracy/cost tradeoff; do
  not assume the first 10% is enough for every metric.

After the `2e20` sweep completed, held-out math endpoint loss remained very
clean under the log-log fit: per-recipe `R^2 ~= 0.998`, monotone endpoints, and
exponents clustered near `b ~= -0.095`. The same small-ladder fit extrapolates
tightly to `1e21` (mean absolute math-loss error `0.0094`) and under-predicts
the complete `1e22` improvement (mean observed-minus-predicted error `-0.0641`,
roughly `-10.7%`). The live W&B snapshot used here has `1e22` coverage at
`11/12` complete-like targets; `p50m50-lr67` is still a best-prefix target in
W&B at step 6382/7647. See `.agents/logbooks/midtraining_delphi.md` sections
`2026-05-21T01:52Z — final-loss scaling-law first pass` and
`2026-05-21T02:45Z — held-out 1e21/1e22 extrapolation overlay` for the recorded
result and commands.

Within-run prefix-prediction first pass (2026-05-21) used five methods:
`last_value`, `linear_tau`, `template_global`, `template_by_mix`, and
`template_by_recipe`. The template methods learn the median fraction of final
improvement achieved by a prefix on the small ladder. For math validation,
10% progress is informative but not enough under small-CV (`template_by_mix`
MAE `0.0353`); the selected accuracy/cost point is `template_by_recipe` at
50% progress (small-CV MAE `0.00387`, held-out complete MAE `0.0210`). Paloma
macro and C4 select `template_by_recipe` at 10% progress with held-out MAE
around `0.024` and `0.022` respectively. See
`midtrain_analysis_outputs/small_final_loss_scaling/trajectory_prediction_summary.md`
for the full leaderboard.

---

## 9. Provenance

- W&B dump: `/tmp/delphi/delphi_runs.json` (24-record JSON with full config + summary per run; regenerate by running the "Pull all runs matching either naming prefix" script in the 2026-04-21 agent session — see entity `marin-community`, projects `marin` and `marin-analysis`, regex above).
- GCS listing: `gcloud storage ls gs://marin-us-central2/<run-name>/checkpoints/` and `.../hf/` per run, parallelized 16-wide. Run-tree sizes were not measured (`gcloud storage du` skipped to avoid cost / latency on 25 B-param checkpoints).
- Source of truth for *code*: `experiments/exp1337_delphi_suite.py` (function `run_optimal_training`, dict `TARGET_BUDGETS`, constants `LABEL = "adamh_scaling_v6"`, `SEQ_LEN = 4096`).
- The "Delphi" codename is documented in `/Users/ahmed/code/marin/.agents/projects/midtraining_math.md` lines 63, 906, 922; it does not appear in W&B project names, tags, groups, or config fields.
- No separate `delphi`, `delphi-lab`, or `delphi-ai` W&B entity exists (verified 2026-04-21).

---

## 10. Open questions / things this doc does *not* yet have

- **Held-out midtrain validation slice** — Plan (proposed 2026-05-01): carve `12_500` sequences (~51.2 M tokens at seq=4096) out of `nemotron_cc_math_v1/4plus` via Levanter's `LmDataConfig.num_validation_sequences` + `shuffle_before_trainval_split=True`. The split runs inside the cache layer in `_split_into_trainval_sets` (`lib/levanter/src/levanter/data/text/datasets.py:519-539`) which applies a **full Feistel permutation** with fixed `PRNGKey(0)` over the entire cache before slicing the val tail. This is strictly stronger mixing than the hierarchical block shuffle (every index can map to every other index, not just within a window) — the val set is N truly-uniform-random sequences pulled from anywhere in the cache, identical across every sweep run as long as the cache content and split key don't change. The per-component hierarchical block shuffle still runs on the train remainder for I/O locality during streaming. Carving from only the math component is intentional — pretrain retention is measured separately by Paloma c4_en. Removes ~0.1 % of 4plus (52 B → still ~52 B), negligible for training.
- **Per-scale midtrain budget table** — Decision (2026-05-01 follow-up): single rule `midtrain_tokens = pretrain_tokens / 5` (i.e. 5/6 pretrain : 1/6 midtrain compute split, K = 0.20 for every scale). The earlier 2/3 : 1/3 (K=0.5) proposal was rejected as too midtrain-heavy; K = 0.20 sits in Mantis-style cooldown territory (~10–20 %). Pretrain steps × BS × 4096 from `experiments/exp1337_delphi_suite.py`:

    | Scale | Pretrain steps × BS | Pretrain tokens | Midtrain (K=0.20) | Midtrain steps @ BS=512 |
    |---|---|---:|---:|---:|
    | 1e20 (3e20-iso d2048-L21) | 47,064 × 128 | 24.67 B | 4.93 B | 2,354 |
    | 1e21-v5 | 22,057 × 512 | 46.27 B | 9.25 B | 4,413 |
    | 1e22-v5 | 38,235 × 1024 | 160.37 B | 32.07 B | 15,294 |
    | 1e23-v5 | 74,884 × 2048 | 628.25 B | 125.65 B | 59,907 |

    Held constant across scales while extrapolating, per §8.3. K = 0.25 (1/5 share, lighter alternative) on the table if K = 0.20 underfits. Sweep tracking in `.agents/logbooks/midtraining_delphi.md` §"2026-05-01 20:30 UTC — new sweep plan".
- **Checkpoint total sizes in bytes** — skipped to keep the audit cheap. Any future agent that cares can `gcloud storage du -s gs://marin-us-central2/<run>/checkpoints/` per run.
- **Per-checkpoint eval curves** — this doc has only the final `_step` eval metrics from W&B summary. If you need mid-training loss/bpb over time, use `wandb.Api().run(...).history()` or scan `run.scan_history(keys=[...])`.
- **Downstream lm-eval scores** (MMLU, HellaSwag, ARC) — these were `None` in the W&B summaries at the time of capture. Some may live on separate `*-lmeval-*` runs (we saw one `8jnuv7ca` for `exp2166-scaling-ladder-nemotron-validation-optimal-1e+18-24e99e-43791_lmeval_mmlu` in the same project) — not audited here.
- **Connection to the `exp2166` earlier scaling-ladder runs** (76 runs with `label=nemo-wider-depth-adapt` or `label=comma-mix`) — those use a *different* heuristic (not CompletedAdamH v6) and are not part of Delphi. They are listed here only so future agents don't confuse them.
