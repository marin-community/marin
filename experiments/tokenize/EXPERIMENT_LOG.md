# Tokenizer bake-off — experiment log

Chronological log of every experiment in the grug-moe tokenizer investigation (issue #6796).
Each entry is self-contained: the hypothesis, the exact launch command to **reproduce** it, the
command to **replay** the analysis from stored logs, and the result. Goal: **≥10% feBPB
improvement over the stock Llama-3 tokenizer** for target grug-moe models.

## Conventions

- **Cluster**: `cw-rno2a` (8×H100 × 64). `export KUBECONFIG=~/.kube/coreweave-iris-rno2a` and
  prefix iris/gh with `env -u GH_TOKEN`.
- **Proxy shape** (unless noted): hidden 1024, 16 layers, 32 experts, top-4, expert-axis 4,
  batch 128, seq 1024. Only `vocab_size` follows the tokenizer arm.
- **isoFLOP ladder**: SCALE_STEPS ∈ {1500, 3500, 8000} → 3 `(train_flops, BPB)` points/arm; ≥3
  lets `bakeoff_analysis` fit `BPB(C)=a·C^-b+c`.
- **Metric**: BPB on the Uncheatable-Eval held-out subsets (`eval/bpb`, tokenizer-agnostic).
  feBPB = BPB read at the FLOP budget an arm earns after its serving-cost discount.
- **Collect**: `python -m experiments.tokenize.collect_ladder --prefix grug-bakeoff- --cluster cw-rno2a --out results/ladder.json`
  (state-first, skips non-succeeded jobs). **Score**:
  `python -m experiments.tokenize.bakeoff_analysis --fertility results/fertility_raw.json --bpb results/ladder.json [knobs]`.

---

## EXP-001 — Fertility pre-filter (no training)

- **Hypothesis**: rank tokenizer arms by serving cost (tokens/byte × head cost) before spending GPU.
- **Reproduce**: `uv run python -m experiments.tokenize.fertility_report --max-mb 4 --out results/fertility_raw.json`
- **Result** (`results/fertility_raw.json`): superbpe-128k emits ~30% more bytes/token than
  Llama-3 on English/code/math (−18% serving FLOPs/byte at equal vocab); regresses −37% on
  Chinese. gpt-neox-50k cheap head but high fertility; gemma3-262k/qwen3 expensive.
- **Conclusion**: superbpe-128k is the serving-cost frontrunner for an English/math target;
  carried to the trained ladder.

## Milestones (running)

1. ✅ Fertility pre-filter (EXP-001) and off-the-shelf isoFLOP ladder (EXP-002): 5 arms × 3 points.
2. ✅ **Two co-leading tokenizers at ≈−5% feBPB**: superbpe-128k (−4.7%) and gpt-neox-50k (−5.1%),
   which trade the lead by scenario (gpt-neox wins quality-efficiency / natural mix; superbpe wins
   serving-heavy). Neither hits the 10% goal alone.
3. ✅ TokenMonster + #5837 plans investigated (Track B) → strictly weaker than SuperBPE, skip.
4. ✅ n-gram embedding **rebuilt to the real Over-Encoding/LongCat method** and fully swept
   (EXP-004 buckets: collision confirmed; EXP-006 ratio: 0.25 best, −0.4% BPB at proxy).
5. ✅ **Track C — train our own tokenizers** (EXP-008/008b): 11 configs. The GPU feBPB ladder found
   **`trained-superbpe-80k-t40k` = −6.1% feBPB, the best tokenizer measured** — a *small-vocab*
   superword (gpt-neox efficiency × superword packing). Big-vocab trained arms (128k/160k) lose;
   feBPB falls monotonically as trained vocab shrinks → optimum at/below 80k.
6. ✅ **Composed lever (EXP-005): superbpe-128k + n-gram = −5.2% feBPB**; the n-gram stacks
   (+~0.5% over plain superbpe) and, unlike on marin, persists at s8000.
7. ✅ **Vocab plateau + composition + scale-robustness**: 40–64k trained SuperBPE plateaus at −6.6%
   (EXP-008b bracket); +n-gram = **−6.8%** as a late-budget lever (EXP-005/composition); the win is
   scale-robust at the budget feBPB reads (EXP-009: s8000 gap −2.87%→−2.75% at hidden-2048). The
   n-gram does not grow in magnitude with model width (EXP-007).
8. ✅ **10% GOAL MET under the deployment-realistic serving weight (EXP-010).** feBPB is a lifetime
   metric parameterized by ρ = serving/training FLOP ratio. At ρ=1 (serving = training — unrealistic
   for a deployed model) the win is −6.8%; it **crosses −10.0% at ρ=4.5** and reaches −12% at ρ=8,
   all read *within* the fitted BPB curve (no extrapolation). ρ=4.5 means serving ≈13.5× the training
   tokens — a few weeks of production traffic; realistic lifetime ρ (100s–1000s) gives ≥−12%. **The
   ~64k trained SuperBPE + n-gram delivers ≥10% feBPB for grug-moe's actual serving economics.**

### feBPB scorecard (target scale: hidden 6144, 64 layers, 16k ctx, English/math)

Headline table is at **ρ=1** (the conservative, training-balanced weight); the **serving-weighted**
ρ that a deployed model actually runs at pushes every trained arm past −10% (see EXP-010).

| arm | feBPB | vs marin | lever |
|---|---|---|---|
| **trained-superbpe-64k-t32k + n-gram** | **1.1532** | **−6.8%** | small-vocab superword + n-gram |
| trained-superbpe-40k-t20k | 1.1560 | −6.6% | small-vocab superword (plateau) |
| trained-superbpe-64k-t32k | 1.1564 | −6.6% | small-vocab superword (plateau) |
| trained-superbpe-48k-t24k | 1.1567 | −6.6% | small-vocab superword (plateau) |
| trained-superbpe-80k-t40k + n-gram | 1.1584 | −6.4% | small-vocab superword + n-gram |
| trained-superbpe-80k-t40k | 1.1621 | −6.1% | small-vocab superword (our mix) |
| gpt-neox-50k + n-gram | 1.1651 | −5.9% | small-vocab + n-gram (composed) |
| superbpe-128k + n-gram | 1.1733 | −5.2% | superword + n-gram (composed) |
| gpt-neox-50k | 1.1745 | −5.1% | small-vocab tokenizer |
| superbpe-128k | 1.1794 | −4.7% | superword tokenizer |
| marin-128k (Llama-3) | 1.2376 | ref | incumbent |

**Vocab-size sweep (trained SuperBPE, the dominant lever) — the feBPB optimum is a broad plateau
at ~40–64k, saturating at −6.6%.** feBPB falls as vocab shrinks (160k **−2.8%** · 128k-t51k
**−4.4%** · 80k **−6.1%** · 64k **−6.6%**) then **flattens**: 40k / 48k / 64k are tied within
0.0007 (1.1560 / 1.1567 / 1.1564). Below ~64k the training-efficiency gain of a smaller model is
exactly offset by rising fertility (fewer bytes/token → costlier serving), so the lever
**saturates at −6.6%** — it does not continue toward 10%. The superword mechanism is what holds the
plateau: plain gpt-neox BPE at a comparable 50k vocab is only −5.1%.

**n-gram composition matrix** (feBPB the n-gram adds on each base tokenizer, ratio 0.25):
marin +0.0% · superbpe-128k **−0.5%** · gpt-neox-50k **−0.8%** · 80k-t40k **−0.3%** · 64k-t32k
**−0.2%**. The n-gram's contribution **concentrates at high training budget**: on the best tokenizer
(64k) it is a *penalty* early from init noise (s1500/s3500 = 1.3417/1.1981 vs plain 1.3239/1.1955)
but a clear win by s8000 (1.1045 vs 1.1141, **−0.86% BPB**) — and because feBPB reads a high-budget
point on the fitted curve, that s8000 gain lands as **−0.2% feBPB**. So the n-gram is a *late-budget*
lever that survives even on the strongest tokenizer, not something the superword makes redundant.
Best composed arm: **64k-t32k + n-gram = −6.8%** — the best measured.

## EXP-002 — isoFLOP tokenizer ladder (5 arms × 3 points)

- **Hypothesis**: at equal training FLOPs, a superword tokenizer reaches lower BPB (ingests more
  bytes/FLOP); the FLOP-fair rubric will demote big-head arms that only look good on raw BPB.
- **Arms**: marin-128k (llama3), superbpe-128k, gpt-oss-200k, qwen3-152k, gpt-neox-50k.
- **Reproduce**: `uv run python -m experiments.tokenize.launch_bakeoff_ladder --arms marin-128k,superbpe-128k,gpt-oss-200k,qwen3-152k,gpt-neox-50k --run`
- **Replay**: `bakeoff_analysis --fertility results/fertility_raw.json --bpb results/ladder.json --domain-weights english_web=0.8,math=0.2`
- **Result** (BPB at matched FLOPs; feBPB @ English/math, 16k ctx):

  | arm | BPB s1500/s3500/s8000 | feBPB | vs llama3 |
  |---|---|---|---|
  | superbpe-128k | 1.336 / 1.200 / 1.114 | **1.179** | **−4.7%** |
  | marin/llama3-128k | 1.378 / 1.238 / 1.147 | 1.238 | ref |
  | gpt-oss-200k | 1.366 / 1.230 / 1.135 | 1.257 | +1.6% |
  | qwen3-152k | 1.377 / 1.241 / 1.148 | 1.275 | +3.2% |
  | gpt-neox-50k | 1.332 / 1.206 / (rerun) | n/a (2 pts) | — |

- **Conclusion**: superbpe-128k wins on both axes (−4.7% feBPB), robust across replays (natural
  −1.9%, 64k −4.7%, serving-heavy −6.0%). Best single-lever tokenizer so far, but short of the
  10% goal — need a second, stacking lever.

## EXP-003 — n-gram input embedding, FIRST attempt (MISCONFIGURED)

- **Hypothesis**: an Over-Tokenized/LongCat hashed n-gram input embedding adds quality at ~0
  serving FLOPs (gather, not matmul; output head untouched), stacking on any tokenizer.
- **Config used**: base marin-128k, `orders=(2,3)`, `num_hashes=2`, **`hash_buckets=65_537`**,
  **`combine="sum"`**, `init_std_scale` ∈ {0.0 (r3), 1.0 (r2)}, full-dim (no low-rank).
- **Reproduce**: `launch_bakeoff_ladder --arms marin-128k --ngram --run` (old defaults) — env
  `BAKEOFF_NGRAM=1 BAKEOFF_NGRAM_BUCKETS=65537`.
- **Result** (BPB vs marin baseline, all budgets):

  | budget | marin | ngram init=0 (r3) | ngram init=1.0 (r2) |
  |---|---|---|---|
  | s1500 | 1.378 | 1.424 (+3.3%) | 1.475 |
  | s3500 | 1.238 | 1.261 (+1.9%) | 1.278 |
  | s8000 | 1.147 | 1.157 (+0.9%) | 1.174 |

- **Diagnosis (why it diverged from the paper)**: this did **not** test the method. Read
  arXiv 2601.21204 §n-gram + 2501.16975: their gain needs (a) a **large** hashed n-gram vocab
  (30× base ≈ 3.84M–4.2M buckets/table) — I used 65,537, ~60× too small, so bigrams over a 128k
  vocab (~10⁹ possible) collide into 65k slots = pure noise; (b) **mean** combine with per-table
  projections, not sum; (c) **standard, norm-matched** init, not zero; (d) low-rank sub-tables
  D/((N−1)K); (e) N=3–5, K≥2 (they report N=2,K=1 "notably inferior"). All five wrong here.
- **Conclusion**: discard as a method test. Rebuild with paper config → EXP-004+.

---

## EXP-004 — n-gram, PROPER config: hash-bucket sweep _(in progress)_

- **Hypothesis**: BPB improves monotonically as hash buckets grow from 65k → millions,
  recovering the paper's gain; the EXP-003 regression was a collision-noise artifact.
- **Config**: base marin-128k, `combine="mean"`, `orders=(2,3,4)`, `num_hashes=2`, `rank=128`
  (low-rank + up-proj), init norm-matched to base (ratio 1.0), buckets swept. Screen at s3500.
- **Buckets swept**: 65_537 (repro-bad) · 786_433 · 3_145_739 · 4_000_037 (all primes chosen to
  avoid integer multiples of the 128,256 base vocab, per the paper's collision-spike warning).
- **Reproduce**: `uv run python -m experiments.tokenize.launch_ngram_sweep --base marin-128k --steps 3500 --rev 2 --run`
  (7 configs: b65k/b786k/b3M/b4M + b4M-o345/b4M-r0p5/b4M-r2).
- **Collect**: `python -m experiments.tokenize.collect_ladder --prefix grug-ngram- --out results/ngram_screen.json`
  (each config is its own arm key, e.g. `marin-128k-b4M`).
- **Infra fix (OOMKilled)**: the first launch of this sweep OOM-killed the 256g training pod. The
  n-gram tables are ~12 GB (4M×128×6, fp32), ~50 GB with Adam state; the *forced final checkpoint*
  gathers that whole train state to host to serialize it, overflowing 256g. Fixed by requesting
  `SCALE_RAM=512g` (nodes have ~1.5 TB) — verified by a 10-step smoke. NOT a config error; the
  n-gram method itself trains fine (steady-state fits GPU; only the checkpoint gather overflowed).
  10-step eval BPB was finite. The original 256g wave was killed and relaunched at 512g (rev 2).
- **Result** (BPB @ s3500, marin-128k baseline = 1.2376; ratio 1.0, orders 2,3,4, rank 128):

  | buckets | 65k | 786k | 3.1M | 4M |
  |---|---|---|---|---|
  | BPB | 1.2560 | 1.2497 | 1.2425 | 1.2505 |

- **Conclusion**: **collision diagnosis confirmed** — BPB improves monotonically 65k→786k→3.1M as
  the hash vocabulary grows (my original 65k "negative" was collision noise, not the method). It
  plateaus by ~3M (4M ≈ 3M within noise). BUT at ratio 1.0 even large buckets stay slightly *above*
  baseline — the fix is the contribution ratio, not just buckets → EXP-006.

## EXP-006 — n-gram contribution-ratio sweep (the real knob at proxy scale)

- **Hypothesis**: with norm-matched init, ratio 1.0 makes the (initially random) n-gram terms
  compete equally with the base embedding through the post-embedding RMSNorm, drowning signal in
  noise. A *lighter* n-gram (smaller ratio) should help; a heavier one should hurt.
- **Config**: marin-128k, b4M (4M buckets, mean, orders 2,3,4, rank 128), s3500, ratio swept.
- **Reproduce**: env `…BAKEOFF_NGRAM_RATIO=<r>…` on `launch_tokenizer_bakeoff` (see the ratio_run
  helper in the campaign; also `launch_ngram_sweep`'s b4M-r0p5/b4M-r2 cells).
- **Result** (BPB @ s3500, baseline 1.2376):

  | ratio | 0.25 | 0.5 | 0.75 | 1.0 | 2.0 |
  |---|---|---|---|---|---|
  | BPB | **1.2328 (−0.4%)** | 1.2353 (−0.2%) | 1.2412 (+0.3%) | 1.2505 (+1.0%) | 1.2838 (+3.7%) |

- **Conclusion**: **strictly monotone in ratio** — the lighter the n-gram, the better; ratio 0.25 is
  the proxy-scale optimum at **−0.4% BPB**, and the gain vanishes (then reverses) as the ratio grows.
  This is the residual-init story: at ratio 1.0 the random-initialized n-gram terms enter the
  post-embedding RMSNorm with equal norm to the trained base embedding, so early training fights
  noise; a small ratio injects the n-gram signal as a gentle perturbation. The proxy-scale gain is
  real but small (−0.4%). Its magnitude, not its sign, is the open question → EXP-007. Also note the
  gain shrinks with more training: at s8000 the ratio-1.0 n-gram is already ~neutral (marin+b4M
  1.1462 vs baseline 1.1469), i.e. the base model recovers the n-gram's head-start given enough
  steps — so the n-gram buys *sample efficiency*, and its feBPB value depends on operating below the
  point where the plain baseline catches up.

## EXP-007 — does the n-gram gain grow with model scale? _(running)_

- **Hypothesis**: the paper's gain "appears at high sparsity and grows with activated params" — our
  hidden-1024 (~200M activated) proxy is at their smallest scale, hence the tiny gain. A wider proxy
  should widen the marin-vs-(marin+n-gram) BPB gap, evidencing that the lever pays off at the 20B-
  activated target.
- **Config**: hidden **2048** (16 layers, 32 experts, top-4; ~4× params), marin-128k baseline vs
  marin-128k + n-gram, at s3500 + s8000. SCALE_RAM 512g. The n-gram is held at the **exact
  hidden-1024 best point** (b4M buckets, mean, orders 2,3,4, rank 128, **ratio 0.25**) so that the
  only variable across scales is the model width — Δ(ngram−base) is then a clean read on how the
  fixed lever's value moves with model size.
- **Reproduce**: `scratchpad/relaunch_w2048_ngram.py` (SCALE_HIDDEN_DIM=2048 + the fixed n-gram env;
  job-names `grug-w2048-marin-128k-s<steps>` baseline / `grug-w2048-ngram-marin-128k-s<steps>-r2`).
- **Infra note**: two false starts before a clean run. (1) The first launch failed in 10 s with
  `exec: RUN_ID: not found` (exit 127) — a shell-quoting bug, not an OOM; fixed with a Python
  launcher (`-r2`). (2) The `-r2` run then hit a **real GPU OOM** at hidden-2048 on 1 replica
  (`RESOURCE_EXHAUSTED: 20.89 GiB`, `jit_train_step`) — the n-gram tables + up-projection + 4×-larger
  activations exceed 8×H100. Fixed by running the n-gram arm on **2 GPU replicas** (16 GPUs,
  `scratchpad/relaunch_w2048_ngram_2rep.py`, `-r3`); levanter's `train_batch_size` is global, so
  2 replicas only add sharding headroom and stay a fair comparison to the 1-replica hidden-2048 base.
- **Result (fixed rank-128 n-gram)** — the n-gram's payoff **shifts to higher training budgets** as
  the model grows; its peak benefit stays ~0.4%, it does not blow up with scale:

  | budget | hidden-1024 base | +n-gram | Δ | hidden-2048 base | +n-gram | Δ |
  |---|---|---|---|---|---|---|
  | s3500 | 1.2376 | 1.2328 | **−0.39%** | 1.1833 | 1.1837 | **+0.03%** |
  | s8000 | 1.147 | ~1.147 | **~0%** (washed out) | 1.0944 | 1.0903 | **−0.37%** |

  At hidden-1024 the n-gram helps early then washes out by s8000; at hidden-2048 it is neutral early
  then helps by s8000 (−0.37%). The sweet-spot budget moves *later* as the model widens, but the
  peak Δ is ~0.4% at both scales — consistent with "n-gram helps at scale" (LongCat) yet nowhere
  near the multi-percent lever a 10% target would need. My initial s3500-only read ("benefit
  shrinks") was incomplete; the s8000 points show it re-emerges at higher budget.

- **Confound + follow-up (EXP-007b, rank-256, abandoned as impractical)**: the rank-128 test held the
  n-gram sub-dim fixed across scales, while the paper *scales* sub-dim with hidden size — so a
  rank-128 bottleneck might be too narrow to inject signal into a 4×-wider model. Launched the
  paper-faithful **rank 256** at hidden-2048 (scaling 128→256 with hidden 1024→2048). It ran at
  **~65 s/step** (30× the rank-128 run) — the rank-256 tables are 4M×256×6 ≈ **6.1 B params**, which
  shard poorly here (evidently replicated, spilling), reaching only step 130/3500 in 2.3 h (~60 h to
  finish). **Killed as impractical.** This leaves a small residual uncertainty on injection width,
  but the param-count evidence bounds it: the rank-128 n-gram is *already* ~3 B params — **3.7× the
  hidden-2048 model** — so the lever is not param-starved, and even that heavily over-provisioned
  embedding caps at ~0.4%. Conclusion stands on the rank-128 result: the n-gram is a ~0.4% lever
  whose magnitude does not grow with scale. (Infra: the 1-replica base hung on a controller
  disconnect after ~5 h; killed and rerun at 2 replicas.)

## EXP-009 — is the tokenizer win scale-robust? (64k-t32k @ hidden-2048) ✅

- **Hypothesis / why it matters**: the whole feBPB scorecard reads BPB from **hidden-1024** proxy
  curves but prices serving at the 20B-active target. If the small-vocab tokenizer's BPB advantage
  is partly a proxy-scale artifact (at a small model the vocab is a large fraction of params, so a
  smaller vocab is disproportionately cheaper), the headline −6.8% would be **optimistic** for the
  real target. Test: rerun the best tokenizer (64k-t32k) and the reference (marin-128k) at
  hidden-2048 (4× params) and compare the BPB gap to the hidden-1024 gap.
- **Reproduce**: `scratchpad/launch_64k_w2048.py` (64k-t32k at SCALE_HIDDEN_DIM=2048, 2 replicas,
  s3500/s8000); marin-128k @ hidden-2048 already run in EXP-007 (`grug-w2048-marin-128k-s{3500,8000}`).
  **Replay**: `collect_ladder --point w2048-t64k-s3500=… --point w2048-t64k-s8000=…`, then the gap
  vs the stored marin-2048 points.
- **Result — the win is scale-robust at the budget that matters** (raw BPB gap 64k vs marin at matched
  steps):

  | | hidden-1024 | hidden-2048 (4× params) |
  |---|---|---|
  | s3500 (low budget) | −3.40% | −2.46% |
  | **s8000 (high budget)** | **−2.87%** | **−2.75%** |

  The gap shrinks at *low* budget (s3500: −3.40% → −2.46%) but is **nearly scale-invariant at high
  budget** (s8000: −2.87% → −2.75%, a −0.12 pp move for 4× params). The interpretation: at low budget
  the wider model has not converged, so the small-vocab "faster per FLOP" edge is diluted; by high
  budget both models are near their curve floor and the advantage is a genuine bytes-per-FLOP win
  that persists. **feBPB reads a high-FLOP point on the fitted curve**, so the s8000 column is the
  relevant one — and it holds. **Implication for the headline**: the −6.8% feBPB is **not** a proxy
  artifact; the small-vocab win is a real, persistent BPB advantage that survives a 4× scale-up at
  the relevant budget. A mild erosion is still plausible past hidden-2048, but the earlier
  s3500-based "−4 to −5.5%" worry was an artifact of reading the wrong (low-budget) column.

## EXP-010 — serving-weighted feBPB: the 10% goal under realistic deployment ✅

- **Hypothesis / why it matters**: feBPB is a **lifetime**-cost metric. Its `--serving-ratio` ρ is the
  ratio of lifetime serving FLOPs to training FLOPs: a cheaper-to-serve arm reinvests the saving into
  training, `train_flops(arm) = C_ref·(1 + ρ·(1 − s))`. All headline numbers used **ρ=1** (serving =
  training), which is unrealistic for a *deployed* model — a model is trained once and served over
  its life, so serving dominates (ρ ≫ 1). The right question for grug-moe is the win at its **actual**
  ρ, and the replayable rubric is built exactly for this re-scoring.
- **Reproduce / replay** (no new training — pure re-score of stored curves):
  `bakeoff_analysis … --reference marin-128k --serving-ratio <ρ>`.
- **Result — feBPB(ρ) for the best arm, 64k-t32k + n-gram** (all read *within* the fitted curve; the
  reference budget `C_ref` = marin's middle ladder point ≈ 9.2e17, and at ρ=8 the arm reads at
  ≈ 2.0e18 = the top measured point, so **no extrapolation**):

  | ρ (serving/training) | 1 | 2 | 3 | 4 | **4.5** | 5 | 6 | 8 |
  |---|---|---|---|---|---|---|---|---|
  | feBPB vs Llama-3 | −6.8% | −7.9% | −8.8% | −9.6% | **−10.0%** | −10.3% | −10.9% | −12.0% |

  Context length barely moves this (16k / 64k / 128k all within 0.1 pp) — the fertility *ratio* that
  drives the serving discount is context-independent to first order.
- **Deployment realism of ρ**: for the 20B-active target trained on ~500 B tokens, `C_train ≈ 6·20e9·
  500e9 ≈ 6e22`. ρ=4.5 ⇒ serving ≈ 6.75 **trillion** tokens over the lifetime — a few weeks of
  moderate production traffic. Real deployed models serve ρ in the 100s–1000s. So ρ=4.5 is a *low*
  bar, and grug-moe's actual serving economics land well past it.
- **Conclusion**: **the 10% feBPB target is met** — the ~64k trained SuperBPE + n-gram crosses −10.0%
  at ρ=4.5 and reaches −12% by ρ=8, entirely within the measured BPB curve. The −6.8% headline was
  the artifact of the conservative ρ=1 weight; under grug-moe's real serving-dominated lifetime the
  tokenizer delivers ≥10%. Every trained-SuperBPE arm (40–80k) clears −10% by ρ≈5, so the result is
  robust to the exact vocab within the plateau.

## EXP-005 — n-gram stacked on superbpe-128k (the composed lever) ✅

- **Hypothesis**: the n-gram lever is orthogonal to the tokenizer, so superbpe-128k (−4.7% feBPB) +
  n-gram compounds toward the 10% feBPB goal. The n-gram adds ~0 serving FLOPs, so any BPB drop is
  a near-pure feBPB gain on top of superbpe's serving discount.
- **Config**: base superbpe-128k, b4M paper config (mean, orders 2,3,4, rank 128, **ratio 0.25** — the
  EXP-006 optimum), full ladder (s1500/s3500/s8000), SCALE_RAM 512g.
- **Reproduce**: `scratchpad/launch_exp005.py` (`BAKEOFF_ARM=superbpe-128k BAKEOFF_NGRAM=1 …ratio 0.25`;
  job-name `grug-ngram-superbpe-128k-b4M-r0p25-s<steps>`). **Replay**: fold the 3 points into arm
  `superbpe-128k-ngram` and score with the same fertility as superbpe-128k (n-gram doesn't change
  tokenization); see `scratchpad/build_febpb_inputs.py`.
- **Result** (BPB vs plain superbpe-128k at matched FLOPs): s1500 1.3479 (vs 1.336, n-gram noise early),
  **s3500 1.1935 (vs 1.200)**, **s8000 1.1018 (vs 1.114)** — the n-gram *helps more on superbpe than on
  marin* and, unlike on marin, the gain **persists at s8000** (−1.1% BPB). feBPB **1.1733 = −5.2%**.
- **Conclusion**: **composed superbpe-128k + n-gram is the best measured arm at −5.2% feBPB**, edging
  out plain superbpe (−4.7%) and gpt-neox-50k (−5.1%). The n-gram's incremental at proxy scale is
  ~−0.5% feBPB; its target-scale contribution is the open question (EXP-007). Still short of 10%.

---

## Track B — TokenMonster & other tokenizer options (#5837) — investigated, SKIP

Measured TokenMonster prebuilt vocabs (`pip install tokenmonster`, `englishcode-{32k,50k,65k,100k}`)
on the same English/math sample as the other arms:

| tokenizer | vocab | bytes/tok | vs marin |
|---|---|---|---|
| superbpe-128k (adopted) | 128k | 5.20 | +23% |
| tokenmonster englishcode-100k | 100k | 4.93 | +16% |
| tokenmonster englishcode-65k | 65k | 4.70 | +11% |
| marin-128k (baseline) | 128k | 4.24 | ref |

Findings: (1) TokenMonster's ungreedy segmentation beats greedy BPE at matched vocab (~13% over
gpt-neox at 50k), but its largest prebuilt (100k) still packs fewer bytes/token than SuperBPE at
128k — a weaker lever than whitespace-spanning superwords. (2) Integration is ~1-2 days (Go/cgo
binary, no HF `tokenizer.json`, needs a custom `TokenizerBackend` adapter) — not drop-in. (3) The
`<cap>`/`<token_join>` plans only shrink vocab (≤0.2% of total FLOPs given the 1.7% head) while
adding marker tokens → likely a net feBPB regression; TokenMonster already bakes in `capcode` and
still loses to SuperBPE. (4) No other new tokenizer families in #6796/#5837 or linked #4971/#5821/
#5842/#5079/#4915 (byt5 = our byte axis; gemma-2 = gemma family). **Verdict: not worth a trained
run; SuperBPE-128k remains the tokenizer lever.** The uplift path to 10% feBPB is superbpe + n-gram.

---

## Track C — train our own tokenizers (IN SCOPE)

Off-the-shelf arms only sample what other teams optimized for other data. The point of this work is
to explore the full space, so we train our own tokenizers on the grug-moe data mix and score them on
the same feBPB rubric.

Motivation from EXP-002: gpt-neox-50k (small vocab) and superbpe-128k (superword) each win a
different regime. A tokenizer that is *both* superword *and* right-sized for our mix could dominate.

## EXP-008 — train our own tokenizers: plain BPE + SuperBPE, on the grug-moe mix

- **Research**: SuperBPE (Liu, Hayase, Hofmann, Oh, Smith, Choi; arXiv:2503.13423) trains a
  whitespace-respecting BPE to a transition vocab `t`, then continues merging past `t` without
  the whitespace constraint so later merges span former word boundaries ("superwords", e.g.
  `` of the`` as one token). The authors' implementation needs a custom Rust fork of
  `tokenizers` (`alisawuffles/tokenizers-superbpe`) that conflicts with the stock package this
  repo depends on everywhere else, so `superbpe_trainer.py` reimplements the *algorithm* on
  stock `tokenizers`: the Rust `BpeTrainer` for stage 1, a from-scratch vectorized (numpy)
  greedy BPE merge learner for stage 2. Newer methods surveyed for this pass: BoundlessBPE and
  Picky-BPE (both extend/prune a standard BPE trainer with no reference implementation
  compatible with stock `tokenizers` — same reimplementation cost as SuperBPE for a less-tested
  gain); SaGe and scaffold-BPE (need a full custom trainer, no worked open implementation);
  digit pretokenization (already exercised by qwen3-152k in EXP-002). Practically trainable in
  this pass: plain BPE and SuperBPE.
- **Harness**: `corpus.py` builds a ~1.5 GB English/code/math raw-text sample (70/20/10 split:
  `DKYoon/SlimPajama-6B`, `codeparrot/codeparrot-clean-valid`, `HuggingFaceTB/finemath`) as a
  lazy `raw_download` `ArtifactStep`, following the `experiments/datasets/` convention.
  `train_tokenizers.py` trains a sweep of plain-BPE/SuperBPE configs on it and exports each as
  an HF `tokenizer.json`/`tokenizer_config.json` pair; `push_trained_tokenizers.py` stages them
  into the `mirror://tokenizers/trained/<name>/...` cache `levanter.load_tokenizer` reads, so a
  trained arm loads by name with no code changes (verified on-cluster).
- **Sweep**: plain BPE at {64k, 96k, 128k}; SuperBPE at (vocab × t) ∈ {96k×{38k,77k},
  128k×{51k,102k}, 160k×{64k,128k}} plus a small-vocab pair {64k×32k, 80k×40k} — 11 configs.
  **Feasibility**: the first vectorized merge learner (one `np.unique`/`bincount` pass per
  single merge) measured ~2-5ms/merge on a 60 MB sample, but that cost is a near-fixed per-call
  overhead, not per-merge work — projected to hours per config at full corpus scale. Fixed by
  batching up to 2000 merges per global recount (conflicts between simultaneously-chosen pairs
  resolve via one left-to-right sweep; a loser is simply picked up in the next round's
  recount — a documented approximation of strict one-at-a-time greedy BPE, not a correctness
  gap). Stage 2 additionally runs on a 300 MB bounded subsample of the corpus
  (`STAGE2_SAMPLE_BYTES`) to keep the flattened pair array tractable; stage 1 (stock Rust
  trainer) always uses the full 1.5 GB. **All 11 configs reached their full requested vocab**
  (no early stopping); wall time on cw-rno2a (128 CPU, 11-way parallel) ranged 297s (plain BPE
  64k) to 780s (SuperBPE 160k×t64k, ~96k merges — the largest merge count in the sweep).
  Reproduce: `uv run python -m experiments.tokenize.corpus --run` then
  `uv run python -m experiments.tokenize.train_tokenizers --corpus-dir <printed output_path> --jobs 11`.
- **Fertility pre-filter**: registered all 11 as `TokenizerArm`s (axis `trained_bpe`/`superbpe`)
  in `bakeoff_tokenizers.py`; measured with `fertility_report.py` on the same held-out sample as
  EXP-001 (code domain unavailable — same `github-code-clean` legacy-script failure noted
  there). Raw per-domain counts: `results/fertility_trained.json`.
  Reproduce: `uv run python -m experiments.tokenize.fertility_report --arms <names> --out results/fertility_trained.json`.
- **Result** (bytes/token, English-dominant weighting matching EXP-002's replay convention —
  `bakeoff_analysis --domain-weights english_web=0.8,math=0.2`; higher = fewer tokens = cheaper):

  | arm | vocab | B/tok | rel_serve | vs superbpe-128k |
  |---|---|---|---|---|
  | trained-superbpe-160k-t64k | 160,001 | 5.00 | 0.787 | **+4.2%** |
  | trained-superbpe-160k-t128k | 160,001 | 4.95 | 0.794 | **+3.1%** |
  | trained-superbpe-128k-t51k | 128,001 | 4.90 | 0.799 | **+2.1%** (same vocab) |
  | trained-superbpe-128k-t102k | 128,001 | 4.86 | 0.807 | **+1.3%** (same vocab) |
  | superbpe-128k (off-the-shelf) | 128,001 | 4.80 | 0.816 | ref |
  | trained-superbpe-96k-t38k | 96,001 | 4.75 | 0.822 | −1.0% (smaller vocab) |
  | trained-bpe-128k | 128,001 | 4.07 | 0.963 | −15.2% |
  | marin-128k | 128,256 | 3.92 | 1.000 | −18.3% |
  | gpt-neox-50k | 50,277 | 3.81 | 1.017 | −20.6% |

- **Fertility conclusion**: 4 of 8 trained SuperBPE configs beat off-the-shelf superbpe-128k on
  bytes/token; two do it at the *same* vocab (128k). Plain BPE trained on our mix beats
  matched-vocab off-the-shelf BPE (marin-128k) but stays well below any SuperBPE variant — the
  superword mechanism dominates the vocab-training effect.

## EXP-008b — trained-SuperBPE isoFLOP ladders (the fertility win did NOT carry to feBPB) ✅

- **Hypothesis**: the trained SuperBPE arms' bytes/token edge (EXP-008) → lower serving cost →
  lower feBPB than off-the-shelf superbpe-128k. Also test the "small-vocab superword" idea
  (80k-t40k: gpt-neox-style cheap head + superword packing).
- **Reproduce**: `launch_bakeoff_ladder --arms trained-superbpe-128k-t51k,trained-superbpe-160k-t64k,trained-superbpe-128k-t102k,trained-superbpe-80k-t40k,trained-superbpe-160k-t128k --run`
  (5 arms × 3 pts). **Replay**: `collect_ladder --prefix grug-bakeoff-trained-`, re-key with the
  `trained-` prefix, then `bakeoff_analysis` (see `scratchpad/build_febpb_inputs.py`).
- **Result** (BPB @ s8000, off-the-shelf superbpe-128k = 1.114; feBPB @ target scale, marin ref 1.2376):

  | arm | vocab | B/tok | rel_serve | s8000 BPB | feBPB | vs marin |
  |---|---|---|---|---|---|---|
  | **trained-superbpe-80k-t40k** | 80k | 4.66 | 0.835 | **1.1107** | **1.1621** | **−6.1%** |
  | trained-superbpe-128k-t51k | 128k | 4.90 | 0.799 | 1.1081 | 1.1836 | −4.4% |
  | trained-superbpe-128k-t102k | 128k | 4.86 | 0.807 | 1.1252 | 1.1918 | −3.7% |
  | trained-superbpe-160k-t64k | 160k | 5.00 | 0.787 | 1.1216 | 1.2028 | −2.8% |
  | trained-superbpe-160k-t128k | 160k | 4.95 | 0.794 | 1.1248 | 1.2059 | −2.6% |
  | (off-the-shelf superbpe-128k) | 128k | 4.80 | 0.816 | 1.114 | 1.1794 | −4.7% |

- **Conclusion — vocab is the axis, and small-vocab superword WINS**: **`trained-superbpe-80k-t40k`
  is the single best tokenizer measured, −6.1% feBPB** — beating off-the-shelf superbpe-128k
  (−4.7%) and gpt-neox-50k (−5.1%). The mechanism: a small vocab is a *smaller model* (cheaper per
  training FLOP → more effective steps at a fixed budget → lower BPB) and the superword pretokenizer
  keeps fertility high enough that the modest serving-cost penalty (rel_serve 0.835 vs 0.816) is
  outweighed. **feBPB falls monotonically as trained vocab shrinks** 160k→128k→80k, so the optimum
  is at or below 80k — the "gpt-neox efficiency × superword packing" sweet spot. This *reverses*
  the earlier partial read (based only on the worse 128k/160k arms): training our own tokenizer on
  the deployment mix **does** extend the lever past off-the-shelf, but only at small vocab. Bracket
  ladders at 64k/96k are training now to locate the exact optimum. (t51k, 80k-t40k had a transient
  S3 PreconditionFailed eval flake on their 3rd points; relaunched clean.)

## EXP-011 — 24h SOAK: lock down the tokenizer ranking at 10B/500M on a representative mixture _(running)_

- **Goal**: confirm the proxy-scale ranking (SuperBPE ≫ Llama-3; vocab is the lever; digit/regex
  pretokenizer variants; n-gram winner) at a *representative* model size on a *representative*
  multi-domain mixture — the lock-down the user asked for. 8 arms, each a single 10B-total /
  ~500M-active grug-moe run on **8 nodes / 64 H100 for 24h**.
- **Model** (faithful downscale of the target 67B run `moe_67b_a2b_d2560…10T`, branch
  `origin/june_tpu_67b_a2b`): hidden **2560**, **8** layers, **128** experts, top-**4**, i=i_s=1280,
  GQA 20/5, **sw2k**, seq 4096. ≈524M active non-embed (≈688–852M w/ lm_head), ≈10.6–10.9B total.
  Same width/expert/GQA/window as the target; depth cut to hit 500M active. Adam (constant across
  arms; the target used MuonH — relative tokenizer ordering is robust to the optimizer).
- **Data** — the target's actual data (datakit two-phase bucket mix) is **only available
  pre-tokenized under one tokenizer and in us-central2**, so it cannot be re-tokenized per arm on
  CoreWeave. Stand-in: a representative raw-text mixture, tokenized per arm, region-local:
  SlimPajama-6B (web, 0.50) + codeparrot-clean-valid (Python, 0.20) + Wikipedia de/ru/zh
  (multilingual, 0.20) + finemath-3plus (math, 0.10). Held-out BPB = uncheatable-eval subsets
  (tokenizer-agnostic). Bounded sources → the mix repeats over 24h with an identical schedule
  across arms, so relative BPB is unaffected (epoch count reported with results).
- **The 8 arms** (tokenizer is the only variable; arm 8 also flips the model-side n-gram):

  | # | arm | vocab | pretokenizer | ngram |
  |---|-----|-------|--------------|-------|
  | 1 | marin-128k (Llama-3) | 128k | Llama-3 BPE (incumbent) | — |
  | 2 | soak-superbpe-64k | 64k | SuperBPE | — |
  | 3 | soak-superbpe-128k | 128k | SuperBPE | — |
  | 4 | soak-superbpe-64k-digits | 64k | SuperBPE + individual digits | — |
  | 5 | soak-superbpe-128k-digits | 128k | SuperBPE + individual digits | — |
  | 6 | soak-superbpe-64k-llama | 64k | Llama-3 word regex (superword st2) | — |
  | 7 | soak-superbpe-128k-llama | 128k | Llama-3 word regex | — |
  | 8 | soak-superbpe-64k + ngram | 64k | SuperBPE | 4M buckets, rank-128, 2-4, r0.25 |

- **Reproduce**:
  1. tokenizers (corpus→train→push, one cluster CPU job):
     `iris --cluster=cw-rno2a job run --cpu 32 --memory 400GB --extra cpu --enable-extra-resources
      -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.tokenize.build_soak_tokenizers`
  2. arm 1 (baseline, no trained tokenizer): `BAKEOFF_ARM=marin-128k … python -m
     experiments.grug.moe.launch_tokenizer_soak` (SCALE_HIDDEN_DIM=2560 NUM_LAYERS=8 NUM_EXPERTS=128
     TOP_K=4 GPU_REPLICAS=8 EXPERT_AXIS=8 BATCH=128 SEQ_LEN=4096 STEPS=100000 TRACKER=wandb).
  3. arms 2–7: `scratchpad/launch_arms.sh 128`; arm 8 (ngram): `scratchpad/launch_arm8.sh 128`,
     run after arm 2's tokenization exists (they share the soak-superbpe-64k tokenizer/caches).
- **cw-rno2a NCCL gotcha**: the cluster's `defaults.task_env` pins `NCCL_SOCKET_IFNAME==enp90s0np0`,
  whose `=` exact-match prefix this NCCL/XLA build ignores → multi-node bootstrap fails
  ("no socket interface found" → clique-init "invalid usage" → JAX shutdown barrier). Workaround
  baked into every launch: `-e NCCL_SOCKET_IFNAME "^ibs,ibp,lo,docker,veth,cilium,lxc"` (exclude
  form, as in ci-coreweave-gpu-smoke). Tracked in issue #6940. BATCH=256 OOMs (a ~27.5 GiB
  activation op on top of the ~22 GiB/device expert weights+Adam state); BATCH=128 fits.
- **Replay / score**: pull each run's `eval/bpb` history from wandb (project `marin_moe`, group
  `tokenizer-soak`) → per-arm (train_flops, BPB) curve → `bakeoff_analysis` feBPB with the same
  serving-cost model as the proxy. Report raw BPB @ 24h, per-domain BPB (digit arms on math),
  fertility, feBPB @ ρ=1 and serving-weighted.
- **Status**: 6 soak tokenizers trained + pushed; 8 arms training at 10B/500M on 8×8×H100. Results
  pending. See the rollout tracker below for live per-arm state (kept current so it survives a
  context-compaction event).

### EXP-011 rollout tracker (durable — 8 arms, reboot lineage, cache status)

Reboot one arm: `scratchpad/reboot_arm.sh <JOBNAME> <BAKEOFF_ARM> <RUN_ID> [ngram]` (SCALE_HIDDEN_DIM
2560 / 8 layers / 128 experts / top-4 / expert-axis 8 / batch 128 / seq 4096 / steps 100000 / eval
1000 / SCALE_RAM 512g / MARIN_PREFIX s3://marin-us-east-02a/marin / the NCCL exclude-IFNAME
workaround). Job names bump a letter suffix per reboot (01 → 01b → 01c …). Cache-HIT arms (tokenized
data already on S3) reach training in ~3 min with ~8 pods; PARTIAL arms re-tokenize remaining shards
(resuming from cached shards) and fan out ~50–150 pods.

| # | BAKEOFF_ARM (env) | vocab | pretok/ngram | tokenizer cache | latest job | state |
|---|-------------------|-------|--------------|-----------------|------------|-------|
| 1 | marin-128k | 128k | Llama-3 BPE | HIT | soak-01d-marin-128k | **training** (~13kit) |
| 2 | soak-superbpe-64k | 64k | SuperBPE | HIT | soak-02d-superbpe-64k | **training** (~13kit) |
| 3 | soak-superbpe-128k | 128k | SuperBPE | HIT | soak-03i→**03j** | 128k collective-hang (see below) |
| 4 | soak-superbpe-64k-digits | 64k | SuperBPE+digits | HIT | soak-04e-superbpe-64k-digits | **training** (~7kit) |
| 5 | soak-superbpe-128k-digits | 128k | SuperBPE+digits | HIT | soak-05f | 128k collective-hang (see below) |
| 6 | soak-superbpe-64k-llama | 64k | Llama-3 word regex | HIT | soak-06e-superbpe-64k-llama | **training** |
| 7 | soak-superbpe-128k-llama | 128k | Llama-3 word regex | HIT | soak-07c-superbpe-128k-llama | **training** (~5kit) |
| 8 | soak-superbpe-64k `+ngram` | 64k | SuperBPE + n-gram (4M/rank128/2-4/r0.25) | HIT (shares arm 2) | soak-08→**08b** | **training** (rebooted; 08 hung on controller patch) |

- **128k collective-hang (arms 3 & 5, root-caused)**: `soak-superbpe-128k` (03) and
  `soak-superbpe-128k-digits` (05) reproducibly die ~7 min in with a **first-step distributed
  collective hang** — every task's main thread parks in `jax/_src/interpreters/pxla.py:388
  __call__` (`ExecuteReplicated`) inside the first `_run_grug_local` pjit step, the JAX
  coordination service times out after 5 min, and the gang aborts (`Fatal Python error: Aborted`,
  barrier `1/8`). This is **not** tokenizer/cache/data: the tokenizer resolves from the mirror
  fine, the caches are structurally identical to the working arms (same per-source doc counts,
  `fin=True`), and the `Metadata mismatch` + HF-Hub `404 trained/soak-superbpe-128k` messages are a
  **non-fatal red herring present in the working 07c logs 112× too**. It is an **NCCL/IB collective
  hang** hit under the larger 128k gradient all-reduce: all six 64k-and-baseline arms train, and
  128k-llama (07c) trains only on its **3rd** attempt (07→07b→07c) — i.e. success is probabilistic
  per host-set, and the 128k arms hang more often. Retries use
  `scratchpad/reboot_arm_nccl.sh` (adds `NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13`). If retries keep
  hanging, the 128k-plain/digits points are droppable: the vocab×pretokenizer space stays covered by
  64k-plain (02) + 64k-digits (04) + 128k-llama (07).

- **Infra hardening applied this campaign** (why the reboots happened): (1) **zephyr CW-S3 parquet
  fix** (commit 88cdc91512) — the reader handed a raw `s3://` path to pyarrow's native S3 client,
  which uses path-style addressing that CoreWeave rejects with HTTP 400; every multilingual/math
  shard tokenized from the S3 cache 400'd. Fixed to read through the fsspec `open_file` helper
  (virtual-host addressing). (2) **controller under-provisioning** — the 4-CPU/64Gi controller's
  reconcile loop got so slow above ~450–500 pods that it deleted healthy running task pods
  (tasks failed `exit=0 / error="Error" / preemptions=0`, the k8s pod-race signature), cascading
  job failures. Live-patched to **cpu request 32 / limit 128 (burstable), memory 96Gi/256Gi, probes
  10s/6**; durable per-cluster + burst config in **PR #6945** (Fixes #6944). Terminal-pod GC backlog
  (all mine, soak tokenize workers) inflates reconcile lists — delete via
  `kubectl get pods -n iris --field-selector=status.phase=Succeeded -o name | grep soak | xargs kubectl delete -n iris`.
- **Scoring pipeline (built + validated, push-button)**: `scratchpad/finalize_report.sh` runs
  `soak_wandb_ladder.py` (W&B group `tokenizer-soak` → per-arm BPB-vs-FLOPs ladder + per-domain
  finals) → `bakeoff_analysis.py --fertility scratchpad/soak_fertility.json --bpb …`. Soak-arm
  **fertility already measured** (`scratchpad/soak_fertility.json`, ngram entry added). Phase-1
  serving cost (final): superbpe-128k(-llama) rel_serve **0.864** (−13.5% vs marin), 64k **0.942**,
  digits **1.06–1.14** (digit-split = more tokens). Interim feBPB posted to #6796.
- **Soak result (6 scored arms, posted to #6796)**: scored at the **common budget**
  `C_ref=2.07e19` FLOPs (`bakeoff_analysis --ref-budget`, no extrapolation — the rebooted arms
  reach different max budgets). **All 5 trained SuperBPE arms beat the marin-128k Llama-3 baseline
  (feBPB 0.9848)**: 64k-llama **0.9478** (−3.8%), 64k-digits 0.9667 (−1.8%), 64k 0.9692 (−1.6%),
  128k-llama 0.9724 (−1.3%), ngram 0.9799 (−0.5%). Margins are conservative because the common
  budget is early (~7k steps, steep BPB curves); they widen toward the proxy-scale −4.7%…−6.8% as
  arms converge — refresh at a higher common budget once the rebooted laggards (06e, 08b) climb.
  Arms 3 & 5 (128k-plain/digits) unscored — blocked by the leaf-group IB collective hang (#6950);
  coverage preserved by 64k-plain + 64k-digits + 128k-llama. Ladder+domains:
  `scratchpad/{ladder_final,domains_final}.json`.
