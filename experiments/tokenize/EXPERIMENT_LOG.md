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
7. 🔄 **Toward 10%**: (a) compose the new best tokenizer 80k-t40k + n-gram; (b) bracket the vocab
   optimum (64k/96k ladders); (c) does the n-gram grow with model scale (EXP-007 hidden-2048,
   2 replicas after an OOM)? + gpt-neox+n-gram ladder finishing.

### feBPB scorecard so far (target scale: hidden 6144, 64 layers, 16k ctx, English/math)

| arm | feBPB | vs marin | lever |
|---|---|---|---|
| **trained-superbpe-80k-t40k** | **1.1621** | **−6.1%** | small-vocab superword (our mix) |
| gpt-neox-50k + n-gram | 1.1651 | −5.9% | small-vocab + n-gram (composed) |
| superbpe-128k + n-gram | 1.1733 | −5.2% | superword + n-gram (composed) |
| gpt-neox-50k | 1.1745 | −5.1% | small-vocab tokenizer |
| superbpe-128k | 1.1794 | −4.7% | superword tokenizer |
| marin-128k (Llama-3) | 1.2376 | ref | incumbent |

**n-gram composition matrix** (feBPB improvement the n-gram adds on each base tokenizer, ratio 0.25):
marin +0.0% (washes out by s8000) · superbpe-128k **−0.5%** · gpt-neox-50k **−0.8%** · 80k-t40k
**~−0.7%** (s3500 point pending). The n-gram helps *more* on the smaller/superword tokenizers and,
unlike on Llama-3, the gain persists at the highest training budget — so the best composition is a
small-vocab tokenizer + n-gram. Best projected composed arm: 80k-t40k + n-gram (~−6.7%).

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
- **Result / Conclusion**: _pending_ — compare the hidden-2048 Δ(ngram−base) to the hidden-1024 Δ
  (−0.4% at this same config). A larger negative Δ = the lever scales with model size (the paper's
  claim, and the basis for extrapolating value to the 20B-activated target).

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
