# Tokenizer FLOP-Equivalent Bake-off for Grug-MoE

Status: draft (weaver #385, issue #6796)
Owner: agent (weaver/tokenizer-research-investigation)
Cluster: cw-rno2a (64× 8×H100, reserved)

## 1. Goal and what is new here

Measure how much real uplift the **grug-moe** family gets from an alternative
tokenizer, where "uplift" is scored on a **compute-equivalent** basis: a
tokenizer that makes the model cheaper to train and serve should be rewarded,
and one that makes it more expensive (large vocab, byte-level) must earn its
keep with a strictly better quality-per-FLOP trade.

A prior six-arm sweep (#5821) already established the honest cross-tokenizer
metric — **bits-per-byte (BPB)** — and ranked existing off-the-shelf tokenizers
on a fixed Grug-MoE ladder at equal *token* budget. After the byte-accounting
fix (PR #6011), that sweep found Gemma-3 (262k) and GPT-OSS (200k) slightly
beat Llama-3 (128k); TokenMonster was worse overall except on code; and
right-to-left digit pretokenization (#4915) helped GPT-OSS-family tokenizers.

This project does three things the prior sweep did not:

1. **Compute-equivalent scoring.** Compare arms at equal *total training FLOPs*
   (including the vocab-dependent LM-head cost that the standard budget
   convention omits), and on a BPB-vs-inference-FLOPs/byte Pareto frontier —
   not equal tokens. This is the rubric the issue asks for.
2. **Build tokenizers, not just borrow them.** Extend the Marin tokenizer with
   derived vocab sizes (roadmap #4971 phase 1, never built), number-aware
   pretokenization, capcode/word-join efficiency tricks (#5837), and greenfield
   axes with no prior art in-repo: **SuperBPE** superword merges and
   **multi-gram input embeddings** ("Over-Tokenized" / the LongCat n-gram idea
   in #6796).
3. **Run it on grug-moe/GPU on cw-rno2a** with consistent, queryable logging so
   every decision is backed by recorded statistics.

Non-goals: production tokenizer rollout; full architecture rewrites (Byte Latent
Transformer, H-Net) are surveyed but out of scope for the first bake-off because
they require model surgery, not a tokenizer swap.

## 2. Measurement rubric (the crux)

### 2.1 Primary metric: bits-per-byte (BPB)

Cross-entropy loss is per-token and therefore not comparable across
vocabularies. BPB normalizes the model's next-token loss by the raw UTF-8 byte
length of the *target* tokens, so it is directly comparable across tokenizers.
Levanter already computes it (`lib/levanter/src/levanter/eval.py`,
`GrugEvalConfig.compute_bpb=True`, default), via a per-token byte-length table
(`byte_length_of_token`, `lib/levanter/src/levanter/utils/hf_utils.py`):

```
bpb = (cross_entropy_bits over targets) / (UTF-8 bytes of targets)
    = mean_token_loss * fertility / ln(2)      # fertility = tokens / byte
```

We report `eval/bpb` (micro), `eval/macro_bpb`, and per-domain `eval/<tag>/bpb`.

Caveat carried from #5821/#6011: BPB byte-accounting is only exact for
well-behaved tokenizers whose tokens decode to their literal source bytes. For
"marker" tokenizers (TokenMonster capcode, our capcode/cap-span variants), we
must count *source* bytes, not decoded marker bytes. Main lacks the #6011 fix,
so exotic marker arms carry a dedicated source-aligned byte-accounting shim
(§5.4) and are validated against a hand-checked corpus before their BPB is
trusted.

BPB itself traces to Gao et al. ("The Pile," arXiv:2101.00027 §3.1) and
`lm-evaluation-harness`; the byte-anchored, per-document-BOS eval protocol
(never concatenate across corpus, non-overlapping windows) follows Paloma
(Magnusson et al., arXiv:2312.10523) — our per-domain, per-arm eval slices
(§4) already follow this. Two named pitfalls from the literature to keep
watching for as we add arms: (1) cross-tokenizer perplexity is only
comparable if both tokenizers can represent the same set of strings (Mielke
et al., arXiv:2112.10508) — true here since all arms retokenize the same raw
bytes, but would break if an arm silently drops/replaces characters; (2)
context-length matching in *tokens* rather than bytes mechanically favors
high-compression tokenizers (flagged by arXiv:2605.01188) — our FLOP/byte
accounting (§2.2) already matches on bytes, avoiding this, but any ad hoc
"just compare at N tokens" shortcut during Phase 0 debugging would reintroduce
it.

### 2.2 The FLOP model and where the tokenizer's cost lives

Levanter's `lm_flops_per_token` (`lib/levanter/src/levanter/utils/flop_utils.py`):

```
flops_per_token = num_layers*(mlp + qkv_proj + dense_proj + attn) + lm_head
lm_head         = 2 * hidden_dim * vocab_size
```

Vocab size enters exactly once, additively, through the output head. The input
embedding is a gather (~0 FLOPs). So a tokenizer changes model compute in two
places:

- **LM-head FLOPs per token**, rising linearly with `vocab_size`.
- **Fertility** `f = tokens/byte`: the number of tokens the model must process
  to cover a fixed amount of text. Coarser tokenizers (lower `f`) process fewer
  tokens per byte.

The grug heuristic's budget convention (`experiments/grug/moe/heuristic.py`)
uses `C = 3 · flops_per_token(**excluding** lm_head) · tokens`. That is fine for
same-vocab scaling but **wrong for a tokenizer bake-off**, because it makes the
LM-head cost of a large vocab free. Our rubric adds it back.

### 2.3 The serving cost model — priced at deployment scale, at deployment context

Cost is a *model*, not a constant, and it is applied at analysis time to the raw
measurements (§2.6). `ServingCostModel` in
`experiments/tokenize/flop_equivalent.py` captures the deployment regime and turns
`(vocab_size, fertility)` into a serving cost per arm:

```
F(vocab)  = forward FLOPs/token at the TARGET model, this context, this sparsity
f         = fertility (tokens/byte), measured on a held-out corpus
infer_flops_per_byte = F(vocab) * f          # the serving cost — fwd only, per byte of text
```

`infer_flops_per_byte` is the honest "cost to serve one byte of text": a byte
tokenizer is penalized on both terms — a high `f` (1.0 token/byte) *and* the full
per-token stack run once per byte.

`F(vocab)` depends on the deployment assumptions, all fields of `ServingCostModel`:

- **Model shape** — the grug-moe we intend to deploy: a **~250B-total / ~20B-active
  MoE** (`TARGET_MODEL_SHAPE`: hidden 6144, 64 layers, 256 experts, top-8). The
  LM-head FLOP fraction shrinks ~10× from the proxy we train to that width
  (llama3-128k 33.5%→1.7%, gemma-262k 50.1%→3.4%). Pricing at the proxy width
  would wildly over-penalize large vocab; at deployment the head is a few percent,
  so **vocab size is nearly free and fertility dominates**. This flips verdicts:
  gemma-3's 262k vocab is *cheaper* to serve than Llama-3's 128k at target scale
  (lower fertility outweighs the trivial extra head), where at proxy scale it
  looked ~28% more expensive.
- **Context window + attention sparsity** — serving is at a **16k-token context**
  (default; `context_len`) with **5:1 local:global attention** (`global_layer_period=6`
  → one full-context layer per six, the rest a `attention_window=4096` sliding
  window). Attention FLOPs/token grow with the positions a token attends over, so
  at long context attention is a large share of cost (7% of forward FLOPs at 4k,
  **10% at 16k, 21% at 64k**), and — crucially — **attention-per-byte scales with
  `fertility · context`**: at a fixed token window an efficient tokenizer packs
  more bytes, so it pays less attention per byte. Long context therefore
  *amplifies* the fertility advantage; the 5:1 sparsity dampens the absolute
  attention cost (most layers are capped at the 4k window) without erasing the
  effect. Byte-level, at fertility 1.0, is penalized ~3× Llama-3 serving cost at
  16k and slightly worse as context grows.
- **Speed factor** — a `speed_factor` multiplier (default 1.0) for
  hardware/kernel efficiency not captured by raw FLOP counting; it cancels in
  relative-cost ratios but is available to set an absolute cost budget.

The proxy model is used only to measure BPB *quality*; every cost multiplier is
`ServingCostModel`. Because all four are fields, the same measured BPB and
fertility can be re-priced at 64k context, a denser attention schedule, a larger
target model, or a different speed assumption without retraining (§2.6).

### 2.4 Compute-equivalent comparison

**Headline comparison — equal total training FLOPs.** Fix a training budget
`C_total`. Each arm trains on `tokens = C_total / (3·F_full)` tokens, i.e. on
`bytes = tokens / f` bytes of text. A low-fertility, modest-vocab tokenizer gets
to see *more* bytes within the same budget; a large-vocab or byte tokenizer sees
fewer. Report held-out BPB. Lower BPB at equal `C_total` is a genuinely
compute-fair win: it already bakes in both the LM-head penalty and the
data-coverage advantage of efficient tokenizers.

**Gold-standard — mini isoFLOP curve.** Because "saw more data" and "cheaper to
serve" both push equal-`C_total` BPB down, we disentangle them by running each
arm at **3 training-FLOP points** (a mini isoFLOP ladder) and fitting
`BPB(C) ≈ a·C^(-b) + c`. The tokenizer whose **BPB-vs-`infer_flops_per_byte`
curve** lies lowest on the Pareto frontier wins. This is the scaling-law-grade
answer and the one we make decisions on.

**Single reportable scalar — FLOP-equivalent BPB (feBPB).** For ranking, model a
fixed *lifetime* FLOP budget split between training once and serving over the
model's life. Anchor on the reference arm (Llama-3/marin): let `C_ref` be its
training budget (the middle isoFLOP point) and let it spend `ρ·C_ref` FLOPs
serving over its life (`ρ` = serving-to-training ratio, default 1.0 — serving and
training weighted equally). The lifetime budget `B = C_ref·(1+ρ)` and the served
byte-volume are held fixed across arms; an arm's serving cost scales by its
relative serving cost `s = arm.infer_flops_per_byte / R*`, and whatever is left
funds training:

```
train_flops(arm) = B − ρ·C_ref·s = C_ref·(1 + ρ·(1 − s))
feBPB(arm)       = BPB_arm read off its fitted curve at train_flops(arm)
```

A cheaper-to-serve arm (`s<1`) gets *more* training than the reference and a
lower feBPB; an expensive one gets less. An arm whose serving cost alone exceeds
the lifetime budget (`train_flops ≤ 0`, e.g. byte-level at `s≈3`, `ρ=1`) is
reported **infeasible** — the honest verdict. `ρ` makes the training/serving
weighting explicit: `ρ→0` recovers the equal-training comparison, large `ρ`
(a model served far more than trained) makes serving efficiency dominate. We
report feBPB alongside the raw Pareto curve — never instead of it. (Earlier
drafts had a serving-cost term that algebraically cancelled, evaluating every arm
at the same training FLOPs; the lifetime-budget form above is the fix.)

### 2.5 Fairness controls

- **Hold the non-embedding proxy model fixed** across arms (same hidden, layers,
  experts, top-k, seq_len) for the BPB *measurement* runs. Only `vocab_size` (→
  embedding/LM-head) and the data tokenization change. Embedding params vary with
  vocab; we log total and non-embedding params so the reader sees the trade. Cost
  scoring is done at `TARGET_MODEL_SHAPE` (deployment scale), not the proxy (§2.3).
- **Same raw corpus bytes**, tokenized per arm — never compare across different
  underlying text.
- **Same optimizer family and LR rule.** LR is set per arm by the grug AdamH
  heuristic as a function of (tokens, hidden, batch); we do a small LR
  sanity-sweep per vocab class (§8) to confirm the heuristic transfers, since it
  was fit at vocab 128k.
- **Same eval set and same held-out bytes** for every arm; BPB is byte-anchored
  so token counts differ but byte coverage of the eval set is identical.
- **Fixed seeds**; ≥2 seeds on the final short-list to get a noise bar.

### 2.6 Replayability — measure once, re-score under any assumption

Every deployment number in §2.3–§2.4 (model size, context, sparsity, speed,
serving ratio ρ, even the eval-domain mix) is an *assumption*, and assumptions
change. So the pipeline never bakes a cost into a training run:

- **Experiments log raw, cost-free measurements.** The fertility pre-filter
  (`fertility_report.py`) writes per-arm, per-domain **token and byte counts**
  (`fertility_raw.json`), not ratios or FLOPs. Training runs log per-arm
  `(training_FLOPs, BPB)` points. Neither carries any deployment-model assumption.
- **A separate analysis step applies the cost model.**
  `experiments/tokenize/bakeoff_analysis.py` reads those raw files and recomputes
  fertility (with an optional domain reweighting), serving cost, the Pareto
  frontier, and feBPB under a `ServingCostModel` supplied on the command line
  (`--context-len`, `--attention-window`, `--global-period`, `--speed-factor`,
  `--target-hidden`, `--target-layers`, `--serving-ratio`, `--domain-weights`).

To answer "what's optimal if we serve at 64k instead of 16k?", or "for a 400B
target?", or "if code is 5× the traffic?", re-run the analysis with new flags —
no GPU time. The winning arm is stable across these because fertility, the
dominant driver, is intrinsic to the tokenizer and scale-invariant; the cost
model only sharpens or softens the margins.

## 3. Tokenizer design space (axes)

Grouped by mechanism and by cost signature. "Cost" = effect on
`infer_flops_per_byte` at fixed model.

### A. Vocabulary size / family (borrow + derive)
- Off-the-shelf baselines: `marin`/Llama-3 (128,256), Qwen3 (151,669, arXiv:2505.09388
  §2 — unchanged from Qwen2's 151,646), Gemma-3 (~256k–262k, arXiv:2503.19786 —
  the report states both figures; treat as a range), GPT-OSS (o200k_base,
  200,019; the open-weight-only `o200k_harmony` variant used by gpt-oss is
  201,088 and not relevant here), gpt-neox (50k). (Re-confirms #5821 under the
  new rubric.)
- **Derived Marin vocab sizes 32k / 64k** (roadmap #4971 phase 1, unbuilt): rank-
  truncate the Llama-3 BPE merge table + reachable-vocab closure. Smaller vocab
  → smaller LM-head → cheaper per token, higher fertility. Tao et al., "Scaling
  Laws with Vocabulary" (arXiv:2407.13623, NeurIPS 2024) fit `Nv_opt ∝
  Nnv^0.83` and report, e.g., a 2.87B model at fixed 2.3e21 FLOPs gains
  **ARC-Challenge 29.1→32.0** just from correcting vocab size (32k→43k) at
  equal compute; extrapolating their curve down implies an optimum in the
  ~30–64k range for our 300M–1B non-embedding size, so this axis is expected
  to *win* on feBPB even if raw equal-token BPB is flat. **Open contradiction
  to track, not resolve by assumption**: a 2026 follow-up ("Compute Optimal
  Tokenization," Limisiewicz et al., FAIR, arXiv:2605.01188, 988 BLT-style
  models 50M–7B) finds the *opposite* trend at their scale — optimal
  compression/vocab **decreases** with more compute — and attributes part of
  the disagreement to Tao et al. matching context length in tokens rather
  than bytes (mechanically favors high-compression/large-vocab tokenizers).
  Our rubric (§2) already matches on bytes/FLOPs, not tokens, which should
  avoid that specific confound, but it means the "30–64k is optimal" claim
  above is a prior from a differently-scoped study, not a settled fact at
  grug-MoE scale — Phase 2's empirical sweep is the actual test.

### B. Pretokenization tweaks (cheap, same vocab-size class)
- **Number-aware / right-to-left digit grouping** (#4915): split numeric runs
  into right-aligned 3-digit groups, bounded to avoid regex backtracking. Prior
  art shows a real BPB win for GPT-OSS-family at 32k. Re-test on marin base.
- **Capcode / word-join** (#5837, TokenMonster-inspired): `<cap>` /`<cap_span>`
  capitalization markers and `<token_join>` compound splitting to reclaim vocab
  slots. Needs source-aligned byte accounting (§5.4).

### C. Superword merges — SuperBPE (greenfield)
- Two-stage BPE: learn subword merges with whitespace pretokenization up to a
  transition point `t`, then lift the whitespace constraint and continue
  merging to `T` to form cross-word "superword" tokens (Liu, Hayase, Hofmann,
  Oh, Smith, Choi, arXiv:2503.13423, COLM 2025). At `T=200k`, the ablated
  transition points are `t∈{80k,160k,180k}`; `t=180k` is the accuracy-best
  point (bytes/token 6.09 vs plain BPE's 4.46, i.e. **~27% fewer tokens**),
  while `t=80k` is the efficiency-best point (bytes/token 6.63, **~33% fewer
  tokens**) but a smaller downstream gain. At an 8B model / 30-task suite,
  `t=180k` gives **+4.0% absolute avg accuracy** (wins 25/30 tasks; MMLU
  +8.2, ARC-Challenge +15.5, CommonsenseQA +20.3) and **~27% less inference
  compute** at matched train budget/model size — enough that an 11B SuperBPE
  model matches an 8B BPE model's train+inference cost. The one regression is
  LAMBADA (−6.4). Same vocab size, lower fertility: a pure win on both
  training and inference FLOPs/byte if quality holds at our scale — the
  single most promising FLOP-equivalent axis. Independent near-concurrent
  corroboration: *Boundless BPE* (arXiv:2504.00178, COLM 2025) relaxes the
  same whitespace constraint via a different construction and reports ≥21%
  Rényi-efficiency / ≥19.7% more bytes/token, converging on the same
  conclusion from an independent method. Caveats: (1) a small (70M-param)
  independent replication confirms the ~20% token reduction but only shows
  **parity**, not a clear loss win, at that scale — the downstream-accuracy
  gain is unverified below 8B, i.e. exactly the gap this bake-off should
  close; (2) a 2026 BPB benchmark (ACL 2026.mellm-1.27) finds SuperBPE
  matches BPE only for English and **underperforms by 0.01–0.06 BPB for
  Hungarian and Mandarin**, so treat any multilingual eval slice (§4) as a
  likely SuperBPE loss, not a bug. We build this ourselves (§5.3).

### D. Multi-gram input embeddings — Over-Tokenized / n-gram (greenfield)
- Decouple **input** vocabulary from **output** vocabulary: hash n-gram
  windows of the base token stream into a fixed-size table `E(idx mod m)`
  (mixed-radix n-gram index, `m` chosen non-multiple-of-V so collisions stay
  "generic") and sum the looked-up vector with the ordinary unigram embedding
  — "Over-Tokenized Transformers" (Huang et al., arXiv:2501.16975, ICML
  2025). Capacity is added via `k` low-rank sub-embedders at staggered moduli,
  not one bigger table. The input side is a gather → **~0 added FLOPs**, cost
  is embedding **parameters/memory** only (e.g. +13.1B params at `m=12.8M`
  atop a 1.3B-active model — size the table to fit HBM, §11). Reported
  effect: a log-linear scaling law, `loss ≈ 2.675 − 0.0256·log10(m)`, holding
  for dense models 151M–1B and `m=1.2M–12.8M`; a 151M model gets **~14%
  perplexity reduction**; OLMoE MoE runs at 500B tokens show val-loss cuts of
  0.08–0.12 and 2.6–5.7× faster convergence to a target loss. The output
  vocab (hence BPB accounting and LM-head FLOPs) is unchanged, so this
  composes with any base tokenizer in axes A–C. This is exactly the
  mechanism the issue calls "n-gram support from longcat": Meituan's LongCat
  team published a direct descendant, "Scaling Embeddings Outperforms Scaling
  Experts" (arXiv:2601.21204) — same hashed-multi-table idea, but the added
  embedding params are **reallocated from MoE experts** (fixed active/total
  ratio) rather than added on top, and N≥3/K≥2 is needed (N=2/K=1 "notably
  inferior"). For our dense-ish grug shape, OT's pure-addition variant (not
  the expert-reallocation variant) is the directly applicable one. Because
  OT's ablations show the gap *widening* past 200B tokens, expect a real but
  smaller effect at our 5–30B-token budget — track loss-vs-tokens curves, not
  a single checkpoint.

### E. Byte / tokenizer-free (reference floor, likely negative — but see caveat)
- Byte-level (256 vocab, fertility 1.0). Under our rubric, priced at deployment
  scale (§2.3), it must overcome **~3× higher inference FLOPs/byte** (at 16k
  context; slightly worse as context grows) and a matching factor less data per
  training FLOP vs Llama-3 (the byte penalty is *larger* at deployment scale than
  at the proxy, because the full non-embedding stack runs once per byte and the
  head savings are negligible at d6144).
  Full byte-latent architectures (BLT: Pagnoni et al., arXiv:2412.09871, Meta
  FAIR, ACL 2025; H-Net: Hwang, Wang, Gu, arXiv:2507.07955) that recover this
  via dynamic patching are **surveyed but out of scope** for the first
  bake-off (model surgery, not a tokenizer swap). We include a plain byte arm
  only as the honest lower-bound reference the rubric is designed to expose.
  Lineage: MegaByte (arXiv:2305.07185) used static fixed-size patches;
  SpaceByte (arXiv:2404.14408) patches per whitespace-delimited word; BLT's
  entropy-based dynamic patching and H-Net's fully learned boundary routing
  are the current state of the art and both report explicit FLOP-matched
  crossover points against BPE Transformers. **Caveat worth flagging**: BLT's
  own crossover language puts byte-level ahead of BPE only above ~2.5–3×
  compute-optimal budget (and a third-party review finds BLT still *worse*
  than BPE-Llama at 1B params, overtaking only near 7B) — consistent with
  "out of scope" here. But H-Net's best (2-stage) variant is reported to
  cross over tokenized Transformers after **~30B training bytes** (≈7–8B
  token-equivalents) — inside this bake-off's 5–30B-token range — so the
  "byte-level is a clear loser at our scale" assumption is not fully settled
  for the most aggressive dynamic-chunking variant, only for plain byte-level
  and for BLT specifically. Recorded as a risk (§11), not an axis to build.

### F. Unigram-LM vs BPE at matched vocab (cheap, strong prior evidence)
- SentencePiece Unigram-LM segmentation instead of BPE, at a matched vocab size.
  This axis is *not* in the original design but the research pass flags it as
  having the cleanest, most-replicated small-scale evidence of any technique
  surveyed: Bostrom & Durrett (arXiv:2004.03720, EMNLP 2020) find Unigram-LM
  beats BPE at a matched 20k vocab on SQuAD (+1.1 F1), MNLI (+0.5–1.4), and
  especially morphologically rich languages (Japanese TyDi QA **+12.3 F1**), and
  a 2025 follow-up (arXiv:2508.08424) reconfirms Unigram > BPE on agglutinative
  languages. Same vocab size and near-identical fertility, so it is almost free
  on the FLOP axis — a near-pure quality test. Cheap to add (SentencePiece
  unigram mode over the same corpus). Include one Unigram arm at the winning
  Phase-2 vocab size, with a multilingual eval slice to exercise its strength.
  (Counterweight to keep in mind: PathPiece, Schmidt et al. arXiv:2402.18376,
  EMNLP 2024, found across 64 models 350M–2.4B that compression and downstream
  accuracy barely correlate — r=0.24 — and its top-5 tokenizers were
  statistically indistinguishable, so segmentation-quality effects at this scale
  may be small. This is a prior for modest effect sizes across all of axes A–F,
  not a reason to skip them; it raises the bar for calling a winner.)

## 4. Dataset and caching

All data is tokenized fresh from HuggingFace **into the CoreWeave bucket**
`s3://marin-us-east-02a/marin` (the cw-rno2a `MARIN_PREFIX`). GCS-pinned llama3
caches are unreachable from CoreWeave and are the wrong tokenizer anyway, so
every arm re-tokenizes. (`tokenized()` keys its cache on `(name, tokenizer,
source, version)`, so distinct tokenizers naturally produce distinct caches.)

- **Train corpus:** SlimPajama (the existing `slimpajama_6b_dataset()` pattern,
  generalized to take a tokenizer + vocab). SlimPajama-6B is small enough to
  re-tokenize per arm on CPU and already the canonical grug GPU-smoke corpus.
  For the larger compute points we scale to a bigger SlimPajama slice or
  FineWeb-Edu sample so we are not token-bound at the top of the isoFLOP ladder.
- **Eval / validation suite (held out):** a fixed set of slices tokenized per
  arm, fresh from HF, for per-domain BPB:
  - English web (held-out SlimPajama / C4 / FineWeb slice)
  - Code (StarCoder sample)
  - Math / numeric (FineMath or Proof-Pile sample) — exercises digit axis B
  - Multilingual (small slice) — exercises coverage differences
  These are `validation=True` tokenize steps so `mixture(ctx, train,
  validation=[...])` wires them as tagged eval sets. Byte coverage of each slice
  is identical across arms; only token counts differ.

Caching is a one-time CPU Fray job per (tokenizer, corpus); training arms then
depend on the materialized `TokenizedCache` handle. We cache before launching
training and reuse across all compute points for an arm.

## 5. Marin tokenizer extensions to build

New code lives under `experiments/tokenizer/` (a tokenizer-construction package)
plus a small `levanter` backend addition where a non-HF tokenizer is needed.
Each built tokenizer is saved to a location the cluster workers can load
(`load_tokenizer` accepts an HF id or a path; we push built tokenizers to the HF
Hub under `marin-community/` or write them to the S3 prefix and reference by
path). Every new tokenizer is registered in `_KNOWN_VOCAB_SIZES`
(`lib/marin/src/marin/processing/tokenize/data_configs.py`) to avoid network
calls at dry-run/fingerprint time.

1. **Derived-vocab builder** (axis A): rank-truncate a base HF BPE tokenizer's
   `model.vocab`/`model.merges` to 32k/64k with reachable-vocab closure and
   special-token remap. (Port the truncation logic from the unmerged
   `tokenizer_sweep.py`, dropping its legacy ExecutorStep API.)
2. **Number-aware pretokenizer** (axis B): right-to-left 3-digit grouping, 510-
   char bounded. Re-usable as a pretokenizer wrapper over any base.
3. **SuperBPE trainer** (axis C): two-stage BPE with a transition point; train at
   target vocab, save an HF-format fast tokenizer.
4. **Capcode / marker tokenizer + source-aligned BPB shim** (axis B/D): capcode
   markers and `<token_join>`; a byte-accounting function that maps marker
   tokens to their source-byte spans so BPB is fair. (Fold in the #6011 idea.)
5. **Multi-gram input embedding** (axis D): a grug model change — an additional
   hashed n-gram embedding table summed into the input embedding, output head
   unchanged. Config-gated so it composes with any base tokenizer.

Intrinsic pre-filter (no training, runs in minutes): for every candidate compute
**fertility (tokens/byte), bytes/token, per-domain fertility, OOV/continued-byte
rate, and `infer_flops_per_byte`** over the fixed eval corpus. This is a cheap
first-pass ranking that decides which arms are worth training compute. (Harness
already prototyped; see §9.)

## 6. Model

Two models with distinct roles (§2.3): a **deployment target** the tokenizer is
priced for, and a small **proxy** we actually train to measure BPB.

- **Target (pricing only, never trained here):** ~250B-total / ~20B-active MoE,
  `TARGET_MODEL_SHAPE` in `experiments/tokenize/flop_equivalent.py` — hidden 6144, 64
  layers, 256 experts, top-8, seq 4096. All cost multipliers use this width.
- **Proxy (trained to measure BPB):** a fixed small Grug-MoE, non-embedding shape
  held constant across arms. Starting point (tunable after the smoke run), ~300M
  active / ~a few-B total:

```
hidden_dim=1024, num_layers=16, num_heads=8, num_kv_heads=2, head_dim=128,
intermediate_dim=512, shared_expert_intermediate_dim=1024,
num_experts=32, num_experts_per_token=4, max_seq_len=2048, sliding_window=2048
vocab_size = <per arm>
```

`vocab_size` is set per arm and must match the tokenizer (grug does not derive it
— we set it explicitly and assert equality against the tokenizer's real vocab at
build time). This size trains a compute point in well under an hour on a couple
of 8×H100 nodes, so a full sweep is affordable. The final short-list is
optionally re-confirmed one width up (d1536/d2048) to check the ranking is
scale-stable, mirroring the #5821 d512/d768/d1024 ladder.

## 7. Training & evaluation procedure

- Launch via a tokenizer-parameterized version of `launch_cw_scale.py` (§9):
  `SCALE_*` env knobs plus a new `TOKENIZER` / `VOCAB_SIZE` / `EVAL=on` set.
- Turn eval **on** (`GrugEvalConfig(compute_bpb=True, steps_per_eval=...)`) — the
  scale launcher currently sets `eval=None`; the bake-off needs periodic
  held-out BPB.
- Optimizer: grug AdamH heuristic per arm (LR from tokens/hidden/batch), with a
  per-vocab-class LR sanity sweep (§8).
- Compute points: 3 per arm for the isoFLOP curve; equal-`C_total` headline uses
  the middle point.
- Metrics logged every step; eval every N steps. Two seeds on the short-list.

## 8. Experimental plan (phases)

**Phase 0 — harness + smoke (fast).** Generalize the launcher and dataset to take
a tokenizer; run a 1-node grug smoke with `marin` tokenizer + eval on, confirm
`eval/bpb` logs and the FLOP/fertility accounting is recorded. Cache the eval
suite for all baseline tokenizers.

**Phase 1 — intrinsic pre-filter (no training).** Compute fertility /
`infer_flops_per_byte` / coverage for every candidate over the eval corpus. Rank;
drop dominated arms. Post the table to the issue.

**Phase 2 — baseline bake-off (borrowed tokenizers, axis A).** Equal-`C_total`
BPB for marin/Llama-3, Qwen3, Gemma-3, GPT-OSS, gpt-neox, and derived marin
32k/64k. One compute point. This reproduces #5821 under the FLOP-equivalent
rubric and validates the harness against a known result.

**Phase 3 — cheap-mechanism axes (B, C, F).** Number-aware and SuperBPE variants
of the best 1–2 base tokenizers from Phase 2, plus one Unigram-LM arm at the
winning Phase-2 vocab size. SuperBPE moves fertility (and thus feBPB) the most;
number-aware and Unigram are near-free quality tests.

**Phase 4 — isoFLOP curves + Over-Tokenized (D).** Run 3-point isoFLOP ladders
for the Phase 2/3 short-list (≤5 arms) and add the multi-gram input-embedding
arm on the best base. Fit BPB-vs-`infer_flops_per_byte`; compute feBPB; pick the
winner. Scale-stability re-confirm one width up.

**Phase 5 — decision + write-up.** Report the Pareto frontier and feBPB ranking,
the recommended tokenizer, and the uplift vs marin baseline in compute-equivalent
terms. Update the issue and the doc.

Hyperparameter search: a small LR sweep per vocab class (Phase 2) is enough; if
the AdamH heuristic proves vocab-sensitive we widen it. Vizier is available if a
larger joint search over (LR, transition-point for SuperBPE, n-gram table size)
is warranted in Phase 4.

## 9. Logging, statistics, and reproducibility

Every trial must record enough to reconstruct its score without rerunning:

- **Tracker:** W&B project `marin_moe`, group `tokenizer-flop-bakeoff`, run name =
  arm slug + compute point + seed. (`SCALE_TRACKER=wandb`.)
- **Per-run config logged:** tokenizer id, vocab_size, measured fertility per
  domain, `F_full`, `train/infer_flops_per_byte`, C_total, tokens, bytes, batch,
  steps, seed, model shape, param counts (total + non-embedding), LR.
- **Per-run metrics:** `train/loss`, `eval/bpb`, `eval/macro_bpb`,
  `eval/<domain>/bpb`, `throughput/{tokens_per_second,mfu,total_gflops}`.
- **Derived, at analysis time:** BPB-vs-C fit per arm, feBPB, Pareto rank. Stored
  as a committed results table (CSV/JSON under the experiment dir) + a summary
  posted to the issue at each phase.
- A single results module reads the W&B runs (or json_logger metrics) and emits
  the ranking deterministically, so the decision is reproducible from logs.

## 10. Execution on cw-rno2a

Submission pattern (CPU launcher dispatches tokenize + GPU train sub-tasks):

```
uv run iris --cluster=cw-rno2a job run --cpu 2 --memory 3GB --extra cpu \
  --job-name grug-tok-<arm> \
  -e TOKENIZER <hf-id-or-path> -e VOCAB_SIZE <n> -e SCALE_TRACKER wandb \
  -e SCALE_GPU_REPLICAS 2 -e SCALE_HIDDEN_DIM 1024 -e SCALE_NUM_LAYERS 16 \
  -e SCALE_NUM_EXPERTS 32 -e SCALE_TOP_K 4 -e SCALE_BATCH <b> \
  -e SCALE_SEQ_LEN 2048 -e SCALE_STEPS <s> -e RUN_ID tok-<arm>-<pt> \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.moe.launch_tokenizer_bakeoff
```

`MARIN_PREFIX=s3://marin-us-east-02a/marin` is injected by the cluster config;
all caches and checkpoints land there. Never bounce the cluster. Use the
reserved warm H100 pool; keep each arm to a couple of nodes so many arms run
concurrently.

## 11. Risks and open decisions

- **BPB accounting for marker tokenizers** (capcode) needs the source-aligned
  shim; until validated, capcode arms' BPB is provisional.
- **AdamH heuristic transfer across vocab sizes** — mitigated by the per-class LR
  sanity sweep; if it fails we fit LR per arm.
- **Over-Tokenized input-embedding memory** — large n-gram tables cost HBM; size
  the table to fit and log the param/memory trade.
- **Ranking scale-stability** — small-model winners may not hold at scale; the
  one-width-up re-confirm in Phase 4 guards this, but the final claim is scoped
  to the tested scale.
- **Cross-arm data identity** — enforce identical raw bytes and identical eval
  slices; a mismatch silently invalidates BPB comparisons.
- **Byte-level "out of scope" is a scoping choice, not a settled negative
  result at our token budget.** H-Net's best (2-stage) dynamic-chunking
  variant is reported to cross over tokenized Transformers at ~30B training
  bytes (≈7–8B token-equivalents, arXiv:2507.07955) — inside our 5–30B-token
  range. We still exclude it (model surgery, not a tokenizer swap), but a
  future bake-off revision that allows architecture changes should not assume
  byte-level is a clear loser at this scale; only plain byte-level and BLT
  (arXiv:2412.09871, worse than BPE-Llama at 1B params in third-party review)
  are safely excluded on current evidence.
- **Vocab-scaling-law direction is disputed at different scales** — see §3.A;
  Tao et al. 2024 (arXiv:2407.13623) and the 2026 BLT-scale follow-up
  (arXiv:2605.01188) disagree on whether optimal vocab/compression grows or
  shrinks with compute. Don't hard-code a "30–64k is optimal" assumption into
  the arm list — the derived 32k/64k/256k sweep in Phase 2 is there precisely
  to settle this empirically at grug-MoE scale.

## 12. Literature backing (background research pass, 2026-07-03)

Full external-literature brief with effect sizes, scale caveats, and a wider
survey (TokenMonster/PathPiece/Bostrom&Durrett segmentation-quality evidence,
Ali et al. and Goldman et al. fertility-vs-performance correlations, and
2025–2026 major-lab tokenizer changes: Claude Sonnet 5/Opus 4.7's confirmed
1.0–1.35x token-count inflation per Anthropic's migration guide, Qwen3/Gemma-3/
Llama-4/GPT-5 vocab sizes and rationale) is recorded in the session that
produced this update; see the `weaver/tokenizer-research-investigation`
research thread. IDs dated 2026+ (e.g. 2601.*, 2605.*, 2508.*) come from that
web-research pass and postdate this author's training cutoff — verify them
against primary sources before citing externally; the load-bearing claims
(SuperBPE, Over-Tokenized, BPB, vocab scaling) rest on the pre-2026 papers.
Key arXiv IDs referenced above, collected for convenience:
SuperBPE 2503.13423, Boundless BPE 2504.00178, Over-Tokenized Transformers
2501.16975, LongCat n-gram/Engram 2601.21204, LongCat-Flash 2509.01322, BLT
2412.09871, H-Net 2507.07955, H-Net++ 2508.05628, MegaByte 2305.07185,
SpaceByte 2404.14408, Tao et al. vocab scaling 2407.13623, Compute Optimal
Tokenization 2605.01188, SCONE embedding scaling 2502.01637, PathPiece/Schmidt
et al. 2402.18376, Bostrom & Durrett 2004.03720, Goldman et al. 2403.06265,
Ali et al. 2310.08754, The Pile (BPB) 2101.00027, Paloma 2312.10523, Mielke et
al. 2112.10508, Qwen3 2505.09388, Gemma 3 2503.19786.
```
