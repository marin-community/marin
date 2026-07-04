# Tokenizer FLOP-equivalent bake-off — findings

Companion results doc to the protocol in
[`20260703_tokenizer_flop_equivalent_bakeoff.md`](20260703_tokenizer_flop_equivalent_bakeoff.md).
That doc is the *design* (what we measure and why); this doc is the *result* (what we found).
The full chronological experiment log, with a reproduce/replay command per experiment, is
[`experiments/tokenize/EXPERIMENT_LOG.md`](../../experiments/tokenize/EXPERIMENT_LOG.md).

## TL;DR

We measured every tokenizer arm under a **FLOP-equivalent BPB (feBPB)** rubric — quality
(bits-per-byte, tokenizer-agnostic) traded against serving cost (inference FLOPs/byte at the
250B-total / ~20B-active deployment target), with the cheaper-to-serve arm earning proportionally
more training budget. Best arms, vs stock Llama-3 (marin-128k, feBPB 1.2376):

| rank | arm | feBPB | vs Llama-3 |
|---|---|---|---|
| 1 | **trained-superbpe-80k-t40k + n-gram** | **1.1584** | **−6.4%** |
| 2 | trained-superbpe-80k-t40k | 1.1621 | −6.1% |
| 3 | gpt-neox-50k + n-gram | 1.1651 | −5.9% |
| 4 | superbpe-128k + n-gram | 1.1733 | −5.2% |
| 5 | gpt-neox-50k | 1.1745 | −5.1% |
| 6 | superbpe-128k (off-the-shelf) | 1.1794 | −4.7% |

**The winning lever is a small-vocab superword tokenizer trained on our own mix.** A ~80k SuperBPE
we trained on the grug-moe data mix beats every off-the-shelf tokenizer: a small vocab is a
*smaller, cheaper model* (more effective training steps per FLOP → lower BPB), and the SuperBPE
superword pretokenizer keeps fertility high enough that the modest serving-cost penalty is
outweighed. Adding a hashed n-gram input embedding on top buys a further ~0.3% (the n-gram and
superword are partially redundant). **Best measured: −6.4% feBPB.**

**On the 10% target.** The measured levers do not reach a 10% feBPB improvement at the flash-scale
proxy — the tokenizer serving-discount lever caps near −5%, small-vocab training-efficiency adds
~1%, and the n-gram adds ~0.3–0.8%. Reaching 10% would require the n-gram's contribution to *grow
substantially with model scale* (the LongCat/Over-Encoding claim). We tested that directly with a
hidden-2048 scale ladder (§N-gram); [SCALE RESULT PENDING]. Our honest recommendation is the
−6.4% arm as a safe, drop-in win, with the scale-dependent n-gram flagged for a target-scale
revisit.

## The rubric: FLOP-equivalent BPB (feBPB)

A cheaper-to-serve tokenizer earns the right to be slightly worse per byte, and a better-per-byte
tokenizer earns the right to be slightly more expensive. We combine the two axes into one number.

- **Quality** is **bits-per-byte**: `bpb = eval_loss / bytes_per_token * log2(e)`. BPB is invariant
  to the tokenizer's segmentation, so a 50k-vocab and a 262k-vocab model are compared on the same
  held-out *bytes* — the only honest way to compare tokenizers on quality.
- **Cost** is **inference FLOPs per byte** at the deployment target: `flops_per_token(vocab) *
  fertility`, `fertility = tokens/byte`, with attention that scales with context length. The LM
  head is only ~1.7% of forward FLOPs at target scale, so **serving cost is dominated by fertility**
  (tokens/byte driving the transformer-layer cost), not by vocab-size head cost.
- **Combine** into a budget-adjusted feBPB: an arm that is `s = infer_cost / reference_cost` as
  expensive to serve is charged for that at training time, `train_flops(arm) = C_ref * (1 + rho *
  (1 - s))`. We fit the isoFLOP curve `BPB(C) = a*C^-b + c` per arm and read each arm's BPB at its
  cost-adjusted budget. Lower feBPB wins.

The cost model (`ServingCostModel`) is attention-aware and **replayable**: every score is
recomputed from stored `(train_flops, BPB)` points and raw token/byte counts under
CLI-configurable deployment assumptions (context, sparsity, target size, serving ratio, domain
mix) — "what if we serve at 64k / a code-heavy mix?" is a re-run, not a retrain.

## Method

- **Model**: grug-moe proxy — hidden 1024, 16 layers, 32 experts, top-4, seq 1024, batch 128. The
  proxy measures the *tokenizer*, not the model; `vocab_size` follows the arm (driving both the
  embedding table and the output head).
- **Data**: SlimPajama-6B as the shared training corpus, tokenized per-arm. Held-out validation is
  the Uncheatable-Eval subsets, tokenized per-arm so every arm is scored on the same raw bytes.
- **isoFLOP ladder**: each arm trained at 3 compute points (1500 / 3500 / 8000 steps) to fit
  BPB-vs-FLOPs and compare at matched training budget.
- **Deployment target for scoring**: 250B-total / ~20B-active MoE (hidden 6144, 64 layers), 16k
  context, Llama-encoder attention, 5:1 local:global sparsity.

## Axis A — serving cost (fertility)

bytes/token on the English/math serving mix (`english_web=0.8, math=0.2`), higher = fewer tokens =
cheaper to serve. `rel_serve` is inference FLOPs/byte relative to marin-128k at the target.

| arm | vocab | bytes/tok | rel_serve |
|---|---|---|---|
| trained-superbpe-160k-t64k | 160k | 5.00 | 0.787 |
| superbpe-128k (off-the-shelf) | 128k | 4.80 | 0.816 |
| trained-superbpe-80k-t40k | 80k | 4.66 | 0.835 |
| marin / llama3-128k | 128k | 3.92 | 1.000 (ref) |
| gpt-neox-50k | 50k | 3.81 | 1.017 |

Superword tokenizers (SuperBPE) pack ~20–28% more bytes/token than Llama-3 on English. A critical
implementation note: SuperBPE repos ship a `GPT2Tokenizer` class, so `AutoTokenizer.from_pretrained`
silently overwrites the superword pretokenizer and measures worse-than-baseline fertility; only the
`from_file` path (which marin's tokenize pipeline uses) honors it.

## Axis B — off-the-shelf tokenizer ladder (EXP-002)

feBPB at the target: **superbpe-128k −4.7%**, **gpt-neox-50k −5.1%** (two co-leaders, trading by
scenario); gpt-oss-200k +1.6%, qwen3-152k +3.2% (large heads / poor fertility). SuperBPE wins by
combining a raw-BPB edge (more bytes/FLOP) with the serving discount; gpt-neox wins by a small,
cheap-to-train vocab. Neither reaches 10% alone.

## Axis C — train our own tokenizer (Track C, EXP-008/008b) — the winning lever

We trained 11 tokenizers (plain BPE + SuperBPE, a from-scratch reimplementation of arXiv:2503.13423
on stock `tokenizers`) on a 1.5 GB English/code/math sample of the grug-moe mix, then ran isoFLOP
feBPB ladders. **Vocab size is the axis, and small-vocab superword wins:**

| trained arm | vocab | s8000 BPB | feBPB | vs marin |
|---|---|---|---|---|
| **trained-superbpe-80k-t40k** | 80k | 1.1107 | **1.1621** | **−6.1%** |
| trained-superbpe-128k-t51k | 128k | 1.1081 | 1.1836 | −4.4% |
| trained-superbpe-128k-t102k | 128k | 1.1252 | 1.1918 | −3.7% |
| trained-superbpe-160k-t64k | 160k | 1.1216 | 1.2028 | −2.8% |
| trained-superbpe-64k-t32k | 64k | [PENDING] | [PENDING] | [PENDING] |

**feBPB falls monotonically as trained vocab shrinks** (160k → 128k → 80k). Two lessons: (1) a
*small* vocab is a *smaller model* — cheaper per training FLOP, so more effective steps at a fixed
budget → lower BPB — and the superword pretokenizer keeps fertility high enough that the serving
penalty is outweighed; (2) *big*-vocab trained SuperBPE (128k/160k) is **worse** than off-the-shelf
superbpe-128k (more bytes/token but worse held-out BPB — it over-merges a small corpus). The win is
specifically at small vocab: the "gpt-neox efficiency × superword packing" sweet spot. 64k/96k
bracket ladders locate the exact optimum [64k PENDING].

## Axis D — hashed n-gram input embedding (Over-Encoding / LongCat)

An input-side hashed n-gram embedding (arXiv 2501.16975 & 2601.21204) adds parameters at **~0
serving FLOPs** (a gather, not a matmul; the output head is untouched), so any BPB gain is a
near-pure feBPB gain. A **first attempt regressed** — but that did not test the method: it used
65k hash buckets (~60× too few → bigram collisions are pure noise), sum-combine, and zero init.
Rebuilt to the paper config (mean-combine, low-rank sub-tables, norm-matched init, orders 2–4) and
swept:

- **Hash buckets** (s3500, marin baseline 1.2376): 65k → 786k → 3.1M gives 1.2560 → 1.2497 → 1.2425,
  monotone — collision diagnosis confirmed, the method recovers as the hash vocab grows.
- **Contribution ratio** (the real knob): 0.25 / 0.5 / 0.75 / 1.0 / 2.0 → 1.2328 / 1.2353 / 1.2412 /
  1.2505 / 1.2838. A *light* n-gram wins (0.25 = −0.4% BPB); a heavy one injects init noise.
- **Composition** (n-gram at ratio 0.25 on each base tokenizer, feBPB added): marin +0.0% (washes
  out by s8000) · superbpe-128k −0.5% · gpt-neox-50k −0.8% · 80k-t40k −0.3%. The incremental
  *shrinks* as the base tokenizer improves — the superword pretokenizer already captures
  multi-token context the n-gram would supply (**partially redundant**), but on every non-marin
  tokenizer the gain persists to the highest budget.
- **Scale** (the 10% question): the paper claims the n-gram gain grows with activated params. We
  ran a hidden-2048 (4× params) ladder, marin baseline vs +n-gram at the fixed hidden-1024 best
  config. [SCALE Δ PENDING — compare to the hidden-1024 Δ of −0.4%; a larger Δ = the lever scales.]

The module (`levanter.grug.NgramInputEmbed`) is implemented, verified a bit-exact no-op when off
and strictly causal, and gated behind `BAKEOFF_NGRAM`.

## Levers investigated and closed out

- **TokenMonster** (#5837) and the linked #4971/#5821/#5842 options: measured, strictly weaker
  than SuperBPE on bytes/token, and not drop-in (Go/cgo, no `tokenizer.json`). Skip.
- **Larger off-the-shelf superword** (superbpe-180k): +0.6% bytes/token over 128k (negligible), and
  the trained 160k result shows bigger vocab hurts feBPB. 128k is the off-the-shelf superword sweet
  spot; small-vocab trained is better still.
- **Plain BPE trained on our mix**: beats matched-vocab Llama-3 on bytes/token but stays well below
  any SuperBPE — the superword mechanism dominates the vocab-training effect.

## Recommendations

1. **Adopt a small-vocab (~80k) SuperBPE trained on the grug-moe mix.** It is the best measured
   tokenizer, **−6.1% feBPB** over stock Llama-3 — better quality per training FLOP *and* competitive
   serving cost. The training harness (`experiments/tokenize/{corpus,train_tokenizers,
   push_trained_tokenizers}.py`) is reproducible; the tokenizer loads by name through the existing
   `levanter.load_tokenizer` path. Confirm the exact vocab optimum from the 64k/96k bracket.
2. **Optionally add the hashed n-gram input embedding at ratio ~0.25** for a further ~0.3% at ~0
   serving cost (−6.4% composed). Its value grows toward the target scale if the scale ladder
   confirms the trend — revisit at full scale.
3. **10% is not reached at proxy scale by tokenizer + n-gram alone.** The honest ceiling here is
   ~−6.4%; closing to 10% depends on the n-gram's scale-dependence (measured in §Axis D) and is a
   target-scale question, not a proxy-scale one.

## Reproduce

```bash
# serving cost, any deployment assumption (no cluster)
uv run python -m experiments.tokenize.bakeoff_analysis \
  --fertility experiments/tokenize/results/fertility_trained.json \
  --domain-weights english_web=0.8,math=0.2
# full feBPB scorecard once ladders are collected (assembler folds composed arms + re-keys trained)
uv run python -m experiments.tokenize.bakeoff_analysis \
  --fertility <assembled_fert.json> --bpb <assembled_bpb.json> \
  --domain-weights english_web=0.8,math=0.2 --reference marin-128k
```

Per-experiment reproduce/replay commands: `experiments/tokenize/EXPERIMENT_LOG.md`.
