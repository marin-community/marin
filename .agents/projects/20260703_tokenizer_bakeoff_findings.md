# Tokenizer FLOP-equivalent bake-off — findings

Companion results doc to the protocol in
[`20260703_tokenizer_flop_equivalent_bakeoff.md`](20260703_tokenizer_flop_equivalent_bakeoff.md).
That doc is the *design* (what we measure and why); this doc is the *result* (what we found).
The full chronological experiment log, with a reproduce/replay command per experiment, is
[`experiments/tokenize/EXPERIMENT_LOG.md`](../../experiments/tokenize/EXPERIMENT_LOG.md).

## TL;DR — corrected result

**Read this first: the headline changed.** The proxy-scale isoFLOP result (a −4.7…−6.8% feBPB win
for small-vocab trained SuperBPE, ≥10% at high serving weight) was measured on tokenizers with an
**English-only stage-2 superword sample** and a **7-domain English+code eval**. Fixing both bugs and
re-running the winning arms as full grug-moe soaks on 8×H100×64 (cw-rno2a, W&B group
`tokenizer-soak`) **did not reproduce a scale-robust win.** The proxy-scale sections below are kept
as the methodology and reasoning trail; the corrected conclusion is here. Full detail in
[#6796](https://github.com/marin-community/marin/issues/6796) comments.

**The SuperBPE advantage is budget-dependent and reverses at scale.** Scored on a domain-fair macro
over the 7 shared English+code domains (the reused Llama-3 `marin-128k` baseline only ever eval'd
those), raw BPB of the 128k SuperBPE arms vs the baseline:

| train FLOPs | marin (Llama-3) | 128k-fixed | 128k-llama | 128k-digits |
|---|---|---|---|---|
| 6e19  | 0.9615 | +0.4% | +1.5% | +0.9% |
| 1.5e20 | 0.9370 | +1.1% | +2.0% | +1.8% |
| 2.6e20 | 0.9062 | +1.9% | +2.7% | +2.3% |

The 128k arms start ~level with Llama-3 at low budget and fall **progressively further behind** — the
gap roughly doubles by 2.6e20. On feBPB at 2.3e20, Llama-3 leads 128k-fixed by **+1.6%** even after
crediting its 17% serving-cost discount.

**Where SuperBPE wins: only small vocab, only at low budget.** At a common 6e19 budget the trained
`64k-fixed` arm beats Llama-3 by ~0.8% feBPB (−0.77%), and 5 of 7 SuperBPE arms nominally edge it
out — but that is the low-budget corner. The 64k arms have no data past ~8e19, so their edge is
unverified at scale, and the 128k reversal argues it would not survive.

**The durable, real benefit is serving density.** SuperBPE packs 17–21% more bytes/token (4.73 vs
3.92 for 128k-fixed), so it is that much cheaper to serve. Whether that justifies adoption is a
serving-economics question — if the deployment is serving-dominated (high serving/training ratio ρ)
the density can pay for a raw-BPB deficit — not a quality-per-training-FLOP win: at ρ=1 SuperBPE does
not beat Llama-3 at scale.

**Secondary findings.** Digit pretokenization loses at every budget. The SuperBPE case-split
pretokenizer beats the Llama word-regex by ~0.8–1% raw BPB at identical fertility (a pure
segmentation effect, mostly from C++), but both 128k variants still lose to the Llama-3 baseline at
scale. Per-domain, SuperBPE is a large C++ win (−7%) and a large Python loss (+9%) that cancel in the
macro. Caveat: `marin-128k` is reused from the original run, not retrained alongside these arms — the
*growing* gap argues against a static confound, but a fresh Llama-3 arm under the identical config
would close it.

**Bottom line.** The original "washout at scale" is **largely real, not just a bug artifact.** The
three fixes narrowed it and made the small-vocab/low-budget corner favorable, but the tokenizer lever
does **not** deliver a scale-robust quality-per-FLOP win for grug-moe. SuperBPE's value is
serving-cost density, weighed against a raw-BPB deficit that grows with training budget.

---

*The sections below are the original **proxy-scale** investigation (hidden-1024 isoFLOP ladders on
SlimPajama). Its headline feBPB numbers are superseded by the corrected re-run above; the rubric,
method, and per-lever decomposition remain valid and are what the re-run was built on.*

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
| trained-superbpe-40k-t20k | 40k | 1.1163 | **1.1560** | **−6.6%** |
| **trained-superbpe-64k-t32k** | 64k | 1.1141 | **1.1564** | **−6.6%** |
| trained-superbpe-48k-t24k | 48k | 1.1130 | 1.1567 | −6.6% |
| trained-superbpe-80k-t40k | 80k | 1.1107 | 1.1621 | −6.1% |
| trained-superbpe-128k-t51k | 128k | 1.1081 | 1.1836 | −4.4% |
| trained-superbpe-128k-t102k | 128k | 1.1252 | 1.1918 | −3.7% |
| trained-superbpe-160k-t64k | 160k | 1.1216 | 1.2028 | −2.8% |

**feBPB falls as trained vocab shrinks (160k → 80k), then flattens into a broad plateau at ~40–64k,
saturating at −6.6%.** 40k / 48k / 64k are tied within 0.0007 feBPB. Three lessons: (1) a *small*
vocab is a *smaller model* — cheaper per training FLOP, so more effective steps at a fixed budget →
lower BPB — and the superword pretokenizer keeps fertility high enough that the serving penalty is
outweighed; (2) below ~64k the shrinking-model gain is **exactly offset** by rising fertility (fewer
bytes/token → costlier serving), so the lever saturates rather than continuing down — this is the
ceiling of the tokenizer lever; (3) *big*-vocab trained SuperBPE (128k/160k) is **worse** than
off-the-shelf superbpe-128k (it over-merges a small corpus). The win is specifically at small vocab:
the "gpt-neox efficiency × superword packing" sweet spot (plain gpt-neox BPE at a comparable 50k is
only −5.1% — the superword mechanism is what holds the plateau).

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
- **Scale** (the 10% question): the paper claims the n-gram gain grows with activated params. We ran
  a hidden-2048 (4× params) ladder, marin baseline vs +n-gram at the hidden-1024 best config. The
  n-gram's payoff **shifts to higher training budgets** as the model widens rather than growing in
  magnitude:

  | budget | h-1024 base | +n-gram | Δ | h-2048 base | +n-gram | Δ |
  |---|---|---|---|---|---|---|
  | s3500 | 1.2376 | 1.2328 | −0.39% | 1.1833 | 1.1837 | +0.03% |
  | s8000 | 1.147 | ~1.147 | ~0% | 1.0944 | 1.0903 | −0.37% |

  At hidden-1024 the n-gram helps early then washes out by s8000; at hidden-2048 it is neutral early
  then helps by s8000 (−0.37%). The peak Δ is ~0.4% at *both* scales — consistent with "n-gram helps
  at scale" (LongCat), but the magnitude does **not** grow, so it is not a multi-percent lever. A
  rank-256 (paper-scaled sub-dim) confirmation to rule out an injection-width confound proved
  impractical — the rank-256 hash tables are ~6.1 B params and shard poorly here (~65 s/step, ~60 h
  to finish), so it was killed. The residual uncertainty is small: the rank-128 n-gram is already
  ~3 B params (3.7× the hidden-2048 model), so the lever is not param-starved, and even that
  over-provisioned embedding caps at ~0.4%.

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

## Recommendations (corrected — after the fixed-tokenizer re-run)

1. **Do not switch grug-moe to SuperBPE for a quality-per-FLOP win.** The fixed-tokenizer re-run
   shows the proxy-scale advantage reversing at scale: the 128k SuperBPE arms fall progressively
   behind Llama-3 (up to +1.9% raw BPB by 2.6e20), and Llama-3 leads on feBPB at scale even after
   its serving-cost discount. The only regime where SuperBPE wins is small-vocab + low-budget, which
   is not the deployment regime and is unverified past ~8e19.
2. **Treat SuperBPE as a serving-cost lever, not a quality lever.** Its durable, real benefit is
   17–21% more bytes/token → cheaper inference. If a specific deployment is heavily serving-dominated
   (very high ρ), the density can justify SuperBPE despite the raw-BPB deficit — decide per
   deployment on the serving-economics numbers, not on a headline feBPB.
3. **Avoid digit pretokenization** — it loses at every budget. If SuperBPE is pursued, the
   case-split pretokenizer beats the Llama word-regex at equal fertility (a small segmentation gain,
   mostly C++), so prefer it over the Llama regex.
4. **Close the last confound before any adoption decision:** retrain a Llama-3 `marin-128k` arm under
   the identical soak config (rather than reusing the original run) to rule out a residual
   training-setup difference, and extend the 64k arms past ~2e20 to confirm whether their low-budget
   edge is real or also reverses.

The harness (`experiments/tokenize/{corpus,train_tokenizers,push_trained_tokenizers}.py`, the
`ServingCostModel`/feBPB scorer, and the soak ladder `soak_wandb_ladder.py`) is reproducible and is
the reusable artifact regardless of the verdict.

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
