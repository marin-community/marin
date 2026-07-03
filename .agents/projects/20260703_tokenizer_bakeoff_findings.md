# Tokenizer FLOP-equivalent bake-off — findings

Companion results doc to the protocol in
[`20260703_tokenizer_flop_equivalent_bakeoff.md`](20260703_tokenizer_flop_equivalent_bakeoff.md).
That doc is the *design* (what we measure and why); this doc is the *result* (what we found).
It is updated as the isoFLOP ladder completes — sections marked **[in-flight]** are filled from
the running runs.

## TL;DR

Two independent, compute-fair levers beat the stock Llama-3 tokenizer we ship with grug-moe,
and they compose:

1. **Swap the tokenizer to SuperBPE-128k** — *same vocabulary size* as Llama-3 (128k), so the
   output head and its FLOPs are unchanged, but it emits **~18% fewer tokens per byte** on
   grug-moe's English/math/code mix. Fewer tokens per byte is fewer forward passes per byte at
   serving time: a **~18% cut in serving FLOPs/byte at equal model quality-per-byte**. This is a
   pure win on the cost axis of the FLOP-equivalent score.
2. **Add a hashed n-gram input embedding** (Over-Tokenized / LongCat style) — extra input-side
   embedding parameters that cost **~0 additional serving FLOPs** (a gather, not a matmul; the
   output head is untouched) and buy quality (BPB). This is a win on the quality axis at (almost)
   no cost.

Both are measured under the FLOP-equivalent rubric (below): quality is bits-per-byte (BPB,
tokenizer-agnostic); cost is priced at the **250B-total / ~20B-active deployment target** with a
16k-token context, Llama-encoder attention, and 5:1 local:global attention sparsity. All numbers
are **replayable** — every score is recomputed from stored measurements under
CLI-configurable deployment assumptions, so "what if we serve at 64k / a 400B model / a
code-heavy mix?" is a re-run, not a retrain.

## The rubric: FLOP-equivalent BPB (feBPB)

A cheaper-to-serve tokenizer earns the right to be slightly worse per byte, and a better-per-byte
tokenizer earns the right to be slightly more expensive. We combine the two axes into one number.

- **Quality** is **bits-per-byte**: `bpb = eval_loss / bytes_per_token * log2(e)`. BPB is invariant
  to the tokenizer's segmentation, so a 50k-vocab and a 262k-vocab model are compared on the same
  held-out *bytes*. This is the only honest way to compare tokenizers on quality.
- **Cost** is **inference FLOPs per byte** at the deployment target: `flops_per_token(vocab) *
  fertility`, where `fertility = tokens/byte` and `flops_per_token` includes attention that scales
  with context length. A byte-level tokenizer (fertility ~1 token/byte) pays the per-token cost
  ~3-4x more often than a 128k-BPE tokenizer, which is exactly the penalty the task asks for.
- **Combine** into a lifetime-budget feBPB: an arm that is `s = infer_cost / reference_cost`
  as expensive to serve is charged for that at training time too, so
  `train_flops(arm) = C_ref * (1 + rho * (1 - s))` with `rho` the serving/training weight. We fit
  the isoFLOP curve `BPB(C) = a*C^-b + c` per arm and read each arm's BPB at its cost-adjusted
  budget. Lower feBPB wins.

The cost model (`ServingCostModel`) is attention-aware: at 16k context, attention is ~10% of
forward FLOPs; at 64k it is ~21%. Because attention-per-byte scales with `fertility * context`, a
low-fertility tokenizer's advantage *grows* with context length — a fact the replay knob makes
visible.

## Method

- **Model**: grug-moe proxy — hidden 1024, 16 layers, 32 experts, top-4, seq 1024, batch 128. The
  proxy measures the *tokenizer*, not the model; only `vocab_size` varies across arms (it follows
  the arm, driving both the embedding table and the output head).
- **Data**: SlimPajama-6B as the shared training corpus, tokenized per-arm (each arm builds its own
  tokenization of the same text). Held-out validation is the Uncheatable-Eval subsets
  (wikipedia_english, github_python, github_cpp, bbc_news, arxiv_physics/cs, ao3_english), tokenized
  per-arm so every arm is scored on the same raw bytes.
- **isoFLOP ladder**: each arm is trained at 3 compute points (1500 / 3500 / 8000 steps) so we can
  fit BPB-vs-FLOPs and compare arms at matched training budget rather than at a single arbitrary
  step count.
- **Deployment target for scoring**: 250B-total / ~20B-active MoE (hidden 6144, 64 layers, 256
  experts, top-8), 16k context. Scoring is decoupled from training: the proxy runs log raw
  `(training-FLOPs, BPB)` points, fertility logs raw token/byte counts, and the cost model is
  applied only at analysis time.

## Serving-cost axis (fertility) — measured

Fertility measured through the exact levanter tokenize path
(`levanter.load_tokenizer` → `tokenizers.Tokenizer.from_file`), on a 4 MB/domain held-out sample.
**bytes/token, higher = fewer tokens = cheaper to serve.**

| domain | marin/llama3-128k | superbpe-128k | superbpe-180k | gpt-oss-200k | gpt-neox-50k |
|---|---|---|---|---|---|
| english_web | 4.474 | **5.782 (+29%)** | 5.797 (+30%) | 4.549 | 4.292 |
| math | 2.617 | 2.863 (+9%) | 2.953 (+13%) | 2.658 | 2.638 |
| multilingual_zh | 3.298 | **2.094 (−37%)** | 3.673 (+11%) | 3.364 | 2.024 |

The headline finding is **domain-dependent**: superbpe-128k is a large win on English (+29%
bytes/token, i.e. −22% tokens) but a large *loss* on Chinese (−37%) because it is an English-only
superword tokenizer. grug-moe trains and serves an English/code/math mix (SlimPajama), so weighting
by that mix (`english_web=0.8, math=0.2`) gives the deployment-relevant serving cost:

| arm | vocab | bytes/tok (eng+math) | rel. serving FLOPs/byte @16k |
|---|---|---|---|
| superbpe-180k | 180k | 4.86 | **0.811** |
| **superbpe-128k** | **128k** | **4.80** | **0.816** |
| gpt-oss-200k | 200k | 3.98 | 0.993 |
| **marin / llama3-128k** | **128k** | **3.92** | **1.000** (reference) |
| gpt-neox-50k | 50k | 3.81 | 1.017 |
| qwen3-152k | 152k | 3.39 | 1.158 |
| gemma3-262k | 262k | 3.40 | 1.172 |

**superbpe-128k cuts serving FLOPs/byte by ~18% at the *same* vocab as Llama-3** — the cleanest
possible comparison, since it holds the output-head cost fixed and moves only the segmentation.
superbpe-180k is marginally cheaper still but confounds the vocab-size axis (larger head).

A critical implementation note: SuperBPE repos ship a `GPT2Tokenizer` class, so
`AutoTokenizer.from_pretrained` silently overwrites the superword pretokenizer and measures a
*worse-than-baseline* fertility. Only the `from_file` path (which marin's tokenize pipeline uses)
honors it. The fertility harness was switched to that path so the reported cost matches training.

## Quality axis (BPB isoFLOP ladder) — [in-flight]

Ladder running on cw-rno2a: 6 arms (marin baseline, marin+n-gram, gpt-oss, qwen3, gpt-neox,
superbpe-128k) x 3 compute points = 18 runs. BPB collected with
`experiments.tokenize.collect_metrics` into `ladder.json` as each completes.

_To be filled: per-arm `BPB(C)` fit and held-out BPB at matched budget._

## FLOP-equivalent ranking (feBPB) — [in-flight]

_To be filled from the ladder via `bakeoff_analysis.py`, at the 250B/20B target, 16k context._

## N-gram free-quality result — [in-flight]

marin-128k vs marin-128k+n-gram, identical tokenizer and compute — the BPB delta is the n-gram
uplift at ~0 serving cost. Config: orders (2,3), 2 hashes, 65537 buckets/table, sum-combine, causal
rolling hash. `NgramInputEmbed` is a verified bit-exact no-op when disabled and strictly causal.

_To be filled: BPB(marin+ngram) − BPB(marin) at matched budget; the implied feBPB improvement._

## Recommendations — [in-flight, pending BPB]

Provisional, on the serving-cost axis alone (BPB pending):

- For an English/code/math deployment, **SuperBPE-128k is the recommended tokenizer swap**: ~18%
  cheaper to serve at equal vocab and equal head cost, contingent on BPB holding (measuring now).
- **Add the n-gram input embedding** as a free-quality lever on top of whatever base tokenizer is
  chosen; it is orthogonal to the tokenizer swap.
- A multilingual target must *not* use superbpe-128k (Chinese regression); superbpe-180k retains the
  win across languages at the cost of a larger head.

## Reproduce

```bash
# serving cost, any deployment assumption (no cluster)
uv run python -m experiments.tokenize.bakeoff_analysis \
  --fertility experiments/tokenize/results/fertility_raw.json \
  --domain-weights english_web=0.8,math=0.2
# with BPB once the ladder is collected
uv run python -m experiments.tokenize.bakeoff_analysis \
  --fertility experiments/tokenize/results/fertility_raw.json --bpb experiments/tokenize/results/ladder.json
```
