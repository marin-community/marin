# Tokenizer FLOP-equivalent bake-off — findings

Companion results doc to the protocol in
[`20260703_tokenizer_flop_equivalent_bakeoff.md`](20260703_tokenizer_flop_equivalent_bakeoff.md).
That doc is the *design* (what we measure and why); this doc is the *result* (what we found).
It is updated as the isoFLOP ladder completes — sections marked **[in-flight]** are filled from
the running runs.

## TL;DR

**Swapping the grug-moe tokenizer from stock Llama-3 to SuperBPE-128k is a clear,
compute-fair win: −4.7% FLOP-equivalent BPB at equal vocabulary.** It wins on *both* axes at
once:

- **Quality**: at identical training FLOPs and identical 128k vocab, SuperBPE-128k reaches **~3%
  lower BPB than Llama-3 at every point on the isoFLOP ladder** — because packing more bytes per
  token lets it ingest ~30% more text per training FLOP.
- **Cost**: it emits **~18% fewer tokens per byte** on grug-moe's English/math/code mix, so it
  makes ~18% fewer forward passes per served byte — a −18% cut in serving FLOPs/byte with the
  output head unchanged (same vocab).

The combined FLOP-equivalent BPB (feBPB) is **1.179 vs Llama-3's 1.238 (−4.7%)**, and SuperBPE
wins under every deployment assumption we replayed (natural multilingual mix −1.9%, 64k context
−4.7%, serving-heavy −6.0%).

A second lever we built and tested — a **hashed n-gram input embedding** (Over-Tokenized /
LongCat style, ~0 added serving FLOPs) — did **not** pay off at this flash-scale proxy: it is a
small BPB *regression* that narrows with training budget, consistent with the literature's
finding that n-gram embeddings help only at larger model/token scale. It is implemented, correct,
and gated off by default; see §N-gram.

Everything is measured under the FLOP-equivalent rubric (below): quality is bits-per-byte (BPB,
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

## Quality axis (BPB isoFLOP ladder) — measured

Full 3-point ladder on cw-rno2a (proxy model held fixed: hidden 1024, 16 layers, 32 experts,
top-4, seq 1024, batch 128; only `vocab_size` follows the arm). Held-out BPB on the
Uncheatable-Eval subsets at each training-FLOP budget (raw points in
`experiments/tokenize/results/ladder.json`):

| arm | s1500 | s3500 | s8000 | BPB vs marin @ matched FLOPs |
|---|---|---|---|---|
| **superbpe-128k** | **1.336** | **1.200** | **1.114** | **−3.0% / −3.0% / −2.9%** |
| marin / llama3-128k | 1.378 | 1.238 | 1.147 | reference |
| gpt-oss-200k | 1.366 | 1.230 | 1.135 | better raw BPB, but +30–60% FLOPs (bigger head) |
| qwen3-152k | 1.377 | 1.241 | 1.148 | ~equal to marin |
| gpt-neox-50k | 1.332 | 1.206 | (rerun) | best raw BPB, but small vocab → expensive serving |

superbpe-128k sits at a strictly lower BPB than marin at every budget, at essentially identical
training FLOPs (same vocab): 3.93e17→1.336, 9.17e17→1.200, 2.10e18→1.114 versus marin's
1.378/1.238/1.147. gpt-oss-200k reaches a comparable BPB only by spending 30–60% more training
FLOPs on its larger output head — exactly the inefficiency the rubric is built to expose.

## FLOP-equivalent ranking (feBPB) — measured

At the 250B/20B deployment target, 16k context, grug-moe's English/math serving mix
(`bakeoff_analysis.py --domain-weights english_web=0.8,math=0.2`; reference = marin-128k):

| arm | vocab | rel. serving cost | feBPB | vs marin |
|---|---|---|---|---|
| **superbpe-128k** | 128k | **0.816** | **1.179** | **−4.7%** |
| marin / llama3-128k | 128k | 1.000 | 1.238 | reference |
| gpt-oss-200k | 200k | 0.993 | 1.257 | +1.6% |
| qwen3-152k | 152k | 1.158 | 1.275 | +3.2% |

superbpe-128k wins by combining its raw-BPB edge with the serving discount reinvested into
training. The win is robust across replayed deployment assumptions:

| scenario | superbpe feBPB | vs marin |
|---|---|---|
| English/math, 16k context | 1.179 | −4.7% |
| natural multilingual mix, 16k | 1.214 | −1.9% |
| English/math, 64k context | 1.180 | −4.7% |
| English/math, serving-ratio 2 | 1.163 | −6.0% |

The margin shrinks on the multilingual mix (superbpe-128k's Chinese fertility is poor) and grows
when serving is weighted more heavily — both the expected directions.

## N-gram result — measured (does not pay off at flash scale)

marin-128k with vs without the hashed n-gram input embedding (identical tokenizer, identical
FLOPs; `init_std_scale=0` clean-ablation start). The n-gram adds input-side embedding parameters
at ~0 serving FLOPs — but at this proxy scale it is a small BPB *regression*, not an uplift:

| budget | marin | marin + n-gram (init 0) | Δ | marin + n-gram (init 1.0) |
|---|---|---|---|---|
| s1500 | 1.378 | 1.424 | +3.3% | 1.475 |
| s3500 | 1.238 | 1.261 | +1.9% | 1.278 |
| s8000 | 1.147 | 1.157 | +0.9% | 1.174 |

Two clean sub-findings: (a) `init_std_scale=0` (start identical to baseline, let the zero tables
grow) beats `init_std_scale=1.0` at every budget — a large nonzero init injects input noise
several times the base embedding and the model spends its budget recovering; (b) even with the
good init, the gap to marin *narrows monotonically with training* (+3.3% → +1.9% → +0.9%). The
converging trend is consistent with Over-Tokenized (arXiv 2501.16975) / LongCat, where n-gram
input embeddings pay off at model and token scales well beyond this flash-scale proxy. The module
(`levanter.grug.NgramInputEmbed`) is implemented, verified a bit-exact no-op when off and strictly
causal, and left gated off — worth revisiting at target scale, not adopted now.

## Recommendations

1. **Adopt SuperBPE-128k as the grug-moe tokenizer for English/code/math.** It is a −4.7%
   FLOP-equivalent improvement over stock Llama-3 at equal vocabulary — better quality per byte
   *and* ~18% cheaper to serve — robust across context length and serving weight. It is a
   drop-in swap: same 128k vocab, same output head, loaded through the existing
   `levanter.load_tokenizer` path (which preserves the superword pretokenizer;
   `AutoTokenizer.from_pretrained` does not — see below).
2. **Do not use superbpe-128k for a multilingual target** — it regresses badly on Chinese
   (−37% bytes/token). superbpe-180k keeps the cross-lingual fertility win but at a larger head
   (180k vocab); re-score it with a trained ladder before committing.
3. **Do not add the n-gram input embedding at flash scale** — it does not help here. Keep the
   implementation for a larger-scale revisit; the converging trend suggests it may pay off past
   ~5e18 training FLOPs / larger hidden size.

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
