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
- **Result**: _pending (512g wave)_
- **Conclusion**: _pending_

## EXP-005 — n-gram (paper config b4M) stacked on superbpe-128k _(pending)_

- **Hypothesis**: the n-gram lever is orthogonal to the tokenizer, so superbpe-128k (−4.7% feBPB) +
  n-gram compounds toward the 10% feBPB goal. The n-gram adds ~0 serving FLOPs, so any BPB drop is
  a near-pure feBPB gain on top of superbpe's serving discount.
- **Config**: base superbpe-128k, b4M paper config (mean, orders 2,3,4, rank 128, ratio 1.0),
  full ladder (s1500/s3500/s8000), SCALE_RAM 512g.
- **Reproduce**: `launch_tokenizer_bakeoff` with `BAKEOFF_ARM=superbpe-128k BAKEOFF_NGRAM=1 …b4M…`
  at each step point (job-name `grug-ngram-superbpe-128k-b4M-s<steps>`).
- **Result / Conclusion**: _pending_

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
