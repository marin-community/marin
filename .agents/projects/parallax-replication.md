# Parallax head-to-head replication (arXiv 2605.29157) on our data

Goal: reproduce the paper's 0.6B Muon comparison, **Transformer vs Parallax (RoPE on ρ)**, with the paper's
architecture and optimizer recipe, on the v16384-tokenized data. First a reduced run (5k steps, ~20B tokens), then
the full 20k steps (78.6B tokens) if the reduced pair looks right.

## Paper setup (Tables 2, 5; Appendix F)

| | value |
|---|---|
| backbone | Qwen3 (torchtitan): 28 layers, d 1024, SwiGLU FFN 3072, 16 Q / 8 KV heads, head dim 128, RoPE θ 1e6 (full rotary), tied embeddings, RMSNorm on q and k |
| Parallax | every layer; `W_R` shares W_Q's head layout (16 × 128); RMSNorm on ρ; RoPE on ρ (same θ); unscaled `r·k` |
| optimizer | Muon lr 5e-3, wd 0.1, momentum 0.95, 5 NS steps (quintic), spectral scaling; non-matrix params: Adam-style with β (0.8, 0.95), ε 1e-7, embedding lr 0.3×, norm lr 0.015×; global grad clip 1.0 |
| schedule | WSD: 0% warmup, linear decay over the last 20%, final lr 0 |
| batch / steps | 3.93M tokens = 960 × 4096, 20,000 steps, 78.6B tokens |
| precision | torchao fp8 linears except the lm_head (we use bf16) |
| evals | LAMBADA ppl + acc, WikiText ppl, 0-shot BoolQ, HellaSwag, PIQA, ARC-e, ARC-c, WinoGrande, OBQA, SciQ (lm-eval-harness) |
| paper result (0.6B, Muon) | Transformer LMB 22.15 / Wiki 23.43 / avg 54.54; Parallax 18.56 / 22.25 / 55.99 |

Differences we accept: our tokenizer (hero-bpe-v16384, vocab 16384 vs 152k) and data, so only the head-to-head gap is
comparable; bf16 instead of fp8 linears.

## Phase 1: Parallax training kernel (JAX Pallas, Triton backend)

Port of `parallax/triton/parallax_train.py` (forward + 3 backward kernels), under
`lib/levanter/src/levanter/kernels/pallas/parallax/` following the `add-pallas-kernel` skill:

- `reference.py`: vanilla JAX oracle (the repo's `parallax_reference`: `out = O1/d1 (1 + d2/d1) - O2/d1`, causal,
  GQA, segment ids).
- `pallas_gpu.py`:
  - forward: online softmax over KV blocks with four accumulators (O1, O2, d1, d2) all rescaled by the same
    `alpha`; saves `o`, `barv = O1/d1` (bf16) and fp32 `m` (log2 units), `d1`, `bart = d2/d1`.
  - backward: preprocess `tau = dO·o`, `beta = dO·barv`; row pass (dQ, dR); column pass (dK, dV) per q head, GQA
    summed after. Formulas: `g1 = p (a - tau + (bart - t) delta)`, `g2 = -p delta`, `a = dO·v`, `delta = a - beta`,
    `dQ = scale Σ g1 k`, `dR = Σ g2 k`, `dK = scale Σ g1 q + Σ g2 r`, `dV = Σ p (1 + bart - t) dO`.
  - Block sizes from the repo's autotune space (BLOCK_M/N ∈ {32, 64, 128}); one tuned entry for (H100, bf16,
    D 128, L 4096).
- `api.py`: `parallax_attention(q, r, k, v, *, segment_ids, implementation=...)` with a `jax.custom_vjp`.
- Tests (`lib/levanter/tests/kernels/test_parallax.py`): value and gradient parity vs the reference on small
  shapes (CPU reference; GPU kernel), max-norm relative error ≤ 1e-2 at the median, like the repo's
  `scripts/parity_train.py`; causal, GQA (16/8), document segments.
- Benchmark vs our FlashAttention path at (B·H, L 4096, D 128): target ≤ ~2.5× FA forward+backward (the extra branch
  doubles the matmuls).

## Phase 2: model and recipe (levanter / marin)

- `Attention._compute_qkv` (`lib/levanter/src/levanter/layers/attention.py:1422`): add `r_proj` (same shape as
  `q_proj`), RMSNorm and RoPE on r; call the Parallax kernel instead of `dot_product_attention` (`:1349`). A
  `ParallaxAttention` subclass plus a config switch read by `LlamaDecoderLayer.init` (`llama.py:310`).
- `Qwen3Config` (`models/qwen.py:297`): 28L, d 1024, FFN 3072, 16/8 heads, head_dim 128, rope theta 1e6,
  `tie_word_embeddings=True`.
- Muon (`optim/muon.py`): add embedding and norm LR multipliers (0.3×, 0.015× of the Muon LR) and Adam β (0.8, 0.95),
  ε 1e-7 for the non-matrix group; make `max_grad_norm=1.0` a global clip (today only in the Adam branch).
  Schedule: `warmup=0, decay=0.2, lr_schedule="linear", min_lr_ratio=0`.
- Data: the flat v16384 cache (`experiments/grug/fast_track/launch.py` `V16384_CACHE_DIR`, `_flat_cache_data_config`),
  built into a `TrainLmConfig` directly (`train_lm` drops `flat_cache`).
- Batch 960 × 4096 on 8×H100 with gradient accumulation.
- Experiment file `experiments/parallax_replication.py`; launch on cw-rno2a, both arms in parallel.

## Phase 3: evals

- In-training lm-eval-harness at the end: `lambada_openai` (0-shot), and 0-shot BoolQ, HellaSwag, PIQA, ARC-e,
  ARC-c, WinoGrande, OBQA, SciQ (`experiments/evals/task_configs.py`).
- WikiText perplexity: levanter's harness lacks `loglikelihood_rolling`, so score WikiText-103 test as a tokenized
  eval set (token-level ppl in our tokenizer; report alongside the paper's word-level ppl with that caveat).
- Also the standard Paloma macro loss.

## Phase 4: runs

1. Reduced pair: 5,000 steps (≈19.7B tokens), same batch, LR and WSD shape. Check: loss curves, Parallax `COR`/probe
   norms growing under Muon (their Fig. 3), the eval gap.
2. Full pair: 20,000 steps if the reduced pair behaves.

## Open items

- `LlamaDecoderLayer` attention hook design (subclass vs config field).
- Gradient-accumulation microbatch size that fits Parallax's extra activations at 0.6B.
- Kernel speed: if the Pallas-Triton port is far from FA, the full run's cost grows; decide after the benchmark.
