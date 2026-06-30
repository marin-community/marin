# Idea 3 — Activation-Aware Muon (Ali Jadbabaie)

## The exact update (from Ali, confirmed)
For a linear layer W (d_out × d_in), momentum/gradient M (d_out × d_in), and the layer's
**input activations** A (d_in × n_tokens_in_batch):

    Σ  = A Aᵀ                         # activation second moment, d_in × d_in (PSD)
    D  = sign( M · Σ^{-1/2} ) · Σ^{-1/2}      # sign = matrix sign = polar factor = NS5 orthogonalization
    W ← W − η · D

i.e. **whiten the gradient on the input/activation side by Σ^{-1/2}, orthogonalize (Muon's
matrix-sign), then apply Σ^{-1/2} again.** Sanity: Σ = I ⟹ D = sign(M) = plain Muon
(verified: cos 1.0000). Σ^{-1/2} computed via eigh(AAᵀ) with damping.

This is the activation-aware analog of the square-root/HJB policy; Ali: "the activation
aware one should work." Kaiyue's earlier torch attempt **diverged** — likely LR / the
exponent / NS error accumulation; the damped eigh form here should be stable.

## Why it's hard: activation capture
The optimizer (optax transform) only sees gradients, not activations, and `AAᵀ` is NOT
recoverable from G alone (G = δAᵀ mixes the output-grad δ and the input A). So we must
capture each linear layer's **input second moment AAᵀ** during the forward and route it to
the optimizer step.

### Capture design (least-invasive faithful path)
1. **Emit per-Linear input Grams from the forward.** Levanter's qwen3/llama transformer is
   `Stacked` (scanned). `Stacked.scan` returns stacked per-block extras — so make each block
   return `(carry, grams_dict)` where grams_dict holds the input Gram for each of its linears
   (attn q/k/v/o, mlp gate/up/down). Gram for input x (NamedArray with axis `In`):
   `hax.dot(x, x.rename({In: In2}), axis=<token axes>)` → (In, In2). Non-scanned linears
   (lm_head) handled outside the scan.
2. **Route Grams as loss aux.** Trainer already does `eqx.filter_value_and_grad(loss_fn,
   has_aux=True)` and loss_fn returns `(loss, metrics)`. Extend aux to `(metrics, grams_tree)`;
   `microbatched` must **sum** grams across microbatches (Gram is additive over tokens).
3. **Thread to optimizer.** In `_train_step`, pass grams_tree to a custom optimizer
   (`optax.GradientTransformationExtraArgs`) via `update(grads, state, params, grams=...)`.
4. **Custom optimizer** (`levanter/optim/activation_aware.py`): for matrix layers, momentum
   M; D = sign(M Σ^{-1/2}) Σ^{-1/2} with NS5 + eigh; AdamW for embeddings/lm_head/biases. EMA
   of Σ per layer in opt state (β≈0.95) for stability. Normalize D's Frobenius norm (the
   Σ^{-1/2} factors change the scale) + aspect scaling, for LR transferability.

### Open knobs to sweep
- LR (primary; Kaiyue diverged → LR matters).
- Σ EMA β; damping λ in (AAᵀ + λI)^{-1/2}.
- Normalization of D (Frobenius to √(min) like Muon).

## De-risk before the full build
A tiny standalone JAX training (small MLP/transformer, trivial activation capture) to confirm
the update trains stably (vs Kaiyue's divergence) before the full levanter integration.

## Test
qwen3-130m, preemptible, marin-community/speedrun, vs Muon best (1.1663). New PR.
Status: formula pinned; optimizer core validated (reduces to Muon, jit-safe). Capture +
plumbing = remaining major work.

## De-risk result (standalone, 2026-06-20)
A standalone MLP on anisotropic-input synthetic regression (derisk_standalone.py): act-aware
Muon (damped-eigh Σ^{-1/2} + Frobenius-norm) **does NOT diverge** at any LR, reaches lower
loss than Muon (0.0015 vs 0.0020 best), and is more robust at high LR (act lr0.1=0.0021 stable
vs Muon lr0.1=0.0076 degrading). ⟹ the construction is sound; Kaiyue's divergence was missing
damping/normalization. Justifies the full 130m levanter integration (capture build).

## Build complete + launched (2026-06-21)
Faithful integration DONE on worktree /tmp/marin-actaware (branch activation-aware-muon):
- `lib/levanter/src/levanter/optim/activation_aware.py` — optimizer (D=NS5(MΣ^{-1/2})Σ^{-1/2},
  damped eigh + Frobenius norm; act-aware on q/k/v/gate/up, Muon on o/down, AdamW rest).
- `lib/levanter/src/levanter/optim/activation_capture.py` — per-layer block-input Grams via
  `Stacked.scan_via` (input_layernorm→q/k/v, post_attention_layernorm→gate/up), summed over tokens.
- trainer.py / trainer_state.py — `compute_activation_grams` flag → grams routed through
  take_step → take_train_step → optimizer.update(grams=).
- CPU smoke test (tiny qwen3) passes end-to-end.
- Sweep: experiments/speedrun/activation_aware_qwen3_scaling/sweep.py (LR × damping).
Validation run launched: /kaiyue/iris-run-job-20260622-004318 (LR=1x, damping=1e-3, v5p-8
preemptible us-central1). Reference: Muon 1.1663 (paloma bpb). Gradient-derived Σ is degenerate
(M(MᵀM)^{-1/2} already the polar factor) → real activation capture is required, hence this build.
