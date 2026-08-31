# Hero d6144 — layer-0 attention gate analysis (step 42000)

`hero_layer0_gate_vocab.png` — the layer-0 headwise attention gate
(`gate = 2·sigmoid(attn_in @ attn_gate)`, identity = 1.0) evaluated over the full
128,256-token vocabulary. The gate is computed pre-attention from each token's own
embedding, so these are the values every token induces in any context.

Findings at step 42000:
- 98.3% of the 6.15M (token × head) gates are shut (< 0.05); mean 0.013, median 0.007.
- Logits (`attn_in @ attn_gate`) have mean −6 and are 98.7% saturated, so the large
  learned layer-0 gate norms (per-head L2 ~7–10) clamp the gate off rather than modulate it.
- Only 546 / 128,256 tokens (0.43%) open any head; each head opens for ≤ 0.33% of the
  vocabulary (head 2 is the most active).

Net: layer-0 self-attention is gated ~off for nearly the entire vocabulary.
