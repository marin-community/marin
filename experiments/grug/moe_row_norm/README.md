# July MoE Row-Norm Variant

This experiment is copied directly from `experiments/grug/moe` at the
`july_baseline` head, commit `52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`.
It keeps the July architecture, data, schedule, initialization, and optimizer
scalars fixed while changing every linear from `Wx` to `v * (Wx)`.

- Every `v` starts at one and uses norm-preserving AdamH.
- Baseline MuonH matrices use per-output-row MuonH. Grug stores matrices as
  `(input, output)` (or `(expert, input, output)`), so row norms reduce over
  stored axis `-2`.
- Matrices assigned to AdamH in July, notably the LM head, stay on AdamH.
- Router and zero-initialized attention-gate matrices retain their July Adam
  routing; their new scale vectors use AdamH.

The launcher runs the d512 variant and an unmodified July d512 control together
on v5p-8 with separate checkpoints and W&B identities:

```bash
uv run python -m experiments.grug.moe_row_norm.launch
```

The paired recipe is 8192-token context, batch 16, 10,980 steps, 256 experts,
top-4 routing, 4:1 GQA, no PKO, no long-window RoPE, and z-loss 0.
