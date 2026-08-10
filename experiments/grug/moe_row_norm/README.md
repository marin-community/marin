# Factorized output scales with row-wise MuonH

Status: active experiment. Origin: [`experiments/grug/moe`](../moe/README.md).

Every linear projection is parameterized as

\[
f(x; v, W) = v \odot (xW),
\]

using Grug's stored `(input, output)` matrix convention. `W` is initialized
identically to the MoE baseline and every `v` starts at one, so the effective
model is unchanged at initialization.

The optimizer preserves one norm for each mathematical output row of `W`.
Because Grug stores transposed linear matrices, this reduces over the
penultimate input axis: axis 0 for `(input, output)` and axis 1 for
`(expert, input, output)`. Each `v` uses AdamH with a fixed L2 norm; expert
scales preserve one vector norm per expert.

Parameter groups retain the baseline exceptions:

- Baseline MuonH matrices use row-wise MuonH.
- The LM-head `W` remains on AdamH.
- Router and zero-initialized attention-gate `W` remain on Adam.
- All `v` parameters use vector AdamH.
- Embeddings, biases, and RMSNorm scales remain on Adam.

Run the two Gate 1 cells with:

```bash
uv run python -m experiments.grug.moe_row_norm.launch \
  --version 2026.08.10 \
  --run \
  --max-concurrent 2
```
