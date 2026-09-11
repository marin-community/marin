# Dense one-layer SGD-H scaling sweep

This variant copies `experiments/grug/base` and runs its dense d512 Transformer
with one layer. It keeps the issue #7856 token horizons, 524,288 tokens per
step (batch 64 × sequence length 8192), seed 0, 1% warmup followed by constant
learning rate, regional datakit input, Paloma evaluation cadence, and v4-8
placement used by the matched one-layer MoE sweeps.

The 25-cell grid crosses `30x / 60x / 150x / 300x / 600x` with
`0.10x / 0.20x / 0.32x / 0.45x / 0.70x`. Dense attention/MLP matrices receive
the raw-gradient Hyperball update (SGD-H); the output projection uses AdamH and
embeddings/norm gains use Adam, matching the parameter-group policy of the MoE
SGD-H ablation.
