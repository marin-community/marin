# Dense one-layer SGD-MH LR scaling

This experiment matches `dense_one_layer_sgdh`, but matrix parameters use a
0.95 momentum trace with Nesterov lookahead before the Frobenius Hyperball
projection. It does not apply Newton--Schulz orthogonalization. Output
projections use AdamH, while embeddings and norm gains use Adam.

The constant and linear-decay launchers each sweep 30x, 60x, 150x, 300x, and
600x token budgets at LR multipliers 0.10, 0.20, 0.32, 0.45, and 0.70. Every
run uses one dense d512 layer, MLP width 1792, batch size 64, sequence length
8192, seed 0, and 1% warmup. The linear schedule decays to 5% of peak LR.
