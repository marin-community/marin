# Dense one-layer MuonH LR scaling

This experiment matches `dense_one_layer_sgdh` except that matrix parameters use
MuonH: momentum and Nesterov are applied before the Newton–Schulz direction, followed
by the Frobenius Hyperball projection. Output projections use AdamH, while embeddings
and norm gains use Adam.

The sweep covers 30x, 60x, 150x, 300x, and 600x token budgets at LR multipliers
0.10, 0.20, 0.32, 0.45, and 0.70. Every run uses one dense d512 layer, an MLP width
of 1792, batch size 64, sequence length 8192, seed 0, and 1% warmup followed by a
constant learning rate.

`launch_linear_decay.py` runs the matched 25-cell comparison with 1% warmup
followed by linear decay to 5% of the peak learning rate. It uses separate
experiment, W&B, and artifact identities from the constant-LR sweep.
