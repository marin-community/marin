# Grug Variant Notes

Use this file for variant-specific guidance and examples.

## `parallax`

Dense Qwen3-0.6B (torchtitan layout and init) with optional Parallax attention (arXiv 2605.29157), for the
paper's Transformer vs Parallax head-to-head on our v16384 data. The Transformer arm runs FA4
(`gpu_fa4_cute`); the Parallax arm runs `levanter.kernels.pallas.parallax` per shard. The optimizer is
`GrugMuonConfig` with the paper's embedding/norm Adam groups (`embedding_lr_mult`, `norm_lr_mult`) and global
clipping; `train.py` adds gradient accumulation (`grad_accum_microbatches`) for the 960-sequence batch.
