# SonicMoE Token Gather/Sum Side-by-Side

This directory keeps the real upstream SonicMoE token gather/sum kernel and the
current JAX/Pallas port in parallel files so source, IR, and timing comparisons
do not depend on a monolithic benchmark script.

Files:

- `real_sonic_token_gather_sum.py`: faithful isolated Triton kernel from
  `sonicmoe/functional/reduction_over_k_gather.py` at commit
  `cfbd65f39b980b85b878b3cccdacb09191e24993`.
- `pallas_token_gather_sum_port.py`: adapter around the Levanter Pallas ports
  in `lib/levanter/src/levanter/grug/sonic_moe.py` using the same fixed-top-k
  reverse-scatter contract. The `pallas_triton_faithful` backend is the
  source-faithful Sonic control; token/block variants are optimized comparison
  candidates.
- `PORT_AWARE_DIFF.md`: semantic block-by-block map from real Sonic source to
  the closest Pallas port blocks.
- `FAITHFUL_GATHER_SUM_BENCH.md`: GH200 sweep results for the source-faithful
  Pallas control against real Sonic.
- `compare_token_gather_sum.py`: runs both sides with identical shape flags in
  separate subprocesses.
- `common.py`: shared CLI/config and timing helpers only.

Example GH200 command:

```bash
uv run --package marin --extra gpu --group dev python \
  .agents/scripts/sonicmoe_compare/compare_token_gather_sum.py \
  --install-deps \
  --tokens 8192 \
  --hidden 2048 \
  --experts 8 \
  --topk 4 \
  --dtype bf16 \
  --weighted \
  --pallas-backends pallas_triton_faithful,pallas_triton_token_kblock,pallas_triton,xla \
  --hidden-block 256 \
  --k-block 4 \
  --warmup 5 \
  --steps 20
```

The harness intentionally prints JSONL records. Use `--write-ir-dir` to keep
Triton artifacts for the upstream Sonic side and XLA/Pallas dumps for the JAX
side.
