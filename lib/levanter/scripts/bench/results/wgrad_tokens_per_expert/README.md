# wgrad-mode speedup vs tokens/expert (GFP8-OPT-P04)

FP8 ragged-dot speedup over the tuned bf16 baseline as a function of per-expert batch size, for the
two Mosaic weight-gradient strategies (`--mosaic-wgrad fp8` vs `bf16`). Establishes where the f8
cast-transpose wgrad overtakes the bf16-fallback wgrad, and confirms the operating regime favors fp8.

- **Date:** 2026-06-30
- **Hardware / recipe:** H100 (cw-us-east-02a), mixed E4M3 act/weight × E5M2 grad, best-vs-best tuned (n=40).
- **Shapes:** real d2560 model (D2560/F1280/E256/topk4), E_local=32 (EP8), tokens/expert ∈ {128,256,512,1024,2048,4096,8192}.
- **Logbook:** `.agents/logbooks/grug-fp8-ragged.md` (GFP8-OPT-P04).

## Data files (raw `orchestrate_fp8_autotune.py` `result_json` payloads)

| file | run | Iris job |
|------|-----|----------|
| `wgrad_fp8.json`  | `--mosaic-wgrad fp8`  | `/matt/iris-run-job-20260630-163824` |
| `wgrad_bf16.json` | `--mosaic-wgrad bf16` | `/matt/iris-run-job-20260630-163830` |

Each carries per-shape `bf16_best`, `fp8_best` (winning configs + median/CI), and `speedup_vs_bf16_best`.

## Result (median speedup over tuned bf16)

| tok/expert | fp8-wgrad | bf16-wgrad |
|---|---|---|
| 128  | 1.010 | 1.097 |
| 256  | 1.061 | 1.117 |
| 512  | 1.115 | 1.171 |
| 1024 | 1.207 | 1.159 |
| 2048 | 1.305 | 1.137 |
| 4096 | 1.380 | 1.151 |
| 8192 | 1.480 | 1.188 |

Crossover between 512 and 1024 tok/expert; fp8-wgrad wins at the operating point (≥1024) and the margin
widens with batch size. bf16-wgrad is flat (its wgrad never gets fp8's large-GEMM scaling).

## Reproduce

Regenerate the plot from the committed data (no cluster needed):

```bash
cd lib/levanter/scripts/bench
uv run --with matplotlib python plot_speedup_vs_tokens.py \
  --series "fp8-wgrad=results/wgrad_tokens_per_expert/wgrad_fp8.json" \
  --series "bf16-wgrad=results/wgrad_tokens_per_expert/wgrad_bf16.json" \
  --out results/wgrad_tokens_per_expert/wgrad_sweep.png
```

Re-run the cluster sweep (regenerate the data) — see `../../FP8_AUTOTUNE_RUNBOOK.md` for auth/wheel
setup; one H100x8 job per wgrad mode:

```bash
export KUBECONFIG=~/.kube/coreweave-iris-gpu
SHAPES="d2560_e32_t128,d2560_e32_t256,d2560_e32_t512,d2560_e32_t1k,d2560_e32_t2k,d2560_e32_t4k,d2560_e32_t8k"
for MODE in fp8 bf16; do
  uv run --no-sync iris --cluster=cw-us-east-02a job run \
    --gpu H100x8 --enable-extra-resources --extra gpu --cpu 32 --memory 128GB --disk 64GB --no-wait \
    -- bash -lc "set -euo pipefail; \
       uv run --no-sync python lib/levanter/scripts/bench/fp8_wheel_cache.py get /tmp/wheels; \
       JAXLIB_WHEEL=\$(ls /tmp/wheels/*.whl | head -1) bash lib/levanter/scripts/bench/mixed_fp8_fork_setup.sh; \
       uv run --no-sync python lib/levanter/scripts/bench/orchestrate_fp8_autotune.py \
         --shapes $SHAPES --mosaic-wgrad $MODE --num-gpus 8 --out-dir /app/scratch/fp8_wgrad_$MODE --worker-timeout 1200"
done
# then extract each job's `result_json {...}` line into wgrad_<mode>.json.
```
