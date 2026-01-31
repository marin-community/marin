# SimPO Sweep Monitoring Guide

## Current Job

**Job ID**: `ray-run-ahmed-sweep_simp-20260129-234457`

**Status**: RUNNING (as of 15:57:30)
- Jobs are training successfully
- At least one run reached step 1300 (beta1_gamma0p2-503109)
- One or more runs may have died during initialization

## What This Job Does

Runs 9 SimPO training experiments on v5p-64 TPUs with different hyperparameters:
- 3 beta values: [0.1, 1.0, 2.0]
- 3 gamma_beta_ratios: [0.2, 0.5, 0.8]
- 9 total combinations (3 × 3 grid)

Each run trains on the Ultrafeedback dataset to step 2150.

## Monitoring Commands

### Check latest logs (last 30 lines with timestamps):
```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml job-logs ray-run-ahmed-sweep_simp-20260129-234457 2>&1 | grep "2026-01-29T" | tail -30
```

### Check for errors or died nodes:
```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml job-logs ray-run-ahmed-sweep_simp-20260129-234457 2>&1 | grep -iE "(died|killed|oom|error)" | tail -20
```

### Check training progress (steps):
```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml job-logs ray-run-ahmed-sweep_simp-20260129-234457 2>&1 | grep "step [0-9]" | tail -20
```

### Check which runs are active:
```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml job-logs ray-run-ahmed-sweep_simp-20260129-234457 2>&1 | grep "beta.*gamma" | tail -20
```

### Check job status:
```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml list-jobs | grep sweep_simp
```

## Expected Run Names

1. ultrafeedback_llama3_8b_beta0p1_gamma0p2
2. ultrafeedback_llama3_8b_beta0p1_gamma0p5
3. ultrafeedback_llama3_8b_beta0p1_gamma0p8
4. ultrafeedback_llama3_8b_beta1_gamma0p2 ✅ (confirmed running, reached step 1300)
5. ultrafeedback_llama3_8b_beta1_gamma0p5
6. ultrafeedback_llama3_8b_beta1_gamma0p8
7. ultrafeedback_llama3_8b_beta2_gamma0p2
8. ultrafeedback_llama3_8b_beta2_gamma0p5
9. ultrafeedback_llama3_8b_beta2_gamma0p8

## Known Status (as of 15:57:30)

**Working:**
- `beta1_gamma0p2-503109`: Step 1300/2150 ✅
- Multiple runs discovered checkpoints and resumed

**Potential Issues:**
- User reported "one of the runs died"
- Need to identify which run failed

## What to Monitor

1. **Training Progress**: Look for "step XXXX" messages
2. **Node Deaths**: Watch for "node...has been marked dead" messages
3. **OOM Errors**: Check for out-of-memory crashes
4. **Checkpoint Saves**: Confirm runs are saving at step intervals (200, 1000, etc.)
5. **WandB Sync**: Ensure metrics are being logged

## Checkpoints Location

All checkpoints are saved to:
```
gs://marin-us-central1/checkpoints/simpo/ultrafeedback_llama3_8b_beta{X}_gamma{Y}-{hash}/checkpoints/
```

Check checkpoint status:
```bash
gsutil ls "gs://marin-us-central1/checkpoints/simpo/"
```

## Configuration

File: `experiments/sweep_simp.py`

Key settings:
- TPU: v5p-64 (32 devices)
- Batch size: 128 (global)
- per_device_parallelism: -1 (auto-determined)
- Training steps: 2150
- Checkpoints: every 10 minutes + every 1000 steps

## If Jobs Fail

1. Check which run failed from logs
2. Check if it's an OOM issue (might need different TPU config)
3. Jobs should auto-resume from latest checkpoint if restarted
4. Can restart individual failed runs with `--force_run_failed True`

## Monitoring Frequency

Check every 30 seconds to ensure jobs stay alive and make progress.

## Latest Known Activity

- 15:47:06-15:47:10: Jobs initialized, loaded checkpoints, started training
- 15:57:12: Saving checkpoint at step 1300
- 15:57:30: Prefetch warning (normal, not critical)

## Stop Monitoring

Stop when all 9 runs complete (reach step 2150) or if multiple runs fail repeatedly.
