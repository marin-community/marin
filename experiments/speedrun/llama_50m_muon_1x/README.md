# 50M Llama with Muon Optimizer (1× Chinchilla)

**Model:** 50M parameter Llama
**Optimizer:** Muon
**Data Scale:** 1× Chinchilla-optimal (1B tokens)
**Expected BPB:** 1.38-1.42

## Rationale

Baseline validation experiment. Tests whether Muon beats Adam at standard Chinchilla-optimal scale before investing in 4× data runs. This experiment establishes whether Muon's advantages hold at the canonical 20:1 token-to-parameter ratio.

## Hyperparameters

- **Learning rate:** 0.020
- **Batch size:** 128
- **Training steps:** 7,629 (1B tokens / 128 batch / 1024 seq_len)
- **Optimizer:** Muon with momentum=0.95, warmup=0
- **Evaluation:** Every 500 steps

## Key Differences from 4x Experiment

- 1/4 the data (1B vs 4B tokens)
- 1/4 the training steps (7,629 vs 30,518)
- Shorter time limit (30 minutes vs 2 hours)
- Same hyperparameters

## Running

```bash
sbatch submit_slurm.sh
```

## Expected Results

If Muon maintains its efficiency advantage at 1× scale, we expect BPB around 1.38-1.42. This would validate proceeding with 4× data experiments.
