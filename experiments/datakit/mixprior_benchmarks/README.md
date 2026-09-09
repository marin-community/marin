# Offline mixture-model replay

`replay.py` fits the production model on prefixes of distinct measured designs
and ranks the remaining designs by posterior mean. Replicates are combined by
inverse observation-noise variance and stay on the same side of each split.
Prefix order follows first appearance in the HF observations.

```bash
uv run python -m experiments.datakit.mixprior_benchmarks.replay \
  --data /path/to/hf-snapshot \
  --swarm h100-d512-from10pct-store-4d2e363d \
  --prefixes 128 384 \
  --output replay.json
```

The output records Spearman correlation, selected-winner rank, and regret
relative to the best measured held-out design. Rank 1 is best; regret is the gap in objective score. Lower rank and regret are better.
Objective and noise calibration use the full campaign's fixed reference data.
These correlated retrospective splits are model-selection evidence; they do
not measure prospective performance or out-of-sample uncertainty calibration.

For a model experiment, pass a different fit function to `replay`. It receives
`(Data, values, variances)` and returns a predictor with `predict(weights)`.
```python
from experiments.datakit.mixprior.hf import load_data
from experiments.datakit.mixprior_benchmarks.replay import replay

# data = load_data(snapshot_root, swarm_id)
results = replay(data, [128, 384], fit_model=my_fit)
```

The production package retains only its selected model and search algorithm.

## Model choice

The MVP retains the direct-weight GP and posterior-mean search. The
[comparison results](model_comparison.json) use 792 observations (777 distinct
designs) from the pinned dataset revision. These two splits show a tradeoff:

| Training designs | Direct GP regret / rank | Quadratic GP regret / rank | Direct / quadratic seconds |
| --- | --- | --- | --- |
| 128 | 2.908 / 62 | 3.503 / 107 | 6.3 / 84.8 |
| 384 | 2.412 / 31 | 1.731 / 14 | 5.0 / 682.1 |

Quadratic chooses a better candidate at the larger prefix and has slightly
lower average regret across the two splits (2.617 versus 2.660). Direct has
better average ranking correlation (0.884 versus 0.786), wins the smaller
prefix, and fits much faster. Direct is the practical choice for this small,
interactive MVP; these results do not establish it as the most accurate model.

Adding or subtracting one posterior standard deviation never improves the
selected candidate in these comparisons. Expected improvement also selects the
same candidates as posterior mean for the direct GP at both prefixes. The core
therefore retains posterior mean. This evaluates selection among measured
designs, not the quality of newly generated continuous-space proposals.
