# Two-Phase-Many Analysis Dataset

This directory contains the derived modeling dataset for joint mixture/scale
analysis. It is intentionally downstream of `run_registry/` and
`metric_registry/`: those directories remain provenance inputs, while this
layer is the fit/packet data contract.

Build:

```bash
uv run --with torch python -m experiments.domain_phase_mix.exploratory.two_phase_many.analysis_dataset.build_analysis_dataset
```

Outputs:

- `nd_scale_runs.csv`: canonical row table for modeling.
- `nd_scale_packet.npz`: packet-compatible arrays.
- `summary.json`: audit counts and source timestamps.

Important convention: `model_size` in this derived dataset means
non-embedding parameter count. Historical scale strings such as `130m_2p6b`
are stable identifiers only; use `scale_display_label`, `non_embedding_params`,
and `experiment_budget` for analysis and plots.

Strong-tier rows are included only when `run_registry/strong_tier_perplexity_ready.csv`
has a target-step label. Completed GRP raw-optimum validation rows are appended
from `run_registry/logical_runs.csv` only when they have an objective metric and
a replay spec that provides explicit budgets, target multiplier, and phase
weights.
