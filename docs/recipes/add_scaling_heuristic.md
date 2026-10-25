# Recipe: Add a Scaling Heuristic

A scaling heuristic maps `(model_size, token_budget)` to training hyperparameters like LR,
beta2, epsilon, batch size, and schedule.

It is not the same thing as a scaling law:

- A scaling law tells you **what** to train for a compute budget.
- A scaling heuristic tells you **how** to train each candidate.

For a given FLOP budget, the heuristic generates a set of candidate runs with different model
sizes and matching hyperparameters. An IsoFLOP sweep compares those candidates and finds the
best one. A scaling ladder then trains compute-optimal models at larger budgets.

You usually need a new heuristic when you introduce a new optimizer, training method, or
architecture. Dataset changes can usually reuse an existing heuristic.

## What you produce

- A **heuristic module** in `experiments/` implementing the `ScalingHeuristic` protocol
  (`marin.scaling_laws.isoflop_analysis.ScalingHeuristic`).
- **Tuned reference hyperparameters** at a fixed scale (from Vizier sweep or equivalent).
- **Validated scaling rules** confirmed at ~1e19 FLOPs and across the full isoflop range.
- A **scaling ladder** training optimal models at larger budgets.
- **Canary update**, if the new setup is intended for the hardware smoke canary.
- A **running experiment record** in the coordinating issue, W&B report, or
  durable session channel.

## Key reference files

```
experiments/grug/moe/heuristic.py              # MuonH MoE heuristic — use as template
experiments/references/completed_adamh.py      # AdamH heuristic

experiments/references/
    reference_hyperparameter_sweep.py  # Vizier hparam tuning at fixed scale
    reference_scaling_suite.py         # completed-run analysis + scaling ladder

experiments/ferries/canary_ferry.py            # daily canary

lib/marin/src/marin/scaling_laws/
    isoflop_analysis.py       # fit_scaling_laws, predict_optimal_config, ScalingHeuristic
    scaling_plots.py          # visualization
    tpu_utils.py              # pick_v4_type, V4_SPEC
    eval_metrics_reader.py    # WandB metrics extraction
```

## Workflow

### 1) Signs of life

First, make sure the idea looks promising at one reference scale. Good signs include:

- A Grug variant experiment (see `.agents/skills/change-grug/SKILL.md`).
- A quick A/B at about 130M parameters against the current best setup.
- Published results or supporting theory.

### 2) Tune reference hyperparameters

Pick one reference model and data budget. The reference sweep uses a ~130M parameter
model and 1B tokens on the Nemotron mix. Sweep the optimizer hyperparameter space with
`experiments/references/reference_hyperparameter_sweep.py`:

```sh
uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.references.reference_hyperparameter_sweep
```

Update `SweepSettings` for your search space, model, budget, and metric.

### 3) Define scaling rules

Implement as a frozen dataclass in `experiments/` that satisfies the `ScalingHeuristic`
protocol. See `experiments/grug/moe/heuristic.py` for the canonical example (MuonH MoE)
and `experiments/references/completed_adamh.py` for the AdamH dense-model heuristic.

The protocol requires `name`, `vocab_size`, `estimate_memory_bytes()`, and
`candidates_for_budget()`. A **candidate** is a specific `(model_config, token_count,
optimizer_config)` triple — for a given FLOP budget, `candidates_for_budget()` generates
candidates spanning different model sizes, each with hyperparameters from the scaling rules.

```python
@dataclass(frozen=True)
class MyHeuristic:
    name: str = "my-heuristic"
    tokenizer: str = "marin-community/marin-tokenizer"

    # Reference point (from step 2)
    reference_batch_size: int = 64
    reference_tokens: float = 2.5e9
    lr_base: float = 0.01

    @property
    def vocab_size(self) -> int:
        return get_vocab_size_for_tokenizer(self.tokenizer)

    def build_optimizer_config(self, batch_size: int, tokens: float) -> OptimizerConfig:
        ...

    def estimate_memory_bytes(self, candidate: CandidateConfig) -> int:
        ...

    def candidates_for_budget(self, budget: float, seq_len: int) -> Iterator[CandidateConfig]:
        ...

my_heuristic = MyHeuristic()
```

Scaling rules to define (notation: `B` = batch size, `B0` = reference, `T` = tokens,
`T0` = reference tokens):
- **LR**: e.g., `lr = lr0 * sqrt(B/B0) * (T0/T)^0.3`
- **Beta2**: constant token half-life: `beta2 = clip(beta2_0^(B/B0), 0.9, 0.9999)`
- **Epsilon**: e.g., `epsilon = eps0 * sqrt(B0 * T / (B * T0))`
- **Batch size**: how it relates to model size and budget
- **Architecture ratios**: `hidden_dim → num_layers`, `num_heads`

Before launching a sweep, print the candidate grids for a few budgets and check the values.

### 4) Run an IsoFLOP sweep

Create a standalone script modelled on `experiments/references/reference_scaling_suite.py`.
It reads completed IsoFLOP runs and returns an `IsoFlopAnalysisArtifact`; the
optimal-training steps depend on that artifact. Replace the reference run paths,
metric key, heuristic, data, and training resources for the new sweep. The
reference suite does not submit candidate runs, so the new experiment must
create and submit those training steps separately.

```python
_ISOFLOP_RUNS: tuple[str, ...] = (
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d512-L6-B32-my-heuristic-v1",
    # ...
)
```

Pass these completed paths to `IsoFlopAnalysisConfig(training_runs=..., output_path=...)`
in an analysis `ArtifactStep`, as `_analysis_step()` does in the reference suite.
Its `run_isoflop_analysis()` reads `tracker_metrics.jsonl` with
WandB fallback, converts the records, calls `fit_scaling_laws()`, and returns
`IsoFlopAnalysisArtifact(scaling_fits=...)`. It writes `fit_curves.json` as a
human-readable sidecar. Downstream steps use `ctx.resolved(analysis).scaling_fits`.
The reference suite reads existing AdamH checkpoint paths retroactively.

What to check after the sweep:
- No loss spikes or divergence at any candidate size.
- New heuristic outperforms (or matches) old heuristic at all sizes.
- Best candidate (minimum loss) is not at a range boundary; widen the grid if it is.

### 5) Scaling ladder

Train compute-optimal models at larger budgets (modelled on `reference_scaling_suite.py`):

1. The analysis step returns fitted values in `IsoFlopAnalysisArtifact`; it also writes
   `fit_curves.json` for inspection.
2. Pass those fitted values to `predict_optimal_config()` for each target budget.
3. Use `candidate.optimizer_config` or recompute via `build_optimizer_config()` when
   batch size must be overridden for available hardware.
4. Train with the appropriate TPU type, TP degree, and gradient accumulation.

```sh
uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G --extra=cpu \
  -- python -m experiments.my_scaling_suite
```

### 6) Check canary fit

Update `experiments/ferries/canary_ferry.py` when the new setup should become the
hardware smoke canary. Its current MoE model has 9.9M parameters. The TPU path
defaults to 250M training tokens; the GPU path defaults to 50 steps. Check the
canary's cost and regression gates before changing its model or token budget.

## Iteration

General rule: **find the smallest scale that reproduces a problem, fix it there, then scale up.**

- **Loss spikes at step 4**: bad scaling rule for some sizes — reproduce cheaply, fix rule.
- **Poor scaling law fits**: outlier runs or degenerate configs at some budgets.
- **Scaling ladder underperforms**: may be extrapolation — add higher-budget sweep points.
- **Worse than old heuristic at some scales**: adjust rules, don't add special cases.

## Definition of Done

- Reference hyperparameters tuned at fixed scale.
- Heuristic module: frozen dataclass, singleton, `candidates_for_budget()`.
- No loss spikes or regressions at ~1e19; competitive with prior heuristic at all sizes.
- Full isoflop sweep with fitted scaling laws and reviewed plots.
- Scaling ladder trained at target budgets.
- Canary updated if the new setup is intended for the hardware smoke canary.
- Report: scaling rules, sweep results, comparison vs baseline, known limitations.
