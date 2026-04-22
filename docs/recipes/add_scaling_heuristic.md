# Recipe: Add a Scaling Heuristic

A scaling heuristic maps `(model_size, token_budget)` to training hyperparameters like LR,
beta2, epsilon, batch size, and schedule.

It is not the same thing as a scaling law:

- A scaling law tells you what to train for a compute budget.
- A scaling heuristic tells you how to train each candidate.

For a given FLOP budget, the heuristic generates a set of candidate runs with different model
sizes and matching hyperparameters. An IsoFLOP sweep compares those candidates and finds the
best one. A scaling ladder then trains compute-optimal models at larger budgets.

You usually need a new heuristic when you introduce a new optimizer, training method, or
architecture. Dataset changes can usually reuse an existing heuristic.

## What you produce

- A **heuristic module** in `experiments/scaling_law_sweeps/<name>.py` implementing the `ScalingHeuristic`
  protocol (defined in `marin.scaling_laws.isoflop_analysis`).
- **Tuned reference hyperparameters** at a fixed scale (from Vizier sweep or equivalent).
- **Validated scaling rules** confirmed at ~1e19 FLOPs and across the full isoflop range.
- A **scaling ladder** training optimal models at larger budgets.
- **Canary promotion** in `experiments/ferries/canary_ferry.py`.
- A **running report** in `.agents/logbooks/<topic>.md`.

## Workflow

### 1) Signs of life

First, make sure the idea looks promising at one reference scale. Good signs include:

- A Grug variant experiment (see `docs/recipes/change_grug.md`).
- A quick A/B at about 130M parameters against the current best setup.
- Published results or supporting theory.

### 2) Tune reference hyperparameters

Pick one reference model and data budget, usually around 130M params and 2.5B tokens on the
Nemotron mix. Then sweep the optimizer hyperparameter space with
`experiments/references/reference_hyperparameter_sweep.py` or an equivalent setup.

```sh
uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.references.reference_hyperparameter_sweep
```

Update `SweepSettings` for your search space, model, budget, and metric.

If useful, tune a few additional scales so you can see how the best hyperparameters move with
model size. That gives you better inputs for step 3.

### 3) Define scaling rules

Start from the tuned reference point and define how each hyperparameter should change with
hidden_dim, batch size, and token count. https://arxiv.org/abs/2512.22382 is a useful
starting point for Adam-style scaling with width and batch size. Use that as guidance, then
adjust based on the empirical results from step 2.

Expect to go back and forth between this step and step 4. That is normal.

Implement as a frozen dataclass in `experiments/scaling_law_sweeps/<name>.py` that satisfies
the `ScalingHeuristic` protocol (`marin.scaling_laws.isoflop_analysis.ScalingHeuristic`).
The protocol requires `name`, `vocab_size`, `estimate_memory_bytes()`, and
`candidates_for_budget()`. A **candidate** is a specific `(model_config, token_count,
optimizer_config)` triple — for a given FLOP budget, `candidates_for_budget()` generates
many candidates spanning different model sizes, each with hyperparameters computed from
the scaling rules. `estimate_memory_bytes()` is used to determine what hardware each
candidate needs. You do not need to inherit from the protocol — structural subtyping
(duck typing) is sufficient. Additional helpers (for example `build_optimizer_config()`)
are optional and only needed if your scaling ladder workflow recomputes optimizer settings
(e.g., when overriding batch size for available hardware).

```python
@dataclass(frozen=True)
class MyHeuristic:
    name: str = "my_heuristic"
    tokenizer: str = "meta-llama/Llama-3.1-8B"

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

# Module-level singleton used by isoflop_sweep.py
my_heuristic = MyHeuristic()

# Module-level function that calls candidates_for_budget() for each budget
# and wraps results into ExecutorSteps for the sweep registry in isoflop_sweep.py
def create_isoflop_sweep_steps(
    tokenized: LMMixtureDatasetConfig,
    experiment_name: str,
    budgets: tuple[float, ...],  # FLOP budgets, e.g. (1e19, 3e19, 9e19)
    ...
) -> tuple[list[ExecutorStep], list[CandidateConfig]]:
    ...
```

Scaling rules to define (notation: `B` = batch size, `B0` = reference batch size,
`T` = tokens, `T0` = reference tokens):
- **LR**: e.g., `lr = lr0 * sqrt(B/B0) * (T0/T)^0.3`
- **Beta2**: e.g., constant token half-life: `beta2 = clip(beta2_0^(B/B0), 0.9, 0.9999)`
- **Epsilon**: e.g., `epsilon = eps0 * sqrt(B0 * T / (B * T0))` (the `seq_len` cancels)
- **Batch size**: how it relates to model size and budget
- **Architecture ratios**: hidden_dim → num_layers, num_heads

See `experiments/scaling_law_sweeps/completed_adamh.py` and
`experiments/scaling_law_sweeps/c_adamc.py` for complete examples.

Before you launch a sweep, print the candidate grids for a few budgets and make sure the
values look sane.

### 4) Validate at ~1e19 FLOPs

Replace the `SCALING_SUITES` entry and `__main__` block in `experiments/isoflop_sweep.py`
with a single-budget sanity check:

```python
# In experiments/isoflop_sweep.py — replace SCALING_SUITES entry
SCALING_SUITES = {
    ...
    "my-heuristic-sanity": my_heuristic_module.create_isoflop_sweep_steps(
        tokenized=nemotron_mix,
        experiment_name="my-heuristic-sanity",
        budgets=(1e19,),
    ),
}

if __name__ == "__main__":
    steps, _ = SCALING_SUITES["my-heuristic-sanity"]
    executor_main(steps=steps)
```

What to check:
- No loss spikes or divergence at any candidate size.
- New heuristic outperforms (or matches) old heuristic at all candidate sizes.
- Best candidate (minimum loss point) is preferably in the middle of the range, not at a boundary; if it lands on a boundary, widen the candidate range and re-check.

Compare against old heuristic runs using a Colab notebook querying the WandB API
([example](https://colab.research.google.com/drive/1sNWqvf09FcDeB3qY1dD3JqZF546348Os)).

If something looks wrong, find the smallest cheap setup that reproduces the problem, fix the
rule there, and then re-run the ~1e19 check.

### 5) Full IsoFLOP sweep

Once the sanity check passes, register the full-budget suite in `experiments/isoflop_sweep.py`:

```python
SCALING_SUITES = {
    "my-heuristic-nemotron": my_heuristic_module.create_isoflop_sweep_steps(
        tokenized=nemotron_mix,
        experiment_name="my-heuristic-v1",
        budgets=LEGACY_BUDGETS,  # FLOP budgets: (3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20)
    ),
}
```

Submit:

```sh
uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.isoflop_sweep
```

The analysis step runs as part of the scaling ladder in step 6, not as a separate manual
step. See `experiments/exp1337_delphi_suite.py` for how `run_isoflop_analysis_step` is wired
as an `ExecutorStep` that consumes the sweep runs. It fits `D* ~ A * C^alpha` and writes
`isoflop_analysis_result.json`, which the scaling ladder then uses for the optimal training
runs.

Use `scaling_plots.py` for visualization (isoflop curves and scaling fit plots).

### 6) Scaling ladder

Train compute-optimal models at larger budgets. See `experiments/exp1337_delphi_suite.py`:

1. Load `isoflop_analysis_result.json` from step 5.
2. `predict_optimal_config()` for each target budget.
3. Use `candidate.optimizer_config` from `predict_optimal_config()`, or recompute with a
   heuristic-specific helper if you need to override batch size for available hardware
   (batch size shouldn't affect quality with appropriate LR/beta2 scaling — this is the
   SDE approximation).
4. Train with appropriate TPU type, TP, gradient accumulation.

### 7) Promote to canary

Update `experiments/ferries/canary_ferry.py` with the new best setup. Keep the canary model
size and FLOP budget roughly fixed (~30M params, ~1B tokens) so the canary remains a fast,
cheap daily check rather than a full training run. Update the optimizer config, model config,
and training parameters to match your new best setup.

## Iteration

General rule: **find a smaller scale that reproduces the problem, fix it there, then scale back up.**

- **Loss spikes at step 4**: bad scaling rule for some model sizes — reproduce cheaply, fix rule.
- **Poor scaling law fits at step 5**: outlier runs or degenerate configs at some budgets.
- **Scaling ladder underperforms**: may be extrapolation — add higher-budget sweep points.
- **Worse than old heuristic at some scales**: adjust the underlying scaling rules, don't add special cases.

## Key files

```
experiments/scaling_law_sweeps/
    completed_adamh.py             # AdamH with sqrt-batch LR scaling
    c_adamc.py                     # Cautious Adam with corrected weight decay

experiments/references/
    reference_hyperparameter_sweep.py  # Vizier hparam tuning at fixed scale

experiments/isoflop_sweep.py           # sweep registry + analysis
experiments/exp1337_delphi_suite.py     # scaling ladder
experiments/ferries/canary_ferry.py     # daily canary

lib/marin/src/marin/scaling_laws/
    isoflop_analysis.py       # fit_scaling_laws, predict_optimal_config, ScalingHeuristic
    scaling_plots.py          # visualization
    tpu_utils.py              # pick_v4_type, pick_v5p_type
    eval_metrics_reader.py    # WandB metrics extraction
```

## WandB conventions

- Use `marin-community/marin` for pretraining sweeps.
- Tag runs with `FLOPs=<training_flops>`, `N=<num_params>`, `B=<batch_size>`, and heuristic name.
- Tag optimal runs with `optimal-training` and `label=<experiment_name>`.
- Compare runs via Colab notebooks querying the WandB API (same Colab linked in step 4).

## Definition of Done

- Reference hyperparameters tuned at fixed scale.
- Heuristic module: frozen dataclass, singleton, `create_isoflop_sweep_steps()`.
- No loss spikes or regressions at ~1e19; competitive with prior heuristic at all sizes.
- Full isoflop sweep with fitted scaling laws and reviewed plots.
- Scaling ladder trained at target budgets.
- Canary promoted in `experiments/ferries/canary_ferry.py`.
- Report: scaling rules, sweep results, comparison vs baseline, known limitations.
