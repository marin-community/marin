# Transfer Bayesian optimization for data mixtures

This package fits a transfer Gaussian process (GP) to data-mixture observations
and recommends one feasible two-phase mixture for a target swarm. Bayesian
optimization uses the GP as its surrogate model and maximizes an acquisition
function over a finite candidate set.

The remote input is a campaign manifest. It selects the target and source
swarms, assigns reference-data roles, defines the objective and constraints,
and points to a swarm registry. The registry maps swarm and content-basis IDs to
hash-verified manifests. A materialized campaign is the downloaded manifest,
registry, and referenced artifacts.

The published campaign is in the
[`grug-moe-mix-swarm`](https://huggingface.co/datasets/marin-community/grug-moe-mix-swarm/tree/6110da69a96d1dcdc2f420187cd818b3a02944b3/registry/v1)
dataset. Candidate generation requires no Marin storage or service credentials.

## Run a campaign

```bash
uv sync --package marin-core --extra mixprior --extra cpu --group test
uv run --no-sync pytest -n 0 -q tests/datakit/mixprior
uv run --no-sync python -m experiments.datakit.mixprior.run_cycle \
  --campaign-uri hf://datasets/marin-community/grug-moe-mix-swarm@6110da69a96d1dcdc2f420187cd818b3a02944b3/registry/v1/transfer_campaign.parquet \
  --campaign-sha256 57a4f13beccf5369e771d0692ffbedd7ce02e76277f8003a7dfe91a7174617ac \
  --output-dir work/rav-candidate \
  --pool-size 65536 \
  --seed 20260821
```

Use `--extra gpu` instead of `--extra cpu` on an x86-64 Linux CUDA 13 host.
The output directory must not exist. Downloads are anonymous, and each URI pins
a Hugging Face dataset commit.

The output directory contains:

| Artifact | Contents |
| --- | --- |
| `candidate.parquet` | Selected phase weights, posterior diagnostics, constraints, and artifact hashes |
| `candidate_pool.npz` | Every feasible candidate evaluated by the acquisition function |
| `acquisition_values.npy` | Acquisition-function value for each candidate |
| `fitted_model.pt` | Fitted surrogate-model class and state dictionary |
| `campaign/` | Materialized campaign and referenced swarm artifacts |
| `cycle.parquet` | Remote campaign URI and generated-artifact hashes |

Inspect the machine-readable records without loading the model:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
from pprint import pprint

from experiments.datakit.mixprior.data import read_record

root = Path("work/rav-candidate")
pprint(read_record(root / "candidate.parquet"))
pprint(read_record(root / "cycle.parquet"))
PY
```

## Bayesian-optimization terms

- An **observation** is one evaluated two-phase mixture and its metric values.
- An **objective metric** is an evaluation metric used to compute the scalar
  objective. The published manifest calls these `response_tasks`; the loader
  translates that compatibility field at the schema boundary.
- An **objective value** is the scalar, higher-is-better value modeled by the
  surrogate. Here it is the negative variance-normalized hinge loss.
- The **surrogate model** is a transfer GP that returns a posterior distribution
  over the objective.
- The **incumbent** is the best observed objective value for the target swarm.
- A **candidate** is an unevaluated feasible mixture.
- The **candidate set** is the finite set over which the acquisition function is
  optimized.
- The **acquisition function** is `PosteriorMean`; the selected candidate is its
  argmax over the candidate set.
- `probability_of_improvement` is the posterior probability that the selected
  candidate's objective exceeds the incumbent.
- A **mixture component** is one weighted entry in a phase mixture. The external
  registry schema calls these entries `cells` and their table `buckets`; loaders
  translate those compatibility fields at the schema boundary.
- A **phase token budget** is a token count. A **phase token fraction** is that
  budget divided by the total token budget.
- A **hinge tolerance** is the allowed standardized improvement before a
  non-linear objective term is capped. The manifest stores it in the
  compatibility field `objective_epsilon`.

## Pipeline stages

The default path has eight stages. Each accepts arrays or typed records, so an
experiment can replace one stage while retaining the others.

| Stage | Default entry point | Input | Output | Alternative |
| --- | --- | --- | --- | --- |
| Campaign loading | `campaign.download_campaign` and `campaign.load_campaign` | Pinned campaign URI or local campaign manifest | `Campaign` | Another transport or swarm selection |
| Objective | `VarianceNormalizedObjective` and `objective.objective_observations` | Evaluation metrics and observation-noise estimates | Objective values and variances | Another scalar objective |
| Feature map | `model.curriculum_features` | Phase weights, component content, phase token fractions | Surrogate features | Another cross-swarm representation |
| Surrogate fit | `model.prepare_hellinger_transfer_data` and `model.fit_additive_hellinger_model` | Campaign | `HellingerTransferData` and `TransferPredictor` | Another surrogate or transfer kernel |
| Candidate generation | `search.campaign_lognormal_pool_inputs` and `search.sample_lognormal_pool` | Target swarm and constraints | Feasible candidate set | Another proposal distribution or design |
| Acquisition | `search.acquire_posterior_mean` and `search.build_candidate_selection` | Fitted surrogate and candidate features | `CandidateSelection` | Another acquisition function |
| Diagnostics | `search.candidate_diagnostics` | Selected candidate and posterior | Diagnostic record | Another report |
| Persistence | `artifacts.write_candidate_bundle` | Campaign, surrogate, candidates, selection, and diagnostics | Hash-linked Parquet bundle | Another audit format |

Array shapes form the boundary between stages:

- Mixture weights have shape `(candidate, phase, mixture_component)`.
- Content matrices have shape `(mixture_component, shared_content_feature)`.
- The modeled objective is the higher-is-better negative hinge loss.
- Human-facing loss values are lower-is-better; conversions between loss and
  objective value must negate the value explicitly.

For example, a hand-authored candidate set can reuse the default objective,
surrogate, acquisition function, diagnostics, and artifact format:

```python
from pathlib import Path

import numpy as np
import torch

from experiments.datakit.mixprior.artifacts import write_candidate_bundle
from experiments.datakit.mixprior.campaign import build_variance_normalized_campaign, load_campaign_inputs
from experiments.datakit.mixprior.model import (
    OBJECTIVE_NAME,
    fit_additive_hellinger_model,
    prepare_hellinger_transfer_data,
)
from experiments.datakit.mixprior.search import (
    candidate_diagnostics,
    default_model_metadata,
    prepare_candidate_features,
    select_posterior_mean,
)

manifest = Path("work/campaign/transfer_campaign.parquet")
campaign = build_variance_normalized_campaign(load_campaign_inputs(manifest))
model = fit_additive_hellinger_model(
    prepare_hellinger_transfer_data(campaign), torch.device("cpu")
)
pool = np.load("my_feasible_pool.npy")
candidates = prepare_candidate_features(campaign.target, pool)
selection = select_posterior_mean(campaign, model, candidates)
diagnostics = candidate_diagnostics(
    campaign.target,
    selection.weights,
    selection.posterior,
    objective_name=OBJECTIVE_NAME,
    hinge_tolerance=campaign.objective.epsilon,
    acquisition_function=selection.acquisition_function,
    selection_rule=selection.selection_rule,
)

write_candidate_bundle(
    campaign_manifest=manifest,
    campaign=campaign,
    model_payload=model.model_state(),
    model_metadata=default_model_metadata(campaign, model),
    pool=pool,
    acquired=selection.acquired,
    selected_weights=selection.weights,
    diagnostics=diagnostics,
    phase_token_fractions=candidates.phase_token_fractions,
    output_dir=Path("work/custom-pool-candidate"),
    seed=7,
    proposal={"kind": "manual", "parameters": {"source": "my_feasible_pool.npy"}},
    acquisition_function=selection.acquisition_function,
    selection_rule=selection.selection_rule,
    dependency_lock=Path("uv.lock"),
)
```

`load_swarm` loads one swarm without constructing an objective.
`build_campaign` accepts a caller-supplied objective and observation-noise
mapping. `assemble_transfer_data` combines precomputed features with objective
observations. `HellingerTransferData` also carries the kernel-calibration
reference. `TransferPredictor` is the posterior-prediction interface used by
acquisition. `build_candidate_selection` accepts the result of any acquisition
function evaluated over a finite candidate set.

## Objective

The loss is a variance-normalized hinge loss over 20 evaluation metrics measured
in bits per byte (BPB). The optimization objective is the negative loss, so it
is maximized. `include_mean` is omitted because the legacy source swarm lacks
that metric. Improvements on code and math contribute an uncapped linear term.
Improvements on the remaining metrics are capped at the configured hinge
tolerance; regressions remain linear.

A pooled per-metric standard deviation from repeated-seed evaluations estimates
observation noise. The scalar objective variance retains the observed
correlation between metrics.

| Swarm role | Rows | Contribution |
| --- | ---: | --- |
| Legacy source swarm | 804 | Shared GP and swarm-specific GP |
| First Harrier reference swarm | 115 | Shared GP, swarm-specific GP, objective reference, and observation-noise estimate |
| Second Harrier kernel-reference swarm | 56 | Shared GP, swarm-specific GP, and kernel calibration |
| Rav target swarm | 3 | Target-specific GP and incumbent |

## Feature map and transfer GP

Each mixture component has a probability distribution over the shared
1,000-feature Luxical basis. A phase mixture is projected into this basis and
square-rooted:

```text
h_phase = sqrt(weights_phase @ component_content) * sqrt(phase_token_fraction)
x = concat(h_phase)
```

Squared Euclidean distance between these features is twice the phase-token-
weighted squared Hellinger distance. The fixed RBF length scale is calibrated
from the kernel-reference swarm with `gamma_factor = 0.25`.

The covariance is

```text
k((x, swarm), (x', swarm'))
  = k_shared(x, x')
  + 1[swarm = swarm'] k_swarm(x, x')
```

The shared kernel transfers content effects between swarms. The swarm-specific
kernel gives each swarm an independent deviation from the shared GP. Model size,
data source, and training-token horizon are provenance fields. They are constant
within each swarm and are therefore not separately identifiable from the four
swarm labels in this campaign.

Every swarm uses the same Luxical basis. The root Hugging Face URI pins one
dataset commit. Registry references verify every manifest, observation table,
mixture-component table, content matrix, and basis lookup by relative path and
SHA-256. Runtime loaders accept the external `mixture-observations-v1` schema.

## Candidate generation and acquisition

Proposal centers include the availability-proportional target design, observed
target designs, and source observations whose mixture components match the
target. Each random proposal retains a 2% availability-proportional floor and
applies a log-normal perturbation with a scale sampled log-uniformly from 0.02
to 2.0.

Candidates must satisfy the simplex constraint for every phase and the maximum
cumulative exposure constraint for every mixture component. Observed target
designs are excluded.

The fitted surrogate predicts the posterior mean over the fixed, seeded
candidate set. `PosteriorMean` is maximized by taking the finite-set argmax.
Candidate generation does not enforce an ordering between quality tiers,
smoothness between phases, or a maximum distance from existing observations.

## Limitations

- Three Rav observations fit the target-specific GP.
- The finite candidate set explores neighborhoods of its proposal centers, not
  the complete Cartesian product of the phase simplices.
- Architecture and training-horizon metadata do not support continuous
  cross-swarm extrapolation.
- The observation-noise estimate is shared across swarms.
