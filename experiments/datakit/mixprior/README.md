# Transfer Bayesian optimization for two-phase data mixtures

This package fits one transfer Gaussian process to evaluated data mixtures and
selects a new mixture for a target swarm. A swarm is a set of runs over one
token store and training recipe. Historical swarms train the shared GP response;
only target-swarm observations define the incumbent.

The input campaign is published in the
[`grug-moe-mix-swarm`](https://huggingface.co/datasets/marin-community/grug-moe-mix-swarm/tree/283eacf18b66b7888b59fd8f889d6be134aee879/registry/v1)
dataset. Every URI pins a dataset commit and every referenced artifact has a
SHA-256 digest. Candidate generation requires no Marin credentials.

## Run a campaign

```bash
uv sync --package marin-core --extra datakit --extra cpu --group test
uv run --no-sync pytest -n 0 -q tests/datakit/mixprior
uv run --no-sync python -m experiments.datakit.mixprior.generate_from_huggingface \
  --campaign-uri hf://datasets/marin-community/grug-moe-mix-swarm@283eacf18b66b7888b59fd8f889d6be134aee879/registry/v1/transfer_campaign.parquet \
  --campaign-sha256 5ed5fb024590dd4707b802caf8fe728be1b8d73375c100139f884b0728f0cca2 \
  --output-dir work/rav-candidate \
  --pool-size-per-seed 65536 \
  --pool-seed 111 \
  --pool-seed 222 \
  --pool-seed 333 \
  --acquisition-seed 7
```

Use `--extra gpu` on an x86-64 Linux CUDA 13 host. The output directory must not
exist.

The output directory contains:

| Artifact | Contents |
| --- | --- |
| `candidate.parquet` | Selected phase weights, posterior diagnostics, constraints, and artifact hashes |
| `candidate_pool.npz` | Every mixture evaluated by the acquisition function |
| `acquisition_values.npy` | Acquisition value for every candidate-pool row |
| `campaign/` | Materialized campaign and referenced swarm artifacts |
| `bundle_manifest.parquet` | Remote campaign URI and generated-artifact hashes |

## Data and objective

Each observation contains exactly two simplex-valued phase mixtures. The
in-memory weight shape is `(observation, phase, mixture_component)`. Each swarm
also records phase token budgets, component token counts, fixed Luxical content
features, model parameters, and physical and simulated training tokens.

The modeled objective is the negative variance-normalized hinge loss over BPB
metrics. Higher objective values are better. The task lists are explicit fields
of `VarianceNormalizedObjective`. Repeated mixtures estimate observation noise.

## Quadratic-exposure GP

For phase $s$ and mixture component $i$, define simulated epochs and available
token share:

$$
e_{s,i}(w)=\frac{B_s w_{s,i}}{V_i},
\qquad
a_i=\frac{V_i}{\sum_j V_j}.
$$

$B_s$ is the phase token budget and $V_i$ is the component's available token
count. The feature map preserves content and repetition separately:

$$
h_s(w)=
\left[
\sum_i a_i e_{s,i}\phi_i,
\sum_i a_i e_{s,i}^2\phi_i
\right],
\qquad
q_s(w)=\sum_i a_i e_{s,i}^2.
$$

$\phi_i$ is the component's fixed Luxical content vector. The learned prior mean
is

$$
m(w)=c-\sum_s b_s q_s(w),
\qquad b_s>0.
$$

The penalty is quadratic in simulated epochs. A scalar linear epoch term was
removed because it is constant on a fixed-budget simplex:

$$
\sum_i a_i e_{s,i}
=\frac{B_s}{\sum_j V_j}\sum_iw_{s,i}
=\frac{B_s}{\sum_jV_j}.
$$

Content-specific benefit is learned by the GP response. The covariance is

$$
k(x,x')=
\sum_{s,t}C_{st}k_c(h_s(x),h_t(x'))
+\mathbf 1[\operatorname{swarm}(x)=\operatorname{swarm}(x')]
\sigma_r^2 k_r(h(x),h(x')).
$$

$C=LL^\top+\operatorname{diag}(v)$ is a learned positive-semidefinite phase
covariance. $k_c$ and $k_r$ are Matérn-5/2 kernels. The first term transfers a
phase-linked content response across swarms. The second lets each swarm learn a
smooth deviation.

Objective values and variances are standardized within each swarm before
pooling. Predictions are transformed back to target-swarm units. Compiled JAX
fits the ten GP parameters by MAP from the prior mode and two fixed prior draws.

## Replace one stage

The package boundaries correspond to the operations an experiment may replace:

| Operation | File | Boundary |
| --- | --- | --- |
| Load local campaign records | `campaign.py` | `Path -> Campaign` |
| Compute objective observations | `objective.py` | `ScalarObjective` |
| Map mixtures, evaluate the GP, and predict | `quadratic_exposure.py` | `quadratic_exposure_features` and `FittedQuadraticExposureGP` |
| Standardize data and fit compiled MAP objectives | `surrogate.py` | `fit_map_restarts` |
| Sample a lattice pool and acquire candidates | `search.py` | `MixturePredictor` and `CandidateSelection` |
| Compose a fitter with an acquisition | `generate_candidate.py` | `search_candidate` |
| Persist a decision | `artifacts.py` | `CandidateDecision` |

`quadratic_exposure.py` contains the package's one supported prior mean and
covariance. A new acquisition implements the selector signature and is passed
to `search_candidate`; it does not modify the GP or artifact code.

## Candidate search

Candidate phases are nonnegative simplex vectors quantized to integer counts
summing to 49,152. These are the only hard constraints.

Half of each proposal draw is global symmetric Dirichlet sampling. The other
half perturbs availability-proportional and observed designs. Three independently
seeded 65,536-row pools are deduplicated before acquisition. Observed target
mixtures are excluded.

The command-line path uses Monte Carlo noisy expected improvement with 1,024
Gaussian samples generated by JAX. Its baseline contains actual target-swarm
observations. Historical observations train the transfer GP and do not enter
the target baseline. `PosteriorMean` is also available for greedy selection.

## Development replay

The replay starts with three target observations, selects batches of five from
a closed pool, reveals their outcomes, and refits. Historical observations stay
available. Mean simple regret over twenty deterministic starts is:

| Target pool | Acquisition | 8 evals | 13 evals | 18 evals | 20 evals |
| --- | --- | ---: | ---: | ---: | ---: |
| 56-row Harrier | PosteriorMean | 1.296 | 0.000 | 0.000 | 0.000 |
| 56-row Harrier | BoTorch qLogNEI | 4.395 | 3.332 | 3.332 | 3.146 |
| 115-row Harrier | PosteriorMean | 7.495 | 0.525 | 0.035 | 0.035 |
| 115-row Harrier | BoTorch qLogNEI | 108.862 | 98.569 | 98.555 | 32.805 |

Mean chronological Spearman is 0.051 on the 56-row pool and 0.834 on the
115-row pool. The low 56-row rank correlation coexists with low regret because
the model elevates a high-value mixture without ordering the rest of the pool.

The model and priors were developed while inspecting both pools. These numbers
measure closed-pool development behavior and predate the JAX acquisition. The
BoTorch qLogNEI replay indicates that the posterior uncertainty was not
calibrated well enough to match the PosteriorMean result.

Run the diagnostics on a CUDA host:

```bash
uv run --no-sync python -m experiments.datakit.mixprior.benchmark rank
uv run --no-sync python -m experiments.datakit.mixprior.benchmark regret --block 0
```

## Current limitations

- The target swarm has three observations.
- The finite candidate pool does not enumerate the full lattice.
- The quadratic term is a soft prior. The GP residual can overturn it.
- MAP fitting does not integrate hyperparameter uncertainty.
- Observation noise is shared across swarms.
- The replay does not measure information value in unobserved regions.
