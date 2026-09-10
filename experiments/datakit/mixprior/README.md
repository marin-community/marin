# Mixture acquisition

Fit completed two-phase mixtures and propose feasible, unobserved mixtures.
Training fits a metric GP with corrections learned from held-out candidate
residuals, then ranks mixtures by their expected objective score.
`generate.py` loads the saved model and scores new mixtures without fetching observations or refitting.

## Run

From the repository root:

```bash
uv run python -m experiments.datakit.mixprior.train \
  --revision 9fa32c159cb3dd123b48f5f14d1f367310325f49 \
  --calibration-revision 9cdd208e42f926f2b23c204085c9dc45f63c6c92 \
  --swarm h100-d512-from10pct-store-4d2e363d \
  --device cpu \
  --output mixture-model

uv run python -m experiments.datakit.mixprior.generate \
  --model mixture-model \
  --batch-size 20 --seed 222 \
  --device cpu \
  --output candidates.parquet
```

On an Iris H100 task, install the `marin-core` GPU extra and pass `--device gpu`
to both commands: `uv run --package marin-core --extra gpu python -m ...`.
That extra installs CUDA-enabled JAX; requesting `gpu` fails if no GPU is
available. Omitting `--device` uses JAX's preferred device. Saved models can be
loaded on either CPU or GPU. Library callers must enable
`jax.config.update("jax_enable_x64", True)` before fitting or loading.

The proportional mixture assigns each cell its fraction of total available
tokens, independently in each phase. Proportional-reference runs use that baseline.

The [HF dataset](https://huggingface.co/datasets/marin-community/grug-moe-mix-swarm)
contains named cells with domain, integer quality level, and available-token
metadata. The loader aligns phase weights by cell name and recomputes language
metric means from their constituents. Use a full dataset commit for
`--revision`. Add `--data /path/to/snapshot` to read a local snapshot containing
`registry/v1/swarms/`. Only `swarm.parquet`, `buckets.parquet`, and
`observations.parquet` are needed for the named swarm. `--calibration-revision`
is required: designs absent from that earlier snapshot supply calibration
residuals. The example uses the 40 newly uploaded candidate designs and fits the
final model on all 832 observations. Use `--calibration-data` for a local snapshot
of the earlier revision. Calibration requires at least eight new distinct designs;
choose a batch representative of future proposals, rather than unrelated ablations.

## Model and objective

The features are square-root cell weights in each phase, square roots of
the arithmetic mean of the two phase weights, and square roots of each domain's
per-phase mass in quality levels at least three (q3/q4 in the pinned swarm). `acquisition.py` fits one metric GP
with separate RBF kernels for the three feature blocks, a bounded domain-and-quality
trend, and a Matérn kernel. It predicts individual metrics and integrates the
clipped objective under their Gaussian posteriors. There is no scalar surrogate
or validation-fold blend. Fitting requires varying features in each block,
finite metric outcomes, and positive replicate counts.

Inputs to `objective.py` must be lower-is-better metric losses. The loader does
not reverse metric signs; convert higher-is-better measurements before fitting
if adding them to the named task lists. `objective.py` subtracts each metric's
proportional-reference mean, divides by
the larger of its reference standard deviation and pooled within-design replicate
noise standard deviation, then clips to ±10. Higher scores are better: targets
receive an improvement reward and regression hinge; guardrails receive the
hinge alone. With standardized target values `t` and guardrail values `g`,
the score is `-mean(t) - mean(max(t, 0)) - mean(max(g, 0))`. Each term has
unit weight, with means taken within its metric group. Edit
the named task lists there to change the objective. Metric fitting currently
requires both groups and zero hinge tolerance.

`observations.py` groups designs by weights rounded to 12 decimal places, in
first-appearance order. Replicates remain together. Metric outcomes use arithmetic means and
replicate counts scale their observation noise. Objective calibration comes from all
observations supplied to the fit. Held-out calibration outcomes do not enter their fold GP fits.

`calibration.py` partitions the designated calibration designs into four folds
with seed 20260912. Each fold's GP excludes that fold, including all its replicates.
For each metric, the correction is the mean held-out residual `b = mean(y - m)`.
The variance multiplier is
`a = max(1, mean(((y - m - b)^2 * (n + 1)/(n - 1) + noise - noise/count) / (v + noise)))`,
where `n` counts calibration designs and the outer mean averages over those
designs separately for each metric. `y` is the held-out design's mean observed
metric, `m` its fold GP prediction, `v` latent posterior variance, `noise` the
per-metric single-run noise variance, and `count` the design's replicate count.
The sample-size factor estimates residual variance and includes uncertainty in
the fitted bias. The noise adjustment converts replicate-averaged errors to the
single-run variance parameterization used at prediction time. All quantities use the objective's standardized metric units.

The final GP refits on every design, including calibration designs. Its mean is
shifted by `b`; latent variance becomes `a * (v + noise) - noise`, treating excess
predictive error as model discrepancy while retaining the replicate-noise estimate.
`predict_metrics(weights)` returns these corrected means and latent variances.
Add `observation_noise` to the variance for marginal intervals on an individual
run. Acquisition integrates the objective using corrected latent moments.
Library callers use `fit_calibrated` with grouped observations and explicit
calibration-design indices. `fit_additive` is the uncorrected base fitter used within calibration. Training takes five metric GP fits; generation only loads and
scores the artifact.
`KernelConfig` supplies Matérn, intercept, feature, and trend amplitudes to both
fold and final fits. These amplitudes are saved with the model and used for
both cross-covariance and its diagonal; loading an artifact preserves its
amplitudes when defaults change. No scalar surrogate or blend is fitted. A later held-out batch is still needed to
check calibration beyond the batch used to develop the correction.

GP conditioning and acquisition scoring use the selected JAX device. Feature
standardization and kernel-length calibration use host NumPy/SciPy. Pairwise distances
use bounded row batches to avoid the earlier
[GPU compilation failure](https://marina.oa.dev/echo/wiki/381).

The model/search interface is `acquisition(weights) -> scores`. Acquisition is
the expected objective under the metric posteriors. Search draws
proportional-centered Dirichlet mixtures, ranks their acquisition scores, and
perturbs the best starts in log space. It excludes observed and duplicate
mixtures, enforces cumulative component epoch limits across both phases, and
returns weights on the 1/49,152 training lattice.

## Artifacts and candidates

Training writes `model.npz` and Marin's `.artifact.json` to `--output`.
`MetricArtifact` records the dataset revision, swarm, observation IDs, component
order, objective metrics, hinge tolerance, distinct-design count, available
tokens, phase budgets, calibration revision, calibration observation IDs, and
fold count. Kernel amplitudes are stored in the NPZ with the fitted state. The NPZ stores the metric acquisition state, correction parameters, and
observed weights. Reusing the directory replaces these files. The new artifact format
requires retraining; artifacts predating persisted kernel amplitudes cannot be
loaded by it.

Move the model directory as a unit. `load_model(Path("mixture-model"))` returns
its metadata, fitted metric model, and observed weights using the directory passed by
the caller. Loading requires neither training data nor fitting. Marin's manual
artifact API does not add step fingerprints or launch provenance automatically.

Generation defaults to 65,536 initial draws, one candidate, seed 111, and at
most 16 cumulative epochs per component. Change these with `--pool-size`,
`--batch-size`, `--seed`, and `--max-epochs`. Batch candidates are ranked
independently; there is no joint exploration or diversity objective.

The output is one Parquet row per candidate, with `swarm_id`, `hf_revision`,
`phase0_weights`, and `phase1_weights`. Each phase maps cell names to weights.
Reusing `--output` replaces that file. Proposed mixtures need training runs to
measure their actual outcomes.

## Validation

The [H100 validation](https://iris.oa.dev/#/job/%2Fheld%2Fmixprior-cleanup-h100-20260910)
fit all 832 observations (817 designs) and exactly reproduced the 20 proposals
from the [calibrated generation run](https://iris.oa.dev/#/job/%2Fheld%2Fmixprior-corrected20-20260909). CPU/GPU metric moments agreed within 2.37e-10 maximum absolute error.
Training took 18.2 seconds and generation 9.0 seconds in a warmed process.

A nested check of the 40 new designs reduced objective RMSE from 2.308 to 1.456
and mean measured-minus-predicted score from +1.710 to -0.167. Nominal 95% metric
interval coverage rose from 80.9% to 94.3%. Five outer folds excluded each scored
design from both GP fitting and the four inner calibration folds. The method
was developed using this batch; future runs are needed for independent validation.
The exploratory benchmark package is not included in this production package.

Run tests with `uv run pytest tests/datakit/mixprior`.
