---
title: Mixprior GP functional-form hill climb
author: held
created: 2026-08-24
updated: 2026-08-24
status: active
tags: [mixprior, gaussian-process, data-mix, ranking]
---

# Mixprior GP functional-form hill climb

## Goal

Improve chronological held-out Spearman correlation on the 56-run
`harrier-store-0381a974-d768` swarm while retaining a Gaussian-process
surrogate.

## Experiment contract

- Run every GP fit and replay benchmark on Iris H100s at production priority.
- Score predictions on later observations after fitting each chronological
  prefix of the target swarm.
- Change one functional-form assumption at a time.
- Before accepting any addition that improves Spearman, ask a subagent to
  challenge whether the added complexity is necessary.
- Development cutoff: 2026-08-25 07:37:15 PDT. After that time, only remove
  code or model terms while preserving the best result.

## Baseline

The current GP combines phase-weighted Hellinger geometry, a shared linear
content kernel over saturating exposure, a learned cooldown multiplier, a
swarm-specific Hellinger residual, and a learned repeat-harm mean.

Earlier H100 replay over target prefixes 1, 3, 5, 10, 20, and 40 measured:

| Form | Mean held-out Spearman |
| --- | ---: |
| Fixed cooldown scale | 0.4430 |
| Learned cooldown scale | 0.4538 |

These numbers are a starting reference. The overnight harness will rerun the
current production implementation before comparing new forms.

## Hypotheses

1. The saturation scale may need to differ from one epoch.
2. A single learned cooldown scale may be sufficient; separate phase content
   coefficients may add variance without improving ranking.
3. The fixed 16-epoch harm knee may be too far into the exposure range to
   affect this swarm.
4. The swarm-specific residual may obscure shared transfer signal.

## Results

### 2026-08-24 22:07 PDT — first H100 batch

The exact same-job baseline and exposure-scale arms currently show:

| Saturation scale (epochs) | Mean held-out Spearman |
| ---: | ---: |
| 0.25 | 0.5444 |
| 0.50 | 0.4930 |
| 1.00 | 0.4538 |
| 2.00 | 0.4453 |

The 0.25 scale improves all six prefix scores relative to 1.0. It is the lower
edge of the first grid, so a finer lower-scale grid is running. A complexity
review found that changing the fixed scale adds no structural complexity, but
the 56-run result is a development score after grid selection and the nested
prefix scores are dependent. Keep every prefix visible and do not describe the
gain as independent validation.

Removal ablations at the 1.0 baseline:

| Form | Mean held-out Spearman |
| --- | ---: |
| Current GP | 0.4538 |
| Remove repeat-harm mean | 0.4538 |
| Remove swarm-specific residual | 0.4542 |

The repeat-harm mean is inert at the current 16-epoch knee. Removing the swarm
residual is a tiny improvement and remains a candidate for the deletion pass.

Lower harm knees at two and four epochs did not improve the baseline. The
eight-epoch result is pending.

### 2026-08-24 22:19 PDT — exposure refinement and first kernel check

The refined exponential-saturation grid found an interior best:

| Saturation scale (epochs) | Mean held-out Spearman |
| ---: | ---: |
| 0.0625 | 0.5589 |
| 0.1250 | 0.5954 |
| 0.1875 | 0.5715 |
| 0.2500 | 0.5444 |
| 0.3750 | 0.5082 |

A rational saturation curve peaked at 0.5936, so it does not justify replacing
the simpler existing exponential form. The 56 rows contain 56 unique phase
mixtures; exact duplicate designs do not dominate the rank statistic.

An initial Matérn-3/2 Hellinger kernel scored 0.5959 at the old 0.25 saturation
scale. Complexity review identified that its numeric lengthscale was calibrated
with the RBF correlation formula, confounding smoothness and effective range.
Keep RBF with 0.125 as incumbent while a median-correlation-matched 2x2
RBF/Matérn by 0.125/0.25 interaction check runs.

### 2026-08-24 22:34 PDT — phase simplification and Hellinger range

The range-matched Matérn-3/2 kernel did not retain the apparent gain. It scored
0.5587 at the 0.125 saturation scale versus 0.5954 for RBF, so the GP remains
RBF-based.

Fixing the cooldown multiplier simplified the model and slightly improved the
development score. At saturation 0.109375, fixed multipliers 1.10, 1.15, 1.20,
1.25, and 1.30 scored 0.5988, 0.6004, 0.6060, 0.6043, and 0.6022. The broad
optimum is around 1.2--1.25. Fixing this value removes a learned kernel
parameter.

Shortening the Hellinger RBF range improved the current fixed-phase model. The
median-distance exponents 0.25, 0.50, and 1.00 scored 0.6043, 0.6397, and
0.6044. A finer range grid and parameter interaction check are running. The
adversarial review requested paired optimizer seeds and emphasized that the six
prefixes are nested, not independent replications. It also questioned whether
the replay reference matches production. The campaign metadata resolves that
point: `kernel_reference_swarm` is `harrier-store-0381a974-d768`, exactly the
56-run swarm used to calibrate the replay lengthscale. This is still a
transductive calibration over known mixture locations, but it matches the
production model's intended reference.

Five optimizer seeds at the 0.109375/1.25 setting range from 0.5628 to 0.6043
mean Spearman, so future comparisons must be paired by seed. Extending the fit
from 2,000 to 5,000 steps reduced Spearman to 0.5601 while increasing the
repeat-harm weight sixfold. A matched no-harm convergence check is needed.

At the current setting, removing the repeat-harm mean changes Spearman by
exactly zero. Removing the swarm residual drops it from 0.6043 to 0.4377, and
removing Hellinger geometry drops it to 0.2977. Keep both covariance terms;
the harm mean is a deletion candidate.

### 2026-08-24 22:55 PDT — robust range result and transfer ablations

The corrected fine Hellinger grid has a broad optimum:

| Median-distance exponent | Mean held-out Spearman |
| ---: | ---: |
| 0.35 | 0.6371 |
| 0.40 | 0.6462 |
| 0.45 | 0.6480 |
| 0.50 | 0.6389 |
| 0.55 | 0.6336 |
| 0.60 | 0.6309 |

The first fine-grid attempt was cancelled because substring parsing made 0.45
match 0.40 and 0.55 match 0.50. The table above is from the corrected rerun.

The 0.50 range gain is stable across all five paired optimizer seeds. The
one-range 0.25 scores are 0.6060, 0.6005, 0.5680, 0.5826, and 0.5904; the
corresponding 0.50 scores are 0.6389, 0.6356, 0.6433, 0.6406, and 0.6390. A
paired 0.45 seed check is running.

Removing either source swarm hurts transfer. With the repeat-harm mean removed,
the full model scores 0.6388 at exponent 0.50; removing the 840-era legacy
swarm scores 0.6133, removing the 115-run first-store swarm scores 0.4787, and
using target-prefix rows alone scores 0.0591. The old observations materially
improve ranking rather than merely adding complexity.

Adding 3, 6, or 12 objective units of independent noise standard deviation did
not improve Spearman. The ten repeated proportional runs in the 115-run swarm
have an observed score standard deviation of 5.93, but their reported eval
standard deviations are already roughly 4.4--8.5. Keep the recorded variance.

Allowing separate shared and swarm-residual Hellinger ranges produced 0.6471
with exponents 0.25 and 0.50. Adversarial review rejected it because it loses to
the simpler single 0.45 range at 0.6480, has a larger selected search space, and
makes two RBF amplitudes and ranges weakly identifiable on this benchmark.

### 2026-08-24 23:11 PDT — local optimum and provisional task correlation

The fixed-shape local grid improves the simple rank-0 GP to 0.6542 at a
cooldown multiplier of 1.125, saturation 0.109375, and Hellinger exponent 0.45.
Neighboring cooldown multipliers 1.15 and 1.175 score 0.6535 and 0.6485, so the
useful region is broad. Exponents 0.425, 0.45, and 0.475 at multiplier 1.15
score 0.6519, 0.6535, and 0.6431.

Alternative exposure curves did not beat exponential saturation. Reweighting
cooldown in the Hellinger geometry also hurt: multiplying its geometric weight
by 2, 4, and 8 reduced 0.6480 to 0.6295, 0.6160, and 0.6197. A separate learned
cooldown multiplier scores 0.6399, below the simpler fixed multiplier.

Changing the swarm residual's `IndexKernel` from rank 0 to rank 1 provisionally
raises 0.6535 to 0.6722; rank 2 scores 0.6437. The gain covers four of six
prefixes, improves mean pairwise accuracy from 0.7295 to 0.7380, and improves
NLL from 6.0478 to 5.8936, but slightly worsens winner regret. Complexity review
notes that the shared RBF already supplies one all-swarms factor, so rank 1 adds
a second weakly identified cross-swarm factor. Paired optimizer seeds and a
simpler single intrinsic-coregionalization kernel are running before acceptance.

### 2026-08-24 23:24 PDT — accept rank-1 swarm correlation

The rank-1 residual passes the requested robustness checks:

- Across five paired optimizer seeds, rank 0 averages 0.6461 and rank 1
  averages 0.6584. Paired gains are +0.0187, +0.0095, +0.0122, -0.0045, and
  +0.0257.
- On the earlier 115-run transition, rank 1 improves the same chronological
  metric from 0.6448 to 0.6793.
- Halving and quartering the 804 legacy rows reduces the rank-1 result from
  0.6722 to 0.6657 and 0.6563, but both remain above the rank-0 development
  score. The effect does not depend on the legacy source overwhelming the
  likelihood.
- Replacing the shared-plus-residual structure with a single rank-1 intrinsic
  coregionalization kernel scores 0.6306. The forced universal shared term is
  useful rather than redundant in prediction.

Retain the rank-1 residual as the only accepted structural addition so far.
Treat only the summed task covariance as meaningful; the individual shared and
residual factors are not scientifically identifiable.

### 2026-08-24 23:37 PDT — promote the replayed form

The rank-1 local grid places the best fixed cooldown multiplier at 1.05:
0.6738 at saturation 0.109375. Nearby multipliers 1.025, 1.075, 1.10, and
1.125 score 0.6695, 0.6727, 0.6731, and 0.6728, so the setting is not an
isolated spike. Hellinger exponents 0.40, 0.425, 0.45, 0.475, and 0.50 score
0.6624, 0.6674, 0.6722, 0.6677, and 0.6688 at the preceding multiplier; keep
0.45.

Learning the cooldown multiplier lowers the rank-1 result from 0.6738 to
0.6617. The production implementation now fixes it at 1.05, removes the
ineffective harmful-exposure mean and feature column, fixes saturation at
0.109375 epochs, calibrates the shared Hellinger exponent at 0.45, and uses a
rank-1 swarm residual. The narrow model tests pass. An exact production-code
replay is running on an Iris H100, while the five-seed final robustness replay
has completed seeds 0--2 at 0.6738, 0.6531, and 0.6671.

The final five-seed scores are 0.6738, 0.6531, 0.6671, 0.6473, and 0.6667
(mean 0.6616). A random swarm-intercept covariance scores 0.5865, and replacing
the joint phase Hellinger RBF with an additive pair of phase RBFs scores 0.4958.
Reject both.

Because the cooldown multiplier is fixed, the two phase-exposure blocks can be
combined before GP construction. The production model now computes
`pretraining + 1.05 * cooldown` in the feature map and uses a stock linear
kernel. This deletes the custom phase-linked kernel and reduces the GP input
width without changing its covariance. All 23 mixprior tests pass; a compact
production-code H100 replay is running.

The compact production-code replay exactly matches the research harness at
0.6738148 mean Spearman, including every prefix score, and completes on one
H100 in 80 seconds including setup. The earlier 115-run transition scores
0.6889, improving on the provisional rank-1 result and providing an independent
campaign check.

Further additions were rejected:

- A task-specific linear usefulness residual scores 0.6683.
- Fixed rational-quadratic Hellinger kernels with alpha 0.5, 1, 2, and 4 score
  0.5904, 0.6372, 0.6576, and 0.6658.
- Downweighting cooldown inside Hellinger geometry by 0.25 or 0.5 scores 0.6539
  and 0.6648.
- Multiplying source or legacy observation variances by 2 or 4 scores between
  0.6701 and 0.6734.
- Equal positive initializations of the rank-1 task factor score between
  0.6629 and 0.6666.
- Harmful-exposure knees at 2, 4, or 8 epochs score 0.6706, 0.6703, and 0.6710.
  Their fitted standardized contributions remain below 0.007, and all trail
  the no-harm model.

### 2026-08-25 00:15 PDT — reject content truncation and two-timescale exposure

Projecting the usefulness vector with an uncentered SVD at ranks 16, 32, 64,
128, and 256 scores 0.6096, 0.6714, 0.6751, 0.6743, and 0.6734. The rank-64
gain over the full 1,000-dimensional linear kernel is only 0.0013 after trying
many variants, while winner rank and regret worsen. Adversarial review also
found that the experimental basis weights bucket rows equally, is
campaign-dependent, and is not invariant to splitting a bucket. Keep the full
linear feature space; it learns one variance rather than one lengthscale per
content dimension, so truncation does not materially simplify GP fitting.

Adding a second, slower exposure-saturation timescale also hurts. Tail scales
at 10 or 20 times the 0.109375-epoch scale with weights 4 or 8 score between
0.6613 and 0.6677, below the single exponential's 0.6738. Keep one saturation
timescale.

Four additional families do not improve the incumbent:

- Replacing phase-separated Hellinger geometry with the aggregate curriculum
  scores 0.4917. Keeping phases distinct is important.
- Square-rooting, raising to the 0.75 power, or normalizing the usefulness
  vector scores 0.6275, 0.6642, and 0.4593. Scaling cooldown exposure before
  saturation scores 0.6702. Keep the direct linear usefulness vector and
  scale cooldown's incremental gain after saturation.
- Centering or standardizing outcomes within each swarm scores 0.5835 and
  0.3438. Asinh response warps score at most 0.5641. Preserve the common
  objective scale.
- Multiplying every recorded observation variance by 0.25, 0.5, or 2 scores
  0.6695, 0.6698, and 0.6748; inferring one homoskedastic noise level scores
  0.6493. The 0.0010 gain at 2x is too small after many comparisons and does
  not improve winner regret, so retain the measured variances unchanged.

### 2026-08-25 00:48 PDT — reject learned independent ranges and epoch kernels

Learning separate Hellinger ranges by marginal likelihood does not improve
chronological ranking. Learning only the shared range scores 0.6554, learning
only the residual range scores 0.5525, and learning both independently scores
0.4015. A single tied learned range remains to be tested.

Rav-style epoch structure does not help within the 56-run regime. An RBF over
token-weighted repetition mass scores 0.4665 when added and 0.5574 when
multiplied with the existing covariance. Linear covariance terms for exposure
past 8.9 epochs score 0.6505 without context scaling and 0.6516 with Rav's
budget/model scaling. Softplus harm means are numerically dominated by their
large raw feature scale. Corrected linear harm-kernel sweeps at knees 1, 2, 4,
and 8 score 0.6251, 0.6597, 0.6483, and 0.6074. Corrected softplus-squared
harm kernels score 0.6589 at knees 2 and 4. All trail 0.6738, consistent with
Rav's finding that epoch correction was an out-of-distribution adjustment that
hurt in-distribution cross-validation.

A quadratic polynomial usefulness kernel scores 0.6484. Keep the linear
usefulness covariance.

### 2026-08-25 01:00 PDT — reject learned range and content entropy

Tying one marginal-likelihood-learned Hellinger range across the shared and
swarm-residual kernels scores 0.3857. Every prefix fit moves the calibrated
median exponent from 0.45 to roughly 0.07--0.10, making the model much smoother
but substantially worse at ranking. Keep the fixed replay-calibrated range.

A normalized content-row entropy statistic initially scores 0.6797. Across
optimizer seeds 0--4 it scores 0.6797, 0.6661, 0.6700, 0.6633, and 0.6703,
compared with incumbent scores 0.6738, 0.6531, 0.6671, 0.6473, and 0.6667.
The earlier 115-run transition falls from 0.6889 to 0.6690. Adversarial review
also finds that the statistic is partly a rank-one anisotropy already
representable by the full usefulness vector, is sensitive to non-identical
bucket repartitioning, and worsens winner metrics. Keep it scratch-only and do
not promote it.

Small changes to cooldown's Hellinger geometry do not help. Relative geometry
multipliers 0.75, 1.25, and 1.5 score 0.6711, 0.6662, and 0.6586. Retain the
phase-token fractions.

### 2026-08-25 01:12 PDT — proportional-distance priors do not transfer cleanly

Adding a linear GP covariance over phase-weighted `KL(mixture || available
token share)` scores 0.6917 on optimizer seed 0. Across five paired optimizer
seeds it averages 0.6723, versus 0.6616 for the incumbent. The gain is almost
entirely in prefixes 3--20; prefix 40 falls from 0.5794 to 0.4676 on seed 0 and
to 0.32--0.44 on the other seeds. More importantly, the earlier 115-run
transition falls from 0.6889 to 0.6321. Do not promote the shared KL feature.

The strong seed-0 result is not explained by an arbitrary extra one-dimensional
direction: four fixed random projections of the existing usefulness vector
score 0.6706--0.6742. However, alternatives do not resolve transfer. Raw
bucket-space Hellinger distance from proportional scores 0.6826, Jensen-Shannon
distance scores 0.6686, shared-content Hellinger distance from proportional
scores 0.6554, and KL of the aggregate two-phase mixture scores 0.6681. A
task-specific rank-1 KL covariance collapses the 56-run score to -0.0234 and
scores 0.6676 on the earlier transition. The cross-schema scale and late-prefix
failure outweigh the development gain.

Replacing raw mixture geometry with marginal saturated-usefulness geometry is
ill-defined in this regime: the 0.109375-epoch curve numerically saturates all
pretraining support for some curricula, leaving zero marginal cooldown mass to
normalize. The failed normalization is itself evidence against treating
marginal usefulness as a probability geometry; no fallback was added.

### 2026-08-25 01:23 PDT — square-root KL improves the 56 replay but fails transfer

Taking the square root of phase-weighted KL is substantially better behaved
than raw KL on the primary 56-run replay. It scores 0.6894 on seed 0 and
0.6705, 0.6787, 0.6666, and 0.6722 on seeds 1--4, for a mean of 0.6755 versus
0.6616 for the incumbent. Every paired optimizer seed improves. The transform
also avoids raw KL's systematic prefix-40 collapse on several seeds.

Adversarial review gives the transform a coherent local interpretation: near
proportional sampling, square-root KL is an information-geometric displacement
magnitude, while raw KL is distance-squared-like. The scalar remains sensitive
to arbitrary bucket repartitioning and can confound specialization intensity
with chronological campaign progress.

The independent transition rejects it. On the earlier 115-run replay,
square-root KL scores 0.6248 versus 0.6889 for the incumbent. Raw and
square-root bucket-space Hellinger, shared-content Hellinger, and Jensen-Shannon
alternatives also fail to improve both transitions. A two-point Rav ordering
flips from wrong to correct under KL-based features, but this is too small to
count as validation. Keep all proportional-distance features out of
production.

Adding a global linear kernel on the rooted Hellinger features scores 0.6726
on the 56 transition and 0.7000 on the earlier transition. Since it does not
improve the primary replay and trades off the two transitions, reject it.

### 2026-08-25 01:36 PDT — phase and interaction additions trade off transitions

Weak log-normal priors on covariance amplitudes do not improve the replay.
A prior on the shared Hellinger amplitude scores 0.6698 on the 56 transition
and 0.6871 on the earlier transition. A prior only on usefulness amplitude is
effectively identical to the incumbent at 0.6738 and 0.6889. Combining both
priors matches the weaker shared-amplitude result. Keep the likelihood fit
unregularized rather than adding inert prior declarations.

A shared Hellinger-by-usefulness interaction is a proper locally linear GP,
but scores 0.6650 and 0.6780. A separate cooldown usefulness residual directly
tests whether late-training content value can depart from the linked phase
coefficient. It improves the earlier transition to 0.6961 but drops the 56-run
transition to 0.6306. The linked fixed phase coefficient transfers more
reliably than either extra degree of freedom, so reject both additions.

Replacing the categorical rank-1 residual with an RBF over model size and
token-horizon context fails sharply, scoring 0.5124 and 0.5731. The two modern
Harrier swarms have identical recorded context, but their residual mixture
surfaces are not interchangeable. Keep model and token context as provenance
and retain the learned swarm covariance.

Median-correlation-matched Matérn Hellinger kernels with smoothness 1/2, 3/2,
and 5/2 score 0.5842, 0.6576, and 0.6635 on the 56 transition. The ranking
improves as the kernel approaches the infinitely smooth RBF incumbent, but none
beats it. Keep the RBF form.

A second shared Hellinger RBF at half the incumbent range scores 0.6742, only
0.0003 above the incumbent on seed 0, while pairwise accuracy, winner metrics,
and the late prefixes worsen. The earlier transition also falls to 0.6853.
Adversarial review rejects the tiny gain as negligible relative to optimizer
seed spread and notes that the two nearby RBF amplitudes are weakly identified.
Do not run a range grid or promote the multiscale form.

Adding a fixed-range RBF residual around the linear usefulness kernel scores
0.6605. The shared Bayesian linear preference is more effective than a local
nonlinear correction in the usefulness representation; reject the residual.

The 115-run and 56-run Harrier swarms have the same 200 domain-quality cell
labels, so two exact-schema residuals were tested while explicitly excluding
the semantically incompatible 840-era buckets. A gated Hellinger RBF over the
aligned raw cells scores 0.6173, and a shared linear kernel over their exact
phase-linked usefulness scores 0.5376. Both are far worse than the shared
Luxical representation. Exact cell identity does not provide a useful shortcut
for transfer between the two token stores; retain only the common content
feature space.

Aggregating the exact-schema residual into five quality-level usefulness
features recovers some performance but still scores only 0.6704, with worse
winner metrics. The quality labels do not justify an extra target-schema
kernel.

A zero-centered Normal prior on the rank-1 task factor scores 0.6718 on the
56-run transition and 0.6041 on the earlier transition. Shrinking transfer
correlation through this parameter harms both replays, so keep the learned task
covariance unregularized.

### 2026-08-25 02:09 PDT — late target observations expose transfer error

At prefix 40, the largest incumbent errors are concentrated in the late PI
suggestions. Trial 44 is observed at -14.3 but predicted at +47.3, while the
observed best trial 40 is +6.0 and predicted at +26.2. Both are close in the
current feature space to target observations with mostly negative outcomes,
yet the shared source signal dominates their posterior means. Increasing source
observation variance by 2x or 4x does not repair the late-prefix rank and harms
the aggregate replay. A task-specific usefulness covariance likewise scores
only 0.6683. This identifies a limitation of the available shared
representation, but does not support a prefix-dependent transfer heuristic.

Two more proper-GP bridges do not repair the late ranking. Reusing the same
rank-1 swarm covariance for a task-specific usefulness residual scores 0.6724
on the 56-run transition, falls to 0.5246 on the earlier transition, and makes
the prefix-40 score worse. Adding an aggregate-curriculum Hellinger RBF while
retaining the incumbent phase-separated RBF scores 0.6706 and 0.6871. The
phase-linked usefulness term already supplies the useful phase pooling; neither
extra covariance is retained.

Replacing the categorical task covariance with an RBF over each token store's
availability-weighted Luxical centroid scores 0.5252 and 0.5731. Aggregate
store composition is not an adequate substitute for the learned swarm
relationship.

### 2026-08-25 02:41 PDT — task-diagonal deletion does not survive paired transfer replay

Removing the independent diagonal variances from the rank-1 task covariance is
the only deletion to improve the seed-0 score on both transitions: 0.6784
versus 0.6738 on the 56-run transition and 0.7166 versus 0.6889 on the earlier
transition. Adversarial review required paired optimizer seeds because the
deletion forces all task-specific residual structure through one factor and
worsens cold-start winner selection.

Across five seeds, the 56-run mean rises from 0.6616 to 0.6670 and four seeds
improve. The independent transition does not confirm the result: its mean falls
from 0.6334 to 0.6224, only three seeds improve, and one seed falls by 0.0968.
The deletion also reduces posterior coverage and worsens some winner metrics.
Keep the diagonal task variances.

Other direct deletions fail. Removing the shared Hellinger output scale scores
0.6272 and 0.6797. Removing the phase-linked usefulness kernel scores 0.4592 on
the 56-run transition despite reaching 0.7202 on the earlier transition.
Replacing the fitted constant mean with zero scores 0.5413 and 0.6774. Retain
all three parts of the production covariance and the constant mean.

A weak log-normal prior on the independent task variances scores 0.6686 on the
56-run transition and 0.6903 on the earlier transition. It trades a 0.0052
primary-replay loss for a 0.0014 calibration gain, so it is not retained. A
full LKJ task-covariance prior was also attempted, but GPyTorch's LKJ prior
kept an internal CPU tensor when the GP was placed on CUDA. Since the simpler
factor and variance priors both fail, no custom GPU compatibility code was
added for this research-only alternative.

### 2026-08-25 02:57 PDT — final pre-cutoff robustness check

An independent scientific review found the covariance search effectively
exhausted. Every remaining untested GP extension would add a weakly identified
nonstationary, high-dimensional, or target-local term without new evidence for
its shape. The least-supported retained choice is instead the fixed cooldown
multiplier: 1.05 was selected on optimizer seed 0 and differs only slightly
from the simpler value 1.0.

A paired five-seed H100 replay of cooldown scales 1.0 and 1.05 is running at
the production saturation and Hellinger settings. The predeclared decision is
to retain 1.05 only if both its mean and median paired Spearman changes are
positive without worse winner selection. Otherwise the extra weighting will
be deleted after the cutoff.

The paired replay retains 1.05. Its five-seed mean Spearman is 0.66160 versus
0.65995 at 1.0. Paired changes are +0.00355, -0.00334, +0.00023, +0.00239,
and +0.00542, giving positive mean and median changes. Aggregate winner rank
and regret are identical between the paired fits for every seed. The effect is
small, but it passes the predeclared test without a winner-selection tradeoff.

The other selected fixed scales also survive paired-seed checks. Saturation
scales 0.09375, 0.109375, and 0.125 have five-seed mean Spearman 0.66007,
0.66160, and 0.65650. Hellinger median-distance exponents 0.425, 0.45, and
0.475 score 0.66102, 0.66160, and 0.65749. The incumbent 0.109375/0.45 setting
has the best mean in both comparisons and a better worst seed than the very
close 0.425 alternative. No scalar setting changes.

### 2026-08-25 07:42 PDT — deletion-only closeout

The deletion phase began after the fixed 07:37:15 cutoff. It removed only
values with no reader or duplicated ownership: the stored target-swarm index,
an unused objective-index property, a duplicate diagnostics summary and its
arguments, and a separately supplied objective payload that could disagree
with the objective owned by the campaign. Candidate artifacts now hash the
campaign's actual objective directly.

All 23 mixprior tests pass after these removals. A fresh end-to-end production
job on one Iris H100 fitted the GP, scored a 4,096-row pool, wrote the complete
candidate bundle, and selected candidate `dc8f667585985b3f`, exactly matching
the pre-deletion H100 run with the same seed and pool. The removals do not
change fitting, ranking, or selection.

### 2026-08-25 09:31 PDT — simplifying ablation search

A new deletion-only model search starts from the verified production GP. The
first screen tests four untried simplifications: fix the linear usefulness
variance after its data-scale initialization, fix all task-specific residual
variances to one, remove outcome standardization, and reduce the optimizer
limit from 2,000 steps to 1,000 or 500. Previously rejected removals are not
repeated.

Each arm first runs seed 0 on both the 56-run validation transition and the
earlier 115-run calibration transition. An arm advances only if it remains
close on both; any apparent improvement then requires paired optimizer seeds
and adversarial review before production changes. The screen runs as
`/held/mixprior-gp-simplification-screen-r2-20260825` on one production-priority
Iris H100 and writes `simplification-screen-r2.json` beside the earlier replay
artifacts.

The seed-0 screen leaves one viable simplification:

| Form | 56 Spearman | 115 Spearman |
| --- | ---: | ---: |
| Incumbent | 0.673815 | 0.676848 |
| Fix usefulness variance | 0.673397 | 0.676788 |
| Fix task variances | 0.632855 | 0.608132 |
| Remove outcome standardization | -0.298265 | 0.194358 |
| 1,000 fit steps | 0.633328 | 0.660658 |
| 500 fit steps | 0.580220 | 0.647906 |

Fixing the linear usefulness variance at its diagonal-matched initialization
removes one learned GP parameter while changing Spearman by less than 0.0005
on either transition and leaving seed-0 winner metrics unchanged. The other
arms lose too much ranking performance and stop here. A five-seed paired replay
of the incumbent and fixed-variance form is running on
`/held/mixprior-gp-fixed-usefulness-paired-seeds-20260825`.

The simplifying search remains restricted to Gaussian-process surrogates. Its
fixed cutoff is 2026-08-25 20:25:49 PDT, ten hours after the constraint was
set. No new model-family search begins after that time.

The fixed usefulness variance passes the paired robustness gate. Across five
optimizer seeds, its mean Spearman change is -0.00016 on the 56-run transition
and -0.00006 on the 115-run transition. Median changes are -0.00006 on both.
Winner rank, winner regret, and coverage are unchanged; pairwise accuracy
changes by less than 0.00008. The small NLL changes are +0.00062 and +0.02156,
the latter against a calibration NLL near 121.

The learned amplitude stays at its diagonal-matched initialization. Across 65
prefix fits spanning both transitions and five seeds, fitted/initial amplitude
ranges from 0.9848 to 0.9988. The same behavior holds after taking every second
or fourth legacy-source row. In five full production candidate generations,
freezing the amplitude selects the same candidate every time; acquisition-score
Spearman is at least 0.9999988 and top-10 overlap is 9/10 or 10/10. Median fit
time falls slightly, by about 0.6 seconds in those production fits.

The production GP now fixes the phase-linked linear amplitude after the
diagonal match. This removes a learned covariance parameter whose optimizer
movement had no operational effect. The other screened simplifications remain
rejected.

All 23 mixprior tests, targeted lint, and pyrefly pass after promotion. A fresh
production-priority H100 run fitted all 978 observations, scored the same
4,096-row seeded pool, wrote the full bundle, and again selected
`dc8f667585985b3f`. The end-to-end result matches the preceding implementation.

The next simplifying screen remains within the same exact GP family. It records
the initialization and fitted value of every remaining covariance parameter to
identify parameters that the likelihood does not use, and separately compares
the explicit 2,000-step Adam fit with BoTorch's standard marginal-likelihood
fit. Only a deterministic, stable parameter or a standard fitter that preserves
both chronological replays can advance.

No second covariance parameter is inert. Across all 13 chronological prefix
fits, the fitted shared-Hellinger log amplitude moves by 0.46--2.56, the
rank-1 task factor moves by 1.02--2.56 times its initial norm, the task diagonal
moves by 0.80--3.80 times its initial norm, and the standardized constant mean
moves by 0.14--2.04. The already-fixed lengthscales and usefulness amplitude are
the only unchanged quantities. Freezing another remaining parameter would
discard a fit the likelihood is actively using.

BoTorch's standard SciPy marginal-likelihood fitter is also rejected. On the
56-run transition it reduces mean chronological Spearman from 0.6734 to 0.5944,
worsens mean winner rank from 7.2 to 14.8, and reduces 90% coverage from 1.00 to
0.73. The calibration half was cancelled after the primary gate failed. Keep
the explicit Torch fitter and its 2,000-step limit.

The final production form also combines pretraining and cooldown gains before
the shared-content projection, replacing two equivalent matrix products with
one. This is algebraically exact. The complete 23-test mixprior suite, targeted
lint, pyrefly, and `git diff --check` pass. A final production H100 end-to-end
run fitted 978 rows, scored 4,096 candidates, and selected
`dc8f667585985b3f` again. No further deterministic GP parameter is supported
for deletion by this screen.

### 2026-08-25 11:50 PDT — restore learned usefulness and epoch harm

The GP again learns the phase-linked usefulness amplitude. Its
diagonal-matched value is only an initialization.

The GP mean now includes a negative repeated-exposure tail. The harm amplitude
starts at 0.05 standardized objective units, and the harmful-epoch knee starts
at 16 epochs. Both positive parameters are fitted by marginal likelihood. The
mean uses the token-share-weighted squared excess above the knee in log-epoch
space. Padding allows swarms with different component counts to share this
permutation-invariant mean.

A production-priority H100 end-to-end fit over all 978 observations succeeded
and scored a 4,096-candidate pool. It learned a 13.414-epoch knee, a 0.4869
harm amplitude, and a 745.04 usefulness variance.

The seed-0 chronological replay does not support the harm term as a ranking
improvement:

| Form | 56 Spearman | 115 Spearman |
| --- | ---: | ---: |
| No learned harm | 0.673815 | 0.676848 |
| Learned knee and amplitude | 0.642799 | 0.674922 |

Across the six 56-run prefix fits, the knee is 13.12--13.14 epochs and the
amplitude is 0.59--0.60. Across the seven 115-run prefix fits, the knee is
12.48--13.31 epochs and the amplitude is 0.52--0.90. The likelihood therefore
uses the harmful tail, but its fitted signal reduces chronological ranking on
the 56-run transition.

The H100 jobs are `/held/mixprior-learned-epoch-harm-e2e-20260825` and
`/held/mixprior-learned-epoch-harm-replay-r2-20260825`. The first replay job
failed because the temporary harness calibrated its Hellinger range from a
one-row observed prefix. The corrected replay uses the full target design set
for transductive kernel calibration, matching the earlier replay contract.

### 2026-08-25 12:10 PDT — scale epoch harm by tokens per total parameter

The harm mean now multiplies the exposure penalty by the swarm's physical
training tokens per total parameter, relative to the campaign geometric mean,
raised to a learned exponent. The exponent starts at zero and may learn either
direction.

The seed-0 chronological H100 replay scored 0.651914 Spearman on the 56-run
transition and 0.738084 on the 115-run transition. The corresponding unscaled
harm scores were 0.642799 and 0.674922. The no-harm scores were 0.673815 and
0.676848.

The exponent is not stable enough to interpret. It stays near -4.0 in the
56-run replay. In the 115-run replay it starts near +3.8, then moves below -4.3
in later prefixes. A full 978-observation fit learned a 12.465-epoch knee,
0.9425 amplitude, and -3.523 exponent. The three Harrier/Rav swarms have
28--30 physical tokens per total parameter; the legacy d512 swarm has 344.
The ratio is therefore mostly a proxy for the legacy task boundary in this
campaign.

The production-priority H100 jobs are
`/held/mixprior-token-total-parameter-harm-e2e-20260825` and
`/held/mixprior-token-total-parameter-harm-replay-20260825`. The replay result
is at
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/token-total-parameter-harm.json`.

### 2026-08-25 12:58 PDT — learn saturation inside the GP kernel

The phase-linked usefulness covariance now receives raw pretraining and final
component exposures. A custom positive-semidefinite kernel applies the
exponential saturation transform, projects the result through each swarm's
fixed Luxical component matrix, and takes a linear covariance in the shared
content space. Its positive saturation scale starts at 0.1 epochs and is
learned jointly with the other GP parameters. This removes the external scalar
profile and the fixed replay-selected denominator.

A full 978-observation H100 fit learned 0.011449 saturation epochs, a 12.640
epoch harm knee, 0.8371 harm amplitude, -3.336 token/total-parameter exponent,
and 412.68 usefulness variance. Its marginal log likelihood was 2.786607,
slightly above the external profile's 2.786464. The 4,096-row end-to-end search
selected candidate `e5c398aa44c2cf99`.

The seed-0 chronological replay scored 0.581387 Spearman on the 56-run
transition and 0.745116 on the 115-run transition. The preceding fixed-scale
form scored 0.651914 and 0.738084. The likelihood prefers a much sharper
saturation curve, but the primary chronological ranking becomes worse. Prefix
fits learned 0.01229--0.01277 saturation epochs on the 56-run transition and
0.01288--0.04410 on the 115-run transition.

The production-priority H100 jobs are
`/held/mixprior-native-saturation-e2e-r2-20260825` and
`/held/mixprior-native-saturation-replay-r2-20260825`. The replay result is at
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/native-saturation.json`.

### 2026-08-25 14:47 PDT — reject the DSP feature-kernel adaptation

Calvin's canonical DSP benefit signal was adapted into the shared-content
kernel as
`(1 + phase_premium * cooldown_share) * (1 - exp(-saturation_rate * epochs))`.
The harmful-exposure mean used DSP's squared softplus in log-epoch space.
Saturation rate, phase premium, harmful-epoch knee, harm amplitude, and the
tokens-per-total-parameter exponent were fitted by marginal likelihood under
priors centered at `0.125`, `1.0`, `16`, `0.05`, and `0` respectively.

The production-priority H100 chronological replay completed as
`/held/mixprior-dsp-gp-replay-20260825-r5`:

| Form | 56-run Spearman | 115-run Spearman |
| --- | ---: | ---: |
| Preceding native saturation kernel | 0.581387 | 0.745116 |
| DSP benefit kernel and softplus harm | 0.397523 | 0.695280 |

The DSP form loses 0.183864 and 0.049836 respectively. On the 56-run prefixes,
the fitted saturation rate stays between 0.02359 and 0.02389 and the phase
premium between 0.477 and 0.484. The fitted harm knee stays near 14.20 epochs
and its amplitude near 0.01883.

This rejects the DSP signal as a covariance feature in the production GP. It
does not reject Calvin's original regression: that model uses nonnegative
benefit and harm coefficients and learns separate saturation rates and
overexposure thresholds by domain. A zero-mean linear GP kernel permits signed
content coefficients and this adaptation shared one saturation rate and phase
premium across content. The faithful GP follow-up would put the signed DSP
structure in the prior mean and use the GP for residuals rather than treating
DSP only as a similarity feature.

Artifact:
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/dsp-gp-replay.json`.

### 2026-08-25 15:20 PDT — reject embedding-conditioned DSP prior mean

The next treatment moved DSP from the covariance into the GP prior mean. Each
mixture component received a token-weighted, standardized 32-dimensional PCA
of its shared Luxical content vector. Two log-linear functions mapped that
embedding to the component's saturation rate and harmful-epoch threshold. The
mean used DSP's positive saturating benefit, learned cooldown premium, and
squared-softplus harm; the Hellinger GP modeled the residual.

The production-priority H100 chronological replay completed as
`/held/mixprior-domain-dsp-mean-replay-20260825-r2`:

| Form | 56-run Spearman | 115-run Spearman |
| --- | ---: | ---: |
| Native saturation kernel | 0.581387 | 0.745116 |
| Embedding-conditioned DSP prior mean | 0.410693 | 0.700389 |

The treatment loses 0.170694 and 0.044727 respectively. The learned embedding
effects are effectively zero: saturation-vector norms are 0.0022--0.0023 on
the 56-run prefixes, and harmful-threshold-vector norms are 0.0004--0.0005.
The base saturation rate stays near 0.0978, the base knee near 14.20 epochs,
the cooldown premium near 0.390, and the benefit amplitude near 0.0381.

This experiment does not support replacing the current covariance with a DSP
prior mean, and it provides no evidence in these aggregate outcomes for
domain-dependent saturation or harm thresholds under the tested Gaussian
shrinkage prior. The treatment is not promoted to the production model.

Artifact:
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/domain-dsp-mean.json`.

### 2026-08-25 15:34 PDT — restore fixed-saturation token-scaled harm baseline

The production model was restored to the fixed 0.109375-epoch exponential
usefulness transform, fixed 1.05 cooldown multiplier, fixed diagonal-matched
usefulness variance, and learned harmful-exposure mean scaled by physical
training tokens per total parameter.

The production-priority H100 replay
`/held/mixprior-fixed-saturation-harm-replay-20260825` reproduced the historical
ranking result:

| Transition | Historical | Restored |
| --- | ---: | ---: |
| 56-run | 0.651914 | 0.652834 |
| 115-run | 0.738084 | 0.738121 |

The small 56-run difference comes from retaining the later simplification that
fixes usefulness variance after its diagonal match. The restored result is the
requested `0.652 / 0.738` model family.

The 0.109375-epoch time constant implies `1 - exp(-1 / 0.109375) = 0.999893`
of modeled usefulness has been captured by one epoch. Treat the restored form
as a ranking baseline, not a credible scientific saturation prior.

Artifact:
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/fixed-saturation-harm-restored.json`.

### 2026-08-25 15:45 PDT — learn saturation from a one-epoch prior

The fixed 0.109375-epoch saturation timescale was replaced with a positive GP
parameter. Its log-normal prior has median one epoch and log standard deviation
one. The usefulness variance remains fixed after diagonal matching, isolating
the exposure-shape change.

The production-priority H100 replay
`/held/mixprior-learned-one-epoch-saturation-replay-20260825` completed with:

| Transition | Fixed 0.109375 epochs | Learned from one-epoch prior |
| --- | ---: | ---: |
| 56-run | 0.652834 | 0.547504 |
| 115-run | 0.738121 | 0.764860 |

Across chronological prefixes, the learned timescale was 2.507--2.840 epochs.
This corresponds to capturing roughly 30%--33% of asymptotic usefulness after
one epoch, rather than assuming essentially complete saturation. The treatment
improves the 115-run replay by 0.026739 and reduces the 56-run replay by
0.105330.

Artifact:
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/learned-one-epoch-saturation.json`.

### 2026-08-25 17:08 PDT — clean-room minimal semantic GP synthesis

Effort: low. Three subagents received only the data shape and scientific
intuitions. They were instructed not to inspect the existing math, code, or
reports. All three independently proposed representing a run as a linear
functional of latent semantic usefulness, with repetition harm kept separate
from the sign-uncertain semantic GP.

The shallowest synthesis uses one latent semantic function shared exactly
between phases. A single positive scalar changes cooldown's contribution. The
expected outcome contains a non-positive quadratic cumulative-exposure term.
Integrating out semantic usefulness gives a standard run-level GP with a linear
semantic kernel. A swarm random intercept handles level shifts.

The adversarial pass found two important constraints. First, a saturating
transform multiplied by a sign-unconstrained GP does not guarantee diminishing
returns for content whose latent utility is negative. Linear semantic benefit
plus a negative quadratic exposure term gives decreasing marginal returns for
every GP draw. Second, cooldown scale is not identifiable when pretraining and
cooldown semantic allocations are collinear. Its prior should remain strong
until the run design varies the phases independently.

The minimal replay compares this model against `harm amplitude = 0`, `cooldown
scale = 1`, target-only fitting, and the current transfer GP. Use both store
transitions and report chronological Spearman, prequential likelihood, winner
rank, and regret. Promotion requires transfer to help both transitions or a
predeclared target transition without degrading calibration enough to indicate
negative transfer.

### 2026-08-25 17:26 PDT — reject the minimal semantic quadratic GP

The clean-room synthesis was implemented as a research-only run-level GP. Its
covariance is a linear kernel over one shared semantic exposure vector,
`pretraining + alpha * cooldown`, plus an independent swarm intercept. Its mean
subtracts a learned positive coefficient times token-share-weighted squared
cumulative epochs. Semantic and repetition features are centered on each
swarm's proportional mixture. The feature map is invariant to splitting an
identical bucket.

The production-priority H100 job
`/held/mixprior-minimal-semantic-gp-replay-20260825` ran seed 0 for the proposed
model and three controls:

| Form | 56-run Spearman | 115-run Spearman | 56 winner rank | 56 regret |
| --- | ---: | ---: | ---: | ---: |
| Restored transfer GP | 0.652834 | 0.738121 | 6.67 | 6.83 |
| Minimal semantic quadratic | 0.426164 | 0.372876 | 32.67 | 130.13 |
| No quadratic harm | 0.416655 | -0.044081 | 40.00 | 179.22 |
| Cooldown scale fixed to one | 0.349094 | 0.535528 | 10.33 | 9.87 |
| Target-only minimal model | -0.077666 | 0.546393 | 29.83 | 42.56 |

The full treatment learned cooldown scale 2.813--2.829 on the 56-run prefixes.
The quadratic harm term contains useful signal: removing it has little effect
on 56-run whole-pool ranking but collapses the 115-run ranking. Learning the
cooldown scale improves 56-run Spearman relative to fixing it at one, while
worsening winner rank and regret substantially.

Transfer helps the minimal model on the revised-store transition
(`-0.078` target-only to `0.426` transfer) and hurts on the first-store
transition (`0.546` target-only to `0.373` transfer). This reproduces the
negative-transfer asymmetry instead of resolving it. The model loses 0.227
Spearman on the primary transition and 0.365 on calibration relative to the
restored GP, so no additional optimizer seeds are warranted. Keep the quadratic
form as an interpretable negative result; do not promote it.

Artifact:
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/minimal-semantic-gp.json`.

### 2026-08-25 18:50 PDT — sequential BO component ablation

Eight production-priority H100 jobs replayed greedy sequential BO over the
observed target-mixture pools. Every arm used seed 0, the same initial target
observation, and posterior-mean selection; the harm-only arm ranked candidates
using the learned prior mean instead. Exact duplicate curricula were removed
from the candidate pool. The 56-run revised-store swarm starts from
`harrier-bo-20260816-trial000-d768-seed0`. The 115-row first-store swarm has 106
unique curricula and starts from the proportional observation.

Simple regret on the 56-run revised-store swarm:

| Model | R@1 | R@5 | R@10 | R@20 | R@40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 176.626 | 4.773 | 0.611 | 0.000 | 0.000 |
| No harm mean | 176.626 | 3.702 | 0.611 | 0.000 | 0.000 |
| Harm mean only | 176.626 | 3.702 | 3.702 | 0.000 | 0.000 |
| No usefulness kernel | 176.626 | 3.702 | 0.611 | 0.000 | 0.000 |
| Usefulness kernel only | 176.626 | 16.217 | 3.702 | 3.702 | 0.000 |
| Full target-only | 176.626 | 18.936 | 18.936 | 5.213 | 0.611 |
| Simple transferable | 176.626 | 3.702 | 0.611 | 0.000 | 0.000 |
| Simple target-only | 176.626 | 20.593 | 20.593 | 5.213 | 0.000 |

The simplified transferable Hellinger GP matches the no-usefulness ablation
at every reported budget. Both outperform the full model at five evaluations
and tie it from ten evaluations onward. Removing the harm mean also improves
five-evaluation regret and otherwise ties the full model. The usefulness-only
model is the weakest transfer model. Historical transfer is the clearest
positive result: both target-only models remain above 18 regret at ten
evaluations, while every transferable model except usefulness-only is at or
below 3.702.

The 115-row replay is uninformative for simple regret: its initial proportional
observation has objective 2.494859, already the maximum of the 106 unique
candidate representatives. Every arm therefore has zero regret at every
reported budget. A future replay should use a non-optimal fixed anchor or a
metric that scores the sequence rather than the incumbent alone.

Submission template:

```text
uv run iris --cluster marin job run --no-wait --job-name mixprior-bo-ablation-<variant>-20260825 --priority production --enable-extra-resources --gpu H100x1 --cpu 8 --memory 64GB --disk 64GB --extra gpu --extra mixprior -- uv run python scratch/bo_ablation_replay.py --variant <variant> --output s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/bo-ablation-replay/<variant>.json
```

All eight jobs succeeded. Their artifacts are the eight JSON files under
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/bo-ablation-replay/`.

### 2026-08-25 20:22 PDT — content-exposure kernel and quality-cell correction

A research-only transferable GP represented each recipe by content and phase
exposure jointly. For item `i`, the exposure coordinate is
`[log1p(e0_i + e1_i), e1_i / (1 + e0_i + e1_i)]`. An RBF kernel over frozen
Luxical content features was approximated with its top 32 eigenfeatures, and a
Matérn-5/2 exposure kernel was approximated with 16 deterministic random
Fourier features. The run-level feature is the natural-token-weighted sum of
content/exposure tensor products. Its covariance has a shared term plus a
same-swarm residual term, so historical swarms remain transfer observations.

The first implementation combined Harrier's five quality cells within each of
40 domains. This discarded quality allocation: equal total tokens in a domain
produced the same exposure, and the five Luxical vectors were replaced by a
fixed availability-weighted centroid. The corrected implementation treats all
200 domain-quality cells separately. A local check moved mixture mass from
`c00q0` to `c00q1` while holding domain mass fixed and observed a nonzero
recipe-feature distance.

The primary 56-run sequential result is mixed:

| Model | R@1 | R@5 | R@10 | R@20 | R@40 | Best-run position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple transferable | 176.626 | 3.702 | 0.611 | 0.000 | 0.000 | — |
| Content exposure, 40 domains | 176.626 | 4.773 | 0.611 | 0.000 | 0.000 | 12 |
| Content exposure, 200 cells | 176.626 | 5.876 | 0.000 | 0.000 | 0.000 | 9 |

The cell-level model is worse at five evaluations but finds the best observed
mixture three evaluations earlier than the domain-aggregated treatment. This is
one deterministic posterior-mean replay over a fixed pool. It is insufficient
to distinguish a useful top-of-list bias from a lucky path.

Chronological held-out ranking is worse for both content-exposure treatments:

| Model | 56-run mean Spearman | 115-run mean Spearman |
| --- | ---: | ---: |
| Simple transferable | 0.412535 | 0.730019 |
| Content exposure, 40 domains | 0.182873 | 0.701998 |
| Content exposure, 200 cells | -0.014756 | 0.621545 |

Because experiment selection values the best discovered incumbent rather than
the ordering of poor mixtures, subsequent comparisons should use simple regret,
time to the best or top-k set, and the intended acquisition function as primary
metrics. Spearman remains a brittleness diagnostic. Repeat the replay across
kernel-feature seeds and target swarms whose initial anchor is not already
optimal before promoting the cell-level kernel.

Actual submissions used federated Iris without a target-cluster constraint:

```text
uv run iris --cluster marin job run --no-wait --job-name mixprior-content-exposure-cells-bo-replay-20260825 --priority production --enable-extra-resources --gpu H100x1 --cpu 8 --memory 64GB --disk 64GB --extra gpu --extra mixprior -- uv run python scratch/bo_ablation_replay.py --variant content_exposure --output s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/content-exposure-cells-bo-replay.json
uv run iris --cluster marin job run --no-wait --job-name mixprior-content-exposure-cells-rank-replay-20260825 --priority production --enable-extra-resources --gpu H100x1 --cpu 8 --memory 64GB --disk 64GB --extra gpu --extra mixprior -- uv run python scratch/content_exposure_rank_replay.py --output s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/content-exposure-cells-rank-replay.json
```

Future GPU submissions should remain federated through `--cluster marin` and
add `--target-cluster cw-us-east-02a`.

Artifacts:

- `s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/content-exposure-cells-bo-replay.json`
- `s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/content-exposure-cells-rank-replay.json`

### 2026-08-25 20:23 PDT — twelve-hour search and ablation window

- Search starts: `2026-08-25T20:23:35-07:00`.
- Stop model-form search and begin complexity ablation: `2026-08-26T06:23:35-07:00`.
- Stop ablation and synthesize the result: `2026-08-26T08:23:35-07:00`.
- Compute cap: eight concurrent H100s.
- Placement: federated Iris via `--cluster marin --target-cluster cw-us-east-02a`.
- Primary metric: simple regret versus target-swarm evaluations. Global
  Spearman is diagnostic only.
- Search constraint: reviewers must prefer a simpler form when regret evidence
  does not distinguish it from a more complex form.

Adversarial review changed the search protocol before the main matrix:

- Call it a closed-pool multi-start replay. Historical target mixtures are
  available from the first step, so it is not a chronological simulation and
  says nothing about candidates absent from the recorded pool.
- Aggregate repeated curricula by inverse objective variance before defining
  regret. Do not let the first row determine the oracle outcome.
- Calibrate kernel distance from observations available to the fit rather than
  from held-out target locations.
- Separate GP-fit, random-feature, anchor, and acquisition seeds.
- Use five outcome-blind anchors. The validation block includes the actual
  proportional anchor; the 115-run stress test excludes proportional by group
  identity because that anchor is already empirically optimal.
- Compare only `simple_transfer` and corrected 200-cell `content_exposure`,
  crossed with posterior-mean exploitation and fixed-seed qLogNEI.
- Run to 20 evaluations first. Primary readouts are mean simple regret through
  evaluation 20 and evaluations to the first top-5 mixture. Report R@5, R@10,
  and exact time-to-best as supporting metrics.

The preliminary eight-seed wave intentionally precedes these fixes and varies
the GP optimizer and 16-feature exposure approximation together. Treat it only
as a brittleness screen. Its jobs are
`/held/mixprior-content-exposure-cells-seed{0..7}-bo-replay-20260825`.

The brittleness screen completed on eight H100s:

| Seed | R@5 | R@10 | Evaluations to best | Mean regret, 1--20 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.876 | 0.000 | 9 | 11.964 |
| 1 | 5.876 | 0.000 | 8 | 15.149 |
| 2 | 36.130 | 20.593 | 38 | 28.437 |
| 3 | 4.773 | 4.773 | 12 | 12.751 |
| 4 | 0.000 | 0.000 | 4 | 11.480 |
| 5 | 5.876 | 5.876 | 26 | 14.414 |
| 6 | 16.217 | 7.633 | 33 | 20.155 |
| 7 | 36.130 | 26.503 | 28 | 26.134 |

Only three of eight seeds reach the best mixture by evaluation ten, and only
four reach it by evaluation twenty. Median R@10 is 5.325. The evaluation-nine
seed-zero result is not robust. Because each seed changes both random features
and optimizer initialization, use this result to establish brittleness rather
than attribute its source.

Artifacts are under
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260825/content-exposure-cells-seeds/`.

### 2026-08-25 21:08 PDT — first paired blocks and isolated kernel treatments

Three isolated changes to the 200-cell content-exposure kernel completed:

| Treatment | R@5 | R@10 | Mean regret, 1--20 | Evaluations to best |
| --- | ---: | ---: | ---: | ---: |
| Log cooldown increment | 18.936 | 18.936 | 24.850 | 38 |
| Shared content plus swarm intercept | 30.954 | 15.065 | 28.738 | 34 |
| 64-rank content, 64 exposure features | 20.593 | 11.468 | 19.085 | 15 |

All three are worse than seed zero of the default 32-rank, 16-feature form.
The high-fidelity treatment also roughly doubles each fit from ten to twenty
seconds. Do not promote any of them.

Blocks zero, one, and three of the corrected paired replay are complete. On the
56-run validation swarm, content exposure beats simple transfer on mean regret
in blocks zero and one, but loses in block three:

| Block | Acquisition | Simple transfer | Content exposure |
| ---: | --- | ---: | ---: |
| 0 | Posterior mean | 14.467 | 10.196 |
| 0 | qLogNEI | 17.536 | 10.742 |
| 1 | Posterior mean | 8.108 | 4.500 |
| 1 | qLogNEI | 13.440 | 3.581 |
| 3 | Posterior mean | 2.241 | 4.557 |
| 3 | qLogNEI | 4.488 | 4.939 |

On the 115-run calibration stress test, content exposure wins all six completed
comparisons, often by a large margin. Blocks two and four remain in flight, so
this is not yet a promotion decision.

Every new job is submitted through federated Iris with production priority and
an explicit `cw-us-east-02a` target. The active search also includes:

- proportional-reference centering for content exposure;
- a learned cooldown phase scale on the simple Hellinger kernel;
- an overall-content plus cooldown-content Hellinger representation;
- paired phase-scale replays on blocks zero, one, and three.

The remaining paired blocks reverse the preliminary impression. On the 56-run
validation target, content exposure wins only two of five anchors under either
acquisition. Posterior-mean results are:

| Model | Median mean regret, 1--20 | Mean mean regret, 1--20 | Median evaluations to top 5 |
| --- | ---: | ---: | ---: |
| Simple transfer | 8.108 | 7.969 | 4 |
| Content exposure | 10.196 | 12.029 | 11 |

qLogNEI is also worse than posterior-mean exploitation for simple transfer in
all five validation blocks. Content exposure wins four of five blocks on the
115-run calibration stress test, but its remaining block is catastrophic and
raises its mean regret above simple transfer. This does not satisfy the
predeclared promotion rule. Retain simple transfer and posterior mean as the
reference while searching simpler phase-sharing changes.

Proportional-reference centering reaches the validation optimum by evaluation
eight in its isolated replay, with R@5 1.014 and R@10 zero. Its mean regret
through evaluation 20 is 18.667 because its second selection is extremely poor.
That is worse than the uncentered seed-zero trajectory and does not warrant a
paired promotion wave.

Learning a cooldown multiplier on the original two-phase Hellinger distance
does not improve robust regret. Its five posterior-mean validation blocks have
median mean regret 9.145 and mean 7.796, versus 8.108 and 7.969 for the fixed
reference. It wins two blocks, loses two, and ties one. Calibration performance
is also effectively unchanged. Prefer the reference because the learned scalar
does not earn its complexity.

Learning the reference kernel lengthscale also fails its isolated screen: R@5
is 3.702, R@10 is 0.611, mean regret through evaluation 20 is 13.173, and the
exact best appears at evaluation 26. The fixed median-distance calibration is
at least as effective in this replay.

Two simpler phase representations passed isolated screens:

| Representation | R@5 | R@10 | Mean regret, 1--20 | Evaluations to best |
| --- | ---: | ---: | ---: | ---: |
| Overall content only | 3.702 | 0.000 | 10.733 | 8 |
| Overall content plus learned cooldown component | 3.702 | 3.702 | 11.882 | 19 |

The overall-only representation is both simpler and better in this screen. A
five-anchor posterior-mean replay is in flight. A second treatment learns the
cooldown weight inside the single overall-content distribution rather than
adding a separate covariance component.

The overall-content representation wins all five matched validation anchors:

| Model | Median mean regret, 1--20 | Mean mean regret, 1--20 | Wins |
| --- | ---: | ---: | ---: |
| Two phase-content distributions | 8.108 | 7.969 | 0/5 |
| One token-weighted overall-content distribution | 3.720 | 4.945 | 5/5 |

The overall model reaches the exact best mixture by evaluation eight in four
blocks and by evaluation five in the fifth. Its improvement comes from a
simpler representation: phase content is aggregated using the actual phase
token fractions before applying the Hellinger RBF.

This representation performs worse on the older 115-run calibration stress
test, where many rows are single-domain cooldown interventions. That divergence
is useful evidence that the older swarm rewards phase detail that does not help
the 56-run target. A second five-anchor replay is running with the three
observed Rav-ladder rows included as historical source data, matching the data
available to the current posterior.

Including the three observed Rav-ladder rows strengthens the result. The
overall-content representation again wins all five matched validation anchors:

| Model | Median mean regret, 1--20 | Mean mean regret, 1--20 | Median evaluations to top 5 |
| --- | ---: | ---: | ---: |
| Two phase-content distributions | 10.018 | 8.579 | 4 |
| One token-weighted overall-content distribution | 2.929 | 4.361 | 2 |

The overall model reaches the exact best mixture by evaluation seven or eight
in every block. The two-phase model does not reach it within twenty evaluations
in four blocks and reaches it at evaluation nineteen in the fifth. Keep the
overall-content RBF as the current search incumbent. Robust paired tests of a
Matérn-5/2 covariance are running before deciding whether the RBF's smoothness
assumption should change.

The Matérn-5/2 covariance loses to the fixed-lengthscale RBF on mean regret in
all five matched starts:

| Model | Median mean regret, 1--20 | Mean mean regret, 1--20 | Wins |
| --- | ---: | ---: | ---: |
| Overall-content fixed RBF | 2.929 | 4.361 | 5/5 |
| Overall-content Matérn-5/2 | 4.377 | 6.587 | 0/5 |

The Matérn model reaches the exact best earlier in some blocks, but only after
larger early errors. The predeclared primary metric rejects it.

Learning the cooldown's contribution to the overall content mixture is worse
than using the actual phase token fractions: isolated mean regret is 14.447,
R@5 is 5.213, R@10 is 0.611, and the exact best is not reached within twenty
evaluations. Learning only the overall RBF lengthscale is a much narrower
treatment: isolated mean regret is 10.456 versus 10.733 for the fixed RBF, and
it reaches the exact best at evaluation five rather than eight. Its improvement
is small enough that it requires the five-start paired test now in flight.

The paired test rejects the learned RBF lengthscale. It wins one of five
matched starts and loses four:

| Model | Median mean regret, 1--20 | Mean mean regret, 1--20 | Wins |
| --- | ---: | ---: | ---: |
| Overall-content fixed RBF | 2.929 | 4.361 | 4/5 |
| Overall-content learned-lengthscale RBF | 3.813 | 4.742 | 1/5 |

The parameter-free Bhattacharyya linear kernel also fails its isolated screen:
R@5 is 20.593, R@10 is 8.390, mean regret through evaluation 20 is 21.097,
and it does not reach the best mixture within twenty evaluations. The
nonlinear RBF relation over rooted content is necessary in this replay.

Multiplying the overall-content RBF by an RBF over the scalar magnitude of the
pretraining-to-cooldown content shift also fails. Its isolated R@5 is 25.299,
R@10 is 1.014, mean regret through evaluation 20 is 16.881, and the exact best
appears only at evaluation 24. Similarity in schedule-shift magnitude is not a
useful extra condition on overall-content transfer here.

Removing the legacy 840-run swarm makes the overall-content model slightly
worse in every matched start:

| Historical sources | Median mean regret, 1--20 | Mean mean regret, 1--20 | Wins |
| --- | ---: | ---: | ---: |
| Legacy + first Harrier store + Rav ladder | 2.929 | 4.361 | 5/5 |
| First Harrier store + Rav ladder | 3.350 | 4.724 | 0/5 |

The old swarm therefore contributes useful transfer after projection into the
shared content space, despite its different original bucket definitions. Keep
it in the incumbent training set.

A global constant mean is modestly worse than the per-swarm constant mean on
the first current-source block: mean regret 10.548 versus 9.757, with both
reaching the exact best at evaluation seven. This is not a search promotion,
but the small gap leaves it as a legitimate simplicity ablation for the final
two-hour window.

The fixed median-distance RBF scale is locally well calibrated. On the first
current-source block, halving it raises mean regret from 9.757 to 10.456; doubling
it raises mean regret to 17.992. The shorter kernel reaches the best two
evaluations earlier, but the primary trajectory metric does not improve. Do not
add a manual scale multiplier.

The incumbent's chronological-prefix rank diagnostic is weak despite its
five-start regret result. Mean Spearman across prefixes 1, 3, 5, 10, 20, and 40
is 0.401 on the 56-run target with Rav sources and 0.200 on the 115-run
calibration target. At prefixes 10 and 20 on the 56-run target, its one-step
posterior-mean winner ranks 44th and 34th among the held-out rows. This does not
reverse the primary closed-pool regret decision, but it is a material warning:
the overall-only geometry succeeds from diverse outcome-blind starts while
failing to rank the full chronological remainder reliably. Any promoted hybrid
must be checked on both diagnostics.

Using phase-sensitive geometry only for the same-swarm residual does not repair
that weakness. It loses all five current-target starts to the overall-only
residual, with median mean regret 5.607 and mean 8.831 versus 2.929 and 4.361.
It is also unstable on the 115-run calibration target: four blocks have mean
regret between 36.128 and 309.769, while one reaches 1.176. The residual phase
detail introduces severe start sensitivity rather than a useful local
correction. Reject it.

The two narrower same-swarm residuals also fail their first matched screen. A
cooldown-content residual has current-target mean regret 17.984 and does not
reach the best mixture within twenty evaluations. A pretraining-to-cooldown
contrast residual has mean regret 18.966 and reaches the best at evaluation
14. Their calibration-block mean regrets are 3.112 and 4.189, but a gain on one
older block does not offset the large current-target regression. Stop both.

Analytic log probability of improvement is worse than posterior-mean
exploitation in four of five matched starts:

| Acquisition | Median mean regret, 1--20 | Mean mean regret, 1--20 | Wins |
| --- | ---: | ---: | ---: |
| Posterior mean | 2.929 | 4.361 | 4/5 |
| Log probability of improvement | 3.114 | 4.835 | 1/5 |

The only log-PI win is 0.007 mean-regret units. Keep posterior mean as the
search acquisition; neither qLogNEI nor log-PI improves this closed-pool
regret objective.

The production functional form is also weaker than the overall-content
incumbent on the first current-target anchor. Removing only its epoch-harm mean
gives mean regret 16.077; the full usefulness-plus-harm form gives 14.760; the
overall-content RBF gives 9.757. The harm mean modestly helps within the
production form on this start, but the phase-sensitive geometry and usefulness
term remain substantially worse as a package. Do not expand either production
variant to the other anchors during the search window.

Adding the saturating usefulness covariance to the overall-content RBF fails
the isolated screen. It reaches the top five at evaluation four and the exact
best at evaluation twelve, but mean regret is 15.536, versus 9.757 for the
overall-content RBF alone. The component therefore does not repair the weak
chronological rank diagnostic without sacrificing the primary search
trajectory. Stop the treatment after this block.

Three independent adversarial reviews selected the same next deletion:
remove the exact-same-swarm RBF residual while retaining the shared
overall-content RBF and per-swarm constant means. The residual duplicates the
shared geometry and is weakly identified; it can discount historical transfer
when only one to five target observations exist. A five-anchor paired replay is
running. Promote the deletion only if it lowers both aggregate mean and median
regret and wins at least four starts; otherwise retain the residual.

The five-anchor replay rejects shared-only covariance. Mean regret rises from
4.361 to 4.828 and median regret rises from 2.929 to 4.268; it wins only the
fourth anchor, by 0.007, and loses the other four. The same-swarm response
surface is therefore useful even though phase-sensitive variants of that
surface are harmful. A standard random-intercept alternative also fails its
first screen: block-zero mean regret is 10.127, versus 10.152 for shared-only
and 9.757 for the incumbent, and it reaches the best at evaluation nine rather
than seven. Stop both deletions.

Tying the shared and same-swarm RBF amplitudes is exactly trajectory-equivalent
to the incumbent on all five anchors: every selected-run sequence and every
mean-regret value matches exactly. The chronological mean-Spearman changes are
negligible, from 0.40066 to 0.40006 on the current target and from 0.19955 to
0.19985 on calibration. Promote the tied form as a strict complexity
reduction: it removes one learned variance scale without changing the primary
search behavior.

A shared linear rooted-content covariance does not improve the incumbent's
first-anchor trajectory. Mean regret rises from 9.757 to 11.188, the exact best
moves from evaluation seven to ten, the first top-five mixture moves from
evaluation two to four, and R@5 rises from 3.702 to 4.773. The broad linear
direction is redundant or harmful beside the overall-content RBF; stop it after
this screen.

The active primary target has 56 distinct curricula and no duplicate mixtures,
so inverse-variance replicate aggregation cannot leak extra feedback into its
replay. The 115-row calibration target has one ten-seed proportional group and
106 unique curricula; replicate aggregation can make that secondary anchor
artificially precise, so calibration results should not drive promotion.

The overall-versus-two-phase result is now being repeated over twenty
outcome-blind starting mixtures rather than five. Blocks zero through four are
unchanged; blocks five through nineteen extend the same SHA-256 ordering. This
robustness wave keeps target, sources, fit seeds, posterior-mean acquisition,
candidate pool, and twenty-evaluation horizon paired within each anchor.

The overall-content GP wins all twenty anchors. Mean 1--20 regret falls from
6.183 to 2.945 and median regret falls from 5.445 to 2.427. Median evaluations
to a top-five mixture fall from four to two; median evaluations to the observed
best fall from beyond the twenty-run horizon to eight. The forced operational
anchor favors overall content by 4.711 regret units. Across the nineteen
hash-selected anchors, the paired mean difference is -3.161 in favor of
overall content; a finite-population-corrected 95% interval over the 55
eligible non-forced anchors is [-3.803, -2.518]. This interval measures anchor
sensitivity within the fixed outcome pool, not generalization to a new swarm
or new training seeds.

A top-five shortlist diagnostic confirms that weak global Spearman mostly
reflects ordering outside the valuable basin. Across chronological prefixes 1,
3, 5, 10, 20, and 40, mean regret of the five highest-posterior candidates is
3.835 for overall content versus 7.431 for the two-phase GP. Overall content
contains the held-out best in its shortlist at prefixes three and five. Its
prefix-ten and prefix-twenty shortlists remain weak, so the diagnostic is not a
blanket calibration claim; it is nevertheless better aligned with the robust
sequential-regret advantage than whole-pool Spearman.

The tied-amplitude form remains exactly trajectory-equivalent across all eight
tested anchors: selected-run sequences and mean regret match bit for bit. End
the check here and keep the tied scale as the search incumbent. It also reduces
end-to-end replay time by 9--15% on every anchor, from 177--189 seconds to
161--171 seconds, because the fit has one fewer covariance parameter.

### 2026-08-25 23:28 PDT — rolling-prefix intervention-family diagnostic

Each model was fit on target rows preceding one operational batch and then had
to select the best member of that batch. This is a post-hoc batch-ranking
diagnostic, not an independent chronological or sequential-regret evaluation:
all three Rav-ladder source observations were available at every cutoff, even
though two were generated from 56- and 57-row target models, and each later fit
observes every member of the preceding batch. The model forms were also chosen
after inspecting this target pool.

| Held-out batch | Two-phase winner regret | Overall-content winner regret |
| --- | ---: | ---: |
| Regret designs, rows 35--39 | 0.000 | 81.728 |
| Monotonic regret, rows 40--43 | 0.000 | 0.000 |
| Final PI, rows 44--47 | 11.307 | 11.307 |
| Final PI follow-up, rows 48--55 | 8.418 | 0.402 |
| Equal-weight mean | 4.931 | 23.359 |

The top-two mean regret is 1.945 for the two-phase model and 3.298 for the
overall model. The overall model makes a catastrophic batch-local choice in
rows 35--39. Only rows 40--43 contain a candidate better than the pre-batch
incumbent, however, and both models select that batch's optimum. The diagnostic
therefore identifies a representation stress case but cannot establish a
simple-regret advantage. The 20-anchor pooled replay remains evidence only for
within-pool search efficiency.

Five paired replays are running to test whether the first-store source causes
negative transfer under the overall-content model. Two additional jobs hold
the operational anchor and all other randomness fixed while varying the GP fit
seed across eight values.

Removing the 115-run first-store swarm sharply worsens pooled sequential
regret. Mean 1--20 regret rises from 4.361 to 12.279, median regret rises from
2.929 to 5.335, and the deletion loses all five matched anchors. The largest
regressions are 17.923 units at the operational anchor and the fifth anchor.
The first-store source is not a removable source of the overall model's
batch-ranking failure.

A reverse hybrid directly addresses the chronological failure with one proper
GP covariance and no learned gate:

$$
K_{ij}=\sigma^2\left[k_{\mathrm{phase}}(x_i,x_j)
+\mathbf{1}[s_i=s_j]k_{\mathrm{overall}}(\bar x_i,\bar x_j)\right].
$$

Historical transfer therefore remains phase-sensitive, while target-swarm
observations can add an overall-content response. Both RBF lengthscales use the
existing median-distance calibration and the two amplitudes are tied. On the
same adaptively reused four batches its top-one regrets are 0.000, 0.000,
11.307, and 0.000, for a mean of 2.827. Its top-two shortlist contains the batch
optimum in all four batches. This is hypothesis-generation evidence only. A
five-anchor pooled replay tests whether the treatment preserves the primary
within-pool trajectory; an independent new swarm remains necessary for
promotion.

The target-only interaction diagnostic localizes the first-batch miss to the
combination of overall geometry and historical transfer. With rows 0--34 as
the only training data, both GPs select the optimal member of rows 35--39.
Across all four batches, target-only top-one mean regret is 2.814 for two-phase
and 8.176 for overall content; their top-two means are 0.164 and 0.101. This
supports the reverse-hybrid separation as a mechanism, but remains post-hoc
evidence on the same target outcomes.

The five-anchor pooled replay decisively rejects the reverse hybrid. Its mean
1--20 regret is 13.481 and median is 15.190, versus 4.361 and 2.929 for the
overall-content incumbent. It loses all five anchors, reaches a top-five
mixture after a median six evaluations rather than two, and never reaches the
observed best within twenty evaluations. The phase-sensitive shared term
preserves the post-hoc batch-ranking pattern at the cost of the primary search
trajectory. Do not promote it.

Its chronological whole-suffix rank diagnostic is also weak: mean Spearman is
-0.045 on the current target and 0.666 on calibration. This again shows that
the older calibration swarm favors phase-sensitive geometry that does not
generalize to the current pooled search trajectory.

Removing the three Rav-ladder source observations leaves every rolling-prefix
batch selection unchanged for all three representations. Their availability
at the earlier cutoffs is historically impossible, but it does not explain the
observed batch-ranking results. The remaining post-hoc reuse and batch-local
metric limitations still apply.

The fixed-anchor fit-seed audit is fully deterministic across seeds zero
through seven. Both the two-phase and tied overall-content models select the
same sequence and produce the same regret for every seed. Overall content has
mean regret 9.757, reaches a top-five mixture at evaluation two, and reaches
the observed best at seven; two-phase has 14.467, four, and beyond twenty.
Optimizer-seed sensitivity does not explain the incumbent's pooled advantage.

### 2026-08-25 23:55 PDT — deployment-pool extrapolation failure

The overall-content model is not yet safe for candidate generation. Four
independent 65,536-row production proposal pools produce four very different
maximizers. Pairwise total variation between selected phase mixtures ranges
from 0.384 to 0.745. Their posterior means are 477.9, 489.8, 541.4, and 614.0,
while the best observed objective is 5.99; posterior standard deviations are
194--336 and reported probabilities of improvement are 0.958--0.994. The
nearest-observation squared Hellinger distances are only 0.033--0.083, so the
failure occurs inside the operational proposal distribution rather than only
at an obviously remote simplex corner.

This invalidates the closed-pool result as an operational promotion criterion:
ranking the 56 observed mixtures does not test the surrogate's extrapolation
between them. A diagnostic rerun is inspecting the learned per-swarm mean and
kernel scale. The likely correction must make the GP revert to a credible
swarm baseline away from observations; clipping acquisition values or adding a
distance threshold would hide the model failure and is not an acceptable fix.

An equally weighted additive phase-plus-overall kernel does not resolve the
representation tradeoff. On the first pooled anchor its mean regret is 10.772
versus 9.757 for overall content, and it reaches the observed best at evaluation
twenty rather than seven. Its post-hoc batch-local mean is 4.771, close to the
two-phase model. Stop the treatment after this screen.

### 2026-08-26 00:09 PDT — Fixed empirical swarm means rejected

Freezing each swarm mean at its empirical value did not repair the deployment-pool extrapolation failure. On the 65,536-row operational pool, the selected posterior mean remained 475.0 with posterior SD 194.8, compared with the observed target maximum of 5.99. The selected mixture and its nearest observed neighbor were also unchanged in substance. The pooled replay anchor was slightly worse than the tied-overall control: mean regret 10.279 versus 9.757, with the observed best reached at evaluation 12 versus 7. The rolling-prefix diagnostic was unchanged from the learned-mean model.

This falsifies the hypothesis that the learned swarm intercept caused the hundreds-scale pool predictions. The correction comes from the covariance and residual observations. The fixed-mean treatment is rejected.

The first phase and Matérn deployment-pool jobs completed model fitting and scoring but failed while serializing a diagnostic that assumed a single `ScaleKernel`. The diagnostic now records all scale kernels, and replacement jobs are running. This failure did not affect model fitting or candidate scores.

Incident record: https://echo.oa.dev/wiki/248

### 2026-08-26 00:24 PDT — Per-swarm affine outcome scaling

The source swarms have radically different objective scales, so the shared GP can use the high-variance first-store residuals to produce hundreds-scale target predictions. Standardizing each swarm's available outcomes and variances before fitting reduced the same operational-pool maximum substantially:

- tied overall RBF on the raw shared scale: posterior mean 475.0, SD 194.8;
- phase RBF on the raw shared scale: 212.0, SD 212.7;
- overall Matérn on the raw shared scale: 354.4, SD 500.5;
- tied overall RBF after per-swarm standardization: 76.4, SD 17.4;
- phase RBF after per-swarm standardization: 46.5, SD 19.7;
- overall Matérn after per-swarm standardization: 35.9, SD 18.7.

The fixed-mean treatment established that centering alone is insufficient; scaling the historical residuals is the operative change. On the first Rav-inclusive pooled replay anchor, the raw tied-overall control still had the best mean regret: 9.757 versus 11.112 for standardized overall RBF, 12.193 for standardized phase RBF, and 10.373 for standardized overall Matérn. Standardized Matérn is the closest safe treatment, so blocks 1–4 are running before deciding whether the deployment calibration benefit justifies the replay cost.

An outcome-blind rolling-origin predictive check changes that conclusion. At target cutoffs 35, 40, 44, and 48, each model was fit without Rav rows and scored the next historical batch in raw objective units. Aggregate marginal negative log predictive density was 124.8 for the raw tied-overall control, 90.4 for standardized overall RBF, 82.1 for standardized phase RBF, and 82.9 for standardized overall Matérn. Mean batch-local winner regret was respectively 23.36, 7.24, 4.93, and 1.10. The raw control's closed-pool trajectory advantage therefore coexists with much worse chronological calibration and next-batch selection. This supports treating the operational-pool failure as a model-selection failure hidden by rank-only closed-pool replay, rather than tolerable GP uncertainty.

The raw reverse hybrid also failed the deployment check, selecting a pool point at posterior mean 377.1 and SD 284.6. Restricting historical transfer to phase-sensitive geometry is insufficient without outcome-scale correction.

The full 20-anchor Rav-inclusive pooled replay confirms that standardized overall Matérn pays no closed-pool regret cost. Mean 1–20 regret is 2.9449 for standardized Matérn versus 2.9453 for raw tied-overall RBF. Median regret is approximately 2.477 versus 2.427; median evaluations to the observed best improve to 6–7 from 8, while mean evaluations are 8.15 versus 7.8. Every anchor reaches the observed best by evaluation 20. Combined with the rolling-origin NLPD and operational-pool checks, standardized overall Matérn becomes the research incumbent.

Two simplifications/local variations did not improve the priority metric:

- Removing the same-swarm residual from standardized Matérn lowered the operational-pool maximum to 24.7 and slightly improved rolling-origin NLPD to 81.5 from 82.9, with identical rolling-origin winner regret. Across five pooled anchors, however, mean regret worsened to 4.615 from 4.363 and the first top-five point moved from evaluation 2 to 3–4. The residual is retained.
- Replacing Matérn-5/2 with rougher Matérn-3/2 lowered the pool maximum to 28.3 but slightly worsened rolling-origin NLPD to 83.8, and its first pooled anchor had regret 10.972 with top-five at evaluation 6 versus 10.373 and evaluation 2 for Matérn-5/2. Matérn-3/2 is rejected.

Historical transfer remains useful after normalization. Fitting standardized Matérn on the target prefix alone gives rolling-origin NLPD 91.1, compared with 82.9 when legacy and first-store source observations are included. Both choose the same batch winners on these four batches, so the observed transfer benefit is predictive calibration rather than batch-local regret.

The predictive-score gain is not variance inflation. Across the four chronological
target batches, standardized Matérn reduced raw-scale RMSE from 279.1, 186.6,
53.6, and 30.7 to 24.6, 14.6, 8.0, and 4.2. Its mean predictive SD also fell
from 145.7, 141.9, 57.5, and 46.0 to 29.1, 27.2, 13.4, and 11.4. The model is
simultaneously sharper and more accurate at every cutoff.

Removing the three Rav-ladder source runs modestly weakens the first four
Rav-inclusive replay anchors. Without Rav, standardized Matérn has mean regret
4.831, median regret 3.461, and reaches a top-five observation after 3--5
evaluations. With Rav, the same anchors have mean regret 4.007, median regret
2.358, and reach a top-five observation at evaluation two. The source set is
small, but it provides useful local transfer after normalization.

Eight replacement jobs initially omitted the `mixprior` dependency extra, then
their next replacements requested only the one-GB Iris default and were killed
while loading the shared feature basis. The corrected jobs explicitly request
the `gpu` and `mixprior` extras and 32 GB of host RAM. This is a launch-spec
error rather than a model or data failure.

The completed 20-anchor cross-swarm calibration replay strongly favors the
standardized Matérn model. When the 115-run first-store swarm is held out as the
target and only the legacy swarm transfers, standardized Matérn lowers mean
regret from 150.63 to 87.40 and median regret from 109.84 to 80.17. It reaches
the observed best within twenty evaluations in 9/20 anchors and a top-five run
in 11/20; the raw tied-RBF control reaches the best in 0/20 and top five in
1/20. The absolute regret remains large because this target contains extreme
bad runs, but the paired improvement is broad rather than driven by one
anchor.

The raw Matérn control confirms that outcome scaling, rather than Matérn
smoothness alone, produces the calibration gain. Raw Matérn has rolling-origin
NLPD 130.31 and mean next-batch winner regret 20.60, versus 82.92 and 1.10 after
per-swarm standardization. Its first two batch RMSEs are 295.3 and 136.6,
compared with 24.6 and 14.6 after scaling.

Freezing the Matérn range at the outcome-blind median-distance calibration is
not a useful simplification. It slightly worsens rolling-origin NLPD from 82.92
to 83.26, raises next-batch winner regret from 1.10 to 7.24, and raises the
first replay anchor's mean regret from 10.37 to 10.93 while delaying the exact
best from evaluation 12 to 20. It also increases the operational-pool winner's
posterior mean from 35.9 to 56.9 rather than making extrapolation safer. Retain
the learned Matérn range.

The standardized Matérn operational-pool scale is stable across three seeds:
posterior-max means are 35.9, 41.0, and 40.0, with SDs 18.7, 23.8, and 22.0.
This is a systematic calibration gap rather than one exceptional proposal
pool. The selected points remain close to observed curricula in overall-content
distance, so simply rejecting distant candidates would not address it.

The same operational-pool failure remains when all historical swarms are
removed. A target-only fit on the 56 current-store observations selects a point
with posterior mean 37.20 and SD 33.39 against an observed maximum of 5.99;
the nearest observation is only 0.0096 away in phase-weighted squared
Hellinger distance. Historical transfer is therefore not the source of the
extreme pool prediction. The target outcome distribution is highly skewed
(minimum -293.98, median -20.74, maximum 5.99), and per-swarm standardization
maps the observed maximum to only 0.83 standard deviations above the mean.
This makes the next treatment a likelihood/noise or outcome-shape question,
not a transfer-strength question.

Adding one learned homoskedastic noise term to the fixed evaluation variances
is not the correction. It lowers the target-only operational-pool maximum from
37.20 to 11.88, but worsens chronological NLPD from 82.94 to 124.83 and mean
winner regret from 1.10 to 1.51. On the first closed-pool anchor it raises mean
regret from 10.37 to 11.27 and no longer reaches the observed best within 20
evaluations. The fitted additional variance is 0.143 on the standardized
scale. The shrinkage is real, but it discards useful target structure and is
rejected.

Replacing raw standardized outcomes with within-swarm Gaussian rank scores is
also worse for the metric that motivated it. Across five Rav-inclusive replay
anchors, mean regret rises from 4.36 to 5.13 and median regret from 2.40 to
3.11. It reaches the observed best later on four of five anchors, while both
models reach a top-five observation at evaluation two. A rank-only latent GP
throws away useful utility spacing even though the downstream decision cares
most about the top of the ordering. It is rejected.

A monotone asinh warp centered and scaled by each swarm's median absolute
deviation retains more utility spacing, but it also loses. Across the same five
anchors, mean regret rises from 4.36 to 4.83 and median regret from 2.40 to
5.74. It reaches a top-five observation at evaluations 3, 8, 3, 3, and 3,
versus evaluation two on every anchor for affine standardization. The warp wins
the proportional anchor and reaches its best much earlier, but loses enough on
the other starts that the aggregate result is clearly worse. Together with the
rank and noise results, this closes the outcome-shape branch: retain affine
per-swarm standardization and interpret raw-unit pool maxima cautiously.

Restoring a rank-one learned cross-swarm covariance inside the Matérn residual
is decisively worse than the exact same-swarm residual. Across the first three
anchors, mean regret rises from 5.03 to 14.79 and median regret from 2.40 to
16.44. On two anchors it reaches neither a top-five run nor the observed best
within twenty evaluations. The richer task covariance helped an earlier,
phase-sensitive functional form, but is unstable after the simpler
overall-content geometry and outcome scaling are in place. Reject it.

A direct target-only trajectory comparison gives mixed but useful evidence for
historical transfer. Because a one-point target-only fit cannot calibrate a
kernel range without an input prior, both arms start from the same two
outcome-blind anchors. Across five starts, transfer lowers mean regret from
6.73 to 5.35, while target-only lowers median regret from 2.01 to 1.65. Transfer
wins one difficult start by 11.04 mean-regret units, but target-only is better
on three of five starts. Combined with the earlier chronological NLPD gain
(82.9 versus 91.1), this supports retaining transfer, while documenting that
its benefit is mostly tail-risk reduction rather than a uniform per-start win.

Across five Rav-inclusive replay anchors, removing Rav-ladder observations
raises mean regret from 4.36 to 5.03 and median regret from 2.40 to 4.10. The
gain is modest but consistent enough to keep Rav as a historical source.

A standardized Matérn variant with overall-content historical transfer and a
phase-sensitive same-swarm residual improves the reused chronological batches:
NLPD is 82.87 versus 82.92 and mean winner regret is 0.16 versus 1.10. It fails
the primary replay screen decisively, however. On the operational anchor its
mean regret is 30.26 versus 10.37, it does not reach the observed best within
twenty evaluations, and its first top-five point moves from evaluation two to
17. The phase residual repeats the earlier raw-scale failure and is rejected.

### 2026-08-26 02:42 PDT — Rational-quadratic kernel screen

Replacing the Matérn-5/2 content kernel with a rational-quadratic kernel does
not materially improve low-budget search. Across the first three paired
anchors, mean 1--20 regret changes from 5.030 to 4.927 and median regret is
unchanged at 2.404. Both kernels reach a top-five run at evaluation two and
reach the observed best at evaluations 12, 6, and 9. The rational-quadratic
fit takes roughly twice as long because it learns an additional scale-mixture
parameter. The negligible trajectory change does not justify that complexity;
retain Matérn-5/2 and stop this treatment after the predeclared screen.

The shared latent phase-response kernel is also rejected. It uses one semantic
Matérn process for both phases and sums all early/late cross-covariances, so it
adds no learned parameters while retaining more phase information than the
incumbent's aggregate content. Across the first three paired anchors, mean
1--20 regret worsens from 5.030 to 5.422 and median regret from 2.404 to 3.704.
One anchor improves, but the other two lose and the first top-five evaluation
is unchanged. Linking the phases through a single latent response is not more
useful than aggregating their token-weighted content for this search.

Adding the overall q0--q4 mass histogram only to the same-swarm residual does
not improve search either. Final exposures reconstruct each recipe's
token-weighted quality histogram; swarms without quality labels receive a
constant unknown-quality coordinate, and the residual distance is calibrated
only from within-swarm pairs. Across the first three anchors, mean regret
worsens from 5.030 to 5.318 and median regret from 2.404 to 2.836. The treatment
improves one anchor slightly, loses the other two, and leaves the first top-five
evaluation unchanged at two. Quality-specific Luxical rows already carry
enough of this distinction for the observed search, so reject the explicit
quality residual.

### 2026-08-26 03:11 PDT — Batch-of-five transfer replay

A realistic batch-of-five replay changes the interpretation of historical
transfer without changing the model. Each of twenty starts begins with five
current-swarm observations, then selects three batches of five by posterior
mean. Transfer and target-only are essentially tied on mean regret across the
four batch boundaries: 3.134 versus 3.137. Transfer wins 6 starts, target-only
wins 7, and 7 tie.

The timing differs. In this closed-pool replay, after the first proposed batch
(10 total evaluations), transfer has mean regret 1.347 versus 1.959 and has
found a top-five run in all 20 starts versus 16/20 for target-only. By
evaluation 20, target-only has lower
mean regret, 0.246 versus 0.667, and has found the exact best in 17/20 starts
versus 14/20. This shows that the transfer arm selected a better first batch;
it does not yet identify whether semantic source signal or generic
regularization caused the difference. The next treatment tests a joint batch
simple-regret acquisition before considering any staged transfer policy.

Greedy discrete qSimpleRegret is worse than selecting the five highest
posterior means. Across five paired batch replays, mean regret rises from 2.248
to 3.315 and median regret from 1.652 to 2.668. Posterior-mean batches win four
starts and tie one; qSimpleRegret reaches the exact observed best in none of the
five starts by evaluation 20, versus three for posterior mean. Its covariance-
aware diversity spends too much of this small batch budget away from the
high-mean region. Keep the simple posterior-mean batch rule.

The five-start staged policy uses transfer for the first proposed batch and
target-only fits afterward. Its mean regret across evaluations 5, 10, 15, and
20 is 2.157, between transfer's 2.248 and target-only's 2.034. It recovers the
target-only result by evaluation 20 but inherits transfer's weaker first batch
on these five starts. This is a post-hoc operational hypothesis, not an
independent validation: the switch point was selected from the same replay,
and later target-only fits are conditioned on candidates chosen by transfer.
The next control permutes source outcome-variance pairs within each historical
swarm while preserving source mixtures and the complete fitting procedure.
This distinguishes semantic transfer from generic regularization or optimizer
effects at the first proposed batch.

The source-permutation control rejects the generic-regularization explanation.
Outcomes and their variances were permuted as unique-mixture blocks within
each historical swarm, preserving replicated seeds. Across eight starts, true
transfer has mean regret 0.768 after the first proposed batch, versus 5.006 for
permuted transfer. It finds a top-five run by evaluation 10 in 8/8 starts,
versus 6/8 for the permutation control. Target-only is slightly better than
true transfer on these starts at evaluation 10 (0.616), consistent with the
larger 20-start near-tie, but the large permutation gap shows that the
historical outcomes contain mixture-linked signal rather than merely adding
generic fitting pressure. The sliding starts overlap and are not independent
trials, so this supports the transfer mechanism without claiming eight
independent replications.

Giving the same-swarm residual rooted raw-mixture geometry does not clear the
predeclared screen. Across five anchors it slightly lowers mean and median
regret through evaluation 10, but wins only 3/5 early-regret comparisons,
raises full-horizon mean regret from 4.363 to 4.621, and fails to reach the
observed best on one anchor where the incumbent reaches it at evaluation 12.
The semantic content projection is lossy, but a raw-mixture residual does not
turn that extra detail into a reliably better search trajectory. Reject it
without running the chronological calibration extension.

A learned historical-source discrepancy nugget improves the first two-anchor
screen on block zero, lowering mean regret from 19.51 to 17.31; the
capacity-matched target-nugget control is identical to the incumbent. All
three still miss a top-five run by evaluation 20 on this difficult start, so
four additional anchors are required before deciding whether the source-only
discount is useful.

The staged policy remains an unnecessary operational complication after eight
starts. It matches target-only by evaluation 20, but its transfer-selected
first batch is worse on these starts: mean regret at evaluation 10 is 0.768
for the staged and transfer arms versus 0.616 for target-only. It provides no
gain over either stationary policy and is rejected.

The first four source-removal starts suggest that the oldest legacy swarm may
hurt the first batch. Removing legacy lowers mean regret at evaluation 10 from
1.231 to 0.925, while removing the first Harrier store worsens it to 1.384.
Including or excluding Rav-ladder data is trajectory-equivalent on these four
starts. Four more starts are running before treating the legacy result as more
than a small-screen hypothesis.

The eight-start source-removal result does not justify changing the source
set. Removing legacy lowers evaluation-10 mean regret by 9.9%, from 0.768 to
0.692, but wins only 2/8 starts and ties five; it misses the predeclared 20%
and 5/8 gates. Removing the first Harrier source worsens evaluation-10 regret
to 0.972. Rav-ladder inclusion is trajectory-equivalent on these eight batch
starts, while earlier sequential replay showed a modest benefit. Retain all
sources rather than introduce an acquisition-specific source policy.

The learned historical-source discrepancy nugget is rejected. Across five
two-anchor starts it raises early mean regret through evaluation five from
12.19 to 13.71 and full-horizon mean regret from 5.35 to 7.11. It wins only
3/5 early comparisons and only one full-horizon comparison. The target-nugget
control has a small inconsistent gain, but it is not broad enough to justify
an additional noise parameter. Together with the earlier generic-noise test,
this closes the likelihood branch.

The first four transfer-window starts show a clear observation-count effect.
With three target observations, transfer lowers mean regret across batch
boundaries from 6.68 to 3.26 and prevents one difficult start from remaining
at 20.52 regret. With ten target observations, target-only is slightly better:
mean boundary regret is 1.74 versus 1.88, and all four target-only runs find
the observed best by evaluation 20 versus three transfer runs. A second set of
four starts is running to confirm that historical transfer is primarily a
cold-start prior rather than a permanently superior surrogate.

Across all eight starts, the effect is confined to the smallest target prefix.
With three observed target mixtures, transfer lowers mean regret at evaluation
8 from 3.365 to 1.591 and at evaluation 13 from 2.692 to 0.616. With four
observed mixtures, target-only is already slightly better: mean regret at
evaluation 9 is 0.742 versus 1.205, and mean regret across all five batch
boundaries is 1.743 versus 1.852. With ten observations, target-only has mean
regret 0.459 at evaluation 15 and zero at evaluation 20, versus 0.432 and
0.153 for transfer. The shared-plus-swarm-residual GP therefore acts as a
useful cold-start prior when the current swarm has only three observations;
ordinary conditioning makes direct historical transfer unnecessary soon
afterward. A final cut-posterior control will test whether historical outcomes
can retain their cold-start value by learning covariance hyperparameters while
being excluded from the target posterior solve.

The cut-posterior control fails at the point where transfer matters most. With
three observed target mixtures, its evaluation-8 mean regret is 8.971, worse
than target-only at 3.365 and full transfer at 1.591. It finds the exact best
by evaluation 20 on only 2/8 starts, versus 7/8 for both ordinary models. The
source-permuted hyperparameter control is also better than the true-source
version, so source outcome geometry is not the useful mechanism. Historical
outcomes help through direct semantic conditioning of the posterior. This
closes the final transfer-mechanism branch; the remaining research time will
increase the replay from eight to all twenty anchor starts without adding
model components.

The exhaustive twenty-start transfer-window replay confirms a sharp crossover.
With three observed target mixtures, transfer lowers mean regret after the
first proposed batch from 3.543 to 1.458, places a top-five run by evaluation
8 on 20/20 starts versus 15/20, and lowers mean regret across all recorded
boundaries from 4.346 to 3.360. It wins 10 starts, loses 5, and ties 5; the
large mean gain is partly tail-risk reduction from one difficult start.

With four observed mixtures, the arms are effectively tied: mean regret after
the first proposed batch is 1.677 for transfer versus 1.934 for target-only,
but mean regret across the full replay is 2.803 versus 2.784. Each arm wins
seven starts and six tie. Transfer puts a top-five run in every first batch,
versus 17/20 for target-only, but target-only finds the exact best by
evaluation 20 on 17/20 starts versus 16/20. The operational result is narrow:
historical transfer is valuable for the first batch with only three current-
swarm observations, becomes a first-batch shortlist aid at four, and should
not be treated as permanently superior once more target data arrive.

The twenty-start two-observation replay strengthens the cold-start result.
Transfer lowers mean regret after the first proposed batch from 5.684 to 2.386
and mean regret across all recorded boundaries from 5.576 to 4.630. It wins
12 starts, loses 3, and ties 5. Both arms find the exact observed best within
twenty evaluations on 19/20 starts. A one-observation comparison is undefined
for this implementation: target-only cannot calibrate either its Hellinger
range or its within-swarm affine outcome scale from one unique recipe. Those
jobs were stopped rather than introduce a special cold-start scale rule.

### 2026-08-26 05:02 PDT — Transfer-window completion

The six- and eight-observation replays fill the gap between the cold-start
result and the earlier five-observation production-shaped replay. With six
observations, transfer lowers first-batch regret at evaluation 11 from 1.849
to 1.406 and places a top-five run in the first batch on 20/20 starts versus
15/20. Target-only then catches up: mean regret across all four boundaries is
2.619 versus 2.734, it finds the exact best by evaluation 20 on 19/20 starts
versus 14/20, and it wins eight paired trajectories while transfer wins four
and eight tie.

With eight observations, transfer again improves the first proposed batch:
regret at evaluation 12 is 1.711 versus 1.918, and first-batch top-five
coverage is 18/20 versus 15/20. Target-only is better over the full trajectory,
with mean boundary regret 2.205 versus 2.468 and the exact best found by
evaluation 20 on 20/20 starts versus 13/20. Target-only wins eleven paired
trajectories, transfer wins two, and seven tie.

The evidence now supports a narrow use of historical transfer. It improves the
first shortlist most strongly with two or three current-swarm observations,
still improves first-batch top-five coverage through eight observations, and
does not improve continued closed-pool search after target observations
accumulate. The replay starts overlap, so the counts describe stability across
initial conditions rather than independent confidence intervals. No further
functional-form hypotheses are queued before the scheduled simplicity
ablation phase.

The ten-observation completion confirms the crossover rather than moving it.
Transfer puts a top-five run in the first batch on 20/20 starts versus 18/20,
but first-batch mean regret is slightly worse, 1.375 versus 1.187. Across the
three recorded boundaries, target-only has mean regret 2.463 versus 2.686 and
finds the exact best by evaluation 20 on 14/20 starts versus 11/20. At ten
target observations, historical transfer no longer improves either regret or
continued search; its only remaining effect is marginal top-five coverage.

### Scheduled simplicity ablations

Three independent adversarial reviews agree on two deletion candidates and
one important distinction. Removing BoTorch's global `Standardize` after the
manual per-swarm affine transform should be almost an affine no-op: the manual
transform already gives the pooled data zero mean and unit population
variance, while the second transform differs mainly by the sample-variance
convention. This deletion must preserve selected sequences, not merely average
regret. It also requires updating every raw-unit evaluation path so target
inverse scaling is applied exactly once; several research diagnostics assume
an outcome transform exists.

Tying the shared and same-swarm Matérn outputscales is a substantive one-
parameter deletion. The current covariance implies transfer correlation
`shared / (shared + residual)`; tying forces it to one half. The isolated arm
must reuse the same Matérn object, inputs, anchors, and fit seeds. It will first
be screened on five paired three-observation starts, where historical transfer
has its strongest demonstrated benefit. The global-transform and scale-tie
deletions will be tested independently and combined only if each passes. Fixing
the remaining amplitude is reserved for a final short screen because it is a
strong prior restriction rather than algebraic cleanup.

The fitted incumbent already learns that one-half correlation to high
precision. Across twenty starts at each target prefix, mean learned correlation
is 0.50011, 0.50014, 0.50023, 0.50032, and 0.50041 for 3, 4, 6, 8, and 10
observations. The full observed range is 0.49977--0.50080. Median shared and
residual outputscales differ by less than 0.001 at every prefix. This makes the
tie a strongly motivated deletion rather than a value chosen from replay
regret, while the planned paired replay will still test whether removing the
flat direction changes optimization numerics or selection.

The twelve- and fifteen-observation completion shows transfer becoming actively
harmful for continued search. At twelve observations, target-only has mean
boundary regret 2.294 versus 2.819, wins 15/20 paired trajectories, and finds
the observed best by evaluation 20 on 13/20 starts versus 4/20. At fifteen
observations, target-only has mean boundary regret 2.533 versus 3.231, wins
17/20 with three ties, and finds the best on 16/20 starts versus 2/20. Both
models already have top-five coverage on every start at fifteen observations.
The operational handoff is therefore explicit: direct historical conditioning
helps construct a cold-start shortlist, but target-only fitting should replace
it once the target swarm has roughly five to ten observed mixtures. One n=12
worker hit a transient Hugging Face CDN disconnect before fitting; its exact
replacement succeeded without a code change.

The complete transfer-window summary is:

| Initial target observations | First-batch regret, transfer | First-batch regret, target-only | Full boundary mean, transfer | Full boundary mean, target-only |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2.386 | 5.684 | 4.630 | 5.576 |
| 3 | 1.458 | 3.543 | 3.360 | 4.346 |
| 4 | 1.677 | 1.934 | 2.803 | 2.784 |
| 5 | 1.347 | 1.959 | 3.134 | 3.137 |
| 6 | 1.406 | 1.849 | 2.734 | 2.619 |
| 8 | 1.711 | 1.918 | 2.468 | 2.205 |
| 10 | 1.375 | 1.187 | 2.686 | 2.463 |
| 12 | 1.940 | 1.321 | 2.819 | 2.294 |
| 15 | 1.518 | 0.122 | 3.231 | 2.533 |

These are closed-pool simple-regret summaries over the same twenty overlapping
starts. They characterize the policy transition and are not independent
replications.

### 2026-08-26 06:50 PDT — Simplicity ablation

Tying the shared and same-swarm Matérn outputscales passes the strict deletion
gate. Across all twenty three-observation starts and all twenty four-observation
starts, the tied model chooses exactly the same sequence as the two-scale
incumbent and therefore has identical regret, top-five coverage, and exact-best
counts. On the four chronological batches it selects the same winners, has the
same 1.1034 mean winner regret and 0.1005 mean top-two regret, and slightly
improves total NLPD from 82.9414 to 82.9350. Mean replay fit time falls from
24.7s to 19.8s at three observations and from 26.3s to 19.7s at four.

Removing BoTorch's second global outcome transform is not promoted under the
predeclared identity rule. It is exact on all twenty three-observation starts,
but changes one selected sequence at four observations. The aggregate regret
curve happens to remain identical and mean fit time falls to 20.2s, but keeping
the transform avoids making a prior-scale change that is only approximately
affine. The combined treatment inherits the same one-sequence difference. The
accepted simplification is therefore the isolated covariance-amplitude tie.

Fixing the tied output scale at 0.5 is rejected. It changes all three screened
selection sequences, is slower than learning the tied scale (32.8s versus
18.5s mean fit time), and only happens to lower mean first-screen regret from
3.839 to 3.633. Unlike tying the two empirically indistinguishable amplitudes,
fixing the remaining amplitude is a strong prior and removes a parameter that
the data can identify.

### 2026-08-26 07:05 PDT — Production promotion

The production surrogate now matches the validated tied model: rooted
token-weighted overall Luxical content, a learned Matérn-5/2 range, zero mean,
manual population standardization within each swarm, the retained BoTorch
`Standardize`, and one learned output scale multiplying a shared kernel plus an
equal same-swarm residual. Posterior means and latent variances are explicitly
returned in raw target-swarm objective units. The exposure-saturation kernel,
harmful-epoch mean, rank-one swarm covariance, phase feature layout, and their
metadata were removed rather than retained as alternate code paths.

The first Iris test submission encoded the job name as part of the command and
failed before pytest. The corrected production-priority H100 submission uses
federated Iris targeting `cw-us-east-02a`.

The complete public Hugging Face campaign then succeeded on that route. It
downloaded the pinned registry anonymously, fit 978 observations across four
swarms on one H100 in 6.3s and 936 optimizer steps, learned lengthscale 0.385155
and outputscale 0.923126, sampled 65,536 feasible curricula, scored the pool,
and wrote candidate ID `44a838c2286fa040`. End-to-end duration was 5m57s; pool
construction dominated runtime. The final exact staged test run passed all 23
mixprior tests on an H100.

The final adversarial review removed two more stale seams. Lengthscale
initialization now uses the same all-training-row overall-content geometry as
the validated replay, so the obsolete kernel-reference role was deleted from
the campaign model and loader. The feature-map factory only returned
`swarm_features` after repeating campaign validation, so fitting now passes
that function directly. A raw-unit prediction test uses a non-identity pooled
BoTorch transform and verifies both the mean and latent-variance inverse maps.

### 2026-08-26 07:34 PDT — Final artifact review

The blocking-only review found no model or geometry defect. It did find that
the recorded marginal likelihood was evaluated after switching the GP to
evaluation mode, which measured a conditioned posterior rather than the
training marginal likelihood. The fit now evaluates that diagnostic in train
mode and switches to evaluation mode only afterward. Candidate diagnostics
also record `uncertainty_kind=latent_function`, making explicit that objective
standard deviation and probability of improvement exclude observation noise.
The incompatible candidate and bundle-manifest wire-format changes now advance
their schema versions to 4 and 3.

The exact updated package passed 23/23 tests on a production-priority H100 via
federated Iris in `cw-us-east-02a`. A second full public-campaign run on the
same route succeeded. It again fit 978 observations across four swarms, learned
lengthscale 0.385155 and outputscale 0.923126, and selected candidate
`44a838c2286fa040`; the corrected training marginal log likelihood is
-1.164323. No `GPInputWarning` was emitted. The candidate is unchanged because
these corrections affect metadata and schema, not posterior fitting or
selection.

### 2026-08-26 08:24 PDT — Scheduled close

The twelve-hour research window closed after its recorded 08:23:35 PDT
boundary. The final H100 test job passed all 23 package tests, the full public
campaign job succeeded, `git diff --check`, the changed-file pre-commit suite,
and targeted Pyrefly all passed, and three independent blocking-only reviews
reported no remaining P0 or P1 issue. No commit or push was performed.

### 2026-08-26 12:08 PDT — Select by batch regret, not global rank

The deployment protocol was clarified after the scheduled close: target
outcomes are observed only for selected mixtures, and five mixtures are chosen
at once without feedback inside the batch. Whole-pool Spearman and
chronological rank summaries are therefore secondary diagnostics. The primary
model-selection replay starts from three observed target mixtures, selects
five from the closed pool by posterior mean, reveals those five outcomes,
refits, and repeats to twenty total evaluations.

The preceding fixed-saturation model initially looked preferable. Its
chronological top-five shortlist regret averaged 2.024 across six prefixes on
the 56-row transition and 4.557 on the 115-row transition, versus 7.038 and
28.443 for the production model. That comparison ranks every unobserved row at
each prefix and does not simulate the actual batch decision loop.

The matched twenty-start batch replay reverses the decision:

| Held-out target | Surrogate | Evaluation 8 | Evaluation 13 | Evaluation 20 |
| --- | --- | ---: | ---: | ---: |
| 56-row Harrier swarm | Tied production GP | 2.039 | 0.462 | 0.101 |
| 56-row Harrier swarm | Fixed-saturation GP | 5.333 | 1.759 | 0.061 |
| 115-row Harrier swarm | Tied production GP | 35.577 | 19.042 | 8.868 |
| 115-row Harrier swarm | Fixed-saturation GP | 68.248 | 30.696 | 10.329 |

At the first selected batch, the production GP wins 19/20 starts with one tie
on the 56-row target and 12/20 with four ties on the 115-row target. At the
second boundary it wins 11/20 with seven ties and 13/20 with two ties. The
fixed-saturation model catches up after enough target outcomes are revealed and
is slightly better at the final 56-row boundary. Early batch regret is the
selection criterion, so retain the simple transferable GP and reject the
restoration.

The exact tied production comparison ran on five production-priority Iris H100
jobs, `/held/mixprior-five-batch-exact-current-g0-r2-20260826` through `g4`.
The fixed-saturation comparison ran in
`/held/mixprior-five-batch-gate-g0-r2-20260826` through `g4`. The earlier
chronological restoration artifact is
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260826/restored-five-candidate-replay.json`.

The exact staged package passed 23/23 tests in
`/held/mixprior-current-tests-r2-20260826`. The public Hugging Face campaign
completed in `/held/mixprior-current-public-campaign-r2-20260826`, fit 978 rows
on an H100, and reproduced candidate `44a838c2286fa040`.

The twelve-hour window was not a clean twelve hours of model progress. The
public campaign, reproducible registry, replay infrastructure, transfer-window
result, and simpler production GP remain useful. The time spent optimizing
whole-pool rank and adding exposure-shape machinery did not improve the actual
batch-selection objective; those experiments are negative evidence rather
than a promoted modeling result.

### 2026-08-26 13:12 PDT — A shrinkable harm mean improves rank and replay regret

I evaluated model changes against two co-primary diagnostics: chronological
held-out Spearman over prefixes 2, 3, 5, 10, 20, and 40, and closed-pool simple
regret when selecting five mixtures at a time from three initial observations.
The first screen used five paired replay starts on both the 56-run validation
swarm and the 115-run calibration swarm.

The old structured model's ranking advantage comes from its epoch-harm mean.
`full` and `harm_only` have identical chronological rankings: mean Spearman is
0.466 on validation and 0.734 on calibration. Removing the mean lowers those
values to 0.256 and 0.679. Removing the usefulness kernel retains 0.209 and
0.762, so the old covariance is not the source of the validation ranking gain.
This ablation ran in `/held/mixprior-old-rank-ablation-20260826`.

Adding the learned `EpochHarmMean` to the tied production covariance while
retaining per-swarm outcome standardization is the first Pareto improvement in
this search. Mean Spearman rises from 0.077 to 0.089 on validation and from
0.196 to 0.250 on calibration. Mean simple regret at evaluations 8, 13, 18,
and 20 changes from 1.065, 0.863, 0.740, and 0.203 to 1.065, 0.740, 0.740, and
0.122 on validation. On calibration it changes from 48.803, 26.761, 12.428,
and 9.646 to 36.124, 12.729, 7.147, and 7.007. The rank job is
`/held/mixprior-shrunk-harm-rank-20260826`; replay blocks 0--4 are
`/held/mixprior-shrunk-harm-regret-b0-20260826` through `b4`.

Other variants did not improve both diagnostics. Robust asinh transforms raise
Spearman but worsen batch regret. A learned empirical swarm-scale exponent
raises Spearman to 0.273 and 0.318 and slightly improves validation regret, but
worsens calibration regret to 99.342, 45.998, 28.635, and 25.589. Raw-output
task covariance variants have the same failure. Fixed usefulness amplitudes
and a multiscale dose kernel also remain behind the tied production GP.

The full twenty-start replay confirms the shrinkable harm mean. Validation
regret at evaluations 8, 13, 18, and 20 changes from 2.039, 0.462, 0.370, and
0.101 to 2.039, 0.277, 0.216, and 0.061. Calibration regret changes from
35.577, 19.042, 13.290, and 8.868 to 26.867, 13.817, 9.071, and 5.846. The
additional paired jobs are `/held/mixprior-shrunk-harm-regret-b5-20260826`
through `b19`. The improvement is therefore not an artifact of the five-start
screen: the first validation batch ties, and every subsequent aggregate
boundary on both swarms improves.

### 2026-08-26 14:02 PDT — Search constraints are mechanical only

The search no longer rejects candidates by epoch count. Repetition remains an
input to the learned GP mean, so evidence can move the harm knee and amplitude
without making the belief a hard rule. The admissible set now has only the
mechanical training constraints: nonnegative phase weights, each phase sums to
one, and every weight is an integer multiple of 1/49,152. Candidate generation
uses largest-remainder apportionment to project each sampled phase onto that
lattice before deduplication and observed-mixture exclusion.

### 2026-08-26 14:52 PDT — Full structured soft-prior model

The production implementation now represents pretraining and cooldown
separately. It computes per-cell exposure, per-phase rooted Luxical content,
per-phase content-weighted log dose, a cross-phase dose interaction, and fixed
log context for active parameters, total parameters, physical tokens, and
simulated tokens. A learned positive-semidefinite stage covariance links the
two phase responses. A same-swarm residual retains architecture, recipe,
tokenizer, and store-specific deviations.

The GP prior mean has learned stage-specific benefit amplitudes and saturation
rates plus learned repetition-harm amplitude, knee, exponent, and
tokens-per-total-parameter scaling. These terms are only a prior mean; the GP
residual can contradict them. No candidate is rejected for epoching or any
other empirical belief. The search domain is two nonnegative simplexes on the
1/49,152 lattice.

Candidate selection now uses qLogNoisyExpectedImprovement with 256 Sobol
samples. Its baseline contains only actual target-swarm observations. Source
observations train the transfer GP but never enter the target incumbent as
counterfactual features.

The first H100 campaign attempts exposed a CUDA-device mismatch inside
LogNormalPrior's cached base distribution. Constructing all structured priors
directly on the training tensor's device fixed it. The corrected public-campaign
smoke `/held/mixprior-structured-full-campaign-smoke-r4-20260826` fit 978 rows
across four swarms in 23.6 seconds, scored 512 candidates with qLogNEI, and
wrote candidate `f93ef1c0d79b2cfc`. It learned saturation rates 3.704 and
3.293 epochs, a 14.978-epoch harm knee, 0.0550 harm amplitude, 2.008 harm
exponent, and stage covariance `[[0.310, 0.556], [0.556, 1.627]]`. The H100
package suite passed in `/held/mixprior-structured-full-tests-r3-20260826`.

### 2026-08-26 15:28 PDT — Structured GP rank and closed-pool replay

The structured model's chronological mean Spearman is 0.165 on the 56-row
Harrier holdout and 0.837 on the 115-row Harrier holdout, compared with 0.089
and 0.250 for the preceding learned-harm GP. The rank replay ran in
`/held/mixprior-structured-full-rank-r2-20260826`; its machine-readable result
is
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260826/structured-full-rank.json`.

The twenty-start replay does not show a uniform search improvement. With
PosteriorMean, structured-model regret at evaluations 8, 13, 18, and 20 is
3.679, 1.180, 0.397, and 0.275 on the 56-row holdout, versus 2.039, 0.277,
0.216, and 0.061 for the simpler learned-harm GP. On the 115-row holdout it is
33.484, 6.988, 0.000, and 0.000, versus 26.867, 13.817, 9.071, and 5.846.
The structured model starts more slowly, but its stronger ranking identifies
the calibration optimum by 18 observations.

The first qLogNEI replay ranked independent one-point scores. I replaced that
approximation with greedy pending-point conditioning inside every batch of
five. The corrected qLogNEI regrets are 9.279, 3.968, 1.554, and 0.549 on the
56-row holdout and 48.794, 8.923, 3.452, and 1.234 on the 115-row holdout.
This acquisition is more exploratory than PosteriorMean and is worse in this
closed candidate pool. The corrected replay jobs are
`/held/mixprior-structured-pending-regret-b0-20260826` through
`/held/mixprior-structured-pending-regret-b19-20260826`; results are under
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260826/structured-pending-regret/`.

### 2026-08-26 15:54 PDT — Steelman audit and exact-current validation

An independent audit found that the structured kernel had an exact scale ridge:
an outer output scale multiplied both the learned stage covariance and the
dose-kernel scale. The outer scale is removed. Stage and dose amplitudes are now
separate, and artifacts record every learned kernel range and amplitude.

The four fixed model/token features now share one strongly regularized context
lengthscale rather than four ARD lengthscales. The registry contains four swarms
but only three distinct recorded contexts, so these inputs support standardized
response-shape transfer rather than separately identified model-size and token-
horizon effects. The token-per-total-parameter harm exponent has a tighter
zero-centered prior. Hyperparameters remain a MAP point fit; acquisition does
not integrate their uncertainty.

Quality ordering is a soft prior rather than a feasibility rule. At equal
exposure, the prior mean gives higher tiers a positive learned phase-specific
bonus. The GP residual can reverse this belief. Candidate generation continues
to enforce only nonnegative phase simplexes on the 1/49,152 lattice.

Candidate pools now mix global Dirichlet draws with local perturbations, shuffle
before truncating small pools, and validate phase/component shape at selection
and persistence boundaries. The focused package suite passes 32 tests, including
kernel PSD and finite-gradient checks, component-order invariance, quality-prior
behavior, and candidate lattice/shape regressions.

Exact-current H100 validation was submitted at production priority through
federated Iris to `cw-us-east-02a`:

- `/held/mixprior-steelman-campaign-smoke-20260826`
- `/held/mixprior-steelman-rank-20260826`

The rank output is
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260826/structured-steelman-rank.json`.
The older structured replay table is historical evidence until the exact-current
replay completes.

### 2026-08-26 17:19 PDT — MAP restarts and acquisition stability

The exact-current prior-mode fit converged naturally after 4,871 Adam steps,
with normalized MLL -0.481274. Two seeded prior-drawn starts found distinct
converged modes. Seed 11 reached -0.476560 and seed 22 reached -0.479769. With
978 observations, seed 11 is 4.61 total log-posterior units above the prior-mode
start and 3.14 above seed 22. The production fit now runs all three starts and
selects the highest-posterior converged result. Artifacts record every start,
its convergence status, and its total log-posterior difference.

The single-start exact replay completed before this change. Its mean Spearman
is 0.232 on the 56-row holdout and 0.810 on the 115-row holdout. PosteriorMean
regret at evaluations 8, 13, 18, and 20 is 1.080, 0.529, 0.336, and 0.092 on
the 56-row holdout and 9.947, 2.759, 0.070, and 0.035 on the 115-row holdout.
These results are retained as a prior-mode optimizer ablation rather than a
production validation.

Three 65,536-row candidate pools showed stable qLogNEI rankings with 256 Sobol
samples, but one pool changed its winner across Sobol seeds. Raising the
production acquisition sampler to 1,024 samples made all three tested Sobol
seeds select the same winner in that pool and increased pairwise acquisition
Spearman from 0.9978--0.9987 to 0.9997--0.9998. The best-of-three replay and
three-pool acquisition audit are being rerun with the 1,024-sample policy.

### 2026-08-26 17:30 PDT — Exact ranking and finite-pool hardening

The best-of-three exact ranking replay completed. Mean chronological Spearman
is 0.180 on the 56-row Harrier transition and 0.810 on the 115-row transition.
The 56-row result is weak at prefixes 2--10 and improves after 20 observations;
the 115-row ranking remains consistently strong. This is a development replay,
because the model family was selected while inspecting both pools.

With one fixed selected MAP fit and 1,024 Sobol samples, the qLogNEI winner is
stable across three Sobol seeds within each 65,536-row pool. The acquisition
maxima still differ across independent pools: the best values are -1.139,
-1.191, and -1.239. The latter two pools therefore miss candidates with 5.3%
and 10.5% higher estimated improvement relative to the best pool. Production
candidate generation now searches the deduplicated union of the three audited
pools instead of choosing one pool. Candidate artifacts separately record all
pool seeds and the acquisition sampler seed. A full union search and the exact
best-of-three regret replay are running on H100s.

### 2026-08-26 18:00 PDT — Final exact replay and production search

All twenty exact best-of-three, 1,024-sample replay blocks succeeded. On the
56-row transition, PosteriorMean mean regret at evaluations 8, 13, 18, and 20
is 3.682, 1.351, 0.559, and 0.488; qLogNEI regret is 4.943, 4.421, 2.433, and
1.462. On the 115-row transition, PosteriorMean regret is 9.947, 2.759, 0.070,
and 0.035; qLogNEI regret is 28.947, 10.662, 6.376, and 6.041. PosteriorMean is
the stronger exploitation policy in both closed pools. qLogNEI remains the
production default because the intended campaign is an open noisy BO problem;
the replay cannot measure information value for mixtures whose outcomes have
never been observed.

The production union search
`/held/mixprior-steelman-union196k-candidate-r7-20260826` succeeded on H100.
It combined three 65,536-row pools, fit all three MAP starts, and selected
candidate `72dc48bdcad57eb6` with log-NEI -1.14059. The focused H100 package
suite passed 36 tests in
`/held/mixprior-steelman-final-tests-r7-20260826`.

An independent steelman review approved the implementation with no remaining
architectural or operational blocker. The reviewer specifically accepted the
soft-prior treatment of scientific beliefs, the exact mechanical search domain,
the shared-content transfer construction, the best-of-three MAP policy, target-
only qLogNEI incumbent, independent seed provenance, and production union
search. Remaining limitations are MAP hyperparameter uncertainty, finite-pool
search, weak fixed-context identification, shared observation noise, three
target observations, ordinal-quality comparability, development-set reuse, and
qLogNEI's worse closed-pool regret relative to PosteriorMean.

### 2026-08-27 00:05 PDT — Minimal quadratic-exposure GP

A smaller surrogate now encodes the proposed rise-then-fall prior without hard
scientific constraints. For phase exposure `e` and token-mass share `p`, its
mean is `c + sum_s(a_s sum_d p_d e_sd - b_s sum_d p_d e_sd^2)`, with positive
MAP priors on `a_s` and `b_s`. Its shared covariance operates on content-
weighted first and second exposure moments with a learned phase covariance; a
same-swarm Matérn component models smooth swarm-specific deviations. Candidate
generation still enforces only the two simplexes and the 1/49,152 lattice.

All 21 production-priority H100 jobs succeeded: one rank replay and twenty
matched regret blocks. The rank artifact is
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260826/quadratic-exposure-rank.json`;
regret artifacts are under
`s3://marin-us-east-02a/marin/mixprior/benchmarks/gp-functional-form-20260826/quadratic-exposure-regret/`.
The focused suite passes 38 tests, and Ruff, Pyrefly, and diff checks pass.

Mean PosteriorMean regret at evaluations 8, 13, 18, and 20 is 1.296, 0, 0,
and 0 on the 56-row transition, compared with 3.682, 1.351, 0.559, and 0.488
for the current surrogate. On the 115-row transition it is 7.495, 0.525,
0.035, and 0.035, compared with 9.947, 2.759, 0.070, and 0.035. Against the
restored 0.652/0.738 rank model on overlapping starts, quadratic-exposure regret
is 1.524, 0, 0, and 0 versus 5.325, 1.914, 0.072, and 0.072 on 56 rows; it is
8.083, 0.482, 0.044, and 0.044 versus 62.199, 29.584, 16.929, and 11.728 on
115 rows.

Mean chronological Spearman is 0.051 on 56 rows and 0.834 on 115 rows. The weak
early global ranking on 56 rows does not prevent low simple regret: the model
reliably elevates one of the best mixtures without correctly ordering the rest
of the pool. This is the relevant success criterion for greedy selection.

The result does not extend to the current qLogNEI policy. Quadratic-exposure
qLogNEI regret is 4.395, 3.332, 3.332, and 3.146 on 56 rows and 108.862,
98.569, 98.555, and 32.805 on 115 rows. The posterior mean is useful, but its
uncertainty is not yet calibrated well enough for qLogNEI. The surrogate is the
best tested greedy PosteriorMean model, not a replacement for the production
acquisition policy without further calibration.
