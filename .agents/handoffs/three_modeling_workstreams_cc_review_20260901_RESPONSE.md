# Claude Code review: three modeling workstreams

Review provenance: Claude Opus 5, maximum effort, read-only tools, subscription-authenticated account
`plambdafour@proton.me`. The reviewer inspected the coupled-onset results and prior modeling artifacts without
editing either worktree.

## Coupled-onset experiment

Bayesian optimization may be run only as a discovery-only surface audit. Its rows must use the existing discovery
seed, cannot alter the fresh-seed confirmation cells, and cannot enter the confirmation statistic. All acquisition
should be on the eligible untied surfaces; the tied coordinate is one-dimensional and already densely covered.
Eight acquisitions at 0.60T are the minimum under-sampling falsifier; eight per arm keeps the refinement balanced.

The fresh confirmation needs three distinct estimands because the discovery result depends on how the tied
comparator is defined:

1. E1, primary optimum-location estimand: the arm-specific tied and untied discovery minima.
2. E2, secondary fixed-policy transport: `c109` tied versus `c016` untied at every onset.
3. E3, secondary fixed-tied comparator: `c109` versus each arm's untied discovery minimum.

The minimal complete inventory is 72 runs: eight reserved seeds for four policies at 0.60T (`c096`, `c042`,
`c109`, `c016`), two at 0.80T (`c109`, `c016`), and three at 0.90T (`c109`, `c067`, `c016`). The primary claim is
the conjunction gain(0.80T) > gain(0.60T) and gain(0.90T) > gain(0.60T), tested as an intersection-union test with
one-sided alpha 0.05 for each contrast. E3 reverses both discovery contrasts, showing that a passing E1 with a
failing E3 is an optimum-location result carried by movement of the tied optimum, not evidence for a general onset
mechanism. C4 remains descriptive and outside candidate selection.

## OLMix single-phase benchmark

Both canonical DCLM and High Quality tables contain all 363 mixtures and all 42 tasks. The earlier partial-row
inventory was withdrawn. Use five geometric outer folds with five repeated partitions, identical across models,
and tune every shape and shrinkage parameter within each outer training split. Report direct 42-task macro BPB
RMSE as primary; Spearman, calibration, and held-fold selection regret are secondary.

The benchmark should include the exact incumbent OLMix law, a linear-exposure baseline, shared-shape DSP, and a
task-joint shared-shape DSP. A challenger must have the same direction of improvement in both swarms fitted
independently. Capacity-matched ridge, a linear shape swap, an inventory or bucket-identity scramble, and a
training-fold outcome permutation distinguish exposure structure from generic regularization. Semantic family
partitions are not allowed.

## State-dependent data exhaustion

The crossed-prefix panel should first report whether the proposed fresh and repeated features are identified after
conditioning on phase-1 exposure and prefix intercepts. The smallest falsifiable head is one nonnegative fresh
benefit block and one nonnegative repeated-damage block, with the same parameter count and fixed nonlinear shapes
as a state-independent one-epoch threshold comparator. Action-blocked prediction at known prefixes is primary;
leave-one-prefix-out prediction anchored by the tied boundary readout is a transfer diagnostic.

Evidence requires corrected repeated-CV intervals showing the state-dependent split beats both cumulative exposure
and the equal-parameter fixed-threshold control, survival of the cap-4 sensitivity, and lower decision regret than
always selecting tied. Failure of any condition rejects promotion of this head on the current panel.
