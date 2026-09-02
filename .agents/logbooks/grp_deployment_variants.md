# GRP Deployment Variants: Research Logbook

## Scope
- Goal: compare observed-only GRP deployment rules on top of the same retuned GRP fit, and pick a final slide-ready procedure.
- Primary metric(s): retrospective `Regret@1`, predicted BPB realism, and deployment movement measured by mean phase TV.
- Constraints: keep the nonlinear retuning procedure fixed; vary only deployment.

## Baseline
- Date: 2026-04-02
- Code refs:
  - `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_grp_retuned.py`
  - `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_grp_observed_hull.py`
- Baseline numbers:
  - raw retuned optimum reaches retrospective `Regret@1 = 0` from `k >= 80`, but predicted optima are unrealistically optimistic (`~1.029`) and the mixture moves a lot.
  - observed-only full hull keeps `Regret@1 = 0` from `k >= 80`, with more realistic predicted BPB (`~1.065`) and lower movement.

## Experiment Log
### 2026-04-02 17:31 - Observed-only deployment variants
- Hypothesis: restricting deployment to observed-run mixtures should preserve retrospective choice quality while stabilizing the optimum and making predicted BPBs more realistic.
- Command:
  - `uv run python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_grp_deployment_variants.py`
- Config:
  - same per-subset GRP retuning as `benchmark_grp_retuned.py`
  - deployment variants:
    - best predicted observed run
    - convex hull of top-4 predicted observed runs
    - convex hull of top-8 predicted observed runs
    - convex hull of top-16 predicted observed runs
    - convex hull of all observed runs
- Result:
  - all variants have the same retrospective `Regret@1` profile: misses only at `k=40,60`, then zero from `k >= 80`
  - after `k >= 80`, movement / predicted-value summary:
    - `top1_observed`: mean move `0.256`, mean predicted `1.0792`
    - `top4_hull`: mean move `0.147`, mean predicted `1.0739`
    - `top8_hull`: mean move `0.195`, mean predicted `1.0702`
    - `top16_hull`: mean move `0.127`, mean predicted `1.0666`
    - `all_observed_hull`: mean move `0.120`, mean predicted `1.0653`
- Interpretation:
  - the full observed-run hull is the cleanest local rule among those tried
  - it matches the other variants on retrospective choice quality, but gives the lowest movement and the most realistic predicted BPB
- Next action:
  - use the observed-only full hull as the documented deployment rule in the slides
  - if needed later, validate a few observed-hull subset deployments by training them

### 2026-04-02 18:12 - More deployment regularizers and representative validation launch
- Hypothesis: a hull over the top actual observed runs should retain the good behavior of the full observed hull, while being cleaner and slightly more local.
- Command:
  - `uv run python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_grp_deployment_variants.py`
  - `uv run python -m marin.run.iris_run --config /Users/calvinxu/Projects/Work/Marin/marin/lib/iris/examples/marin.yaml -- --no-wait --job-name dm-genericfamily-top8actual-hull-subset-optima-$(date +%Y%m%d-%H%M%S) --region us-east5 --zone us-east5-a --cpu 2 --memory 16GB --disk 20GB --extra marin:tpu --extra marin:eval -- python -m experiments.domain_phase_mix.launch_two_phase_many_genericfamily_top8actual_hull_subset_optima --tpu-type v5p-8 --max-concurrent 4`
- Config:
  - added these observed-only variants:
    - `top4_actual_hull`
    - `top8_actual_hull`
    - `top16_actual_hull`
    - `all_hull_disp0.01`, `all_hull_disp0.02`
    - `all_hull_to_bestactual0.02`, `all_hull_to_bestactual0.05`
- Result:
  - after `k >= 80`:
    - `all_observed_hull`: mean predicted `1.0653`, mean move `0.1200`, mean support `8.75`
    - `top8_actual_hull`: mean predicted `1.0661`, mean move `0.1100`, mean support `6.75`
    - `top16_actual_hull`: mean predicted `1.0660`, mean move `0.1096`, mean support `9.13`
  - dispersion / locality penalties reduced nearest-TV distance but generally worsened predicted BPB and often collapsed toward a single observed run
  - launched representative subset validation for `top8_actual_hull`:
    - parent job `/calvinxu/dm-genericfamily-top8actual-hull-subset-optima-20260402-181147`
- Interpretation:
  - `top8_actual_hull` is the best cleaner policy: essentially tied with `top16_actual_hull`, but simpler and sparser
  - the full observed hull still wins slightly on pure predicted BPB, but the difference is negligible
- Next action:
  - validate representative `top8_actual_hull` subset runs first
  - only run an all-subset sweep if the representative validations look materially better than the raw retuned subset optima
