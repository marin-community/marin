# muon_epsilon sweep at d512: Research Logbook

Experiment ID prefix: `MOE-MEPS`

Linked issue: marin-community/marin#5933 (sub-issue of #5358).

## Scope
- Goal: Determine the sensitivity of the May MoE Recipe to the MuonH
  branch's `muon_epsilon` (eps inside the Newton-Schulz step on matrix
  params) at d512 / 2.19e17.
- Primary metric: `eval/paloma/macro_loss`.
- Secondary metrics: `throughput/tokens_per_second`, optimizer / weight
  health watchers if any deviate.
- Constraints: only the `muon_epsilon` knob varies. Model shape, AdamH
  `epsilon` (~7.68e-16 from the scale heuristic), learning rates, betas,
  warmup, schedule, parallel_mode all match the May Recipe d512 baseline.

## Sweep axis
| tag | muon_epsilon |
| --- | --- |
| `muon-eps-1e-6` | 1e-6 |
| `muon-eps-1e-10` | 1e-10 |
| `muon-eps-1e-16` | 1e-16 |

Default (not in sweep) is `1e-8`, served by the existing `direct_launch.py`
baseline.

## Baseline
- Branch baseline (per user override): `grug_moe_may_recipe`.
- Reference run: `direct_launch.py` (d512 / 2.19e17, `muon_epsilon=1e-8`).
- Code refs: `experiments/grug/moe/optimizer.py:239` (`muon_epsilon` field),
  `experiments/grug/moe/muon_epsilon_sweep.py`.

## Kickoff
- Branch: `muon_epsilon_sweep` (off `grug_moe_may_recipe`).
- Submission: single `iris job run --no-wait --zone us-east5-a` invoking
  `python -m experiments.grug.moe.muon_epsilon_sweep`. Coordinator fans
  out the three training jobs via `ThreadPoolExecutor` and blocks on all
  of them.
- Stop criteria for kickoff: per `experiments/grug/moe/agent.md`, stop
  once the Iris coordinator is submitted; any follow-up requires user
  direction.

## Experiment Log
### Kickoff
- Hypothesis: at d512 the MuonH branch is robust to `muon_epsilon` over
  several orders of magnitude; large eps may damp updates and underfit,
  tiny eps may amplify ill-conditioned directions.
- Result: TBD.
