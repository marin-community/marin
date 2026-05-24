# GLU vs SwiGLU activation: Research Logbook

Experiment ID prefix: `MOE-GLU`

Linked issue: TBD (filled in after `gh issue create`).

## Scope
- Goal: Test whether replacing the silu gate in the MLP (both shared expert
  and routed experts) with a plain sigmoid — i.e. swapping SwiGLU
  (`silu(gate) * up`) for the classic GLU (`sigmoid(gate) * up`) — clears
  gate 1 of the May MoE Recipe.
- Primary metric: `eval/paloma/macro_loss` + effective speedup
  (`experiments/grug/moe/agent.md` §Effective speedup calculation).
- Secondary metric: `throughput/tokens_per_second`. GLU saves an
  elementwise mul vs SwiGLU; expect a tiny throughput delta.
- Constraints: only `mlp_gate_activation` differs from the May Recipe
  baseline. All other recipe knobs identical.

## Implementation
- Added `sigmoid` to `ActivationFunctionEnum`
  (`lib/levanter/src/levanter/utils/activation.py`) so this can flow
  through `moe_mlp(activation=...)` without ad-hoc shims.
- Added `mlp_gate_activation: ActivationFunctionEnum = silu` to
  `GrugModelConfig`. Threaded into:
  - `MoEMLP.__call__` (routed experts) — via `self.cfg.mlp_gate_activation`.
  - `Block.__call__` (shared expert) — via `self.mlp.cfg.mlp_gate_activation`.

## Baseline
- Branch baseline (per user override): `grug_moe_may_recipe`.
- Reference runs: `direct_launch.py` (d512) and `direct_launch_d768.py`
  (d768), both `mlp_gate_activation=silu`.

## Kickoff
- Branch: `glu_activation` (off `grug_moe_may_recipe`).
- Submission: single `iris job run --no-wait --zone us-east5-a` invoking
  `python -m experiments.grug.moe.glu_activation_sweep`. Coordinator fans
  out 2 child training jobs (d512, d768) via `ThreadPoolExecutor`.
- Stop criteria: per `experiments/grug/moe/agent.md`, stop after the
  coordinator is submitted; gate-2 promotion requires user direction.

## Experiment Log
### Kickoff
- Hypothesis: SwiGLU's small advantage over GLU in dense LMs may or may
  not translate to QB-routed MoE with shared experts; the gate-1 signal
  will tell us whether to invest further (e.g. ReGLU, GeGLU, NoGLU).
- Result: TBD.
