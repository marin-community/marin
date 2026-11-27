# Post-training experiments

## Checkpoint evaluation cohorts

Snowball's `train-long-checkpoint-eval` preset uses three fixed agentic cohorts.
[`rl_checkpoint_cohorts.yaml`](rl_checkpoint_cohorts.yaml) pins every package;
[`rl_checkpoint_eval.py`](rl_checkpoint_eval.py) binds those packages to the
curated evaluation datasets and verifier settings.

| Cohort | Curated dataset | Pinned tasks | Verifier deadline |
| --- | --- | ---: | --- |
| `bfcl_parity` | BFCL Parity Pi | 30 | 300 seconds |
| `swebench_verified` | SWE-bench Verified Mini SWE | 30 | 3,000 seconds |
| `openthoughts_tblite` | OpenThoughts-TBLite | 30 | Task deadline, capped at 3,600 seconds |

The package list is stratified by BFCL tag, SWE repository, and TBLite category.
Each task has one trial. The checkpoint evaluation uses the same frozen policy
as EvalLite and scores each cohort separately. See the
[RL experiment guide](../../docs/recipes/rl-experiments.md#agentic-checkpoint-evaluation)
for credentials, serving limits, and checkpoint eligibility.
