# Debugging log for qsplit240 520M pilot

Investigate the failed Iris job `/calvinxu/dm-qsplit240-520m-chinchilla-pilot-20260408-182741`, determine whether the failure was caused by experiment code or infrastructure, and recover it before handing it off for babysitting.

## Initial status

At `2026-04-09 00:54 PDT`, `iris job list --json --prefix /calvinxu/dm-qsplit240-520m-chinchilla-pilot-20260408-182741` reported:

- parent job state: `JOB_STATE_FAILED`
- child jobs:
  - `train_lm_baseline_proportional-6e6375`: `JOB_STATE_KILLED`
  - `train_lm_baseline_unimax-20e093`: `JOB_STATE_KILLED`

The parent job error showed a bundle staging failure on a restarted executor:

- `RuntimeError: Failed to fetch 0c7ee2ad5ab5403aa16735216923ee2105bc269b99f24c72c65ecb2c19b9fb1c: timed out`
- stack trace rooted in `iris.cluster.bundle.BundleStore._fetch_from_controller()`

## Hypothesis 1

The 520M pilot did not fail because of bad experiment code. The parent executor was restarted multiple times and then timed out fetching its staged workspace bundle from the controller while rehydrating on a worker.

## Changes to make

No code changes to the qsplit launcher itself.

Operational recovery:

- confirm the failure from Iris job state and error text
- inspect the 520M launcher for any obvious configuration issue
- resubmit the parent executor with the same canonical job name so Iris replaces the finished failed job

## Results

Checked `experiments/domain_phase_mix/launch_two_phase_many_qsplit240_520m_chinchilla_pilot.py`; there was no launcher bug or region mismatch.

Resubmitted with:

```bash
uv run python -m marin.run.iris_run \
  --config /Users/calvinxu/Projects/Work/Marin/marin/lib/iris/examples/marin.yaml \
  -- \
  --job-name dm-qsplit240-520m-chinchilla-pilot-20260408-182741 \
  --cpu 1 \
  --memory 4GB \
  --disk 20GB \
  --region us-central1 \
  --zone us-central1-a \
  --no-wait \
  -- python experiments/domain_phase_mix/launch_two_phase_many_qsplit240_520m_chinchilla_pilot.py
```

At `2026-04-09 00:56 PDT`, the resubmission succeeded and `iris rpc controller get-job-state` reported:

- `/calvinxu/dm-qsplit240-520m-chinchilla-pilot-20260408-182741`: `JOB_STATE_RUNNING`

Conclusion:

- root cause was Iris control-plane / bundle-fetch instability during executor restart, not a bug in the 520M pilot launcher
- practical fix was resubmission once controller responsiveness improved enough to accept the launch

## Future Work

- [ ] If this failure mode repeats, investigate whether Iris bundle fetch should fall back to storage instead of relying on controller HTTP for large restart bursts.
- [ ] Check whether executor preemption behavior for long-running CPU parents is still too fragile under controller load.
