# Daily Ferry Log

This log is the canonical summary of completed daily ferry runs.

Sealing policy (for new runs):
- seal each run with a pushed git tag that points to the commit containing the exact `experiments/ferries/daily.py` used for launch
- use canonical daily seal tags in the format `ferry/daily/YYYYMMDD/<run_slug>`
- keep detailed launch/debug narrative in the run issue
- open a PR that updates only this file (`docs/experiments/daily-ferry-log.md`)

## Runs

| Date (UTC) | Run | Issue | W&B | Experiment File | Experiment Record | Seal Tag | Summary |
|---|---|---|---|---|---|---|---|
| 2026-02-20 (completed 2026-02-21) | `ferry_daily_125m_20260220-202930-daily-ferry-vmem50m-fusedargmax-5bbb39` | [#2940](https://github.com/marin-community/marin/issues/2940) | [run](https://wandb.ai/marin-community/marin/runs/ferry_daily_125m_20260220-202930-daily-ferry-vmem50m-fusedargmax-5bbb39) | [`experiments/ferries/daily.py`](https://github.com/marin-community/marin/blob/main/experiments/ferries/daily.py) | [daily-da79b9](https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/daily-da79b9.json) | `N/A` (pre-sealing policy) | Inaugural daily 125M ferry. Initial TPU vmem failures were mitigated; final run reached `SUCCEEDED` with eval harness completion and stable end-of-run metrics. |
