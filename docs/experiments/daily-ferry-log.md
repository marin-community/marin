# Daily Ferry Log

This log is the canonical summary of completed daily ferry runs.

Sealing policy (for new runs):
- seal each run with a pushed git tag that points to the commit containing the exact `experiments/ferries/daily.py` used for launch
- use canonical daily seal tags in the format `ferry/daily/YYYYMMDD/<run_slug>`
- keep detailed launch/debug narrative in the run issue
- open a PR that updates only this file (`docs/experiments/daily-ferry-log.md`)
- extract canonical metric keys with:
  - `uv run python scripts/ferries/daily_analysis.py --run <wandb_run_url_or_path> --format markdown`
- if interesting, include optional analysis observations from the script output comparing this run against recent completed daily ferries

## Runs

## 2026-02-20 (completed 2026-02-21)

- [Issue #2940](https://github.com/marin-community/marin/issues/2940)
- Summary: Inaugural daily 125M ferry run.
- [Run `ferry_daily_125m_20260220-202930-daily-ferry-vmem50m-fusedargmax-5bbb39`](https://wandb.ai/marin-community/marin/runs/ferry_daily_125m_20260220-202930-daily-ferry-vmem50m-fusedargmax-5bbb39)
- [Experiment Script `experiments/ferries/daily.py`](https://github.com/marin-community/marin/blob/main/experiments/ferries/daily.py)
- [Data browser `daily-da79b9`](https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/daily-da79b9.json)
- Seal tag: `N/A` (pre-sealing policy)
- Metrics:
  - `eval/paloma/c4_en/bpb`: `1.12509`
  - `eval/bpb`: `1.17049`
  - `eval/uncheatable_eval/bpb`: `1.13019`
- Notes:
  Initial TPU vmem failures were mitigated; the final run reached `SUCCEEDED` with eval harness completion.
