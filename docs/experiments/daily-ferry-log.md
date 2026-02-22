# Daily Ferry Log

This log is the canonical summary of completed daily ferry runs.

Sealing policy (for new runs):
- seal each run with a pushed git tag that points to the commit containing the exact `experiments/ferries/daily.py` used for launch
- use canonical daily seal tags in the format `ferry/daily/YYYYMMDD/<run_slug>`
- keep detailed launch/debug narrative in the run issue
- open a PR that updates only this file (`docs/experiments/daily-ferry-log.md`)

## Runs

## 2026-02-22 (completed 2026-02-22)

- [Issue #2954](https://github.com/marin-community/marin/issues/2954)
- Summary: Daily 125M ferry with LR delta (`learning_rate` `3e-3 -> 9e-3`). Primary run reached end-of-train eval (`global_step=5153`) and logged final metrics; Ray terminal status was `STOPPED`.
- [Run `ferry_daily_125m_20260222-004403-daily-ferry-lr3x-a0df25`](https://wandb.ai/marin-community/marin/runs/ferry_daily_125m_20260222-004403-daily-ferry-lr3x-a0df25)
- [Experiment Script `experiments/ferries/daily.py` @ launch commit](https://github.com/marin-community/marin/blob/1b03fc1b3df7316b151d4bbfd98dce64a036751a/experiments/ferries/daily.py)
- [Data browser `daily-1f1377`](https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/daily-1f1377.json)
- Seal tag: `ferry/daily/20260222/20260222-004403-daily-ferry-lr3x`
- Metrics:
  - `global_step`: `5153`
  - `eval/loss`: `3.46797`
  - `eval/paloma/macro_loss`: `3.71091`
  - `eval/uncheatable_eval/bpb`: `1.13664`
- Notes:
  W&B marked the run `crashed`, but final eval metrics were recorded at the terminal training step. Subsequent retries (`r1-r3`) were intentionally stopped after confirming the primary run had already produced complete final eval output.

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
