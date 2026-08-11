# H100 evidence image NVTX smoke failure

GitHub Actions run
[31472791216](https://github.com/marin-community/marin/actions/runs/31472791216)
used exact source `4611929eab07dc8fa11294c2456b4cdfde5a46bc` to build the
dedicated H100 evidence image. The sole image job failed in the CPU-only NVTX
smoke before BuildKit exported or pushed an image. All five legacy image jobs
were skipped. The workflow was not rerun.

`run.json` is the terminal machine-readable workflow snapshot. `raw-job.log`
is the complete GitHub job log. `failure-analysis.txt` states the supported
control-flow boundary without treating the generated local Nsight report as
accepted evidence.
