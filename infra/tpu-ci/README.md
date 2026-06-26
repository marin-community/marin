# TPU CI Runner Fleet

Self-hosted, preemptible TPU GitHub Actions runners that execute the TPU smoke
and canary jobs.

**Operating it** — first-time bring-up or full rebuild, rotating the GitHub PAT
when runners stop registering, recovering stuck runners, or debugging a single
CI VM: see [`.agents/runbooks/tpu-ci-fleet.md`](../../.agents/runbooks/tpu-ci-fleet.md).
That runbook is the operational entry point; the scripts below own the details.

| File | What it owns |
|---|---|
| `setup.py` | Fleet CLI — `setup-controller`, `teardown`, `build-image`, `update-token` (PAT rotation), `restart-runners`, `controller-logs`. Command docstrings are the authority. |
| `vm_manager.py` | Per-VM CLI — `check-logs`, `debug-setup`, `debug-tpu`. |
| `config.py` | Single source of truth for fleet shape: controller VM, `TPU_ZONES_CONFIG`, labels. Do not restate these values elsewhere. |
| `clean-tpu.sh` | Wedged-TPU device reset on a runner. |
