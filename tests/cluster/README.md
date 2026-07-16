<!--
Copyright The Marin Authors
SPDX-License-Identifier: Apache-2.0
-->

# Cluster smokes

Tests here submit a job to a standing Marin cluster and assert on the result. They
carry the `cluster` marker and are deselected by default (see `addopts` in the root
`pyproject.toml`), so a normal `pytest` run does not touch a cluster.

They target a long-lived cluster that is already running, not an ephemeral cluster
the test starts itself (those are the Iris smokes in `iris-smoke-*.yaml`, marker
`requires_cluster`):

- `marin` — the GCP cluster (TPU pools). IAP-fronted, reached over HTTPS.
- `cw-us-east-02a` — the CoreWeave cluster (H100). Kube-fronted, reached via kubeconfig.

## Fixtures (`conftest.py`)

- `iris_client` — opens the `marin` (TPU) cluster and installs it as the current Fray
  client, so `StepRunner` smokes (evalchemy, SFT) submit there.
- `marin_gpu_client` — opens `cw-us-east-02a` (GPU).
- `run_test_job` — submits a `JobRequest`, bounds its queue/runtime waits, and
  terminates it on interruption. Used by the vLLM e2es.

`iris_client` and `marin_gpu_client` build on `open_cluster_client(name)`, which skips
when no credential reaches the cluster: a missing kubeconfig (`ConfigException`,
CoreWeave) or no ambient service-account creds (`IapCredentialsUnavailable`, `marin`).
So `-m cluster` is a no-op on a machine without credentials.

## Running locally

```bash
# TPU smokes against the marin cluster (needs a login or ambient GCP creds).
# No MARIN_PREFIX: the tests pin a region (smoke_region) that binds both the slice
# and the storage root.
HF_TOKEN=... WANDB_API_KEY=... \
  uv run pytest tests/cluster/evals tests/cluster/sft \
    -m cluster -o addopts= --import-mode=importlib --timeout=0 -vv -s

# GPU e2es against CoreWeave (needs ~/.kube/coreweave-iris):
uv run pytest tests/cluster/vllm -m cluster -o addopts= --import-mode=importlib -vv -s
```

`-o addopts=` clears the default marker filter and the 600s session cap; `-m cluster`
then selects these tests; `--import-mode=importlib` is re-added because `-o addopts=`
drops it and the `tests.cluster.*` imports need it. `--timeout=0` disables the 60s
per-test cap (a separate `timeout` ini that `-o addopts=` does not clear) for the
TPU smokes, which carry no per-test timeout marker; the vLLM e2es set their own.

## CI

`.github/workflows/marin-cluster-smoke.yaml` runs these nightly and on
`workflow_dispatch`. It authenticates to the `marin` cluster with the existing
`IRIS_CI_GCP_SA_KEY` service account; its ADC mints the IAP edge token, the same path
the canary ferry uses, so no Workload Identity Federation setup is required. The 8×H100
CoreWeave e2es are gated behind a `run_gpu` dispatch input and do not run nightly.
