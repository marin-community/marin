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
- `cw-us-east-08a` — the CoreWeave GB200 cluster (NVL72 racks). Federated from the `marin`
  hub the same way, via `target_cluster=MARIN_GB200_CLUSTER`.

## Fixtures (`conftest.py`)

- `iris_client` — opens the `marin` (TPU) cluster and installs it as the current Fray
  client, so `StepRunner` smokes (evalchemy, SFT) submit there.
- `marin_gpu_client` — opens `cw-us-east-02a` (GPU).
- `run_test_job` — submits a `JobRequest`, bounds its queue/runtime waits, and
  terminates it on interruption. Used by the vLLM e2es.

No workflow runs the `cluster` marker today: #8605 deleted Marin Cluster Smoke after 38
consecutive scheduled runs timed out. Everything here is launched by hand until #8704
gives accelerator tests a runner again.

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

## Snowball training-forward parity

`vllm/test_snowball_training_parity.py` scores the model the training path builds, not the inference load path.
It loads the Snowball HF export as `train_lm` does (HF config reload, `preserve_hf_runtime_settings`,
`fp32_fields=("router_bias",)`, `p=f32,c=bf16`), takes the trainer's compute view and runs a forward-only pass on
one H100x8 node with an expert-parallel mesh (data 2 x expert 4). The runtime settings are the identity recipe's:
dropless MoE capacity, FP32 router dot, FP32 combine, FP32 activations and `torch_order` attention (FA4 for the
16K and 32K prompt buckets, where `torch_order` cannot fit). Every position of the 64 representative prompts is
compared with a stored vLLM reference. Each prompt category, and the positions past the 2,048-token window, must
have p99 <= 0.70 and max <= 3.5 nats of absolute log-prob difference over vLLM's top-20 tokens above p = 1e-3.
Every layer must drop zero valid MoE assignments.

The reference is a content-addressed public file, pinned by `VLLM_POSITION_REFERENCE_SHA256` in
`lib/marin/src/marin/testing/inference/snowball.py`. It records the vLLM release and commit, the parallel layout,
the attention backend and the model sha256, so a stale reference can be told apart from a regression. Until the
sha is pinned the test skips. To make one, submit the generator, then copy the written file to the URL its log
prints and pin the sha:

```bash
uv run python tests/cluster/vllm/generate_snowball_position_reference.py \
  --output s3://marin-us-east-02a/<scratch>/snowball-position-reference.json --run
```

Run the check by hand:

```bash
uv run pytest tests/cluster/vllm/test_snowball_training_parity.py \
  -m cluster -o addopts= --import-mode=importlib -vv -s
```

It is not on a schedule. Whether it joins the Snowball vLLM parity workflow, and how often, is an owner decision.
The workflow selects tests with `-k "$SELECTOR"` over one test file; adding this check means a
`workflow_dispatch` input (off by default) that adds this file and its `--include` to the submitted pytest command.
