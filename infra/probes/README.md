# probes

Synthetic infra canary: an always-on daemon that runs **collectors** against
Iris and Finelog on a fixed cadence. A collector emits one or more `Sample`s
(`metric`, `value`, JSON `labels`, `collected_at`) — the single shape for both
up/down health checks and numeric gauges.

Health checks (emit a `probe_up` 1/0 sample; the runner adds `probe_latency_ms`):

- `controller-ping` — `list_workers()` on the Iris controller (cadence 60s).
- `finelog-write` — write a nonce and read it back (60s).
- `iris-job-submit/<zone>` — submit a tiny job per zone, wait for SUCCEEDED (300s).

Gauges:

- `provisioning` — accelerator provisioning stats over a trailing 3h window,
  recomputed every 15 min. The controller's autoscaler emits one structured row
  per slice provisioning outcome to the `iris.provisioning` finelog namespace;
  this collector reads that namespace with one bounded query and rolls it up by
  `(resource_type, scale_group, zone)` plus a fleet series — emitting
  `provision_*` count/latency/success-ratio gauges. See
  `iris.cluster.controller.autoscaler.provisioning` for the outcome vocabulary
  and `src/provisioning.py` for the emitted metrics.
- `workers` — worker-fleet snapshot from `list_workers()` (60s). Rolls the
  healthy workers into fleet resource totals (`worker_healthy`,
  `worker_cpu_millicores`, `worker_memory_bytes`, `worker_tpu_chips`, all
  labelled `scope=fleet`) plus a per-region healthy head count
  (`worker_healthy{region=…}`).
- `jobs` — root-job-state breakdown from one raw-SQL `GROUP BY` (120s). Splits
  into a live in-flight snapshot (`job_inflight{state=…}`) and a trailing-24h
  terminal window (`job_terminal_24h{state=…}`), each with a `scope=fleet` total.
  Runs the controller's `ExecuteRawQuery` RPC over a dedicated connect client
  (the same call the `iris query` CLI makes). See `src/cluster.py` for the
  emitted metrics.

Each sample is logged to stdout (`probe <name>: ok|fail [<ms>ms] start=<utc>`),
written to the `infra.canary.metrics` finelog namespace (query it with
`finelog query <cluster> 'SELECT ... FROM "infra.canary.metrics"'`, slicing
labels with DuckDB `json_extract`), and appended to a daily JSONL that rolls up
to `gs://<us-central1 data bucket>/infra/probes/dt=<date>/` at UTC rollover.

Standalone package (own `pyproject.toml`/`uv.lock`): pulls `marin-iris`,
`marin-finelog`, `marin-rigging` from the rolling GitHub releases via
`find-links`. Bump to today's nightly with `uv lock -U` inside `infra/probes/`.

## Run

```bash
cd infra/probes
uv run python -m infra_probes --iris-endpoint http://<controller>:10000
# --zone defaults to europe-west4-b, us-west4-a
```

## Deploy

Single COS VM `infra-probes` (us-central1-b), one container, `restart=always`.
`deploy/deploy.py` is a click CLI; run it with `uv run` from `infra/probes/`.

```bash
cd infra/probes
uv run deploy/deploy.py build    # build + push :sha and :latest
uv run deploy/deploy.py apply    # roll the VM to this HEAD's :sha image
uv run deploy/deploy.py status   # VM state + recent logs
```

Project, region, zone, VM name, and repo default to the prod values and can be
overridden per-command (`--project`, `--zone`, …) or via `MARIN_PROBES_*` env vars.

### One-time VM creation

`create` provisions the service account (image pull, Cloud Logging, GCS
roll-ups), its IAM bindings, and the COS VM in one shot:

```bash
uv run deploy/deploy.py create    # --iris-endpoint / --machine-type to override
```

The VM gets a `/var/lib/probes` host mount that persists the JSONL across
container restarts, plus a startup-script that makes it writable by the uid-1000
container.
