# probes

Synthetic infra canary: an always-on daemon that runs three probes against Iris
and Finelog on a fixed cadence.

- `controller-ping` — `list_workers()` on the Iris controller.
- `iris-job-submit/<zone>` — submit a tiny job per zone, wait for SUCCEEDED.
- `finelog-write` — write a nonce and read it back.

Each result is logged to stdout (`probe <name>: ok|fail [<ms>ms] start=<utc>`),
written to the `infra.canary.probes` finelog namespace, and appended to a daily
JSONL that rolls up to `gs://marin-us-central1/infra/probes/dt=<date>/` at UTC
rollover.

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
uv run deploy/deploy.py apply    # roll the VM to :latest
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
