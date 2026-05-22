# probes

Synthetic infra canary. A small always-on daemon that exercises critical Marin
infrastructure on a fixed cadence and records per-probe latency / outcome
samples so we can measure detection time and SLOs without waiting for a human
to notice.

v1 probes:

- **`controller-ping`** — `RemoteClusterClient.list_workers()` against the Iris controller.
- **`iris-job-submit/<zone>`** — submits a tiny CPU job in each GCP zone, polls to terminal, asserts SUCCEEDED.
- **`finelog-write`** — pushes a unique-nonce `LogEntry` and reads it back to verify the indexer is alive.

Samples land in local SQLite (canonical) and are mirrored best-effort to Finelog under `marin.canary` for query.
Heartbeat rows land in `marin.canary.meta` so an external `absent()` alarm can detect a dead daemon.

Design + spec: `.agents/projects/infra_canary/`.

## Layout

```
infra/probes/
├── pyproject.toml          # standalone; not a root workspace member
├── uv.lock                 # own lockfile, pinned to today's marin-* nightly wheels
├── deploy/
│   ├── Dockerfile          # multi-stage; build context = infra/probes/
│   ├── Dockerfile.dockerignore
│   └── deploy.sh           # build / apply / status
├── src/probes/
│   ├── probe.py            # Probe protocol + dataclasses
│   ├── daemon.py           # scheduler loop, hard-deadline cancellation, heartbeat
│   ├── cli.py              # click CLI, builds the 3 probes inline
│   ├── checks/             # ControllerPing, IrisJobSubmit, FinelogWrite
│   └── store/              # sqlite (canonical) + finelog (secondary)
└── tests/
```

This package is **standalone** — not a member of the root marin uv workspace. It pulls `marin-iris`, `marin-finelog`, and `marin-rigging` from the per-package rolling GitHub releases (`marin-iris-latest`, `marin-finelog-latest`, `marin-rigging-latest`) via `find-links` in its own `pyproject.toml`. To bump to today's nightly: `uv lock -U` inside `infra/probes/`.

## Running locally

`--once` runs each spec once (ignoring cadence), flushes stores, exits 0. This is the path CI uses as a pre-push gate.

```bash
cd infra/probes
uv run python -m probes \
  --iris-endpoint https://iris-controller.internal:10001 \
  --zone us-central1-a --zone europe-west4-b \
  --sqlite-path /tmp/samples.sqlite \
  --once
```

A blocking run is the same command without `--once`. Send SIGTERM/SIGINT to shut down gracefully.

## Configuration

All flags also read from env vars; on the VM these are set by the GCE
instance metadata, so the container starts with no per-deploy config file.

| Flag                    | Env var                            | Default                             |
| ----------------------- | ---------------------------------- | ----------------------------------- |
| `--iris-endpoint`       | `MARIN_PROBES_IRIS_ENDPOINT`       | *(required)*                        |
| `--finelog-endpoint`    | `MARIN_PROBES_FINELOG_ENDPOINT`    | falls back to `--iris-endpoint`     |
| `--zone` (repeatable)   | `MARIN_PROBES_ZONES` (comma-sep)   | *(required)*                        |
| `--sqlite-path`         | `MARIN_PROBES_SQLITE_PATH`         | `/var/lib/probes/samples.sqlite`    |
| `--heartbeat-seconds`   | `MARIN_PROBES_HEARTBEAT_SECONDS`   | `30`                                |
| `--once`                | —                                  | (off)                               |
| `--log-level`           | —                                  | `INFO`                              |

To add a probe: edit `_build_specs` in `src/probes/cli.py`. No registry indirection.

## Deployment

A single Container-Optimized OS GCP VM named `probes` runs one container. There is no staging tier — the probes themselves run against prod infra; CI's `--once` gate validates new images before push.

Operator interface is `infra/probes/deploy/deploy.sh`:

```bash
infra/probes/deploy/deploy.sh build   # docker build/push to Artifact Registry (tags :<sha> and :latest)
infra/probes/deploy/deploy.sh apply   # roll the VM to :latest
infra/probes/deploy/deploy.sh status  # VM state + last 50 lines of container logs
```

Environment overrides: `MARIN_PROBES_PROJECT`, `MARIN_PROBES_REGION`, `MARIN_PROBES_ZONE`, `MARIN_PROBES_VM`, `MARIN_PROBES_REPO`.

### One-time VM creation

```bash
PROJECT=hai-gcp-models
ZONE=us-central1-a
VM=probes
IMAGE=us-central1-docker.pkg.dev/${PROJECT}/marin/probes:latest
SA=probes@${PROJECT}.iam.gserviceaccount.com

# 1. Service account (one-time)
gcloud iam service-accounts create probes \
  --project=${PROJECT} --display-name="probes daemon"
gcloud projects add-iam-policy-binding ${PROJECT} \
  --member="serviceAccount:${SA}" --role="roles/iap.tunnelResourceAccessor"

# 2. PD-SSD for SQLite state (survives VM recreate)
gcloud compute disks create probes-state \
  --project=${PROJECT} --zone=${ZONE} \
  --size=20GB --type=pd-ssd

# 3. Create the COS VM with the container, state disk, and config env vars.
gcloud compute instances create-with-container ${VM} \
  --project=${PROJECT} --zone=${ZONE} \
  --machine-type=e2-small \
  --service-account=${SA} \
  --scopes=cloud-platform \
  --container-image=${IMAGE} \
  --container-restart-policy=always \
  --container-mount-disk=name=probes-state,mount-path=/var/lib/probes \
  --container-env=MARIN_PROBES_IRIS_ENDPOINT=https://iris-controller.internal:10001 \
  --container-env=MARIN_PROBES_FINELOG_ENDPOINT=https://iris-controller.internal:10001 \
  --container-env=MARIN_PROBES_ZONES=us-central1-a,europe-west4-b \
  --disk=name=probes-state,device-name=probes-state \
  --tags=probes
```

After this, every rollout is `infra/probes/deploy/deploy.sh apply`. To
change zones / endpoints, `gcloud compute instances update-container ${VM} --container-env=...` (or recreate).

## Querying samples

Local (canonical):

```bash
gcloud compute ssh probes -- \
  docker exec $(docker ps -q --filter ancestor=...probes) \
    sqlite3 /var/lib/probes/samples.sqlite \
    "SELECT datetime(timestamp_us / 1000000, 'unixepoch'), probe_name, outcome, latency_ms \
     FROM probe_samples ORDER BY timestamp_us DESC LIMIT 20;"
```

Finelog (secondary): query namespace `marin.canary` via the standard Finelog SQL surface; heartbeat rows are under `marin.canary.meta`.

## Tests

```bash
cd infra/probes
uv run pytest tests
```

No external dependencies. The integration test that runs the daemon against a live Iris dev cluster is the CI pre-push gate (`python -m probes --once`).
