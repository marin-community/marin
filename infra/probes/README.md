# probes

Synthetic infra canary. A small always-on daemon that runs three probes
against Iris and Finelog on a fixed cadence and logs the result of each.

v1 probes:

- **`controller-ping`** — `RemoteClusterClient.list_workers()` against the Iris controller.
- **`iris-job-submit/<zone>`** — submits a tiny CPU job in each GCP zone, polls to terminal, asserts SUCCEEDED.
- **`finelog-write`** — pushes a unique-nonce `LogEntry` and reads it back to verify the indexer is alive.

Results land on stdout as structured log lines (Docker → Cloud Logging on COS). Each line is `probe <name>: ok|fail [<wall_time_ms>ms] start=<utc-iso>`.

Design + spec: `.agents/projects/infra_canary/` (kept for historical context — the implementation is now much smaller than what the spec described).

## Layout

```
infra/probes/
├── pyproject.toml          # standalone; not a root workspace member
├── uv.lock                 # own lockfile, pinned to today's marin-* nightly wheels
├── deploy/
│   ├── Dockerfile          # build context = infra/probes/
│   ├── Dockerfile.dockerignore
│   └── deploy.sh           # build / apply / status
├── src/
│   └── marin_infra_probes.py  # ProbeResult, Probe, ProbeRunner, the three probes, main()
└── tests/test_runner.py
```

The package is **standalone** — not a member of the root marin uv workspace. It pulls `marin-iris`, `marin-finelog`, and `marin-rigging` from the per-package rolling GitHub releases (`marin-iris-latest`, `marin-finelog-latest`, `marin-rigging-latest`) via `find-links` in its own `pyproject.toml`. To bump to today's nightly: `uv lock -U` inside `infra/probes/`.

## Public API

`src/marin_infra_probes.py` defines:

```python
@dataclass
class ProbeResult:
    is_success: bool
    name: str                       # stamped by the runner, which owns this metadata
    started_at: datetime            # UTC wall-clock time at probe start
    wall_time: float | None = None  # filled in by the runner once the run completes

# A probe fn reports only whether the probe succeeded; the runner stamps the rest.
ProbeFn = Callable[[], bool]

@dataclass
class Probe:
    name: str
    fn: ProbeFn
    timeout: float
    cadence: float

class ProbeRunner:
    def add_probe(self, name: str, fn: ProbeFn, *, timeout: float, cadence: float) -> None: ...
    def run(self) -> None: ...   # blocks forever; Ctrl-C (KeyboardInterrupt) kills the process

# Concrete probes (callables you pass to add_probe):
def probe_controller_ping(iris: RemoteClusterClient) -> bool: ...
def probe_iris_job_submit(iris: RemoteClusterClient, zone: str) -> bool: ...
def probe_finelog_write(finelog: LogClient) -> bool: ...
```

To add a probe, write a function returning `bool` and call `runner.add_probe(name, fn, timeout=..., cadence=...)`.

## Running locally

```bash
cd infra/probes
uv run python -m marin_infra_probes \
  --iris-endpoint https://iris-controller.internal:10001
```

Send SIGTERM/SIGINT to shut down. Tail stdout to see results.

| Flag                  | Env (n/a, all flag-driven) | Notes                                  |
| --------------------- | -------------------------- | -------------------------------------- |
| `--iris-endpoint`     | *(required)*               | Iris controller RPC endpoint; the finelog address is resolved from its endpoint registry (`/system/log-server`). |
| `--zone` (repeatable) | `europe-west4-b`, `us-west4-a` | One canary job per zone every 5 min.   |

## Deployment

Single Container-Optimized OS GCP VM named `infra-probes`, one container, `restartPolicy=always`, no orchestrator. State (logs) goes to Cloud Logging via Docker's stdout pickup.

```bash
infra/probes/deploy/deploy.sh build   # docker build, push :sha and :latest
infra/probes/deploy/deploy.sh apply   # roll the VM to :latest
infra/probes/deploy/deploy.sh status  # VM state + last 50 container log lines
```

Environment overrides: `MARIN_PROBES_PROJECT`, `MARIN_PROBES_REGION`, `MARIN_PROBES_ZONE`, `MARIN_PROBES_VM`, `MARIN_PROBES_REPO`.

### One-time VM creation

Set flags on the VM via container args; cloud-init isn't needed.

```bash
PROJECT=hai-gcp-models
ZONE=us-central1-b
VM=infra-probes
IMAGE=us-central1-docker.pkg.dev/${PROJECT}/marin/infra-probes:latest
SA=infra-probes@${PROJECT}.iam.gserviceaccount.com

gcloud iam service-accounts create infra-probes \
  --project=${PROJECT} --display-name="probes daemon"

gcloud compute instances create-with-container ${VM} \
  --project=${PROJECT} --zone=${ZONE} \
  --machine-type=e2-small \
  --service-account=${SA} \
  --scopes=cloud-platform \
  --container-image=${IMAGE} \
  --container-restart-policy=always \
  --container-arg="--iris-endpoint=http://iris-controller.internal:10001" \
  --tags=infra-probes
```

After this, every rollout is `infra/probes/deploy/deploy.sh apply`.

## Querying results

```bash
gcloud compute ssh infra-probes -- docker logs $(docker ps -q --filter ancestor=...infra-probes) | tail -200
```

Or via Cloud Logging once Docker stdout is shipped.

## Tests

```bash
cd infra/probes
uv run pytest tests
```
