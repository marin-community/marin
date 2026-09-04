# App-declared Marina runners: contract

## Manifest schema

`infra/marina/apps/<app>/app.toml` accepts zero or more array-of-table entries:

```toml
[[jobs]]
name = "sync"
runner = "hourly"
schedule = "0 * * * *"
command = ["python", "-m", "echo.sync.main"]
timeout = 7200
cpu = 4
memory_gib = 4
secrets = ["MARINMIRROR_TOKEN"]
```

Every field is required except `secrets`, which defaults to an empty list. `name` and `runner` match `^[a-z][a-z0-9-]*$`. `schedule` is a non-empty Cloud Scheduler unix-cron string. `command` is a non-empty array of non-empty strings. `timeout`, `cpu`, and `memory_gib` are positive integers. `secrets` is an array of distinct environment-variable names. Job names are unique within one app. Every job assigned to one runner across all discovered apps must declare the same schedule.

## Python API

File: `infra/marina/src/marina/manifest.py`

```python
@dataclass(frozen=True)
class AppJob:
    name: str
    runner: str
    schedule: str
    command: tuple[str, ...]
    timeout: int
    cpu: int
    memory_gib: int
    secrets: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppManifest:
    name: str
    title: str
    description: str
    root: Path
    connect_src: tuple[str, ...] = ()
    build_command: str | None = None
    jobs: tuple[AppJob, ...] = ()


@dataclass(frozen=True)
class BoundJob:
    app: AppManifest
    job: AppJob


@dataclass(frozen=True)
class JobRunner:
    name: str
    schedule: str
    jobs: tuple[BoundJob, ...]

    @property
    def timeout(self) -> int:
        """Return the sum of child timeouts plus the fixed migration and runner overhead."""

    @property
    def cpu(self) -> int:
        """Return the largest child CPU requirement."""

    @property
    def memory_gib(self) -> int:
        """Return the largest child memory requirement."""


def job_runners(apps: Sequence[AppManifest]) -> tuple[JobRunner, ...]:
    """Group jobs by runner in runner-name order.

    Jobs within a runner are ordered by app name and then job name. Raise ValueError when
    declarations assigned to one runner use different schedules.
    """
```

File: `infra/marina/src/marina/cli.py`

```python
@cli.command("run")
@click.argument("runner")
@click.option("--apps-dir", type=click.Path(path_type=Path), default=DEFAULT_APPS_DIR, show_default=True)
@click.option("--reader")
@click.option("--migrate-only", is_flag=True)
def run_jobs(runner: str, apps_dir: Path, reader: str | None, migrate_only: bool) -> None:
    """Run every app job assigned to `runner` in stable order.

    Acquire a PostgreSQL advisory lease scoped to `runner`; return successfully if another
    execution owns it. Apply every app migration and reader grant while holding the lease.
    Stop after migration when `migrate_only` is true. Otherwise run each command as a child
    process in its app directory with its declared timeout and only its declared registered
    secrets. Attempt later jobs after a timeout, spawn error, or nonzero exit. Raise
    ClickException after completion with each failed `<app>.<job>` name. Raise UsageError if
    the runner has no jobs.
    """
```

File: `infra/marina/apps/evaldash/ingest.py`

```python
def main() -> int:
    """Reconcile every configured EvalDash record prefix into PostgreSQL once.

    Require a PostgreSQL Marina database configuration and the production Postgres store.
    Configure CoreWeave S3 credentials, preserve the existing per-prefix failure behavior,
    dispose the engine, and return zero after a pass whose prefixes all succeeded. After
    attempting and persisting every prefix, raise RuntimeError naming failed prefixes when
    any prefix listing failed. Configuration and database failures propagate.
    """
```

File: `infra/marina/apps/evaldash/app.py`

```python
class PgRecordStore(RecordStore):
    def refresh_if_due(self) -> bool:
        """Check for and install a newer catalog generation at most once per cache interval.

        Serialize checks and snapshot loads within the process. Install a fetched snapshot
        only when its generation is newer than the current generation. Clear catalog_error
        after a successful check. On failure, record the error, advance the retry deadline,
        and keep serving the last committed snapshot.
        """
```

Every PostgreSQL read method reaches `refresh_if_due()` through the store's snapshot boundary. `build_api(...) -> ASGIApp` returns the existing background-loop wrapper only for the local memory store. In PostgreSQL mode, `/status` reads persisted prefix status and `/refresh` reloads any committed generation, queues `EVALDASH_INGEST_JOB`, and returns HTTP 202 with the Cloud Run operation name.

## Cloud resources

The `marin-marina` stack declares:

- Cloud Run service `marina`: `cpu_idle=true`, startup CPU boost enabled, minimum instances `1`, maximum instances `4`, concurrency `8`, CPU `4`, memory `4Gi`.
- Cloud Run job `marina-hourly`: args `marina run hourly --reader <reader-group>`, timeout equal to the child timeout sum plus fixed overhead, retries `0`, CPU `4`, memory `4Gi`, Marina service account, Cloud SQL volume, database environment, and `MARINMIRROR_TOKEN`.
- Cloud Scheduler job `marina-hourly-trigger`: schedule `0 * * * *`, UTC, OAuth as the Marina service account, invoking `marina-hourly:run`.
- Cloud Run job `marina-evaldash`: args `marina run evaldash --reader <reader-group>`, CPU `1`, memory `1Gi`, and the two CoreWeave secrets.
- Cloud Scheduler job `marina-evaldash-trigger`: schedule `*/10 * * * *`, UTC, invoking `marina-evaldash:run`.
- Deploy command: execute `marina-hourly` with args `marina,run,hourly,--reader,<reader-group>,--migrate-only` and wait when the image reference changes. The service revision depends on this execution. Scheduled and deploy executions use the same runner lease and both migrate before app code.

The stack no longer declares `marina-migrate`, `marina-echo-sync`, or `marina-echo-sync-trigger`. Pulumi deletes those resources during apply.

## Errors

- Manifest loading raises `ValueError` with the manifest path for unknown job keys, missing fields, invalid names, empty command elements, duplicate secret names, or duplicate job names.
- Runner grouping raises `ValueError` naming the runner and conflicting schedules. Deployment raises `ValueError` for an unknown logical secret or a missing `hourly` runner.
- `marina run` raises `click.UsageError` for an unknown runner and `click.ClickException` after all commands finish if any child times out, cannot spawn, or exits nonzero.
- A scheduled execution that cannot acquire its runner lease exits successfully without running migrations or app commands.
- EvalDash catalog-refresh failures preserve the last loaded snapshot and appear in `/status` through `catalog_error`.

## Out of scope

- Parallel execution inside one runner.
- Per-job service accounts, network attachments, or retry policy.
- Scaling the Marina web service to zero.
- Applying the Pulumi stack or deleting live resources outside the normal deployment workflow.
