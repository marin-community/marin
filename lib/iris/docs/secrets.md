# Secret delivery

Iris delivers entries in `EnvironmentSpec.secrets` as read-only files. By default,
the submitting client includes its current rigging secrets bundle and available
`HF_TOKEN`, `WANDB_API_KEY`, `CW_KEY_ID` and `CW_KEY_SECRET` defaults, using file-first lookup with environment
fallback. Explicit secrets or environment entries take precedence.
`SecretBehavior.ISOLATE` sends only explicitly supplied secrets.
Iris supplies no cluster default application credentials. Ordinary `env_vars`,
CLI `-e` values and cluster `defaults.task_env` retain environment delivery.

## Submit and read secrets

Use the default HF/W&B/CoreWeave forwarding, or pass other credentials explicitly:

```python
from iris.cluster.types import EnvironmentSpec
from rigging.secrets import require_secret

environment = EnvironmentSpec(
    secrets={"SERVICE_KEY": require_secret("SERVICE_KEY")},
    env_vars={"SERVICE_URL": "https://service.example"},
)
# Pass environment=environment to client.submit(...).
```

The application in the task reads `require_secret("SERVICE_KEY")`. Rigging
checks the named file under `/run/rigging/secrets`, then falls back to the same
environment variable if the file is absent. `RIGGING_SECRETS_DIR` selects another
directory, which is useful for local development. `get_secret` returns `None`
when absent; `require_secret` rejects missing or empty values. Unreadable or
malformed files fail immediately. `secret_path(name)` returns a path without
reading its value. `list_secret_names()` lists mounted names without reading
values; it excludes Kubernetes projection metadata and does not guess which
environment variables contain secrets.
If only an environment value exists, `secret_path` still returns the prospective
file path; it does not create a file. The CLI also supports source references without putting values in argv:
`iris job run --secret SERVICE_KEY env:SERVICE_KEY -- python app.py`.
Repeat `--secret NAME SOURCE` for other credentials; `-e` supplies environment values.

Fray exposes the same explicit `secrets` map through `create_environment` and
`EnvironmentConfig` for its Iris backend. Fray's in-process `LocalClient` runs
in the caller's environment; it provides no container or credential isolation.
Ordinary child jobs, Fray actors and Zephyr workers inherit the submitting
process's mounted bundle by default. The client resolves inheritance before
submission; the controller delivers the submitted map. Child values override
inherited values, and grandchildren read the resulting child's bundle.
The submitting process must have access to the files or environment sources it
forwards. To opt out, submit with:

```python
from iris.cluster.types import EnvironmentSpec
from rigging.secrets import SecretBehavior, require_secret

environment = EnvironmentSpec(
    secret_behavior=SecretBehavior.ISOLATE,
    secrets={"SERVICE_KEY": require_secret("CHILD_SERVICE_KEY")},
)
```

Iris and Fray use the same `rigging.secrets.SecretBehavior` enum.
Fray accepts it through `create_environment(secret_behavior=...)`. `create_actor` and
`create_actor_group` accept `environment=...` for overrides and isolation.
The CLI equivalent is `--secret-behavior isolate`.
`ISOLATE` affects only the secret map; ordinary environment inheritance is unchanged.

Ordinary child jobs also inherit job environment and setup scripts. Sandboxes
inherit neither parent `env_vars` nor secrets. A sandbox always receives only
its own explicit secrets, even with `INHERIT`, and no cluster credentials. See
[container profiles](container-profiles.md).

## Source references

Rigging also resolves explicit source references:

```python
from rigging.secrets import resolve_secret_spec

credential = resolve_secret_spec((
    "file:/private/service-key",
    "env:SERVICE_KEY",
    "gcp-secret://projects/example/secrets/service-key/versions/1",
)).value
```

Lists select the first present source. Missing env variables, missing files and
cloud NOT_FOUND responses advance to the next reference. Denied, unreachable,
unreadable or malformed sources fail immediately. An empty source is present.
The result's `.source` identifies the selected reference; `.value` contains the
credential and is excluded from its representation. Cloud sources require
`marin-rigging[secrets]` and access to the referenced version.
If every reference is absent, resolution raises `SecretResolutionError`.

Kubernetes Secrets and CSI providers can supply files using the same named
lookup without giving applications Kubernetes API access.

## Application clients

Iris runtimes use rigging's shared file configuration to set `HF_TOKEN_PATH`
before Python starts when a task receives `HF_TOKEN`. Hugging Face clients use
their native file discovery, including datasets and `hf://` filesystems. An
explicit `HF_TOKEN_PATH` is preserved.
Other clients can receive credentials at their application boundary, such as
`wandb.Settings(api_key=require_secret("WANDB_API_KEY"))`.
Training environment planning does not require a W&B key. Online authentication
errors surface when the logger initializes in the task; offline logging and
native W&B login remain available.

Rigging's S3 clients read named AWS credentials or an explicitly configured
fsspec credential set. Credentials remain together as one provider's key,
secret and optional token. Custom consumers must read rigging or receive a
credential explicitly. Supplying a value in `env_vars` keeps it in the process
environment even if its name looks sensitive.
The named AWS credentials are `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
the optional `AWS_SESSION_TOKEN`.
CoreWeave tasks require the submitter's complete `CW_KEY_ID`/`CW_KEY_SECRET`
pair. Rigging prepares a native AWS credentials file alongside the named files;
Iris sets `AWS_SHARED_CREDENTIALS_FILE` so boto3, TensorStore and other native
clients use the submitted pair. Explicit CW environment entries retain their
precedence and supply the same file. The output uploader receives only that
storage file, through a separate projection of the attempt Secret. Tasks never
mount the operator's storage Secret.

Supply complete pairs when overriding generic AWS credentials. Named AWS
secrets override storage credentials for rigging clients;
native boto3 clients continue using `AWS_SHARED_CREDENTIALS_FILE` unless the
application configures them explicitly. AWS values in `env_vars` override the
native file for both clients.

## Runtime and persistence

VM workers write per-attempt files on host tmpfs, with directory mode `0700`
and file mode `0400`, owned by the task user. Docker mounts the directory
read-only. Files are removed after the attempt; startup removes orphaned
directories while preserving adopted running attempts. ProcessRuntime remaps
the directory to its host path and retains its existing host environment; it is
a development backend, without container isolation.

Kubernetes creates an attempt Secret owned by the Pod. The task mounts it
read-only with mode `0440` and a supplemental group. A required volume prevents
startup before the Secret exists. Init containers and sidecars do not mount
application secrets. The output uploader mounts only the derived storage file.
Pod garbage collection and Iris cleanup remove the Secret.

Secret values still travel through authenticated job RPCs and remain in the
controller database and checkpoints. Display APIs expose secret names but
redact every value in the secret map. This change does not encrypt persisted job
state or introduce a vault. Restrict access to controller storage, backups and the Kubernetes API;
enable Kubernetes encryption at rest where supported. Authorized applications
and host administrators can read mounted credentials.

## Deployment and migration

Apply the IaC namespace Role/RoleBinding `iris-task-secret-delivery` for
per-attempt Secret creation before deploying the Kubernetes controller.
CI bootstraps these objects in `iris-ci`. Before the first IaC apply on
`cw-us-west-04a`, adopt its existing `task-secret-role` and
`task-secret-binding` resources using the
[Pulumi import procedure](https://github.com/marin-community/marin/blob/main/infra/pulumi/README.md#adopting-live-resources).
The other CoreWeave stacks create them normally. No GCP IAM changes are needed.

Deploy updated controller, worker and task images. For GCP workers, use
`iris --config <cluster.yaml> controller worker-restart`; see
[Iris operations](../OPS.md) for deployment commands. Existing running containers
keep their original environment. No HF/W&B cloud resources or grants are needed
unless an application chooses a cloud reference.

Remove `defaults.inject_env` and any former `defaults.secrets` or worker secret
source configuration. HF/W&B/CoreWeave credentials are forwarded from the user's client;
pass other application credentials in the root submission. Each submitting
client forwards its current bundle unless isolated. Operator object-storage
credentials remain in an infrastructure-only read-only mount for the controller
and storage health agents. Tasks and their output uploaders use the submitter's
credentials. Resume legacy jobs with personal CoreWeave keys and resubmit queued
jobs lacking them before rollout; existing submitted values are not refreshed.
Controller auth source references remain in
`auth.signing_key` and `auth.principal_directory`; Kubernetes deployment resolves
them into a controller-only Secret and renders file references.

Update all submitters and runtimes together: an old backend does not understand
the new secret map. SkyRL uses editable Iris/Rigging sources from the submitted
Marin checkout; no library publication is needed. Rotate a source, then submit a
new job to capture its new value; existing job retries retain the submitted value.
