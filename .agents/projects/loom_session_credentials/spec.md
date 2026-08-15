# Loom configuration ownership contract

## Loom Settings surface

The existing route query ids remain stable. Only labels and component placement change.

| Query id | Label | Contents |
|---|---|---|
| `agents` | Agents | Profiles, profile environment, MCPs, custom agents, metadata settings |
| `sessions` | Sessions | Server and session lifecycle settings |
| `github` | Connections | GitHub App, GitHub behavior, Slack connection and behavior |
| `watches` | Watches | Watch settings |
| `workspace` | Workspace | Editor and terminal appearance |
| `environment` | Session environment | Readable non-secret values on the default profile |
| `access` | Access | Current user, password, personal GitHub PAT, approved users, Loom API tokens, authentication settings |
| `diagnostics` | Diagnostics | Logs, background tasks, and build information |

`GithubConnectionPanel.vue` owns GitHub App configuration. `AccountPanel.vue` no longer reads or writes deployment GitHub App configuration.

The personal PAT creation URL is:

```text
https://github.com/settings/personal-access-tokens/new?name=Loom&description=Interactive%20Loom%20sessions&contents=write&issues=write&pull_requests=write
```

The URL is guidance, not validation. Token storage remains write-only. Existing sessions are unchanged after token save or removal.

## Documentation contract

`docs/configuration.md` defines the four owners: user, profile, deployment, and repository. It separately documents registered-setting precedence. Credential docs use `personal GitHub token`, `profile GH_TOKEN`, and `GitHub App installation token`; they do not call all three “the GitHub token.”

## Marin Pulumi input

```python
@dataclass(frozen=True)
class HomeFileConfig:
    path: str
    project: str
    secret: str
    version: str
    mode: str

    @classmethod
    def parse(cls, path: str, value: object) -> "HomeFileConfig": ...

    def manifest(self) -> dict[str, str]: ...
```

`DeploymentConfig` adds:

```python
home_files: tuple[HomeFileConfig, ...] = ()
```

Pulumi key `homeFiles` maps a relative path to an object with:

- `secretRef`: the sole secret input, a required full GCP Secret Manager version resource with a numeric version; parsing decomposes it into the dataclass's `project`, `secret`, and `version` fields;
- `mode`: required four-digit octal string, allowed values `0400` and `0600`.

Paths use POSIX separators, are relative to `/home/app`, and reject empty segments, `.`, `..`, NUL, a leading slash, duplicates, and ancestor/descendant overlap within one manifest.

The VM metadata key `loom-home-files` contains sorted JSON entries:

```json
[
  {
    "mode": "0600",
    "path": ".kube/coreweave-iris",
    "project": "hai-gcp-models",
    "secret": "loom-coreweave-iris-kubeconfig",
    "version": "1"
  }
]
```

## Materialization contract

`infra/loom/materialize_home_files.py` has a host-side prepare command:

```text
materialize_home_files.py prepare --image=<digest> --manifest=<manifest-file> --state-dir=/var/lib/loom/home-files
```

The prepare command:

1. validates the complete manifest before changing the volume;
2. resolves each Secret Manager version into a mode-0600 temporary host file;
3. runs the pinned Loom image as root, without a network, with the target volume mounted at `/home/app`; Docker seeds a new named volume from the image's `/home/app`, and the command fails when the mounted path is absent or is not a directory;
4. uses directory file descriptors with `O_NOFOLLOW` to reject a target path that crosses a symlink;
5. copies through a temporary file, restores the uid and gid read from the mounted `/home/app`, and atomically renames it with the declared mode;
6. removes only regular files present in the previous ledger and absent from the new manifest;
7. writes `/var/lib/loom/home-files/managed-paths.json` atomically with mode `0600` and managed paths only; the root-owned state directory is mounted into the applicator but not ordinary sessions;
8. deletes all host temporary payloads on exit.

`infra/loom/startup-script.sh` pulls the pinned image, reads the manifest, stops the `loom` and `caddy` Compose services, runs `prepare`, and calls `docker compose up -d` only after preparation succeeds. An empty manifest prunes all previously managed files and leaves all unmanaged home contents unchanged. A stale managed file that blocks a new managed directory, or vice versa, is removed before staging; empty ancestors are pruned only as needed, and unmanaged content makes the transition fail closed. Any validation, Secret Manager, or install failure leaves Loom and Caddy stopped and the activation unsuccessful.

Detached session containers are not stopped. During rotation, readers holding an open file descriptor continue reading the previous payload after the atomic rename; later opens see the new version. Payload bytes never appear in command arguments, stdout, metadata, or the ledger.

## IAM

The Loom VM service account receives `roles/secretmanager.secretAccessor` on each distinct `(project, secret)` referenced by a profile or home file. The Pulumi program does not create secret versions.

## Production kubeconfig migration

The code PR does not upload the kubeconfig. After review, an operator creates `loom-coreweave-iris-kubeconfig`, uploads the existing `/home/app/.kube/coreweave-iris` as numbered version 1, and adds the `homeFiles` entry. A preview must show the secret IAM member and `loom-home-files` metadata. Activation replaces the existing path atomically.

## Errors

- Invalid paths, modes, or unpinned references raise `ValueError` during Pulumi program evaluation.
- Missing or inaccessible secret versions fail the startup script with the reference and target path but no payload.
- A symlink in the target path fails activation without following or deleting it.

## Out of scope

- Per-profile filesystem isolation or file visibility.
- Creating, reading, or rotating secret payloads from Pulumi.
- Validating all GitHub fine-grained PAT permissions.
- Removing existing profile or daemon `GH_TOKEN` fallbacks.
- Changing registered-setting precedence or route query ids.
