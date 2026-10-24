# Iris cluster configs

`iris --cluster=<name>` resolves `<name>` against the top-level `*.yaml` files
here (plus `~/.config/marin/clusters`, which wins on conflict). Subdirectories
are not searched.

Naming convention:

- **Live clusters** — bare names. GCP: `marin.yaml`, `marin-dev.yaml`.
  CoreWeave: `cw-<region>.yaml` (e.g. `cw-us-east-02a.yaml`, `cw-rno2a.yaml`).
- **`ci-*.yaml`** — CI and test-harness configs; not real long-lived clusters.
- **`examples/`** — reference templates. Deliberately outside name resolution;
  pass an explicit path to use one.

`auth.user_roles` assigns roles to verified IAP email addresses. An entry takes
precedence over `auth.admin_users` and the default role. Supported roles are
`admin`, `user`, and `dashboard`.

To keep a personal email out of Git, use its encrypted IAM principal reference
in `auth.user_roles`, `auth.admin_users`, `auth.allowed_submitters`, or
`user_budgets[].user_ids`:

```yaml
auth:
  principal_directory: gcp-secret://projects/hai-gcp-models/secrets/iris-principal-directory/versions/latest
  user_roles:
    "principal:human-NNN": user
```

Replace `human-NNN` with the account's registered IAM principal ID.
The deployed controller image must include `auth.user_roles` support in both
Python and Rust.

The `marin` Pulumi stack derives the directory from `iam_data.yaml` and publishes
it in Secret Manager. The payload is secret in previews and encrypted in state;
see [Pulumi conventions](../../../infra/pulumi.md).
Apply that reviewed stack before deploying configs that use the directory.

GCP controllers load the reference at startup using their service account.
On CoreWeave the deploy operator reads the same reference and writes the value
into the `iris-controller-secrets` Kubernetes Secret. The rendered config uses
a controller-only file reference; tasks do not mount this Secret. Operators need
directory-read access; ordinary submitters do not need it for login or submission.
Use tracked configs for normal start and restart commands. Preflight rejects a
missing, malformed, or incomplete directory before stopping a controller.
Startup resolves role, admission, and budget identities before persistent-state
changes. ConfigMaps and VM bootstrap configs retain the references.

Pass application credentials explicitly through `EnvironmentSpec.secrets`; see
[secret delivery](../docs/secrets.md). Cluster configs have no application secret
defaults. `task_env`, `.marin.yaml` and job `-e` values use environment delivery.
