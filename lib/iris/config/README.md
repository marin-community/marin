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
  user_roles:
    "principal:human-079": user
```

The deployed controller image must include `auth.user_roles` support in both
Python and Rust.

Before deployment, resolve the references with the existing IAM tool:

```bash
uv run --package marin-iac --extra deploy python infra/pulumi/iam_principal.py \
  render-iris lib/iris/config/marin.yaml --output /private/configs/marin.yaml
```

The output directory must exist outside the checkout. The command creates a new
mode-`0600` file containing the resolved emails. Pass that file to the normal Iris
rollout command with `--config`. Controllers reject unresolved auth references.
Regenerate the deployment files from the current tracked configs for each rollout;
the account policy remains in Git.
