# GitHub resources

This Pulumi project records Marin's existing GitHub Actions secrets as external resources.
Declarations live in [`Pulumi.marin-community.yaml`](Pulumi.marin-community.yaml); values do not.

Secret resources use Pulumi lookups. The stack can read their metadata, but cannot create, update,
delete, or rotate them. The audit checks declarations against workflow references and live GitHub
scope metadata.

```bash
uv run --package marin-iac python infra/pulumi/github/audit.py
uv run --package marin-iac python infra/pulumi/github/audit.py --live

cd infra/pulumi/github
pulumi login gs://marin-iac-state
pulumi stack select marin-community  # use `stack init` on first setup
pulumi preview
pulumi up
```

The provider reads `GITHUB_TOKEN`; the stack config sets `github:owner`.

To add a secret, create or rotate it through an approved external path, then add a `present`
declaration. Record a pinned Secret Manager version when one exists; it is recovery metadata and is
never dereferenced by this program.

To remove a secret, first confirm it is an unreferenced `remove-candidate` with the live audit. Delete
it externally, then remove its declaration. For example:

```bash
for secret in GCP_SA_KEY GCP_SA_LOGGING_KEY; do
  gh secret delete "$secret" --repo marin-community/marin
done
```

Keep service-account JSON and SSH credentials until their workflows use OIDC. GitHub variables,
environments, repository settings, and other non-secret resources can be added here as normally
managed Pulumi resources.
