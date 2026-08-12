# GitHub resources

This Pulumi project manages Marin's GitHub repository policy and records existing GitHub Actions
secrets as external resources. Declarations live in
[`Pulumi.marin-community.yaml`](Pulumi.marin-community.yaml); plaintext values do not.

Most secret resources use Pulumi lookups. The stack can read their metadata, but cannot create,
update, delete, or rotate them. The dedicated updater's environment secret is managed from
GitHub-sealed ciphertext. The audit checks both forms against workflow references and live GitHub
scope metadata.

```bash
uv run --package marin-iac python infra/pulumi/github/audit.py
uv run --package marin-iac python infra/pulumi/github/audit.py --live

cd infra/pulumi/github
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

## Dependency updater app

The external-runtime and native-package dependency workflows share a private, repository-scoped
GitHub App instead of the Nightshift app. It may write repository contents and pull requests; it has
no webhook, OAuth, administration, or organization permissions. Its private key is available only
through the `external-runtime-updater` Actions environment, whose deployment policy accepts `main`
and rejects pull-request branches.

Pulumi gives the app a pull-request-only bypass of the one-review rule and no required-CI bypass.
Organization admins retain an always-on emergency bypass on both rulesets. The CI ruleset requires
GitHub Actions' own `marin-integration`, `marin-lint`, `rust-checks`, and `unit-tests` runs; matching
context names from another integration do not satisfy it.

App registration and installation remain owner-managed because GitHub's repository-selection
endpoint requires a user-scoped token unsuitable for unattended Pulumi runs. The installation must
select only `marin`. To recreate or rotate the app credential:

1. Verify that the app has only Contents and Pull requests read/write permission and remains
   installed only on `marin`. Record the app ID and slug from its settings page.
2. Generate a private key and seal it to the protected environment's Actions public key. `--no-store`
   prints ciphertext without creating the secret. Record the matching public-key ID:

   ```bash
   repository_id=$(gh api repos/marin-community/marin --jq .id)
   gh api "repositories/$repository_id/environments/external-runtime-updater/secrets/public-key" \
     --jq .key_id
   gh secret set DEPENDENCY_UPDATER_PRIVATE_KEY \
     --repo marin-community/marin --env external-runtime-updater --no-store \
     < /path/to/private-key.pem
   ```

3. Update `externalRuntimeUpdater` in `Pulumi.marin-community.yaml` with the app metadata, public-key
   ID, and sealed ciphertext. The ciphertext is safe to commit: only GitHub can decrypt it, and
   Pulumi never receives the private key plaintext.

   ```yaml
   marin-github:externalRuntimeUpdater:
     repository: marin-community/marin
     appId: 123456
     appSlug: marin-external-runtime-updater
     reviewRulesetId: 785435
     actionsKeyId: example-key-id
     encryptedPrivateKey: example-base64-ciphertext
   ```

4. Run `pulumi preview`, verify the ruleset bypass actors, then run `pulumi up` and the live audit:

   ```bash
   pulumi preview
   pulumi up
   uv run --package marin-iac python audit.py --live
   ```

Delete the downloaded PEM after the ciphertext is recorded. To rotate the key, generate a new PEM,
repeat the sealing command with the current Actions public-key ID, update the two stack fields, and
run `pulumi up` again.
