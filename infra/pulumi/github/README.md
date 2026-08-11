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

## External runtime updater app

`Ops - External Dependency Update` uses a private, repository-scoped GitHub App instead of the
shared Nightshift app. The app may write repository contents and pull requests; it has no webhook,
OAuth, administration, or organization permissions. Its private key is available only through the
`external-runtime-updater` Actions environment, whose deployment policy accepts protected branches
and rejects pull-request branches. Pulumi makes that app the only integration that bypasses the
one-review rule. A separate ruleset has no bypass actors and requires GitHub Actions' own
`marin-integration`, `marin-lint`, `rust-checks`, and `unit-tests` check runs before any pull request
can merge; matching context names from another integration do not satisfy it.

GitHub App registration and initial installation require an organization owner to confirm them in
GitHub. GitHub does not expose that confirmation through the Pulumi provider. Bootstrap the app
once before enabling the stack resource:

1. Run `pulumi up` once with `externalRuntimeUpdater.enabled: false`. This creates the protected
   Actions environment needed to seal and hold the app key.
2. Open the [preconfigured organization app registration](https://github.com/organizations/marin-community/settings/apps/new?name=marin-external-runtime-updater&description=Advances%20Marin%27s%20immutable%20external%20runtime%20pins&url=https%3A%2F%2Fgithub.com%2Fmarin-community%2Fmarin&public=false&webhook_active=false&contents=write&pull_requests=write),
   verify that only Contents and Pull requests have read/write access, and create the private app.
3. Generate a private key on the app's General page. Install the app on `marin-community`, selecting
   only the `marin` repository. Record the app ID, the slug from its settings URL, and the numeric
   installation ID from the installation URL.
4. Seal the private key to the protected environment's Actions public key. `--no-store` prints
   ciphertext without creating the secret. Record the matching public-key ID:

   ```bash
   repository_id=$(gh api repos/marin-community/marin --jq .id)
   gh api "repositories/$repository_id/environments/external-runtime-updater/secrets/public-key" \
     --jq .key_id
   gh secret set EXTERNAL_RUNTIME_UPDATER_PRIVATE_KEY \
     --repo marin-community/marin --env external-runtime-updater --no-store \
     < /path/to/private-key.pem
   ```

5. In this directory, replace the disabled bootstrap object with the values from GitHub. The
   ciphertext is safe to commit: only GitHub can decrypt it, and Pulumi never receives the private
   key plaintext.

   ```yaml
   marin-github:externalRuntimeUpdater:
     enabled: true
     repository: marin-community/marin
     appId: 123456
     appSlug: marin-external-runtime-updater
     installationId: 789012
     reviewRulesetId: 785435
     actionsKeyId: example-key-id
     encryptedPrivateKey: example-base64-ciphertext
   ```

6. Run `pulumi preview`, confirm that the existing app installation and `protect main` ruleset are
   imports rather than replacements, then run `pulumi up`. Run the live credential audit afterward:

   ```bash
   pulumi preview
   pulumi up
   uv run --package marin-iac python audit.py --live
   ```

Delete the downloaded PEM after the ciphertext is recorded. To rotate the key, generate a new PEM,
repeat the sealing command with the current Actions public-key ID, update the two stack fields, and
run `pulumi up` again.
