# GitHub resources

This Pulumi project manages Marin's GitHub repository policy and records existing GitHub Actions
secrets as external resources. Declarations live in
[`Pulumi.marin-community.yaml`](Pulumi.marin-community.yaml); plaintext values do not.

Most secret resources use Pulumi lookups. The stack can read their metadata, but cannot create,
update, delete, or rotate them. Credentials with `management: sealed` are managed from ciphertext
sealed to the destination's GitHub Actions public key. `management: pulumi` records a secret owned
by another resource in this program, such as the dependency updater's environment secret.
The audit checks all three forms against workflow references and live GitHub scope metadata.

```bash
uv run --package marin-iac python infra/pulumi/github/audit.py
uv run --package marin-iac python infra/pulumi/github/audit.py --live

cd infra/pulumi/github
pulumi stack select marin-community  # use `stack init` on first setup
pulumi preview
pulumi up
```

The provider reads `GITHUB_TOKEN`; the stack config sets `github:owner`.

To add an externally managed secret, create or rotate it through an approved external path, then add
a `present` declaration. Record a pinned Secret Manager version when one exists; it is recovery
metadata and is never dereferenced by this program.

Use `management: sealed` when Pulumi should restore the declared value after GitHub-side drift.
Record `key_id` and `value_encrypted` from the same GitHub Actions public key. The ciphertext is safe
to commit because only GitHub can decrypt it. Keep a recoverable plaintext copy in Secret Manager;
do not put plaintext in Pulumi config, resource arguments, or state.

GitHub resolves a repository secret before an organization secret with the same name. Keep each
secret name at one scope unless a repository override has a separate owner and rotation path.

## Slack alert webhook

`SLACK_WEBHOOK_URL` uses a dedicated incoming webhook bound to `#marin-alerts`. Its recoverable
plaintext is `marin-github-alerts-slack-webhook` in Secret Manager; Pulumi stores only ciphertext
sealed to the organization Actions public key. Do not reuse `marin-grafana-slack-webhook`: that
retired webhook is bound to `#marin-eng`.

To rotate the webhook, add a Secret Manager version and seal that version to the organization
Actions public key without writing plaintext to disk:

```bash
gh api orgs/marin-community/actions/secrets/public-key --jq .key_id
gcloud secrets versions access <version> \
  --project=hai-gcp-models \
  --secret=marin-github-alerts-slack-webhook \
  | gh secret set SLACK_WEBHOOK_URL --org marin-community --no-store
```

Update the organization credential declaration with the printed public-key ID, ciphertext, and
pinned Secret Manager version:

```yaml
- name: SLACK_WEBHOOK_URL
  scope: organization
  presence: present
  management: sealed
  visibility: all
  key_id: <public-key-id>
  value_encrypted: <sealed-ciphertext>
  source_kind: gcp-secret
  source_ref: gcp-secret://projects/hai-gcp-models/secrets/marin-github-alerts-slack-webhook/versions/1
  disposition: keep
  note: Routes GitHub Actions alerts to #marin-alerts.
```

Run `pulumi preview`, `pulumi up`, and the live audit. A GitHub-side edit changes
`remote_updated_at`; the next update reapplies the sealed value.

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

The `LOOM_FORK_FERRY_PROFILE` repository variable is derived from the production Loom
stack's `githubFederationProfiles` output. Deploy the Loom stack first when changing the
mapping or profile name; the stack reference then keeps the workflow variable aligned with
the profile authorized by Loom.

## Dependency updater app

The external-runtime and native-package dependency workflows share the private
`marin-external-runtime-updater` GitHub App. The app may write repository contents, workflows, and
pull requests; it has no webhook, OAuth, administration, or organization permissions. Each
repository receives a separate `external-runtime-updater` environment whose deployment policy
accepts `main` and rejects pull-request branches. The same private key is sealed independently to
each environment's Actions public key.

`dependencyUpdater.repositories` is the installation allowlist. Pulumi selects exactly those
repositories, creates their environments and variables, adopts or creates their review rulesets,
and manages every configured environment ciphertext. The provider's installation-selection
resource requires a user-scoped organization-owner token and does not support GitHub App
authentication. Run this stack with an owner `GITHUB_TOKEN` that can administer every configured
repository and the App installation.

Pulumi verifies that the installed app identity matches the declared app, uses selected-repository
scope, is not suspended, and has Contents, Pull requests, and Workflows write permission. App
registration changes and permission approval remain owner operations because GitHub does not expose
the registration as a provider resource.

The app receives a pull-request-only bypass of the one-review ruleset. Organization admins retain
an always-on emergency bypass. Required-CI rulesets contain only the organization-admin bypass and
bind check names to GitHub Actions' integration ID. A repository with classic protection on `main`
must declare `classicRequiredChecks`; preview fails when a matching classic rule is present but
unmanaged. This preserves the second review layer that previously blocked an otherwise green updater
pull request; the [2026 updater bypass incident](https://echo.oa.dev/wiki/107) records that failure
and the required separation between review and CI bypasses.

Bootstrap a repository in two updates because an environment must exist before GitHub exposes its
public key:

1. Add the repository to `dependencyUpdater.repositories`. Set `reviewRulesetId` when adopting an
   existing review ruleset. Declare `classicRequiredChecks` only when importing an exact classic
   `main` rule. Omit `privateKey` during this first update.
2. Run `pulumi preview` and `pulumi up`. Inspect all imported protection changes. This creates the
   environment, restricts it to `main`, and reconciles the App installation allowlist.
3. Generate a new App private key and immediately create its recovery secret with the PEM. Later
   rotations add a version to the same secret.

   ```bash
   gcloud secrets create marin-external-runtime-updater-private-key \
     --project=hai-gcp-models \
     --replication-policy=automatic \
     --data-file=/path/to/marin-external-runtime-updater.pem
   ```

   ```bash
   gcloud secrets versions add marin-external-runtime-updater-private-key \
     --project=hai-gcp-models \
     --data-file=/path/to/marin-external-runtime-updater.pem
   ```

4. Seal the same PEM to each repository environment without creating the GitHub secret out of band.
   `--no-store` prints the ciphertext; record it with the public-key ID under that repository's
   `privateKey` declaration:

   ```bash
   updater_repository=marin-community/harbor
   updater_repository_id=$(gh api "repos/$updater_repository" --jq .id)
   gh api \
     "repositories/$updater_repository_id/environments/external-runtime-updater/secrets/public-key" \
     --jq .key_id
   gh secret set DEPENDENCY_UPDATER_PRIVATE_KEY \
     --repo "$updater_repository" \
     --env external-runtime-updater \
     --no-store \
     < /path/to/marin-external-runtime-updater.pem
   ```

   ```yaml
   - repository: marin-community/harbor
     reviewRulesetId: 19130649
     privateKey:
       actionsKeyId: example-key-id
       encryptedPrivateKey: example-base64-ciphertext
   ```

5. Run `pulumi preview`, `pulumi up`, and the live credential audit:

   ```bash
   pulumi preview
   pulumi up
   uv run --package marin-iac python audit.py --live
   ```

Delete the downloaded PEM after Secret Manager and every environment ciphertext are updated. Rotate
the key by adding a Secret Manager version, sealing it to every current environment public key, and
updating all `privateKey` declarations in one reviewed change.
