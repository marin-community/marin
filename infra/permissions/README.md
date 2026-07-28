# Marin permissions

This Pulumi project owns additive IAM grants shared by deployment workflows. It does not own
the workload identity pool, state bucket, or KMS key — those are existing shared resources. Most
deploy accounts are existing too (created out-of-band; this stack only grants IAM on them). An
account with `create_account: true` is the exception: this stack owns it end-to-end, creating the
`google_service_account` alongside its grants.

The `hai-gcp-models` stack lets GitHub OIDC tokens for the `main` branch impersonate deployment
accounts. Each account can update the shared Pulumi state bucket and encrypt or decrypt stack
secrets with `marin-iac-key`. The IAM resources are protected, so a stack destroy fails instead
of removing workflow authentication.

The Iris CI account can read only the `iris-cw-us-west-04a-signing-key` secret needed to start
the CoreWeave smoke-test controller.

The Ducky account also receives `roles/iam.serviceAccountTokenCreator` from the same exact
GitHub subject. Rigging needs it to mint the service-account ID token accepted by Iris's IAP
edge; Grafana does not mint IAP tokens and does not receive that role.

The `pulumi-ci` account backs `ops-iac-preview.yaml` (`infra/pulumi/README.md`'s CI preview) and
is created by this stack (`create_account: true`) rather than pre-existing. It binds two
`github_subjects`: `pull_request` for the PR trigger, and `*main_subject` for
`workflow_dispatch` — whose OIDC subject follows the dispatching ref, not the event name, so a
manual run only authenticates when dispatched from `main`. Its grants are preview-only:
`kms_access: decrypt_only` (never encrypt/write secrets) and `state_access: preview` (read
state, write only the `.pulumi/locks/` prefix — never state content). Its custom
`marinGcpResourcePreviewer` role can read only the Artifact Registry repositories and reserved
addresses declared by the GCP stack, including the location metadata those reads require. Every
other account defaults to `kms_access: encrypt_decrypt` / `state_access: apply`, matching
`pulumi up`, and `github_subjects` normally holds just the one main-branch subject.

The Grafana deploy account can list Secret Manager metadata for its optional-secret probe. A
custom role lets it manage IAM policies on the four secrets wired into Cloud Run without
reading their payloads; the role is granted only on those secret resources. It can upload
images only to the `marin-grafana` Artifact Registry repository. A separate custom role lets
it read and update IAP policies on web services without granting access to those services or
control over IAP tunnels.

The secret IAM list is an explicit permission allowlist rather than a value derived from the
Grafana deployment. A newly wired runtime secret therefore fails closed until its deploy-account
IAM management is reviewed here.

## Apply

Select the existing stack and review its plan before applying changes:

```bash
uv sync --package marin-iac --extra deploy
cd infra/permissions
pulumi login gs://marin-iac-state
pulumi stack select hai-gcp-models
pulumi preview
pulumi up
```

The stack was bootstrapped directly with the shared `marin-iac-key` KMS provider. If the
backend metadata must be recreated, initialize it with:

```bash
pulumi stack init hai-gcp-models \
  --secrets-provider=gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key
```

A normal preview is a no-op. Review every create or update before applying because this stack
controls deployment identities and access to shared state, KMS, and service secrets.

The shared bucket and key are a deliberate trust boundary: either deploy account can access
state from other projects in the backend. Splitting state prefixes and keys requires a separate
backend migration; Grafana also reads the Cloud SQL stack through a stack reference.

## Human access inventory

[`user-access-inventory.yaml`](user-access-inventory.yaml) records a read-only snapshot of
Compute Admin and KMS access. It has no effect on GCP. Human access can be
managed here later with non-authoritative `gcp.projects.IAMMember` and
`gcp.kms.CryptoKeyIAMMember` resources after the entries have been reviewed. Do not use an
authoritative project or role binding: the live project contains unrelated members that such a
binding would remove.
