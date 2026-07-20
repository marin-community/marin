# Marin permissions

This Pulumi project owns additive IAM grants shared by deployment workflows. It does not own
the existing GitHub workload identity pool, service accounts, state bucket, or KMS key.

The `hai-gcp-models` stack lets GitHub OIDC tokens for the `main` branch impersonate the Ducky
and Grafana deployment accounts. Each account can update the shared Pulumi state bucket and
encrypt or decrypt stack secrets with `marin-iac-key`. The IAM resources are protected, so a
stack destroy fails instead of removing workflow authentication.

The Ducky account also receives `roles/iam.serviceAccountTokenCreator` from the same exact
GitHub subject. Rigging needs it to mint the service-account ID token accepted by Iris's IAP
edge; Grafana does not mint IAP tokens and does not receive that role.

The Grafana deploy account can list Secret Manager metadata for its optional-secret probe. A
custom role lets it manage IAM policies on the four secrets wired into Cloud Run without
reading their payloads; the role is granted only on those secret resources. It can upload
images only to the `marin-grafana` Artifact Registry repository. A separate custom role lets
it read and update IAP policies on web services without granting access to those services or
control over IAP tunnels.

`secret_access_grants` maps a deploy service account to the secret payloads it resolves during
deployment. Add entries only after the named Secret Manager resources and versions exist. The
permissions stack uses resource-level `roles/secretmanager.secretAccessor` members; it does
not grant payload access across the project.

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
