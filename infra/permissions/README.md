# Marin permissions (retired)

This Pulumi project no longer owns or applies IAM grants. The `marin` stack in
[`infra/pulumi`](../pulumi/README.md) is the sole repository owner of IAM on
`hai-gcp-models`, including deployment identities, shared state, KMS, Secret Manager, and IAP
access. Declare changes in `infra/pulumi/src/iac/gcp/iam_data.yaml` through the central grant
workflow.

Do not preview or update the retired `hai-gcp-models` permissions stack. Its program exits with
a migration error so IAM ownership cannot diverge again. Historical configuration and the
read-only `user-access-inventory.yaml` remain here only for migration reference.
