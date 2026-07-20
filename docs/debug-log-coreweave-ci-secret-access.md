# Debugging log for CoreWeave CI secret access

Restore the CoreWeave smoke test after GitHub credentials became available.

## Initial status

The controller reached GCP authentication, then received
`secretmanager.versions.access` denied for `iris-cw-us-west-04a-signing-key`.

## Hypothesis 1

The `iris-ci-smoke@hai-gcp-models.iam.gserviceaccount.com` account lacks payload access to the
signing key resolved during `iris cluster start`.

## Changes to make

Grant `roles/secretmanager.secretAccessor` only on that secret through the shared permissions
Pulumi stack.

## Results

The production preview contained one `SecretIamMember` create and 19 unchanged resources. The
apply completed, and the secret IAM policy now contains the scoped accessor binding. CoreWeave CI
still needs to rerun against the applied policy.

## Future work

- [ ] Replace the remaining JSON service-account credential with GitHub OIDC.
