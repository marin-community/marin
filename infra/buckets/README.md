# Shared data buckets

This Pulumi project owns the shared Marin buckets and their complete lifecycle policies across
GCS, CoreWeave AI Object Storage, and Cloudflare R2. The `marin-buckets` stack is operated
manually because the CoreWeave provider requires a broad account token.

Bucket names, stores, locations, and endpoints come from the reviewed `data:` blocks in
[`config/marin.yaml`](../../config/marin.yaml) and
[`config/coreweave.yaml`](../../config/coreweave.yaml). The implementation is the shared
[`DataBuckets`](../pulumi/src/iac/buckets/__init__.py) component. GCS bucket IAM remains in the
`marin-iac/marin` stack with the other project grants.

## Credentials

Install the Pulumi providers and authenticate to GCP from the repository root:

```bash
uv sync --package marin-buckets
gcloud auth application-default login
gcloud auth application-default set-quota-project hai-gcp-models
```

Export an AI Object Storage account token only in the operator process:

```bash
export COREWEAVE_API_TOKEN=<coreweave-object-storage-token>
```

Cloudflare account tokens are stored in GCP Secret Manager. Use the read token for previews and
the write token for updates:

```bash
export CLOUDFLARE_API_TOKEN="$(gcloud secrets versions access latest \
  --project=hai-gcp-models --secret=cloudflare-r2-pulumi-read-token)"

export CLOUDFLARE_API_TOKEN="$(gcloud secrets versions access latest \
  --project=hai-gcp-models --secret=cloudflare-r2-pulumi-write-token)"
```

The provider credentials remain in the process environment. They are not Pulumi configuration,
resource inputs, or stack outputs. GitHub Actions and `pulumi-ci` have no access to them.

## Preview and apply

Run a credentialed preview:

```bash
pulumi -C infra/buckets preview --stack marin-buckets --diff
```

Stop on any bucket replacement or deletion. An approved update is applied manually:

```bash
pulumi -C infra/buckets up --stack marin-buckets
```

Cloudflare replaces the complete R2 lifecycle configuration. Audit the live rules before the
first lifecycle update:

```bash
curl -fsS \
  "https://api.cloudflare.com/client/v4/accounts/74981a43be0de7712369306c7b19133d/r2/buckets/marin-na/lifecycle" \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" | jq .result.rules
```

The declared policy preserves seven-day incomplete-multipart cleanup and the nine
`tmp/ttl=Nd/` expiration prefixes from `config/marin.yaml`.

## Automation boundary

Pull-request CI does not preview or update this stack. Reviewers use the code diff and an
operator-posted preview for changes under `infra/buckets` or `infra/pulumi/src/iac/buckets`.
Routine `marin-iac` previews do not initialize the CoreWeave or R2 providers.
