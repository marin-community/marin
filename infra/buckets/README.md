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

The declared R2 policy preserves seven-day incomplete-multipart cleanup and the nine
`tmp/ttl=Nd/` expiration prefixes from `config/marin.yaml`.

CoreWeave bucket lifecycle configurations apply the same nine expiration prefixes and abort
incomplete multipart uploads after seven days.

`marin-us-east-08a` is shared scratch storage for every CoreWeave cluster. It is declared as
`data.stores.coreweave.temp_bucket` in `config/coreweave.yaml`, separately from the region
mirror set. Permanent east-region artifacts remain in `marin-us-east-02a`; `hero-checkpoints`
is reserved for hero checkpoints.

Before switching existing deployments, import the existing `marin-us-east-08a` bucket into
this stack and apply its lifecycle and access policy. Review the preview for bucket
replacement or deletion. The standard prefixes affect only `tmp/ttl=Nd/` objects; existing
data stays at its current location. Roll out the CoreWeave task defaults after the bucket
policy is active. Already running jobs retain their environment.

Installed consumers need a `marin-rigging` release containing both the routing code and its
bundled `config/coreweave.yaml`. Update MarinSkyRL's exact rigging override and lockfile;
update Harbor's and Evalchemy's rigging dependency floors and lockfiles. These dependencies
come from PyPI, so changing their Marin Git pins alone does not update storage routing.

CoreWeave bucket access policies preserve S3 access for principals in the Open Athena organization.
Only `marin-us-east-02a` denies every principal the `s3:DeleteObject` and
`s3:DeleteObjectVersion` actions under `marin/grug/hero-*`. The `hero-checkpoints` bucket and
temporary checkpoint staging paths remain deletable. Before a policy update, audit each bucket's
live access policy; Pulumi replaces the complete bucket policy.

## Automation boundary

Pull-request CI does not preview or update this stack. Reviewers use the code diff and an
operator-posted preview for changes under `infra/buckets` or `infra/pulumi/src/iac/buckets`.
Routine `marin-iac` previews do not initialize the CoreWeave or R2 providers.
