# OpenBao to CoreWeave credential pilot

This probe tests whether an OpenBao workload identity can exchange its JWT for
temporary CoreWeave AI Object Storage credentials. It checks that the resulting
principal can read one test prefix, then keeps one S3 client running past its
first credential's expiration.

Run this on a disposable host with Docker Engine 28 or later, Caddy, uv, and a
public DNS name you control. Keep port 18200 blocked from external access.
Use a new private CoreWeave bucket named `marin-openbao-pilot-<suffix>`
with no bucket policy or broad organization grant covering the pilot role.
Use the bucket's availability zone as the signing region. An organization
administrator must register the issuer and create the test policy.

## Start the issuer

From this directory, set the public hostname and a unique test bucket name.

```bash
export PILOT_ISSUER_HOST=openbao-pilot.example.net
pilot_bucket=marin-openbao-pilot-example
pilot_dir=$(mktemp -d)
(umask 077; openssl rand -hex 32 > "$pilot_dir/root-token")
export BAO_DEV_ROOT_TOKEN_ID="$(cat "$pilot_dir/root-token")"
docker run --rm -d --name marin-openbao-pilot --log-driver none \
  --publish 127.0.0.1:18200:8200 --env BAO_DEV_ROOT_TOKEN_ID \
  ghcr.io/openbao/openbao:2.7.2 server -dev -dev-listen-address=0.0.0.0:8200
unset BAO_DEV_ROOT_TOKEN_ID
curl --fail --silent --retry 5 --retry-connrefused --noproxy '*' \
  http://127.0.0.1:18200/v1/sys/health > /dev/null

uv run --script pilot.py bootstrap \
  --public-origin "https://$PILOT_ISSUER_HOST" --bucket "$pilot_bucket" \
  --root-token-file "$pilot_dir/root-token" --credentials-file "$pilot_dir/approle.json" \
  > "$pilot_dir/bootstrap.json"
```

OpenBao runs in memory and listens on loopback at the host boundary. Bootstrap
creates one AppRole with permission to mint identity tokens for the pilot and
writes its credentials to a private file. Its secret ID expires after four
hours. The root token is only used during bootstrap. Read `bootstrap.json`
for the issuer, audience, JWKS URL, and CoreWeave policy. AppRole login creates
the entity whose UUID appears in `sub`. Restarting OpenBao creates a new
signing key and entity, so remove the
old federation configuration and policy before bootstrapping again.
[OpenBao identity tokens](https://openbao.org/docs/secrets/identity/identity-token/)

Run `caddy run --config Caddyfile` on the same host with
`PILOT_ISSUER_HOST` set and TCP ports 80 and 443 reachable for public HTTPS.
The proxy exposes only the GET discovery and JWKS paths. From another machine,
fetch the discovery URL at `<issuer>/.well-known/openid-configuration` and its
advertised `jwks_uri` before registering federation.

## Configure the test bucket and federation

1. As the test bucket owner, upload `allowed/read.txt` and `outside/read.txt`
   with the exact bytes from `printf 'openbao-pilot\n'`. Keep both objects in
   the bucket for the run so the outside-prefix check exercises authorization
   on an existing object.
2. In CoreWeave Workload Federation, register the `issuer` and `audience`
   from `bootstrap.json` exactly. Use
   `jq '{policy}' "$pilot_dir/bootstrap.json" > "$pilot_dir/policy.json"`
   for the new organization access policy request. The policy grants the
   derived `role/<issuer>:<entity UUID>` principal credential exchange and
   `s3:GetObject` on `<test-bucket>/allowed/*`.
   [CoreWeave OIDC federation](https://docs.coreweave.com/products/storage/object-storage/auth-access/workload-identity-federation/use-oidc-tokens)
   [Create a CoreWeave organization policy](https://docs.coreweave.com/products/storage/object-storage/auth-access/organization-policies/manage)
3. Run the probe on the OpenBao host with your CoreWeave organization ID and
   the test bucket's availability zone.

```bash
uv run --script pilot.py probe --credentials-file "$pilot_dir/approle.json" \
  --organization "$coreweave_org_id" --region "$coreweave_zone"
```

The probe requires a successful sentinel read and explicit `AccessDenied` for
the outside-prefix read, put, and delete. Put and delete target separate,
random, unused keys under `allowed/`. The probe refreshes the OpenBao JWT file
while the SDK manages storage credential refresh. It passes only after a read
starts past the first storage credential's expiration and the same client has
exchanged a different JWT for a new key with a later expiration. The default
wait limit is 90 minutes. Set `--max-wait-minutes` above the initial credential
lifetime if needed.

## Read the sentinel through fsutil

The probe uses an isolated botocore environment to test the CoreWeave exchange
and SDK refresh directly. Use the same test identity to exercise Rigging's
bucket-routed client. From this directory, declare the disposable bucket in a
local cluster config and mint a fresh JWT.

```bash
pilot_config_dir="$HOME/.config/marin/clusters"
mkdir -p "$pilot_config_dir"
pilot_config="$pilot_config_dir/$pilot_bucket.yaml"
(set -C; cat > "$pilot_config" <<EOF
data:
  scheme: s3
  region_buckets:
    pilot: {bucket: $pilot_bucket, store: coreweave, signing_region: $coreweave_zone}
EOF
)

uv run --script pilot.py issue-token --credentials-file "$pilot_dir/approle.json" \
  --token-file "$pilot_dir/oidc.jwt"
AWS_CONTAINER_CREDENTIALS_FULL_URI="https://api.coreweave.com/v1/cwobject/temporary-credentials/oidc/$coreweave_org_id" \
AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE="$pilot_dir/oidc.jwt" \
CW_S3_ENDPOINT="https://cwobject.com" \
  uv run --package marin-rigging fsutil cat --raw "s3://$pilot_bucket/allowed/read.txt" \
  > "$pilot_dir/read-back.txt"
printf 'openbao-pilot\n' | cmp - "$pilot_dir/read-back.txt"
```

Levanter uses the same token file and credential endpoint for TensorStore
operations on declared CoreWeave buckets. TensorStore 0.1.85 accepts the empty
`Token` field in CoreWeave's credential response. Later operations can use a
refreshed credential. Retries within an operation can retain the credential
that operation started with, so an in-flight operation can still fail at expiry.

## Remove the pilot

Remove the dedicated CoreWeave access policy and Workload Federation
configuration. Delete the test bucket and all pilot objects, including any put
object created by an unexpectedly authorized probe. Stop Caddy and run
`docker stop marin-openbao-pilot`. Remove the temporary directory and its
root token and AppRole files, and remove the local cluster config if created.
