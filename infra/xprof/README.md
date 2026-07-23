# Hosted XProf

The always-on Iris job `/ops/xprof` serves:

```text
https://iris.oa.dev/proxy/xprof
```

Levanter writes XPlane and optional HLO files under
`tmp/ttl=Nd/xprof/<run_id>` in the `MARIN_PREFIX` backend. The service accepts
only the buckets in `Pulumi.xprof-marin.yaml` and rejects paths outside that TTL
prefix. Iris authenticates browser requests. The service uses workload identity
for GCS and injected credentials for CoreWeave S3.

## Deploy

`.github/workflows/ops-xprof.yaml` deploys changes merged to `main`. To deploy
manually:

```bash
uv sync --package marin-iac --extra deploy --frozen
pulumi login gs://marin-iac-state
cd infra/xprof
pulumi stack select xprof-marin
pulumi preview
pulumi up
```

Before the first deploy, initialize the stack:

```bash
pulumi stack init xprof-marin \
  --secrets-provider='gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key'
```

Commit the generated `encryptedkey` field in `Pulumi.xprof-marin.yaml`.

The deploy requires `CW_KEY_ID` and `CW_KEY_SECRET`. Increment
`xprof:deploy_generation` to resubmit the service without a code or configuration
change.

`infra/xprof/config.py` pins XProf because the gateway calls its standalone WSGI
API. Before upgrading it, test the proxy rewrite against XProf's compressed
`index.html` and `bundle.js`.
