# Hosted XProf

The production XProf viewer runs as the always-on Iris job `/ops/xprof`. Pulumi
owns its job specification, retry policy, resources, and stable endpoint:

```text
https://iris.oa.dev/proxy/xprof
```

Training profiles are not copied through the deploy. Levanter uploads XPlane and
optional HLO files to `tmp/ttl=Nd/xprof/<run_id>` in the storage backend selected
from `MARIN_PREFIX`, then logs an `/open?uri=...` URL. The viewer validates the
bucket and TTL path, stages the tree on its Iris disk in a background task, and
redirects the browser to the complete XProf application. Background staging keeps
large profiles outside the Iris proxy's per-request timeout.

The service accepts only the Marin GCS and CoreWeave object-storage buckets listed
in `Pulumi.xprof-marin.yaml`. It rejects non-TTL paths before storage access. The
GCP Iris workload identity supplies GCS access; Pulumi resolves the CoreWeave access
key and secret from the deploy environment and injects them into the job. The Iris
controller proxy supplies user authentication.

## Deploy

Merges to `main` that touch the service or its storage dependencies run
`.github/workflows/ops-xprof.yaml`. To deploy manually:

```bash
uv sync --package marin-iac --extra deploy --frozen
pulumi login gs://marin-iac-state
cd infra/xprof
pulumi stack select xprof-marin
pulumi preview
pulumi up
```

Initialize the backend state once before the first workflow or manual deploy:

```bash
pulumi stack init xprof-marin \
  --secrets-provider='gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key'
```

The initialization adds this stack's encrypted data key to
`Pulumi.xprof-marin.yaml`; commit that generated `encryptedkey` field.

`CW_KEY_ID` and `CW_KEY_SECRET` must be available to the deploy process. They are
resolved before the Iris submission and do not enter Pulumi state. Increment
`xprof:deploy_generation` to resubmit a wedged service whose code and configuration
are otherwise unchanged.

The local staging cache is disposable. An Iris restart clears it; the TTL object
tree remains the source of truth and is staged again on the next open.

`infra/xprof/config.py` pins the service's XProf package because the gateway wraps
its standalone WSGI surface. On an XProf upgrade, verify the proxy-path rewrite
against the real compressed `index.html` and `bundle.js` before changing that pin.
