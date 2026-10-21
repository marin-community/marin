Build, check and export every guest port using Iris BATCH workers on `cw-us-east-02a`.
Assets and catalogs publish to durable `s3://marin-us-east-02a/marin/shellsim/ports`;
snapshots share assets under `https://ports.oa.dev`. The CLI uses the worker image
from the `cw-us-east-02a` Pulumi stack; `--worker-image` overrides it for experiments.

```sh
MARIN_PREFIX=s3://marin-us-east-02a/marin \
  uv run python -m infra.shellsim.ports --source-revision <full-sha> \
    --version 2026.10.12 --run
```

Omit `--run` to print the plan. Use a new version when source or runtime changes.

Pulumi builds one Shellsim image, runs its proxy command, and exports the same
digest as `workerImage`. The proxy requires the `shellsim-ports-readonly`
secret (`CW_KEY_ID`/`CW_KEY_SECRET`) and ambient provider credentials. Cilium limits
ingress to Traefik/trusted Iris clients and egress to DNS/the bucket endpoint.

```sh
cd infra/shellsim
pulumi stack init cw-us-east-02a --secrets-provider=gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key
pulumi preview --stack cw-us-east-02a
```

Preview does not build or push images. Apply requires separate authorization.
