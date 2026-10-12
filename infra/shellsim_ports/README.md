Serve `https://ports.oa.dev` from durable `s3://marin-us-east-02a/marin/shellsim/ports`.
The deployment requires the existing `shellsim-ports-readonly` secret containing
prefix-scoped `CW_KEY_ID`/`CW_KEY_SECRET`, plus ambient Kubernetes, Cloudflare, and
registry credentials. Cilium permits Traefik and labeled trusted Iris clients;
egress permits cluster DNS and HTTPS to the configured bucket's CoreWeave host.

Initialize the stack once with the repository KMS provider, then preview:

```sh
cd infra/shellsim_ports
pulumi stack init cw-us-east-02a --secrets-provider=gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key
pulumi preview --stack cw-us-east-02a
```

Preview does not build or push the image. Apply only after separate authorization.
