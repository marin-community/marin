Build and check every guest port using Iris BATCH workers on `cw-us-east-02a`.
Build caches stay on CoreWeave; releases publish to
`s3://marin-na-public/shellsim/ports`, served directly from R2 at
`https://public.oa.dev/shellsim/ports`. Readers need no storage credentials.

```sh
MARIN_PREFIX=s3://marin-us-east-02a/marin \
  uv run python -m infra.shellsim.ports --source-revision <full-sha> \
    --version 2026.10.23 --run
```

Omit `--run` to print the plan. Use a new version when source or runtime changes.
The CLI reads `workerImage`, `publicationPrefix` and `url` from the
`cw-us-east-02a` Pulumi stack. Explicit `--worker-image`, `--publication-prefix`
and `--url-prefix` override those outputs.

The publisher needs `CW_KEY_ID`/`CW_KEY_SECRET` for the build store and
`R2_KEY_ID`/`R2_KEY_SECRET` with object read/write access to `marin-na-public`. Credentials
are needed only while publishing; rotating them does not interrupt downloads.
Export uploads missing shared assets before writing a versioned catalog, then
promotes the stable catalog. Assets are immutable and cached for one year;
catalogs and package indexes require revalidation. Pin
`releases/<version>/catalog.json` for a reproducible environment.

Apply the shared `infra/buckets` stack first. Its `publicUrl` and `publicPrefix`
outputs supply the release locations through a StackReference. The Shellsim
stack builds the worker image and removes its former Kubernetes proxy, ingress,
network policy and DNS record. For an existing deployment, first build the
pinned image with `pulumi up --target <ports-image-URN>`; use
`pulumi stack --show-urns` to find that resource. Publish with explicit
`--publication-prefix s3://marin-na-public/shellsim/ports` and
`--url-prefix https://public.oa.dev/shellsim/ports`, then verify the catalog
and a package download before applying the full stack update. This rollout
sequence is manual. Keep `--source-revision` and the stack's
`source_revision` on a Shellsim revision with S3 HTTP publication metadata and
directory-index objects.

```sh
cd infra/shellsim
pulumi preview --stack cw-us-east-02a
```

Preview does not build or push images. Apply requires separate authorization.
