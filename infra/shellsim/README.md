Build, check and export every guest port using Iris BATCH workers on `cw-us-east-02a`.
Assets and catalogs publish to durable `s3://marin-us-east-02a/marin/shellsim/ports`;
snapshots share assets under `https://ports.oa.dev`. The pinned worker image matches
Shellsim 0.1.35; update the locked runtime and image together.

```sh
MARIN_PREFIX=s3://marin-us-east-02a/marin \
  uv run python -m infra.shellsim.ports --source-revision <full-sha> --version 2026.10.12 --run
```

Omit `--run` to print the plan. Use a new version when source or runtime changes.
