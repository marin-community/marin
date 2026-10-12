Build every guest port, including guest Clang, through Buildomatic's BATCH workers
and sccache in `s3://marin-us-east-02a/marin/shellsim/buildomatic/v2`. Checks precede
publication. Release catalogs live under `marin/shellsim/ports/releases/<version>`;
assets and the stable catalog use the durable `marin/shellsim/ports` prefix, outside
the bucket's scratch TTL prefixes. Assets retain immutable URLs and are copied only
when missing.

Run from a Marin checkout with the locked Shellsim runtime. Its wheel omits recipes,
so provide the full Shellsim source commit SHA containing the matching ports workflow.
The worker image is pinned to the Buildomatic 0.1.35 image; update the runtime and
image together when native compiler identity changes. The default prints a plan:

```sh
MARIN_PREFIX=s3://marin-us-east-02a/marin \
  uv run python -m experiments.ports.release --source-revision <full-sha> --version 2026.10.12
```

Add `--run` to execute. Worker count, CPUs, RAM, disk, and persistent driver work
directory are CLI options. Use a new release version for changed source or runtime.
This workflow does not apply Pulumi changes.
