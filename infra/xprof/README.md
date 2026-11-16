# Hosted XProf

The always-on Iris job `/ops/xprof` serves:

```text
https://iris.oa.dev/proxy/xprof
```

Levanter writes XPlane profiles with optional HLO metadata under
`tmp/ttl=Nd/xprof/<run_id>` in the `MARIN_PREFIX` backend. The gateway only opens
`gs://` or `s3://` roots containing that `ttl=Nd/xprof/<run_id>` layout. Iris
authenticates browser requests. The gateway stages profiles under its workdir
and forwards viewer requests to an xprof-rs process bound to loopback. The
viewer is pinned to [xprof-rs v0.1.1](https://github.com/Locamage/xprof-rs/releases/tag/v0.1.1)
at source commit `133df8c837f5d19a4aef142dd735864df8592092`. The deploy
downloads its x86_64 Linux archive, checks SHA-256, and bundles a compressed
executable that is unpacked when the service starts.

The shared service disables `/capture_profile`, which otherwise connects to a
request-supplied gRPC address. It serves existing `.xplane.pb` profiles.
xprof-rs does not read `.xplane.riegeli` files. The service change does not
replace the optional Python `xprof.convert` package used by Marin's offline
profile summaries.

## Hosted profile workflow

Open the profile path in the hosted viewer and retain the run ID, storage root, and
profile format with the result. Large multi-host profiles can spend substantial time in
overview processing and trace-summary generation. A slow overview or summary indicates
profile-processing work; it does not establish that the training job stalled.

The hosted service has dedicated memory, disk, and proxy-timeout settings for these
profiles. xprof-rs parses the XPlane files and serves XProf's frontend. The
gateway retains the `/open?uri=...` and `/progress?uri=...` links emitted by
Levanter, including its asynchronous profile download.

## Deploy

Changes deploy automatically from `main` through
`.github/workflows/ops-pulumi-rollout.yaml`. Dispatch that workflow to redeploy the
current `main` revision with its GitHub-held credentials. A local deploy is for an
unmerged checkout and requires `CW_KEY_ID` and `CW_KEY_SECRET` in the operator
environment:

```bash
uv run --all-packages --extra deploy marin-deploy xprof rollout
```
