# Hosted XProf

The always-on Iris job `/ops/xprof` serves:

```text
https://iris.oa.dev/proxy/xprof
```

Levanter writes XPlane profiles with optional HLO metadata under
`tmp/ttl=Nd/xprof/<run_id>` in the `MARIN_PREFIX` backend. The gateway only opens
`gs://` or `s3://` roots containing that `ttl=Nd/xprof/<run_id>` layout. Iris
authenticates browser requests.

## Hosted profile workflow

Open the profile path in the hosted viewer and retain the run ID, storage root, and
profile format with the result. Large multi-host profiles can spend substantial time in
overview processing and trace-summary generation. A slow overview or summary indicates
profile-processing work; it does not establish that the training job stalled.

The hosted service has dedicated memory, disk, and proxy-timeout settings for these
profiles. Its summary path decodes Perfetto trace JSON into compact typed events and
computes exclusive-time and breakdown data in one pass.
