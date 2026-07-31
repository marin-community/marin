# Hosted XProf

The always-on Iris job `/ops/xprof` serves:

```text
https://iris.oa.dev/proxy/xprof
```

Levanter writes XPlane profiles with optional HLO metadata under
`tmp/ttl=Nd/xprof/<run_id>` in the `MARIN_PREFIX` backend. The gateway only opens
`gs://` or `s3://` roots containing that `ttl=Nd/xprof/<run_id>` layout. Iris
authenticates browser requests.
