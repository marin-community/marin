# Hosted XProf

The always-on Iris job `/ops/xprof` serves:

```text
https://iris.oa.dev/proxy/xprof
```

Levanter writes XPlane profiles with optional HLO metadata under
`tmp/ttl=Nd/xprof/<run_id>` in the `MARIN_PREFIX` backend. The gateway can open
any `gs://` or `s3://` path available to its GCS workload identity or CoreWeave
S3 credentials. Iris authenticates browser requests.
