---
date: 2026-07-24
system: gcp
severity: degraded
resolution: mitigated
pr: none
issue: none
---

# TL;DR

- Grafana fired `WebhookEndpointsEmpty`, `K8sClusterUnreachable`, and related fleet alerts for all three CoreWeave clusters after a token rotation.
- `/tmp/ops-token` ended with `\n`. Secret Manager preserved that byte, and Cloud Run injected it unchanged into `CW_READ_TOKEN`.
- `kubectl` token-file tests passed because client-go trims token-file whitespace. The Grafana bridge passed the raw environment value to `httpx`, which rejected the Authorization header before sending a request.
- Secret version 3 removed the newline. Cloud Run revision `marin-grafana-cw-token-v3` returned healthy values on the next evaluation cycle with no Kubernetes query failures.
- Secret versions 1 (admin/write) and 2 (malformed read token) were disabled after version 3 passed the live checks.
- The rejected header appeared in Cloud Logging. The token must be replaced even though the service is functional.
- Local hardening strips `CW_READ_TOKEN` at load time and prevents transport exception text from entering logs.

# Original problem report

Grafana sent `WebhookEndpointsEmpty cw-rno2a` and
`WebhookEndpointsEmpty cw-us-east-08a` for
`kueue-system/kueue-webhook-service` at 12:26 PM Pacific. The operator asked to
prioritize the alert fix over the ongoing CoreWeave token-management work.

# Investigation path

1. Direct reads with `/tmp/ops-token` showed one Ready and Serving EndpointSlice
   and one Ready `kueue-controller-manager` replica on `cw-us-east-02a`,
   `cw-us-east-08a`, and `cw-rno2a`. The token could list EndpointSlices.

2. The bridge's exact `K8sFleet.alert_webhook_ready()` path returned `value=1`
   for all three clusters. This ruled out Kueue readiness and Kubernetes RBAC.

3. Cloud Run logs from revision `marin-grafana-cw-token-v2` showed repeated
   `Illegal header value b'Bearer <redacted>\\n'` failures on every Kubernetes
   path. The application logged the complete malformed header.

4. The earlier access audit had used `kubectl` with `tokenFile:
   /tmp/ops-token`. Client-go removed the trailing newline, so `/version`,
   namespace enumeration, and authorization reviews succeeded.

5. Secret Manager version 3 was created from the same token with CR/LF removed.
   Revision `marin-grafana-cw-token-v3` became Ready at
   `2026-07-24T19:30:20Z` and received 100% of traffic.

6. The new revision logged zero Kubernetes failures. The post-ready webhook
   evaluation at `2026-07-24T19:31:18Z` sent three notifications after the
   bridge returned healthy values, clearing the stale firing set.

# User course corrections

- The investigation was moving from live RBAC into durable IaC and CoreWeave
  token automation. The operator reported the alert storm and asked to fix it
  first. This exposed the malformed runtime secret before more time was spent
  on the separate token-lifecycle design.

# Root cause

`BridgeConfig.from_environment()` previously copied `CW_READ_TOKEN` without
normalization. Secret Manager and Cloud Run preserve secret bytes, including a
terminal newline. `httpx` rejected the resulting Authorization header locally,
so every bridge read failed without reaching a Kubernetes API server.

`K8sSource._get()` also copied `httpx.TransportError` text into `K8sError`.
httpcore's invalid-header exception included the rejected Authorization header,
which put the bearer token in Cloud Logging.

# Fix

The live repair added Secret Manager version 3 without CR/LF and deployed
`marin-grafana-cw-token-v3`.

Each Grafana cluster received a token-specific `marin-grafana-node-reader`
ClusterRole and ClusterRoleBinding. The role contains only `get`, `list`, and
`watch` on `nodes`. `infra/pulumi/src/iac/coreweave/rbac.py` and the three
CoreWeave cluster configs now declare these objects for Pulumi adoption.

`infra/grafana/src/config.py:229` now strips the environment value:

```python
cw_read_token=(os.environ.get("CW_READ_TOKEN") or "").strip() or None
```

`infra/grafana/src/k8s_source.py:247` now records only the transport exception
class, never its text. Regression coverage lives in
`infra/grafana/tests/test_config.py` and
`infra/grafana/tests/test_k8s_source.py`. The focused suite passed 43 tests.

The token remains usable but appeared in Cloud Logging. Mint a replacement,
update the token-specific Kubernetes RBAC subject, add a newline-free Secret
Manager version, and deploy a fresh revision.

# How OPS.md could have shortened this

Add a credential-rotation check near the CoreWeave Kubernetes guidance in
`lib/iris/OPS.md`: reject CR/LF in bearer-token files before loading them into
environment-backed secrets, and smoke the same runtime client used by the
service. Include this non-secret check:

```bash
test "$(tr -d '\r\n' < TOKEN_FILE | wc -c | tr -d ' ')" = \
  "$(wc -c < TOKEN_FILE | tr -d ' ')"
```

Also note that `kubectl --token-file` trims whitespace and therefore does not
validate byte-for-byte behavior of a Cloud Run secret environment variable.

# Artifacts

- `infra/grafana/src/config.py`
- `infra/grafana/src/k8s_source.py`
- `infra/grafana/tests/test_config.py`
- `infra/grafana/tests/test_k8s_source.py`
- `infra/pulumi/src/iac/coreweave/rbac.py`
- https://grafana.oa.dev/
