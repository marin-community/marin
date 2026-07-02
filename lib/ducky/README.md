# ducky

Ad-hoc DuckDB SQL over object-store parquet, as an always-on Iris service. Paste SQL
in the dashboard or hit the API; the full result spills to a TTL'd GCS path and a capped
preview comes back inline.

Dashboard: <https://iris.oa.dev/proxy/ducky/>

## Query API

Queries run asynchronously: `POST /query {"sql": …}` returns `202 {"query_id"}`, then poll
`GET /result/{query_id}` until `status != "running"`.

```
POST /proxy/ducky/query   {"sql": "SELECT 42 AS answer"}
  → 202 {"query_id": "…"}
GET  /proxy/ducky/result/<id>
  → {"status":"done","columns":["answer"],"rows":[[42]],"total_rows":1,
     "result_path":"gs://…/ducky/<hash>.parquet","elapsed_ms":267}
```

Pass `{"use_cache": false}` on the POST to force a fresh run (by default an identical prior
query's result is reused).

### From the CLI (auto-tunnel)

```bash
ducky query --cluster marin "SELECT count(*) FROM read_parquet('gs://marin-…/*.parquet')"
```

Opens a controller tunnel for you, prints a table (or `--format json`); `--no-cache` forces
a fresh run. No IAP token needed — the tunnel is loopback-trusted.

### Directly through IAP (no tunnel)

The service sits behind IAP at `https://iris.oa.dev/proxy/ducky/`. Send a Google-signed
OIDC ID token as a bearer. **The audience must be the desktop OAuth client**
(`MARIN_DESKTOP_OAUTH_CLIENT` in `rigging/auth.py`) — IAP's browser-redirect client-id is
*not* an accepted bearer audience and returns
`401 Invalid bearer token. Audience doesn't match the allowlisted oauth clients`. The
caller's identity (service account or user) must already be IAP-authorized.

```python
import time
import httpx
import google.auth.transport.requests
import google.oauth2.id_token

AUD = "748532799086-qf8m6mvovtdmd71npm07gk1ohijsr3q5.apps.googleusercontent.com"  # MARIN_DESKTOP_OAUTH_CLIENT
token = google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), AUD)
headers = {"Authorization": f"Bearer {token}"}
base = "https://iris.oa.dev/proxy/ducky"

qid = httpx.post(f"{base}/query", json={"sql": "SELECT 42 AS answer"}, headers=headers).json()["query_id"]
while True:
    result = httpx.get(f"{base}/result/{qid}", headers=headers).json()
    if result["status"] != "running":
        break
    time.sleep(1)
print(result["columns"], result["rows"])
```

`gcloud auth print-identity-token --audiences=$AUD` mints the same token for a service-account
credential. `Proxy-Authorization: Bearer …` works interchangeably with `Authorization`.

## Deploy

```bash
uv run ducky deploy --cluster marin        # builds the dashboard, auto-tunnels, submits
```

Replaces a running instance by default; `--keep` makes it an idempotent watchdog resubmit
(only recreates a gone/terminal job). Config comes from `DUCKY_*` env vars — see `config.py`.

## Notes

- ducky reads only object-store URIs on the bucket allowlist (`DUCKY_ALLOWED_BUCKETS`);
  local-file access is blocked. Queries against buckets outside the allowlist are refused
  before execution.
- A datakit **clustered-store** bucket (`…/cluster=C/quality=Q/`) holds levanter
  `JaggedArrayStore` (zarr) caches, not parquet — `read_parquet` finds nothing there. Query
  the upstream parquet attribute datasets instead.
