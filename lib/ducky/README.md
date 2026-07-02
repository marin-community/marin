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

## Pre-baked views & queries

ducky ships a small catalog of named views over common Marin data sources, plus
click-to-fill example queries surfaced in the dashboard (`GET /api/catalog`). The views
are ordinary DuckDB views, so you can `SELECT` from them without spelling out a
`read_parquet` glob:

- **finelog** (`finelog.log`, `finelog."iris.task"`, `finelog."iris.worker"`,
  `finelog."iris.task_status"`, `finelog."iris.profile"`, `finelog."iris.provisioning"`,
  `finelog."zephyr.stage"`, `finelog."zephyr.worker"`) over
  `DUCKY_FINELOG_ROOT/<namespace>/seg_L*.parquet`. The namespace directory names contain
  dots, so quote them: `SELECT * FROM finelog."iris.task"`.
- **datakit** normalized parquet — a curated subset (`datakit.finetranslations`,
  `datakit.finepdfs`, `datakit.nemotron_cc_v2_high_quality`, …) over
  `DUCKY_DATAKIT_ROOT/<name>_<hash>/outputs/main/*.parquet`. The full ~100-source set
  lives in `lib/marin/src/marin/datakit/sources.py`; use the dashboard's **Browse
  normalized datasets** example query to list what's present.

Roots are config (`DUCKY_FINELOG_ROOT` / `DUCKY_DATAKIT_ROOT`); set either to empty to
disable that source's views. The datakit root defaults to `<MARIN_PREFIX>/normalized` (the
datakit corpus is canonical in eu-west4); finelog defaults to its marin deployment at
`gs://marin-us-central2/finelog/marin`.

Configuring a catalog root **declares that prefix readable**: the roots join
`DUCKY_ALLOWED_BUCKETS` to form the effective read-allowlist, so a pre-baked view and a
literal `read_parquet` of the same prefix behave identically (a view can't reach a bucket a
literal URI couldn't). This is how finelog stays queryable even though it lives in
us-central2, cross-region from a us-east5 ducky — the egress is a deliberate choice encoded
in `finelog_root`, not a silent bypass, and *other* buckets in that region stay gated. To
avoid the finelog egress entirely, set `DUCKY_FINELOG_ROOT=` (empty) or point it at a
same-region copy. (An empty `DUCKY_ALLOWED_BUCKETS` still means allow-all.)

Views are bound (and their parquet footers cached) at startup; a view over an
absent/unreachable dataset is logged and skipped rather than failing the service. The
`/api/catalog` endpoint and dashboard advertise only the views that were actually created. Repeat
queries are fast because ducky enables DuckDB's parquet-footer cache
(`parquet_metadata_cache`) and HTTP metadata cache, so re-reading the same files skips the
footer round-trips.

The example queries are written to be cost-aware: DuckDB reads parquet column-wise with
projection and predicate pushdown, so the bytes fetched scale with the columns and row
groups a query touches. The big `data`/`text` blob dominates size (≈80% of the finelog
`log` table), so the examples project explicit small columns, prefer `count(*)`/`GROUP BY`
over a small column (answered from footers / cheap columns), and truncate text with
`left(...)` under a `LIMIT` — never a bulk `SELECT *` over a multi-GB dataset.

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
