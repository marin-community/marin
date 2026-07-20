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

Result caching lives in the scratch bucket, so it **survives restarts**. Next to each spilled
result ducky writes a small `ducky/cache/<sql_hash>.meta.parquet` sidecar; on an exact-SQL repeat
it reads that sidecar back and returns the result without re-scanning the source — even after the
(preemptible) service restarts. The sidecar shares the `ducky/` prefix, so the scratch bucket's
lifecycle rule reaps it alongside the result it points at; a hit older than `result_ttl_days` is
ignored. Set `DUCKY_PERSIST_CACHE=0` to disable caching. Caching keys on the exact SQL text, so
identical SQL reuses a prior result even if the underlying data changed — use `use_cache: false`
to force a fresh read.

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

## Query log

Every submitted query is recorded to finelog under ducky's own namespace, `ducky.query` —
one row per query with its SQL text and terminal outcome (`status`, `cached`, `elapsed_ms`,
`total_rows`, `result_bytes`, `result_path`, `error`). It's a durable, queryable record of
what runs through ducky, so the query mix can be reviewed later for hot paths and
optimization opportunities. Writes are fire-and-forget (finelog buffers on a background
thread and swallows transport failures), so logging never blocks or fails a user's query;
if there's no in-cluster finelog endpoint it's disabled with a warning. The namespace is
itself queryable via the pre-baked `finelog."ducky.query"` view (see the **ducky query
history** example) — note the view only materializes once at least one segment exists and
ducky has (re)started to bind it.

## Pre-baked views & queries

ducky ships a small catalog of named views over common Marin data sources, plus
click-to-fill example queries surfaced in the dashboard (`GET /api/catalog`). The views
are ordinary DuckDB views, so you can `SELECT` from them without spelling out a
`read_parquet` glob:

- **finelog** (`finelog.log`, `finelog."iris.task"`, `finelog."iris.worker"`,
  `finelog."iris.task_status"`, `finelog."iris.profile"`, `finelog."iris.provisioning"`,
  `finelog."zephyr.stage"`, `finelog."zephyr.worker"`, `finelog."ducky.query"`) over
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
in `finelog_root`, not a silent bypass. To avoid the finelog egress entirely, set
`DUCKY_FINELOG_ROOT=` (empty) or point it at a same-region copy. (An empty
`DUCKY_ALLOWED_BUCKETS` still means allow-all.)

### Cross-region reads are opt-in

Read access on the literal URIs in a query is enforced in two stages:

- **`DUCKY_ALLOWED_BUCKETS`** is the outer bound — the prefixes ducky may read at all
  (default `gs://marin-,s3://marin-`, i.e. any marin GCS bucket plus the R2/CoreWeave S3
  stores). A URI outside it is hard-refused.
- Among the allowed URIs, whether a **GCS** read is same-region or egress-costly cross-region
  is decided at query time by `rigging.filesystem.is_cross_region_url`, which compares the
  bucket's live GCS location metadata against this VM's region (multi-region aware, and
  honoring the `MARIN_I_WILL_PAY_FOR_ALL_FEES` override). A cross-region GCS read must **opt
  in** with a leading comment:

  ```sql
  -- cross-region: allow
  select count(*) from read_parquet('gs://marin-us-central2/some/data/*.parquet');
  ```

  Without the comment the query is refused with a message telling you to add it. The directive
  must sit in the query's leading comment block (before the first statement line); `cross
  region` and `allowed` are accepted too.

**S3 reads (R2/CoreWeave) are never cross-region-gated** — rigging models GCS regions only, so
any allowed `s3://` URI is read freely. Pre-baked catalog views bypass the opt-in (their URIs
never appear literally in the SQL), and a literal `read_parquet` of a configured catalog root
(`finelog_root`/`datakit_root`) is likewise exempt — that egress is a deliberate config choice,
so the view and the literal read behave identically. An empty `DUCKY_ALLOWED_BUCKETS` disables
enforcement entirely (allow-all), which also makes the cross-region gate inert. Because region
is resolved from live metadata, no per-region tuning of the allowlist is needed when ducky is
deployed to a different region.

The gate **fails closed**: on a GCP VM, if a bucket's location can't be resolved (e.g. the
service account lacks `storage.buckets.get`, or a transient metadata failure), the read is
treated as cross-region and requires the opt-in — a metadata failure can't silently bypass the
gate. Only off-GCP (local smoke), where there is no VM region to compare against, is region
gating skipped entirely, leaving just the outer allowlist. So the deployed task's service
account needs `storage.buckets.get` on the marin buckets, or every GCS read will demand the
opt-in comment.

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

Ducky deploys through Pulumi: the `infra/ducky` project declares it as an always-on Iris
job (`iac.iris.service.IrisService`), with the job shape and `DUCKY_*` task environment in
the committed `infra/ducky/Pulumi.ducky-marin.yaml` and secret values in Secret Manager.
CI rolls the stack on merge to main (`ops-ducky.yaml`); to force a redeploy with unchanged
code, dispatch that workflow with a `deploy_generation` override. The deploy builds the
Vue dashboard itself on every roll (node/npm required on the deploying machine). For a
manual roll:

```bash
uv sync --all-packages --extra deploy
cd infra/ducky
pulumi login gs://marin-iac-state
pulumi stack select ducky-marin
pulumi up
```

The stack uses the shared `marin-iac-key` KMS secrets provider. The operator needs
`roles/cloudkms.cryptoKeyEncrypterDecrypter` on that key; no passphrase is used.

Runtime config still arrives as `DUCKY_*` env vars — see `config.py`; the stack yaml is
where the values are set.

## Notes

- ducky reads only object-store URIs on the bucket allowlist (`DUCKY_ALLOWED_BUCKETS`); local-file
  access is blocked. Queries against buckets outside it are refused before execution, and
  cross-region GCS reads among the allowed buckets need a `-- cross-region: allow` opt-in comment
  (see above).
- A datakit **clustered-store** bucket (`…/cluster=C/quality=Q/`) holds levanter
  `JaggedArrayStore` (zarr) caches, not parquet — `read_parquet` finds nothing there. Query
  the upstream parquet attribute datasets instead.
