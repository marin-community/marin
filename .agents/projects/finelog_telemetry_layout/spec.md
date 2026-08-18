# Finelog Telemetry Layout Contract

## Public schema API

`lib/finelog/src/finelog/schema.py` and
`lib/finelog/rust/src/store/schema.rs` expose these additive schema fields:

```python
@dataclasses.dataclass(frozen=True)
class Schema:
    columns: tuple[Column, ...]
    key_column: str = ""
    projections: tuple[CoveringProjection, ...] = ()
    grouped_extrema: tuple[GroupedExtrema, ...] = ()
    sort_columns: tuple[str, ...] = ()
    max_row_group_rows: int = 0
```

`key_column` remains the event/range and retention key. `sort_columns` controls
physical compaction order; the compactor appends `seq`. An empty sort list uses
the historical key-plus-sequence order. `max_row_group_rows == 0` uses the
Finelog default. Explicit values must be between 16,384 and 1,048,576 rows.
Registration uses last-nonempty/nonzero-writer-wins semantics. Empty values from
older clients retain the registered policy; this API has no reset-to-default
operation.

Sort columns must exist, must not be maps, must not repeat, and must not name
the implicit `seq` column. Violations raise the existing schema-validation
error before registration mutates the catalog.

## Protobuf

Both Finelog schema proto copies define these fields with the same tags:

```proto
message Schema {
  repeated Column columns = 1;
  string key_column = 2;
  repeated CoveringProjection projections = 3;
  repeated GroupedExtrema grouped_extrema = 4;
  repeated string sort_columns = 5;
  uint32 max_row_group_rows = 6;
}
```

Older clients omit fields 5 and 6. New servers retain the registered layout in
that case. Older servers ignore the unknown protobuf fields.

## Telemetry HTTP resource

The version-1 telemetry request keeps its existing envelope. `resource` accepts
these optional strings in addition to `service` and `attributes`:

```text
run_id
job_id
execution_uid
region
node_name
process_index
```

For each field, explicit input wins. Otherwise Finelog reads the same key from
`resource.attributes`. Explicit
fields are inserted under their canonical keys in the serialized
`resource_attributes_json`, replacing a conflicting value at that key. The JSON
column is a canonicalized attribute map, not a copy of the original conflicting
envelope. Empty or oversized explicit values fail through the existing
telemetry string-validation error path.

Identity meanings are producer-independent:

| Field | Meaning |
|---|---|
| `run_id` | Stable logical training, evaluation, or serving run across scheduler attempts |
| `job_id` | Scheduler/root-job identity owning the resource |
| `execution_uid` | One execution or attempt of the job |
| `region` | Infrastructure region where the resource runs |
| `node_name` | Host or Kubernetes node identity |
| `process_index` | Producer process/rank index within the resource |

Values are opaque strings. Producers own their values and must keep them stable
for the lifetime of one telemetry resource.

## `telemetry_v1` persisted schema

The table adds nullable `COLUMN_TYPE_STRING` columns in this order after
`service`:

```text
run_id, job_id, execution_uid, region, node_name, process_index
```

Existing Parquets are projected onto the effective schema with nulls for these
columns. New rows populate them from the normalized resource dimensions. The
JSON columns remain present; explicit dimension conflicts are canonicalized as
described above.

Telemetry layout policy is:

```text
key_column: timestamp_ms
sort_columns: service, run_id, name, timestamp_ms
implicit tie-breaker: seq
max_row_group_rows: 131072
```

The row ceiling applies to L0 and merged Parquet writers. L0 preserves append
order; a multi-input compaction sorts its output by the configured columns and
`seq`. Single-input level promotions rename the existing file and preserve its
layout. The global Parquet layout revision remains unchanged, so this policy
does not guarantee complete convergence and does not schedule terminal files
for an immediate rewrite.

The `training-status` projection includes `run_id` and `job_id`.
The `training-run-attribution` projection includes `run_id`, `job_id`,
`node_name`, and `process_index`. Accelerator and node projections used by
first-party dashboards include `node_name` where needed.

## Exact index bundle

The exact-posting section keeps its existing payload encoding and changes its
method version from 1 to 2. Its header `coverage` bytes contain UTF-8 JSON:

```json
{
  "columns": {
    "name": ["phase", "step"]
  }
}
```

Column names are ordered map keys. Values are sorted and represent posting keys
present in that segment. Before loading the payload, the planner checks whether
at least one exact query constraint is covered. Version 1 preserves the old
payload-loading behavior. Missing, corrupt, or unknown-future coverage disables
that accelerator for the segment and leaves source scanning available.

Before constructing any source, projection, or postings plan, integer
predicates on the resolved key column are compared with the immutable segment
bounds captured in the same query snapshot as the paths. Disjoint segments are
omitted. Missing or invalid bounds retain the segment as a scan fallback. An
empty registered key hint resolves to the historical `timestamp_ms` key.

`needs_rebuild` requires method version 2 plus parseable coverage for each
configured exact column. The normal maintenance backfill therefore replaces
version-1 bundles without a blocking namespace migration.

## Catalog and rollback

Catalog JSON adds `sort_columns` and `max_row_group_rows`, both with defaults for
old catalog rows. New columns use an existing type and are nullable. Parquet
readers tolerate both the old and new column sets, and sorting/row-group changes
do not alter logical rows.

Rolling back to the prior binary is data-readable: unknown catalog JSON fields
are ignored, registered extra string columns remain in the effective schema,
and old writers null-fill them. HTTP write compatibility holds only while
producers remain attribute-only; the previous parser rejects explicit promoted
fields. The old process may replace version-2 `.fidx` bundles with version 1 and
does not apply the new sort policy. A later roll forward re-registers the policy
and rebuilds indexes. The `safe_deploy` digest rollback remains the required
operational path.

First-party Grafana queries require the promoted columns. Production rollout
must update Finelog servers before deploying those query changes. Training and
Zephyr alert queries use `COALESCE(job_id, json_get(...))` so pre-deploy and
rollback rows retain job identity. Other identity-filtered views contain only
post-deploy samples during transition and must roll back with their consumer if
the Finelog server rolls back.

## Out of scope

- Changing the local-hot/GCS-archive serving contract.
- Uploading `.fidx` bundles or filtered projections to GCS.
- Hive partitioning, service-specific physical files, Iceberg, or Lance.
- A forced rewrite of existing terminal telemetry Parquets.
- Catalog-bound timestamp file elimination and `LIMIT`-aware file ordering.
- A general arbitrary `run_tags` map or separate run-discovery catalog.
