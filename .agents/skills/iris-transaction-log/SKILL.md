---
name: iris-transaction-log
description: Query the Iris controller transaction log (job/task CREATE/UPDATE/DELETE history). Use when asked to inspect what changed on the controller, who/what mutated a job or task, or when investigating state regressions.
---

# Skill: Query Iris Transaction Log

The Iris controller records every job/task mutation (CREATE, UPDATE, DELETE)
into a `txn_actions` SQLite table. Entries are exposed via the
`GetTransactions` RPC and surfaced on the dashboard `Transactions` tab. This
skill wraps that RPC for agent use.

Each `TransactionAction` has:

- `timestamp` — RFC3339 timestamp
- `action` — e.g. `CREATE`, `UPDATE`, `DELETE`
- `entity_id` — job or task ID
- `details` — JSON string with action-specific fields

Entries are returned in reverse chronological order.

## Fetching transactions

Use the generic RPC CLI. Request a large limit and filter client-side with
`jq` — the server RPC only supports a `limit` field.

```bash
# Most recent 50 transactions (server default)
iris rpc controller get-transactions | jq '.actions'

# Pull a larger window for filtering
iris rpc controller get-transactions --limit 2000 > /tmp/txns.json
```

The response is JSON of the form `{"actions": [ {timestamp, action, entity_id, details}, ... ]}`.

## Filtering

All filters run client-side against the fetched window. If your window is too
small, increase `--limit`.

### By entity ID (job or task)

```bash
jq --arg id "job-abc123" '.actions | map(select(.entity_id == $id))' /tmp/txns.json
```

### By action type

```bash
# Only deletes
jq '.actions | map(select(.action == "DELETE"))' /tmp/txns.json

# Creates or updates
jq '.actions | map(select(.action == "CREATE" or .action == "UPDATE"))' /tmp/txns.json
```

### By time range

Timestamps are RFC3339 strings; string comparison is safe.

```bash
jq --arg since "2026-04-07T00:00:00Z" \
   '.actions | map(select(.timestamp >= $since))' /tmp/txns.json

jq --arg since "2026-04-07T00:00:00Z" --arg until "2026-04-07T12:00:00Z" \
   '.actions | map(select(.timestamp >= $since and .timestamp < $until))' /tmp/txns.json
```

### Combined

```bash
jq --arg id "task-xyz" --arg since "2026-04-07T00:00:00Z" '
  .actions
  | map(select(.entity_id == $id and .timestamp >= $since))
  | map(. + {details: (.details | fromjson)})
' /tmp/txns.json
```

The trailing `fromjson` step parses the `details` field for readable output.

## Tips

- The `details` field is a JSON-encoded string. Use `fromjson` in `jq` to
  inspect its structure.
- To find every mutation touching a specific job, filter on `entity_id`
  without constraining `action`.
- If you need more than the most recent few thousand entries, paginate by
  re-running with a larger `--limit`; the RPC does not currently support
  cursor-based pagination.
- The dashboard `TransactionsTab.vue` is the visual equivalent; prefer this
  skill for programmatic inspection in agent contexts.

## Related

- `debug-iris-controller` — deeper controller state inspection via offline
  snapshots and process RPC.
- `debug-iris-job` — runtime debugging of a single task/container.
