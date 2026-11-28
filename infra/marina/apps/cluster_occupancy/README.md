# Cluster Occupancy

A Marina app for one job: show which users held which parts of the CoreWeave
clusters (cw-us-east-08a, cw-us-east-02a, cw-rno2a) at which time. Nodes are
blocks colored by the user holding them, grouped by NVLink-domain rack where
the node reports one; a time slider drags through history and a picker sets
the window start in US Eastern (UTC on hover).

## Data sources

- **Now** — the slider's right end is the live snapshot. Every cluster's
  controller database answers read-only SELECTs through the Iris raw-query
  RPC (running tasks joined with jobs, job_config and the current task
  attempt), so rno2a maps in real time even though its controller does not
  publish mirror rows. The legend counts GPUs held right now.
- **History** — anything left of "now" reads the Finelog `marin` hub's
  `iris.controller_db__*` mirrors, taking the latest snapshot per key by
  `ts`. A one-hour window follows the reference query's membership rule; the
  legend counts GPU-hours in the window. Node capacity is the most GPUs seen
  held on a node at once, and nodes never observed are not shown.

Coverage is explicit: a cluster or period the mirror does not reach shows a
hatched "no data" state, never an idle-looking map. rno2a history is always
marked unavailable.

Read-only everywhere: the app never writes Finelog namespaces or controller
state, and usernames shown inside the logged-in app come straight from the
data. Tests and journeys use synthetic users only.

## Layout

- `backend/app.py` — Finelog-backed API (`/coverage`, `/occupancy`, `/now`,
  `/health`) with short-TTL caches for coverage and node inventory.
- `backend/live.py` — the live controller read and its normalization.
- `web/` — Vue 3 + Vite frontend on the shared Marina tokens.
- `tests/` — API tests against an embedded Finelog server and fixture rows.
- `journeys/` — browser journeys (load, hover, slider, picker, unmirrored
  states, light/dark contrast) run by `marina journey cluster_occupancy` in
  CI.

## Environment

- `OCCUPANCY_FINELOG_URL` / `OCCUPANCY_IAP_CLUSTER` — point the mirror client
  at a specific hub; otherwise it resolves `finelog-marin` through GCE.
- `OCCUPANCY_LIVE_ROWS_FILE` — fixture rows (JSON: cluster → row lists) that
  stand in for the controllers; tests and journeys set it, production never
  does.

Live controller reads resolve each cluster config and credentials the way the
`iris` CLI does; production deployment needs those credentials available to
the Marina runtime.
