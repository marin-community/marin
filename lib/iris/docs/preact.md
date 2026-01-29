# Iris Dashboard: Preact+HTM Architecture

## Decision

Replace inline HTML/CSS/JS in Python strings with Preact+HTM components served as static ES modules. No build step, no npm, no node — vendored ES modules with browser-native `importmap` resolution.

## File Structure

```
lib/iris/src/iris/cluster/static/
├── vendor/
│   ├── preact.mjs           (Preact 10.25.4)
│   ├── preact-hooks.mjs     (Preact Hooks)
│   └── htm.mjs              (HTM 3.1.1)
├── shared/
│   ├── styles.css           (consolidated CSS)
│   ├── rpc.js               (controllerRpc/workerRpc Connect RPC wrappers)
│   ├── utils.js             (formatBytes, formatDuration, formatRelativeTime, stateToName, etc.)
│   └── components.js        (InfoRow, InfoCard shared UI components)
├── controller/
│   ├── app.js               (main app: tab routing via hash, data fetching, refresh)
│   ├── jobs-tab.js          (jobs table with progress bars, sorting, pagination)
│   ├── workers-tab.js       (workers table with health indicators)
│   ├── endpoints-tab.js     (endpoints table)
│   ├── vms-tab.js           (VMs table flattened from scale groups, pagination)
│   ├── autoscaler-tab.js    (status, scale groups, actions log, logs)
│   ├── job-detail.js        (job info cards, task logs viewer, tasks table)
│   └── vm-detail.js         (VM info cards, bootstrap logs)
├── worker/
│   ├── app.js               (stats bar + task table via workerRpc)
│   └── task-detail.js       (task status, resources, build info, logs tabs)
    └── log-viewer.js        (shared log viewer with prefix filter, auto-refresh)
```

## Python Layer

Python dashboard files serve thin HTML shells via `html_shell(title, app_script)` from `dashboard_common.py`. Each shell includes:
- A `<div id="root">` mount point
- An `importmap` mapping `preact`, `preact/hooks`, `htm` to vendored ES modules
- A `<script type="module">` entry point

Data flows via Connect RPC — both controller and worker JS call their respective RPC services directly.

## Key Files

| File | Role |
|------|------|
| `dashboard_common.py` | `html_shell()`, `static_files_mount()`, logs page |
| `controller/dashboard.py` | Page routes + `/health` + `/api/logs` + RPC mount |
| `worker/dashboard.py` | Page routes + `/health` + `/api/logs` + RPC mount |

## Testing

- `pytest lib/iris/tests/cluster/controller/test_dashboard.py` — controller API + page tests
- `pytest lib/iris/tests/cluster/worker/test_dashboard.py` — worker RPC + page tests
- `uv run lib/iris/scripts/screenshot-dashboard.py --output-dir /tmp/screenshots` — visual validation
