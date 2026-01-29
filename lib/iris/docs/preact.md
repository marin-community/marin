# Iris Dashboard: Preact+HTM Migration

**Issue:** [#2525](https://github.com/marin-community/marin/issues/2525)
**Approach:** Preact+HTM (Option B from issue)
**Log:** `lib/iris/logs/dashboard-preact/summary.md`

# Multi-stage execution

<coordinator>

You are a top-level agent working on a complicated plan. Your job is to
orchestrate sub-agents to do work. You do minimal work yourself: even if you see
a lint warning, send the ml-engineer to fix it.

When talking to sub-agents, tell them they are not the coordinator and give them
instructions in a <subagent>...</subagent> block. Subagent instructions always
link to the high-level plan as well as provide instructions for where to write
progress logs (described below), in addition to any context you think is
relevant.

Always pass this full plan to subagents with their appropriate context at the end.

Use the following agents for your work:

- ml-engineer: lints, simple migrations and refactorings
- senior-engineer: larger scoped tasks and complex changes requiring judgement, validation of results.

Apply the following process:

* Maintain an execution log in logs/plan-name/summary.md .
With each change in status, append to the log the changes made and any issues or concerns you have encountered.

* Break down your work into many fine-grained tasks. Each task should be
independently testable and verifiable, and contribute meaningfully to the sucess
of the plan.

* For each task:
  - Send off the task, along with the high-level plan summary and link to the planning document to the appropriate sub-agent(s) for the task
  - On completion, send the task changes to the senior-engineer for validation
  - Send any requested changes to the a sub-engineer for fixes
  - Continue until the senior-engineer is satisified.
  - Subagents should maintain their own log in logs/plan-name/step-name.md

* When you believe the plan is complete, send the plan and the execution log to the senior-engineer for final review.

Remember:

* No making changes yourself, you are a coordinator, not an executor.
* Provide sub-agents with the planning document, and summary of the current task.
* When compacting conversations, be sure to re-read the summary log

ALWAYS provide sub-agents with @AGENTS.md @lib/iris/AGENTS.md @lib/iris/README.md as context
</coordinator>

## Overview

Replace ~2400 lines of inline HTML/CSS/JS in Python strings with Preact+HTM components served as static ES modules. Python dashboard files shrink from ~2400 → ~210 lines (API endpoints + static serving only). No build step, no npm, no node.

## Target File Structure

```
lib/iris/src/iris/cluster/static/
├── vendor/
│   ├── preact.mjs           (~4KB)
│   ├── preact-hooks.mjs     (~1KB)
│   └── htm.mjs              (~700B)
├── shared/
│   ├── styles.css           (consolidated CSS from all templates)
│   ├── rpc.js               (controllerRpc/workerRpc wrappers)
│   └── utils.js             (escapeHtml, formatBytes, formatDuration, formatRelativeTime, stateToName)
├── controller/
│   ├── app.js               (main app: tab routing, data fetching)
│   ├── jobs-tab.js
│   ├── workers-tab.js
│   ├── endpoints-tab.js
│   ├── vms-tab.js
│   ├── autoscaler-tab.js
│   ├── job-detail.js
│   └── vm-detail.js
├── worker/
│   ├── app.js               (main app: stats + task list)
│   ├── task-list.js
│   └── task-detail.js
└── logs/
    └── app.js               (shared logs viewer)
```

## Key Python Changes

### `dashboard_common.py` (~50 lines after)
```python
from pathlib import Path
from starlette.staticfiles import StaticFiles
from starlette.responses import HTMLResponse

STATIC_DIR = Path(__file__).parent / "static"

def static_files_mount():
    return Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static")

def html_shell(title: str, app_script: str) -> str:
    return f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8"><title>{title}</title>
  <link rel="stylesheet" href="/static/shared/styles.css">
</head><body>
  <div id="root"></div>
  <script type="importmap">{{"imports": {{
    "preact": "/static/vendor/preact.mjs",
    "preact/hooks": "/static/vendor/preact-hooks.mjs",
    "htm": "/static/vendor/htm.mjs"
  }}}}</script>
  <script type="module" src="{app_script}"></script>
</body></html>"""
```

### `controller/dashboard.py` (~80 lines after, down from 1731)
- Routes: `/` → `html_shell("Controller", "/static/controller/app.js")`
- Routes: `/job/{job_id}` → `html_shell("Job Detail", "/static/controller/job-detail.js")`
- Routes: `/vm/{vm_id}` → `html_shell("VM Detail", "/static/controller/vm-detail.js")`
- Keep: `/health`, `/api/logs`, `/logs`, RPC mount
- Add: `static_files_mount()`

### `worker/dashboard.py` (~80 lines after, down from 510)
- Routes: `/` → `html_shell("Worker", "/static/worker/app.js")`
- Routes: `/task/{task_id}` → `html_shell("Task Detail", "/static/worker/task-detail.js")`
- Keep: all `/api/*` REST endpoints, `/health`, `/logs`, RPC mount
- Add: `static_files_mount()`

## Execution Steps (Spiral)

### Step 1: Foundation — vendor deps + static serving + hello world
**Agent:** ml-engineer
**Deliverables:**
- Vendor `preact.mjs`, `preact-hooks.mjs`, `htm.mjs` into `static/vendor/`
- Add `html_shell()` and `static_files_mount()` to `dashboard_common.py`
- Add `/test-preact` route to controller serving a hello-world component
- Verify: visit `/test-preact`, check no console errors, no CDN requests
**Validation:** ml-engineer reviews

### Step 2: Shared CSS + JS utilities
**Agent:** ml-engineer
**Deliverables:**
- Extract `static/shared/styles.css` from all 5 templates (consolidate duplicates)
- Create `static/shared/rpc.js` with `controllerRpc()` / fetch wrapper
- Create `static/shared/utils.js` with `escapeHtml`, `formatBytes`, `formatDuration`, `formatRelativeTime`, `stateToName`
- Verify: import from hello-world component, check formatting output in console
**Validation:** senior-engineer reviews

### Step 3: Logs viewer migration
**Agent:** ml-engineer
**Deliverables:**
- Create `static/logs/app.js` — Preact component for log viewer
- Update `dashboard_common.py`: replace `LOGS_HTML` with `html_shell()` call
- Remove `LOGS_HTML` constant
- Verify: `/logs` page works on both controller and worker
- Run: scripts/review-dashboard.sh , validate screenshots
**Validation:** senior-engineer reviews

### Step 4: Worker dashboard migration
**Agent:** senior-engineer
**Deliverables:**
- Create `static/worker/app.js`, `task-list.js`, `task-detail.js`
- Rewrite `worker/dashboard.py` to API-only + static serving (~80 lines)
- Remove `DASHBOARD_HTML`, `TASK_DETAIL_HTML` constants
- All `/api/*` endpoints unchanged
- Run: scripts/review-dashboard.sh , validate screenshots
**Validation:** senior-engineer reviews

### Step 5: Controller dashboard — tabs + main page
**Agent:** senior-engineer
**Deliverables:**
- Create `static/controller/app.js`, `jobs-tab.js`, `workers-tab.js`, `endpoints-tab.js`, `vms-tab.js`, `autoscaler-tab.js`
- Tab switching via hash routing (`#jobs`, `#workers`, etc.)
- All tabs fetch via `controllerRpc()` and render with Preact
- Pagination on jobs and VMs tabs
- Run: scripts/review-dashboard.sh , validate screenshots
**Validation:** senior-engineer reviews

### Step 6: Controller detail pages
**Agent:** ml-engineer
**Deliverables:**
- Create `static/controller/job-detail.js`, `vm-detail.js`
- Job detail: task table, expandable logs, retry info
- VM detail: status, logs viewer
- Rewrite `controller/dashboard.py` to API-only + static serving (~80 lines)
- Remove all `*_HTML` constants
- Verify: `pytest lib/iris/tests/cluster/controller/test_dashboard.py` passes
- Run: scripts/review-dashboard.sh , validate screenshots
**Validation:** senior-engineer reviews

### Step 7: Screenshot validation + cleanup
**Agent:** senior-engineer
**Deliverables:**
- Run `uv run lib/iris/scripts/screenshot-dashboard.py`
- Read all generated screenshot images, compare to expected layout
- Add worker dashboard screenshot captures if missing from script
- Remove test route (`/test-preact`)
- Final `pytest` run for all dashboard tests
- Update `lib/iris/README.md` if needed (static file structure)
**Validation:** final senior-engineer review of entire changeset

## Files Modified

| File | Action |
|------|--------|
| `lib/iris/src/iris/cluster/dashboard_common.py` | Rewrite: add `html_shell`, `static_files_mount`, remove `LOGS_HTML` |
| `lib/iris/src/iris/cluster/controller/dashboard.py` | Rewrite: remove all HTML constants, API-only |
| `lib/iris/src/iris/cluster/worker/dashboard.py` | Rewrite: remove all HTML constants, API-only |
| `lib/iris/src/iris/cluster/static/**` | New: all JS/CSS components |
| `lib/iris/scripts/screenshot-dashboard.py` | May need updates for worker screenshots |

## Verification

1. `pytest lib/iris/tests/cluster/controller/test_dashboard.py` — all API tests pass
2. `pytest lib/iris/tests/cluster/worker/test_dashboard.py` — all API tests pass
3. `scripts/review-dashboard.sh` -- no reported failures
4. `./infra/pre-commit.py --all-files` — lint passes


## Final Verification 

`uv run lib/iris/scripts/screenshot-dashboard.py` — screenshots capture all pages

review resulting images yourself for quality.