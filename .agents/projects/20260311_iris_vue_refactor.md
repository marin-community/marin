# Iris Dashboard: Vue 3 + Rsbuild Migration Plan

**Issue:** [#3423 — iris: cleanup dashboard, make it structured](https://github.com/marin-community/marin/issues/3423)
**Status:** Planning
**Stack:** Vue 3 + TypeScript + Rsbuild + Tailwind CSS v4

---

## 1. Goals

1. Replace Preact+HTM with structured Vue 3 SFCs (Single File Components)
2. Lazy per-tab data fetching instead of "fetch everything on load"
3. New task detail page with memory/CPU history, profiling links, logs
4. Expandable job listing (top-level jobs → click to expand tasks)
5. Remove job-based log viewer; replace with task-detail-page log view
6. Build-on-demand via Rsbuild (wired into CLI, Docker, and test conftest)
7. TypeScript throughout for type-safe RPC types and component props

## 2. Architecture Overview

```
lib/iris/dashboard/                     # NEW: frontend source (lives outside src/iris to avoid Python packaging)
├── package.json                        # Rsbuild, Vue, Tailwind, TypeScript deps
├── tsconfig.json
├── rsbuild.config.ts
├── postcss.config.cjs
├── tailwind.config.ts
├── src/
│   ├── main.ts                         # Vue app entry
│   ├── App.vue                         # Root layout: header + tab router
│   ├── router.ts                       # Vue Router (hash mode for static serving)
│   ├── types/
│   │   ├── rpc.ts                      # TypeScript types mirroring cluster.proto messages
│   │   └── status.ts                   # Status enums, color maps, display names
│   ├── composables/
│   │   ├── useRpc.ts                   # Connect RPC wrapper with reactive state
│   │   ├── useAutoRefresh.ts           # Polling composable with configurable interval
│   │   └── useRelativeTime.ts          # Reactive relative time formatting
│   ├── components/
│   │   ├── layout/
│   │   │   ├── AppHeader.vue           # Cluster name, status indicator, nav
│   │   │   ├── TabNav.vue              # Tab navigation bar
│   │   │   └── PageShell.vue           # Back link, title, breadcrumbs
│   │   ├── shared/
│   │   │   ├── StatusBadge.vue         # Semantic status pill (running, failed, etc.)
│   │   │   ├── InfoCard.vue            # Key-value card with title
│   │   │   ├── InfoRow.vue             # Label + value row
│   │   │   ├── DataTable.vue           # Sortable, paginated table
│   │   │   ├── MetricCard.vue          # Big number + label + optional sparkline
│   │   │   ├── ResourceGauge.vue       # Horizontal progress bar with label
│   │   │   ├── LogViewer.vue           # Filterable log viewer with level coloring
│   │   │   ├── EmptyState.vue          # Centered empty state with icon
│   │   │   ├── Sparkline.vue           # Inline SVG trend chart
│   │   │   └── ConstraintChip.vue      # Constraint key=value pill
│   │   ├── controller/
│   │   │   ├── JobsTab.vue             # Job listing with expandable rows
│   │   │   ├── JobRow.vue              # Single job row with task progress badges
│   │   │   ├── JobDetail.vue           # Full job detail page
│   │   │   ├── TaskTable.vue           # Task table within job detail
│   │   │   ├── TaskDetail.vue          # NEW: full task detail page
│   │   │   ├── TaskResourceChart.vue   # NEW: historical CPU/memory chart
│   │   │   ├── FleetTab.vue            # Workers + VMs unified view
│   │   │   ├── WorkerDetail.vue        # Worker detail page
│   │   │   ├── AutoscalerTab.vue       # Autoscaler status + scale groups + actions
│   │   │   ├── ScaleGroupTable.vue     # Scale group waterfall table
│   │   │   ├── EndpointsTab.vue        # Registered endpoints
│   │   │   ├── StatusTab.vue           # Cluster overview
│   │   │   ├── TransactionsTab.vue     # Transaction history
│   │   │   └── UsersTab.vue            # User listing
│   │   └── worker/
│   │       ├── WorkerApp.vue           # Worker dashboard root
│   │       ├── WorkerTaskDetail.vue    # Worker-side task detail
│   │       └── WorkerStatusPage.vue    # Worker health/status
│   └── styles/
│       └── main.css                    # Tailwind directives + custom utilities
├── public/
│   └── (empty — no static assets needed)
└── dist/                               # Build output (gitignored)
    ├── controller/
    │   ├── index.html
    │   └── static/
    │       ├── js/
    │       └── css/
    └── worker/
        ├── index.html
        └── static/
            ├── js/
            └── css/
```

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Bundler | Rsbuild | Rust-based speed, first-class Vue plugin, Tailwind integration |
| Language | TypeScript | Type-safe RPC types, component props, IDE support |
| CSS | Tailwind v4 | Utility-first, consistent design tokens, small output |
| Routing | Vue Router (hash mode) | Works with static file serving, no server config needed |
| State | Composables (no Pinia) | Dashboard is read-heavy; composables + `ref()` suffice |
| RPC | Keep Connect RPC | Thin `fetch()` wrapper stays, add TypeScript types |
| Build output | Gitignored, built on demand | No node in runtime; build in Docker multi-stage + conftest |
| Dark mode | Light only (for now) | Tailwind makes retrofitting trivial later |

## 3. Design System

### Aesthetic direction: Precision Engineering

An engineer-focused dashboard that prioritizes data density, scanability, and clarity.
Not a GitHub clone — something closer to Linear or Vercel's dashboard aesthetic.

### Typography

```css
/* Tailwind config */
fontFamily: {
  sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
  mono: ['"JetBrains Mono"', '"SF Mono"', 'Menlo', 'monospace'],
}
```

- **Plus Jakarta Sans** — geometric, modern, excellent at small sizes. Available via Google Fonts or self-hosted.
- **JetBrains Mono** — purpose-built for code/data. Used for IDs, timestamps, metrics, logs.

Both loaded via `@fontsource` packages (no external CDN dependency).

### Color palette

```typescript
// tailwind.config.ts
const colors = {
  // Base
  surface: {
    DEFAULT: '#ffffff',
    raised: '#f8f9fb',
    sunken: '#f1f3f5',
    border: '#e2e5e9',
    'border-subtle': '#eef0f3',
  },
  // Text
  text: {
    DEFAULT: '#1a1d23',
    secondary: '#5c6370',
    muted: '#9ca3af',
  },
  // Accent
  accent: {
    DEFAULT: '#2563eb',     // Blue-600: links, active states, primary actions
    hover: '#1d4ed8',
    subtle: '#eff6ff',
    border: '#bfdbfe',
  },
  // Semantic
  status: {
    success: '#16a34a',
    'success-bg': '#f0fdf4',
    'success-border': '#bbf7d0',
    warning: '#ca8a04',
    'warning-bg': '#fefce8',
    'warning-border': '#fef08a',
    danger: '#dc2626',
    'danger-bg': '#fef2f2',
    'danger-border': '#fecaca',
    purple: '#7c3aed',
    'purple-bg': '#f5f3ff',
    'purple-border': '#ddd6fe',
    orange: '#ea580c',
    'orange-bg': '#fff7ed',
    'orange-border': '#fed7aa',
  },
}
```

### Spacing & sizing

- Base unit: 4px (Tailwind default)
- Border radius: `rounded` (4px) for chips/badges, `rounded-lg` (8px) for cards
- Shadows: Minimal — prefer 1px borders over drop shadows
- Max content width: `max-w-7xl` (1280px)
- Table density: `text-[13px]` body, `text-xs` for secondary data

### Component style patterns

Every component uses scoped `<style>` only for component-specific animations.
All layout and visual styling uses Tailwind utility classes.

```vue
<!-- Example: StatusBadge.vue -->
<template>
  <span :class="[
    'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full',
    'text-xs font-semibold tracking-wide uppercase',
    statusClasses[status]
  ]">
    <span :class="['w-1.5 h-1.5 rounded-full', dotClasses[status]]"
          v-if="showDot" />
    {{ displayName }}
  </span>
</template>
```

### Status color mapping

| Status | Text | Background | Border | Dot |
|--------|------|-----------|--------|-----|
| running | `text-accent` | `bg-accent-subtle` | `border-accent-border` | `bg-accent` |
| succeeded | `text-status-success` | `bg-status-success-bg` | `border-status-success-border` | `bg-status-success` |
| failed | `text-status-danger` | `bg-status-danger-bg` | `border-status-danger-border` | `bg-status-danger` |
| pending | `text-status-warning` | `bg-status-warning-bg` | `border-status-warning-border` | `bg-status-warning` |
| building | `text-status-purple` | `bg-status-purple-bg` | `border-status-purple-border` | `bg-status-purple` |
| assigned | `text-status-orange` | `bg-status-orange-bg` | `border-status-orange-border` | `bg-status-orange` |
| killed | `text-text-muted` | `bg-surface-sunken` | `border-surface-border` | `bg-text-muted` |

## 4. RPC Layer

### Current state

```javascript
// shared/rpc.js — 28 lines
export async function connectRpc(service, method, body) {
  const resp = await fetch(`/${service}/${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!resp.ok) throw new Error(`RPC ${method}: ${resp.status}`);
  return resp.json();
}
```

### Migration: typed RPC composable

```typescript
// composables/useRpc.ts
import { ref, type Ref } from 'vue'

interface RpcState<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  refresh: () => Promise<void>
}

export function useControllerRpc<T>(method: string, body?: Record<string, unknown>): RpcState<T> {
  return useRpc<T>('iris.cluster.ControllerService', method, body)
}

export function useWorkerRpc<T>(method: string, body?: Record<string, unknown>): RpcState<T> {
  return useRpc<T>('iris.cluster.WorkerService', method, body)
}

function useRpc<T>(service: string, method: string, body?: Record<string, unknown>): RpcState<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function refresh() {
    loading.value = true
    error.value = null
    try {
      const resp = await fetch(`/${service}/${method}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body ?? {}),
      })
      if (!resp.ok) throw new Error(`${method}: ${resp.status} ${resp.statusText}`)
      data.value = await resp.json()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  return { data, loading, error, refresh }
}
```

### TypeScript types for RPC responses

Create `types/rpc.ts` with interfaces matching the protobuf messages used by the dashboard.
These are manually maintained — not auto-generated — since only a subset of fields are used.

```typescript
// types/rpc.ts — key interfaces (excerpt)

export interface JobStatus {
  job_id: string
  name: string
  state: number
  submitted_at: string
  started_at?: string
  finished_at?: string
  error?: string
  tasks: TaskSummary[]
  resources: ResourceSpec
  constraints?: ConstraintSpec[]
  user?: string
}

export interface TaskStatus {
  task_id: string
  job_id: string
  state: number
  worker_id?: string
  started_at?: string
  finished_at?: string
  error?: string
  resources_used?: ResourceUsage
  profiling_url?: string
  build_info?: BuildInfo
}

export interface ListJobsResponse {
  jobs: JobStatus[]
  total: number
}

// ... (full set derived from cluster.proto messages actually used in the dashboard)
```

### New RPC endpoints needed

Per issue #3423, the job listing should support hierarchical expansion:

| Endpoint | Request | Response | Purpose |
|----------|---------|----------|---------|
| `ListJobs` (modified) | `{ sort, limit, offset, state_filter? }` | `{ jobs, total }` | Top-level job list without full task arrays |
| `GetJobTasks` (new) | `{ job_id, limit, offset }` | `{ tasks, total }` | Paginated task list for a specific job |
| `GetTaskDetail` (new) | `{ task_id }` | `{ task, resource_history, log_summary }` | Full task detail with resource usage history |
| `GetTaskResourceHistory` (new) | `{ task_id }` | `{ samples: [{ts, cpu_pct, memory_bytes}] }` | Time-series resource data for charts |

These require corresponding additions to `cluster.proto` and service implementations.

## 5. Component Migration Plan

### Phase 1: Scaffold + Shared Components

Set up the project, build toolchain, and shared component library.

#### 5.1 Project initialization

```bash
cd lib/iris/dashboard
npm create rsbuild@latest -- --template vue-ts
```

Then configure:

**`rsbuild.config.ts`:**
```typescript
import { defineConfig } from '@rsbuild/core'
import { pluginVue } from '@rsbuild/plugin-vue'

export default defineConfig({
  plugins: [pluginVue()],
  source: {
    entry: {
      controller: './src/controller-main.ts',
      worker: './src/worker-main.ts',
    },
  },
  output: {
    distPath: {
      root: 'dist',
    },
    assetPrefix: '/static/',
    // Produce controller/index.html and worker/index.html
  },
  html: {
    template: './src/template.html',
    templateParameters: {
      title: 'Iris Dashboard',
    },
  },
})
```

**`tailwind.config.ts`:**
```typescript
import type { Config } from 'tailwindcss'

export default {
  content: ['./src/**/*.{vue,ts,tsx,html}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"SF Mono"', 'Menlo', 'monospace'],
      },
      colors: {
        surface: { /* ... as defined in section 3 */ },
        accent: { /* ... */ },
        status: { /* ... */ },
      },
    },
  },
  plugins: [],
} satisfies Config
```

**`package.json` scripts:**
```json
{
  "scripts": {
    "dev": "rsbuild dev",
    "build": "rsbuild build",
    "build:check": "vue-tsc --noEmit && rsbuild build",
    "preview": "rsbuild preview"
  }
}
```

**Font installation:**
```bash
npm install @fontsource-variable/plus-jakarta-sans @fontsource-variable/jetbrains-mono
```

#### 5.2 Shared components (build first, use everywhere)

| Component | Props | Behavior |
|-----------|-------|----------|
| `StatusBadge` | `status: string, size?: 'sm'\|'md'` | Maps status to semantic color scheme |
| `InfoCard` | `title: string` | Card container with titled header, slot for rows |
| `InfoRow` | `label: string` | Flex row: label left, value (slot) right |
| `DataTable` | `columns, rows, sortable?, paginated?` | Generic sortable table with pagination |
| `MetricCard` | `value: string\|number, label, variant?` | Big metric display with semantic coloring |
| `ResourceGauge` | `label, used, total, unit` | Horizontal gauge bar with percentage |
| `LogViewer` | `logs, filterPrefix?, autoRefresh?` | Filterable log viewer with level coloring |
| `EmptyState` | `message, icon?` | Centered empty state |
| `Sparkline` | `data: number[], color?, width?, height?` | Inline SVG line chart |
| `ConstraintChip` | `constraint: string` | Styled key=value pill |

### Phase 2: Controller Dashboard

Migrate each tab as an independent component that fetches its own data.

#### 5.3 App shell + routing

```typescript
// router.ts
import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  { path: '/',           component: () => import('./components/controller/JobsTab.vue') },
  { path: '/fleet',      component: () => import('./components/controller/FleetTab.vue') },
  { path: '/autoscaler', component: () => import('./components/controller/AutoscalerTab.vue') },
  { path: '/endpoints',  component: () => import('./components/controller/EndpointsTab.vue') },
  { path: '/status',     component: () => import('./components/controller/StatusTab.vue') },
  { path: '/transactions', component: () => import('./components/controller/TransactionsTab.vue') },
  { path: '/users',      component: () => import('./components/controller/UsersTab.vue') },
  // Detail pages
  { path: '/job/:jobId', component: () => import('./components/controller/JobDetail.vue') },
  { path: '/job/:jobId/task/:taskId', component: () => import('./components/controller/TaskDetail.vue') },
  { path: '/worker/:workerId', component: () => import('./components/controller/WorkerDetail.vue') },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})
```

Each tab is lazy-loaded. Data is fetched in `onMounted()` via the `useControllerRpc` composable.

#### 5.4 Jobs tab — hierarchical expansion

This is the most complex tab and the primary improvement from #3423.

**Current behavior:** Fetches all jobs with all task data in one RPC call.
**New behavior:** Fetches job summaries only; clicking a job name expands inline to show tasks.

```vue
<!-- JobsTab.vue — conceptual structure -->
<template>
  <div class="space-y-4">
    <div class="flex items-center justify-between">
      <h2 class="text-lg font-semibold text-text">Jobs</h2>
      <div class="flex items-center gap-3">
        <select v-model="stateFilter" class="...">
          <option value="">All states</option>
          <option value="running">Running</option>
          <!-- ... -->
        </select>
        <AutoRefreshBadge :active="autoRefresh" @toggle="autoRefresh = !autoRefresh" />
      </div>
    </div>

    <DataTable :columns="columns" :rows="jobs" :loading="loading" sortable paginated>
      <template #row="{ row }">
        <JobRow
          :job="row"
          :expanded="expandedJobId === row.job_id"
          @toggle="toggleExpand(row.job_id)"
          @navigate="router.push(`/job/${row.job_id}`)"
        />
      </template>
      <template #expanded="{ row }">
        <!-- Lazy-loaded task list for this job -->
        <TaskTable :job-id="row.job_id" />
      </template>
    </DataTable>
  </div>
</template>
```

#### 5.5 Task detail page (NEW)

The headline new feature. Replaces the job-based log viewer.

```vue
<!-- TaskDetail.vue — structure -->
<template>
  <PageShell :title="taskId" back-to="/job/...">
    <!-- Status + timing header -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
      <InfoCard title="Status">
        <InfoRow label="State"><StatusBadge :status="task.state" /></InfoRow>
        <InfoRow label="Worker"><router-link ...>{{ task.worker_id }}</router-link></InfoRow>
        <InfoRow label="Started">{{ formatRelative(task.started_at) }}</InfoRow>
        <InfoRow label="Duration">{{ formatDuration(task.duration) }}</InfoRow>
        <InfoRow label="Retries">{{ task.retry_count }}</InfoRow>
      </InfoCard>

      <InfoCard title="Resources">
        <ResourceGauge label="CPU" :used="task.cpu_used" :total="task.cpu_requested" unit="cores" />
        <ResourceGauge label="Memory" :used="task.memory_used" :total="task.memory_requested" unit="bytes" />
        <ResourceGauge label="Disk" :used="task.disk_used" :total="task.disk_requested" unit="bytes" />
      </InfoCard>

      <InfoCard title="Links">
        <InfoRow label="Profiling"><a :href="task.profiling_url">xprof →</a></InfoRow>
        <InfoRow label="W&B"><a :href="task.wandb_url">Weights & Biases →</a></InfoRow>
        <InfoRow label="Logs"><a :href="task.log_url">Full logs →</a></InfoRow>
      </InfoCard>
    </div>

    <!-- Resource history chart -->
    <TaskResourceChart :task-id="taskId" class="mb-6" />

    <!-- Logs (replaces old job-based log viewer) -->
    <LogViewer :task-id="taskId" />
  </PageShell>
</template>
```

#### 5.6 Other tabs (migrated 1:1)

| Tab | Migration notes |
|-----|----------------|
| `FleetTab` | Direct port. Fetches workers + VMs on mount. |
| `AutoscalerTab` | Direct port. Scale group table + actions log. |
| `EndpointsTab` | Simple table, trivial migration. |
| `StatusTab` | Cluster summary metrics. |
| `TransactionsTab` | Transaction history table. |
| `UsersTab` | Simple user list. |
| `WorkerDetail` | Direct port with utilization panel and log viewer. |
| `JobDetail` | Simplified — task table links to TaskDetail instead of inline logs. |

### Phase 3: Worker Dashboard

The worker dashboard is smaller (~486 lines currently) and shares many components.

| Component | Migration notes |
|-----------|----------------|
| `WorkerApp` | Stats bar + task table. Separate entry point (`worker-main.ts`). |
| `WorkerTaskDetail` | Worker-side task view. Shares components with controller TaskDetail. |
| `WorkerStatusPage` | Health/status page, simple port. |

### Phase 4: Build Integration

#### 5.7 CLI: `iris build dashboard`

Add a new CLI subcommand to `lib/iris/src/iris/cli/build.py`:

```python
@build.command("dashboard")
@click.pass_context
def build_dashboard(ctx: click.Context):
    """Build Vue dashboard assets via Rsbuild."""
    dashboard_dir = find_iris_root() / "dashboard"
    if not (dashboard_dir / "package.json").exists():
        raise click.ClickException(f"Dashboard source not found at {dashboard_dir}")

    # Install deps if needed (check node_modules existence)
    if not (dashboard_dir / "node_modules").exists():
        click.echo("Installing dashboard dependencies...")
        subprocess.run(["npm", "ci"], cwd=dashboard_dir, check=True)

    click.echo("Building dashboard...")
    result = subprocess.run(["npm", "run", "build"], cwd=dashboard_dir, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(result.stderr, err=True)
        raise click.ClickException("Dashboard build failed")

    click.echo("Dashboard built successfully.")
```

#### 5.8 Python serving layer changes

Update `dashboard_common.py` to serve built assets:

```python
# dashboard_common.py — updated

DASHBOARD_DIR = Path(__file__).parent.parent.parent.parent.parent / "dashboard" / "dist"
# Falls back to a well-known location in Docker images
DOCKER_DASHBOARD_DIR = Path("/app/dashboard/dist")

def _resolve_dashboard_dir() -> Path:
    """Find the built dashboard assets directory."""
    for candidate in [DASHBOARD_DIR, DOCKER_DASHBOARD_DIR]:
        if candidate.is_dir() and (candidate / "controller").is_dir():
            return candidate
    raise FileNotFoundError(
        "Dashboard assets not found. Run 'iris build dashboard' or 'npm run build' in lib/iris/dashboard/."
    )

def html_shell(title: str, dashboard_type: str = "controller") -> str:
    """Serve the built index.html for controller or worker dashboard."""
    dist = _resolve_dashboard_dir()
    index_path = dist / dashboard_type / "index.html"
    return index_path.read_text()
```

The key change: instead of generating an HTML shell in Python, we serve the Rsbuild-generated
`index.html` directly. The Python layer becomes even thinner.

#### 5.9 Docker integration

Add a frontend build stage to `Dockerfile.controller`:

```dockerfile
# --- Frontend build stage ---
FROM node:22-slim AS dashboard-build
WORKDIR /dashboard
COPY lib/iris/dashboard/package.json lib/iris/dashboard/package-lock.json ./
RUN npm ci
COPY lib/iris/dashboard/ ./
RUN npm run build

# --- Python stage (existing) ---
FROM python:3.11-slim AS runtime
# ... existing setup ...

# Copy built dashboard assets
COPY --from=dashboard-build /dashboard/dist /app/dashboard/dist
```

Same pattern for `Dockerfile.worker` (worker dashboard entry point).

#### 5.10 Test conftest integration

Ensure dashboard assets are built before E2E tests that render pages.

```python
# tests/e2e/conftest.py — add dashboard build fixture

import subprocess
from pathlib import Path

IRIS_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = IRIS_ROOT / "dashboard"

@pytest.fixture(scope="session", autouse=True)
def _ensure_dashboard_built():
    """Build dashboard assets before any E2E test runs."""
    dist = DASHBOARD_DIR / "dist"
    if dist.is_dir() and (dist / "controller" / "index.html").exists():
        return  # Already built

    if not (DASHBOARD_DIR / "package.json").exists():
        pytest.skip("Dashboard source not found")

    # Install deps
    if not (DASHBOARD_DIR / "node_modules").exists():
        subprocess.run(["npm", "ci"], cwd=DASHBOARD_DIR, check=True, timeout=120)

    # Build
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=DASHBOARD_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"Dashboard build failed:\n{result.stderr}")
```

The fixture is `scope="session"` so it only runs once per test session, and `autouse=True`
so every E2E test gets it automatically.

For unit tests (`tests/cluster/controller/test_dashboard.py`), the dashboard doesn't need
to be built since those tests hit RPC endpoints directly, not rendered HTML.

#### 5.11 E2E test updates

The `wait_for_dashboard_ready()` function in conftest needs updating since Vue
mounts differently than Preact:

```python
def wait_for_dashboard_ready(page) -> None:
    """Wait for Vue to mount and render the dashboard."""
    if _is_noop_page(page):
        return
    page.wait_for_function(
        "() => document.getElementById('app') !== null"
        " && document.getElementById('app').children.length > 0",
        timeout=30000,
    )
```

## 6. Migration Strategy

### Approach: parallel development, single cutover

Build the Vue dashboard alongside the existing Preact dashboard. The Python serving layer
switches atomically once the Vue build is ready and tested.

### Phase sequence

| Phase | Scope | Estimated components | Dependencies |
|-------|-------|---------------------|--------------|
| **P1** | Scaffold + shared components | 12 components | None |
| **P2a** | Jobs tab + job detail | 4 components | P1 |
| **P2b** | Task detail (NEW) | 3 components | P1, new RPC endpoints |
| **P2c** | Fleet + worker detail | 3 components | P1 |
| **P2d** | Autoscaler tab | 2 components | P1 |
| **P2e** | Remaining tabs | 4 components | P1 |
| **P3** | Worker dashboard | 3 components | P1, some P2 components |
| **P4** | Build integration | CLI, Docker, conftest | All above |
| **P5** | Cutover + cleanup | Remove Preact code | All above |

### Proto changes (Phase P2b)

Add to `cluster.proto`:

```protobuf
message GetJobTasksRequest {
  string job_id = 1;
  int32 limit = 2;
  int32 offset = 3;
}

message GetJobTasksResponse {
  repeated TaskStatus tasks = 1;
  int32 total = 2;
}

message GetTaskResourceHistoryRequest {
  string task_id = 1;
}

message ResourceSample {
  google.protobuf.Timestamp timestamp = 1;
  double cpu_percent = 2;
  int64 memory_bytes = 3;
}

message GetTaskResourceHistoryResponse {
  repeated ResourceSample samples = 1;
}
```

Add corresponding RPC methods to `ControllerService`:

```protobuf
service ControllerService {
  // ... existing methods ...
  rpc GetJobTasks(GetJobTasksRequest) returns (GetJobTasksResponse);
  rpc GetTaskResourceHistory(GetTaskResourceHistoryRequest) returns (GetTaskResourceHistoryResponse);
}
```

## 7. File Cleanup (Phase P5)

After cutover, remove:

```
lib/iris/src/iris/cluster/static/          # Entire directory (vendor/, shared/, controller/, worker/)
lib/iris/docs/preact.md                    # Old architecture doc
```

Update:
- `dashboard_common.py` — remove `html_shell()` template generation, simplify to serving built files
- `controller/dashboard.py` — remove page routes (Vue Router handles routing), keep RPC mount
- `worker/dashboard.py` — same treatment

## 8. Development Workflow

### Local development

```bash
cd lib/iris/dashboard
npm install
npm run dev     # Starts Rsbuild dev server with HMR on port 3000
```

The Rsbuild dev server proxies RPC requests to the Iris controller. Add to `rsbuild.config.ts`:

```typescript
server: {
  proxy: {
    '/iris.cluster.ControllerService': 'http://localhost:8080',
    '/iris.cluster.WorkerService': 'http://localhost:8081',
    '/api': 'http://localhost:8080',
  },
},
```

This allows developing the frontend independently of the Python server.

### Production build

```bash
npm run build   # Produces dist/controller/ and dist/worker/
```

Or via CLI:
```bash
iris build dashboard
```

### Testing

```bash
# Unit tests (no dashboard build needed)
uv run pytest lib/iris/tests/cluster/ -o "addopts="

# E2E tests (auto-builds dashboard via conftest fixture)
IRIS_SCREENSHOT_DIR=/tmp/screenshots uv run pytest lib/iris/tests/e2e/ -o "addopts="

# Visual review
./lib/iris/scripts/review-dashboard.sh /tmp/screenshots
```

## 9. Detailed Component Specifications

### 9.1 `LogViewer.vue`

Replaces the current `log-viewer.js` (153 lines) and the job-based log viewer.
Used in TaskDetail, WorkerDetail, and AutoscalerTab.

```vue
<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'

const props = defineProps<{
  taskId?: string
  workerId?: string
  source?: 'controller' | 'worker'
  maxHeight?: string
}>()

const filter = ref('')
const level = ref('info')  // debug | info | warning | error
const tailLines = ref(500)

// Fetch logs via appropriate RPC endpoint
const { data: logData, refresh } = useControllerRpc<LogResponse>('GetTaskLogs', {
  id: props.taskId,
  tail_lines: tailLines.value,
})

// Auto-refresh every 30s when viewing
const { active: autoRefreshActive } = useAutoRefresh(refresh, 30000)

const filteredLogs = computed(() => {
  if (!logData.value?.task_logs) return []
  return logData.value.task_logs
    .flatMap(batch => batch.logs)
    .filter(entry => {
      if (filter.value && !entry.data.includes(filter.value)) return false
      return levelPriority(entry.level) >= levelPriority(level.value)
    })
})
</script>

<template>
  <div class="space-y-2">
    <div class="flex items-center gap-3 text-sm">
      <input
        v-model="filter"
        type="text"
        placeholder="Filter logs..."
        class="w-64 px-3 py-1.5 bg-surface border border-surface-border rounded
               text-sm font-mono placeholder:text-text-muted
               focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
      />
      <select v-model="level" class="px-2 py-1.5 border border-surface-border rounded text-sm">
        <option value="debug">Debug</option>
        <option value="info">Info</option>
        <option value="warning">Warning</option>
        <option value="error">Error</option>
      </select>
      <span class="ml-auto text-xs text-text-muted">
        {{ filteredLogs.length }} lines
      </span>
    </div>

    <div :class="['overflow-y-auto rounded-lg border border-surface-border bg-white',
                   maxHeight ? `max-h-[${maxHeight}]` : 'max-h-[60vh]']">
      <div v-if="filteredLogs.length === 0"
           class="py-12 text-center text-text-muted text-sm">
        No log entries
      </div>
      <div v-for="(entry, i) in filteredLogs" :key="i"
           :class="['px-3 py-0.5 font-mono text-xs leading-relaxed',
                     'hover:bg-surface-sunken',
                     logLevelClass(entry.level)]">
        <span class="text-text-muted mr-2">{{ formatLogTime(entry.timestamp) }}</span>
        <span class="whitespace-pre-wrap break-all">{{ entry.data }}</span>
      </div>
    </div>
  </div>
</template>
```

### 9.2 `DataTable.vue`

Generic sortable, paginated table used by JobsTab, FleetTab, etc.

**Props:**
```typescript
interface Column {
  key: string
  label: string
  sortable?: boolean
  align?: 'left' | 'center' | 'right'
  width?: string
  mono?: boolean  // Use monospace font
}

interface Props {
  columns: Column[]
  rows: any[]
  loading?: boolean
  sortKey?: string
  sortDir?: 'asc' | 'desc'
  pageSize?: number
  emptyMessage?: string
}
```

**Emits:** `sort(key, dir)`, `page(offset)`

**Slots:** `#cell-{key}` for custom cell rendering, `#row` for full row override,
`#expanded` for expandable row content.

### 9.3 `TaskResourceChart.vue` (NEW)

SVG-based chart showing CPU and memory usage over time. No charting library needed —
a simple SVG line chart is sufficient for this data.

```vue
<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type { ResourceSample } from '@/types/rpc'

const props = defineProps<{ taskId: string }>()

const { data, refresh } = useControllerRpc<{ samples: ResourceSample[] }>(
  'GetTaskResourceHistory',
  { task_id: props.taskId }
)

useAutoRefresh(refresh, 10000)  // Refresh every 10s
onMounted(refresh)

// Normalize samples to 0-1 range for SVG path generation
const cpuPath = computed(() => generatePath(data.value?.samples ?? [], 'cpu_percent', 100))
const memPath = computed(() => generatePath(data.value?.samples ?? [], 'memory_bytes'))
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-white p-4">
    <h3 class="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-4">
      Resource Usage Over Time
    </h3>
    <div class="grid grid-cols-2 gap-6">
      <div>
        <div class="flex items-baseline justify-between mb-2">
          <span class="text-xs text-text-muted">CPU</span>
          <span class="text-sm font-mono font-semibold text-accent">
            {{ currentCpu }}%
          </span>
        </div>
        <svg viewBox="0 0 200 60" class="w-full h-16">
          <path :d="cpuPath" fill="none" stroke="currentColor"
                class="text-accent" stroke-width="1.5" />
          <path :d="cpuAreaPath" class="fill-accent/10" />
        </svg>
      </div>
      <div>
        <div class="flex items-baseline justify-between mb-2">
          <span class="text-xs text-text-muted">Memory</span>
          <span class="text-sm font-mono font-semibold text-status-purple">
            {{ formatBytes(currentMemory) }}
          </span>
        </div>
        <svg viewBox="0 0 200 60" class="w-full h-16">
          <path :d="memPath" fill="none" stroke="currentColor"
                class="text-status-purple" stroke-width="1.5" />
          <path :d="memAreaPath" class="fill-status-purple/10" />
        </svg>
      </div>
    </div>
  </div>
</template>
```

## 10. Checklist

### Pre-implementation
- [ ] Confirm resource history data is available from workers (or needs collection)
- [ ] Review proto changes with team
- [ ] Set up `lib/iris/dashboard/` project skeleton
- [ ] Verify Rsbuild + Vue + Tailwind + TypeScript build works

### Phase 1: Shared components
- [ ] `StatusBadge.vue` — status → color mapping
- [ ] `InfoCard.vue` + `InfoRow.vue` — card layout
- [ ] `DataTable.vue` — sortable, paginated table
- [ ] `MetricCard.vue` — big number display
- [ ] `ResourceGauge.vue` — horizontal progress bar
- [ ] `LogViewer.vue` — filterable log viewer
- [ ] `EmptyState.vue` — empty state display
- [ ] `Sparkline.vue` — inline SVG chart
- [ ] `ConstraintChip.vue` — constraint pill
- [ ] `AppHeader.vue` + `TabNav.vue` — layout shell
- [ ] `PageShell.vue` — detail page wrapper
- [ ] Composables: `useRpc`, `useAutoRefresh`, `useRelativeTime`

### Phase 2: Controller dashboard
- [ ] `App.vue` + `router.ts` — root layout with Vue Router
- [ ] `JobsTab.vue` + `JobRow.vue` — expandable job list
- [ ] `JobDetail.vue` — job detail page (simplified, links to task detail)
- [ ] `TaskTable.vue` — task list within job detail
- [ ] `TaskDetail.vue` — NEW task detail page with resource charts + logs
- [ ] `TaskResourceChart.vue` — SVG resource history chart
- [ ] `FleetTab.vue` — workers + VMs
- [ ] `WorkerDetail.vue` — worker detail with utilization panel
- [ ] `AutoscalerTab.vue` + `ScaleGroupTable.vue` — autoscaler status
- [ ] `EndpointsTab.vue` — endpoints table
- [ ] `StatusTab.vue` — cluster overview
- [ ] `TransactionsTab.vue` — transaction history
- [ ] `UsersTab.vue` — user list

### Phase 3: Worker dashboard
- [ ] `WorkerApp.vue` — separate entry point
- [ ] `WorkerTaskDetail.vue` — worker-side task view
- [ ] `WorkerStatusPage.vue` — health/status

### Phase 4: Build integration
- [ ] `iris build dashboard` CLI command
- [ ] `Dockerfile.controller` multi-stage frontend build
- [ ] `Dockerfile.worker` multi-stage frontend build
- [ ] E2E conftest `_ensure_dashboard_built` fixture
- [ ] Update `wait_for_dashboard_ready()` for Vue mount
- [ ] Update `review-dashboard.sh` if needed

### Phase 5: Cutover + cleanup
- [ ] Update `dashboard_common.py` to serve built assets
- [ ] Update `controller/dashboard.py` — remove page routes
- [ ] Update `worker/dashboard.py` — remove page routes
- [ ] Remove `lib/iris/src/iris/cluster/static/` directory
- [ ] Remove `lib/iris/docs/preact.md`
- [ ] Update `lib/iris/AGENTS.md` and docs references
- [ ] Run full E2E test suite + visual review

### CI / repo
- [ ] Add `lib/iris/dashboard/` to `.gitignore` for `dist/` and `node_modules/`
- [ ] Add dashboard build step to CI pipeline
- [ ] Update pre-commit to lint/format Vue files (optional: eslint + prettier for .vue)

## 11. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Node dependency in Docker builds | Multi-stage build isolates Node to build stage only |
| E2E tests break during migration | Build both dashboards; switch serving layer atomically |
| Resource history data not available | Start with basic task detail (status, logs, links); add charts when data pipeline exists |
| Tailwind class conflicts with existing CSS | New dashboard is completely separate; no coexistence needed |
| Rsbuild Vue plugin maturity | Rsbuild 1.x is stable; Vue plugin is officially maintained by the Rsbuild team |
| Font loading latency | Self-host via `@fontsource`; fonts bundled in the build output |

