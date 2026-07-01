# buoy → Vue dashboard migration

Migrate buoy's frontend from the single inline vanilla-JS SPA
(`lib/buoy/src/buoy/static/index.html`, ~700 lines) to the house standard:
**Vue 3 + rsbuild + Tailwind + TypeScript**, served from the Starlette app.

## Why

- Matches the existing standard used by every other in-repo dashboard:
  `lib/finelog/dashboard`, `lib/iris/dashboard`, and — most recently and most
  analogous — **ducky (`lib/ducky/dashboard`, PR #6762)**, an ad-hoc service with
  a dashboard just like buoy.
- Kills the maintainability debt of one 700-line inline HTML string (all CSS + JS
  in one file); gives components, composables, TypeScript, and a real build.
- The design doc always intended a Vue SPA.

## Reference template

**ducky (#6762)** is the closest template — mirror it. Key facts learned from it:

- `lib/ducky/dashboard/`: `rsbuild.config.ts`, `tailwind.config.ts`,
  `tsconfig.json`, `postcss.config.cjs`, `env.d.ts`, `package.json`,
  `src/main.ts`, `src/App.vue`, `src/template.html`, `src/styles/main.css`,
  `src/components/*.vue`, `src/composables/*.ts`, `src/utils/*.ts`.
- Build: `npm run build` (`rsbuild build`) → `dashboard/dist/`; `build:check` runs
  `vue-tsc --noEmit` first. `dist/` and `node_modules/` are **gitignored**.
- Proxy sub-path: `rsbuild` `output.assetPrefix: 'auto'` + a `<base href="/">` in
  `template.html` that the **server rewrites at serve time from
  `X-Forwarded-Prefix`** so the bundle works under `/proxy/<name>/`. (buoy
  currently special-cases the proxy path; this replaces that.)
- Serving: `StaticFiles` **Mount** on `dashboard/dist` + an index handler that
  serves `dist/index.html` with the rewritten base href. Uses the shared
  **`iris.cluster.dashboard_common`** (`public` / `requires_auth` / `on_shutdown`).
- Deploy: `deploy.py` runs `npm run build` **before bundling** the workspace
  (with a `--skip-build` flag); the Iris container serves the pre-built `dist`
  (no npm on the worker).

## Target structure

```
lib/buoy/dashboard/
  package.json          # vue, @rsbuild/*, tailwind, vue-tsc, ts, plotly.js-dist-min
  rsbuild.config.ts     # entry src/main.ts, distPath dist, assetPrefix auto, title "buoy"
  tailwind.config.ts    # buoy palette via CSS vars (navy #0d3b66 accent)
  postcss.config.cjs
  tsconfig.json
  env.d.ts
  .gitignore            # node_modules, dist, .rsbuild
  src/
    main.ts
    App.vue             # layout shell: <Sidebar/> + <MainPane/>
    template.html       # <base href> rewritten from X-Forwarded-Prefix
    styles/main.css     # CSS vars (light; dark optional) + tailwind layers
    api.ts              # typed fetch wrapper over /api/*
    types.ts            # Manifest, RunRef, Metric series, etc.
    composables/
      useRuns.ts        # entity/project/user pickers + run list (server-side filter)
      useRun.ts         # load a run: mirror→poll→manifest/config/summary (mirrorJob)
      useLiveRefresh.ts # running-run 30s browser-driven re-mirror
    components/
      Sidebar.vue       # brand + entity/project/user fields + run list + collapse
      RunHeader.vue     # info pills + refetch button
      Tabs.vue          # summary / charts / profile (profile only when present)
      SummaryTab.vue    # searchable summary metrics + config tables
      ChartsTab.vue     # metric chip picker + chart grid
      MetricChart.vue   # one Plotly chart + per-chart bar (log x/y, smoothing,
                        #   full-screen, close, pan+scrollZoom)
      ProfileTab.vue    # xprof iframe (via /wrap → /xprof proxy)
      LoadingOverlay.vue
```

## Serving changes (`app.py`)

- Drop `INDEX_HTML = ...read_text()` + the `index` HTMLResponse.
- Add `StaticFiles` Mount on `_dashboard_dist()` and an index handler that reads
  `dist/index.html` and rewrites `<base href="/">` → the `X-Forwarded-Prefix`
  value (mirror ducky's `_index_html`). Keep all `/api/*`, `/wrap`, `/xprof`
  routes unchanged.
- Evaluate adopting `iris.cluster.dashboard_common` (`public`/`requires_auth`)
  for a consistent auth story (currently buoy has none — behind the proxy only).

## Build / deploy changes (`launch.py`)

- Add a pre-bundle `npm --prefix lib/buoy/dashboard install && … run build` step,
  with `--skip-build` to reuse an existing `dist` (mirror ducky's `deploy.py`).
- `dist/` gitignored; the built assets travel in the workspace bundle to Iris.

## CI

- Add a dashboard `build:check` (`vue-tsc --noEmit && rsbuild build`) step, gated
  on `lib/buoy/dashboard/**`, mirroring finelog/ducky's dashboard CI.

## Feature-parity checklist (must all survive the cutover)

- [ ] entity / project / user pickers with server-side filtering + status glyphs
- [ ] left sidebar run list + collapse toggle; deep-link `?entity&project&run&user`
- [ ] 3 tabs; **profile tab only when a profile exists**
- [ ] summary: searchable summary-metric + config tables
- [ ] charts: chip multi-select + searchable add; per-chart log x/y, EMA smoothing,
      full-screen, close, **pan default + scroll/pinch zoom**, single blue line,
      default-large sizing; "no data yet" for a starting run
- [ ] columnar `/api/metrics` consumption (`{x,y}`)
- [ ] profile: embedded xprof via `/wrap`→`/xprof` (Perfetto trace viewer)
- [ ] big centered loading overlay with mirror progress; sidebar stays usable
- [ ] live auto-refresh of running runs (browser-driven `mirrorJob`, no server watcher)
- [ ] refetch-artifacts button

## Phased migration (each phase builds + runs)

1. **Scaffold** (this step): dashboard build tree + config + `App.vue` shell +
   `api.ts` + `useRuns` so the sidebar/run-list renders against the live API.
   Old inline SPA stays the served frontend until parity.
2. Summary tab + RunHeader + `useRun` (mirrorJob).
3. Charts tab + `MetricChart` (Plotly wrapper) — the biggest chunk.
4. Profile tab (xprof iframe) + LoadingOverlay + `useLiveRefresh`.
5. **Cutover**: switch `app.py` to serve `dist`; add the `launch.py` build step;
   add CI; delete `static/index.html`.
6. Deploy + full parity smoke on Iris.

## Open decisions

- **Plotly**: bundle `plotly.js-dist-min` (offline, no CDN) vs keep the CDN
  `<script>`. Lean bundle (matches "self-contained", and the current CDN tag is
  the only external dependency). Cost: ~1 MB gz in the bundle.
- **Dark mode**: ducky ships a light/dark toggle; buoy is light-only today. Start
  light-only (CSS vars ready for dark later).
- **Auth**: adopt `dashboard_common.requires_auth` now, or keep proxy-only and add
  later. Likely a follow-up.

## Status

- 2026-07-01: plan written; phase 1 (scaffold: build tree + sidebar/run-list) done.
- 2026-07-01: phase 2 done — `useRun` (mirror→poll→load, nav-race guarded),
  `RunHeader` (info pills + refetch), `Tabs` (profile only when present),
  `SummaryTab` (searchable summary + config). `build:check` green. Next: phase 3
  (charts + `MetricChart` Plotly wrapper).
