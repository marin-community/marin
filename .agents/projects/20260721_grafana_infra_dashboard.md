# Compact Grafana infra dashboard

## TL;DR

The Grafana port preserved most data sources but lost the old status page's information density and several operator-facing details. At 1440×900, the old page showed the seven-day nightly matrix, CI health, Iris reachability, and fleet capacity. The current `infra.json` needs about four viewports to cover the same sections. It also omits nightly durations and links, the commit-history strip, aggregate CPU/memory/chip capacity, detailed provisioning outcomes, and the W&B hero-run context.

Use a hybrid Grafana design:

1. Add one purpose-built React panel plugin, `marin-infra-panel`, with three concrete views: the nightly matrix, the equal-width CI commit strip, and W&B token-axis charts. These are the three legacy views that Grafana's built-in panels cannot reproduce without dropping information or distorting the x-axis.
2. Keep the remaining dashboard in built-in Grafana panels. Recompose it as a compact overview with a single health rail, ferry status, fleet capacity, job state, provisioning, probes, and history charts. Link to the specialized K8s, Iris, Fleet, Training, and Pipelines dashboards for drill-down.
3. Do not enable raw HTML or embed a second Vue application. Grafana sanitizes Text-panel HTML by default; disabling sanitization would turn provisioned dashboard content into executable page code. Grafana's supported custom visualization API is React-based, and a panel plugin receives dashboard query data without bypassing Grafana's data-source boundary.

Legacy reference: https://loom.rjp.io/s/aoe4sxhi/artifacts/legacy-infra-dashboard

## Audit

The comparison uses `42d1ce5e95^:infra/status-page`, the last bespoke status-page source, its checked-in 1440×900 Playwright snapshot, and the current `infra/grafana/dashboards/infra.json`.

| Operator need | Legacy status page | Current Grafana `Infra` | Proposed result |
|---|---|---|---|
| Nightly regressions | 12 lanes × 7 days in 329 px; group/subgroup headers; status icon; duration; schedule and expected-duration evidence; attempts/recovery; collisions; click-through details | 14 Grafana grid rows; color only; no duration, grouping, attempt state, or link | Custom panel and enriched bridge rows restore the full matrix contract |
| Main CI | Latest commit plus a 100-commit equal-width colored strip; direct GitHub links | One 8-row stat plus an 8-row table | Custom commit-strip view restores equal-width cells, latest-commit context, and links in 3 grid rows |
| Service health | Prod/dev Iris and finelog reachability plus current latency, p50/p90/p99 or p50/max history, sample span/count, URLs, and update age | Three 4-row prod-Iris stats; synthetic probe latency has different semantics | Compact health rail plus a bridge-owned rolling health projection for the four service/environment pairs |
| Fleet capacity | Healthy total plus aggregate CPU, memory, chips; by-region history | Healthy total and a separate 8-row region table | Four compact stats and a small region bar/table; keep the 24h history chart |
| Job state | In-flight and trailing-24h totals with colored proportions | Plain 8-row table | Two compact bar-gauge/table views, split by bucket |
| Provisioning | Fleet success, ready/stockout/error/preempt counts, pools placing, pools with no ready outcome and a stockout/error, p50/p95, per-pool rows, fleet/per-region history | One fleet success-ratio history line | Fixed latest-cycle bridge projection restores fleet/pool rows; history restores fleet and per-region series |
| Synthetic probes | Colored pills with latency and sample age | Plain table | Compact table with colored background, latency unit, and age |
| Training context | Three W&B hero-run charts and report link in the Iris section | Removed from Infra; `training.json` is generic telltale data and is not equivalent | Restore train cross-entropy, Paloma macro loss, and MFU against cumulative tokens, following the report's pinned runset |
| Navigation | Direct links embedded in panels | No dashboard links | Header links to K8s, Iris, Fleet, Training, Pipelines, GitHub Actions, and iris.oa.dev |

The old page's strengths were hierarchy, density, and direct links. Its main weakness was duplication: it implemented charts, refresh state, authentication, and data fetching outside Grafana. The redesign keeps Grafana's shell and native panels where their constraints fit, and uses custom code only for the three layouts that native panels cannot reproduce faithfully.

## Grafana extension decision

Grafana supports four relevant surfaces:

- Text panels accept Markdown or sanitized HTML. `disable_sanitize_html` permits raw HTML, but that is a global security escape hatch and still provides no supported Vue lifecycle or data-frame API. Reject.
- Canvas panels place data-bound elements freely. They are useful for fixed diagrams, but do not repeat a dynamic 12×7 matrix or produce grouped headers. Reject.
- App plugins can add custom pages, panels, data sources, Scenes, and navigation. This would recreate the retired status application inside Grafana and require app enablement/provisioning. Reject for this scope.
- Panel plugins are supported React components that receive Grafana data frames, dimensions, theme, and time-range state. Choose one narrow internal plugin for the three proven bespoke views: `nightlies`, `commits`, and `wandb`. Each view has a separate input contract and component; there is no generic rendering DSL.

Official references:

- https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/text/
- https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/canvas/
- https://grafana.com/developers/plugin-tools/tutorials/build-a-panel-plugin
- https://grafana.com/developers/plugin-tools/key-concepts/plugin-types-usage
- https://grafana.com/developers/scenes/scene-app

The plugin will be private and image-bundled. It has a locked Node dependency tree, `plugin.json` with `type=panel`, ID `marin-infra-panel`, and a Grafana dependency covering the pinned 13.1.1 runtime. A Node Docker stage builds with Grafana's supported plugin webpack tooling, and the runtime stage receives only `dist/` under `/var/lib/grafana/plugins/marin-infra-panel`. Grafana requires plugin signatures by default. For the first internal version, `GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=marin-infra-panel` allows only this image-reviewed plugin; no plugin is downloaded at runtime.

Private signing is the hardening path, not part of this change. It requires `GF_SERVER_ROOT_URL=https://grafana.oa.dev/`, a signing command whose `--rootUrls` value matches it exactly, and a Grafana Cloud access-policy token from the organization that owns the plugin-ID prefix. Until that credential exists, startup and browser smoke tests must fail if the allowlist is wrong or the plugin is absent.

## Data and rendering design

`project_nightlies()` already returns the base long rows. The current bridge immediately pivots them to status-only wide rows for the state-timeline panel, discarding useful fields. Restore the remaining legacy projection before serving long rows: run attempt count, actor, SHA, fetched time and source error, expected instant, schedule label, overdue provenance, expected-duration range/provenance/evidence, prior attempt outcomes/links/errors, recovered state, and colliding scheduled-run links. Attempt calls remain bounded to selected reruns and match the retired page's four-request concurrency limit. Then delete `nightly_matrix()`.

Each row retains:

```text
ts, date, lane_id, lane, label, group, subgroup, lane_order,
repository, workflow, workflow_url, schedule_label, expected_at,
expected_min_seconds, expected_max_seconds, expected_provenance,
expected_evidence_urls, state, status_code, healthy, due,
duration_state, duration_seconds, run_id, run_attempt, conclusion,
sha, actor, url, recovered, prior_attempts, attempt_error,
collision_urls, source_fetched_at, source_error
```

The custom panel groups rows by configured lane order and descending date. It renders:

- fixed date column, then proportional lane columns;
- Marin/Forks and Training/Data/Cluster/Evaluation/RL/Inference headers;
- status icon plus formatted duration inside each 32 px cell;
- green success, amber slow, stronger amber very-slow, red failure/missing, blue not-yet-due, muted unscheduled;
- a patterned outline for suspicious too-short successes;
- a keyboard-accessible detail card with full lane name, conclusion/state, duration class, schedule/baseline provenance, SHA/actor, prior attempts, collision links, and source error/age;
- one-click GitHub run link when present, otherwise workflow link;
- `Today: healthy/due` summary and a compact legend.

The matrix is a semantic table: caption, row headers, group/column scopes, and keyboard-focusable anchors for actionable cells. Every cell has a non-color status icon and an accessible name containing lane, UTC date, status, and duration. Focus is visible. Details open from focus/click as well as hover and render through a portal so panel overflow cannot clip them. Below 1200 px, the table keeps a fixed date column and gains horizontal scrolling instead of shrinking text into unreadability.

Normalization fails visibly on missing required fields, duplicate `(lane_id,date)` cells, unknown enums, or incompatible multiple frames. It handles reordered fields and partial lane-source failures. Tests cover running/cancelled/timed-out/missing/unavailable/recovered states, collision and attempt links, narrow width, light/dark contrast, keyboard navigation, and an axe scan scoped to the panel.

The commit view receives the `/builds` rows including hidden `url` and `avatar_url` fields. It uses equal flex widths, so commit cadence cannot distort the strip. Each cell is a direct link with SHA, status, relative time, and headline in its accessible name. The latest commit line retains the legacy author decoration and finalized success rate.

The W&B bridge port remains anonymous and follows the report's first runset on every refresh. It returns long `chart, run, run_state, tokens, value` rows plus report title/URL. Three plugin instances select the named chart and render raw plus debiased-EMA series using cumulative tokens as numeric x. The y-domain keeps the legacy warmup clipping rule. The report link and non-running run states remain visible.

## Fixed bridge projections

Native Grafana panels should not reconstruct latest-cycle EAV data. Add fixed read-only endpoints, backed by the existing finelog source:

```text
GET /overview/probes
  probe, up, latency_ms, collected_at

GET /overview/provisioning
  one `scope=fleet` row plus one row per resource_type/scale_group/zone:
  collected_at, window_hours, ready, stockout, error, preempted,
  outcomes, success_ratio, pools_placing, pools_no_ready_outcome,
  latency_p50_seconds, latency_p95_seconds

GET /overview/provisioning/history
  collected_at, scope(fleet|region), region, success_ratio

GET /overview/control_plane
  collected_at, environment(prod|dev), service(iris|finelog), reachable,
  latency_ms, p50_ms, p90_ms, p99_ms, max_ms, sample_count, sample_span_ms, url

GET /overview/wandb
  report_title, report_url, chart, run, run_state, tokens, value
```

Provisioning selects one shared `MAX(collected_at)` over `provision_%` in a six-hour bounded window before pivoting. It never mixes each metric's independent latest row. Per-pool success is computed as `ready/outcomes`; zero outcomes produce null. `pools_no_ready_outcome` is the UI label for `provision_pools_stockout_dead`, avoiding the misleading word "stuck." Empty or stale cycles render a timestamped no-data state. History groups `provision_ready` and `provision_outcomes` by cycle and restores both fleet and region series.

Probe projection pairs latest `probe_up` and latency by probe, retains `collected_at`, and marks samples older than two expected probe intervals stale. Service health preserves the old four-series semantics. Its bounded in-memory history is sampled by the 30-second dashboard refresh and has the same restart-loss caveat as the retired status page; synthetic probe history remains the durable companion.

## Dashboard layout

Target: at 1440×900, the full nightly matrix, CI section, health rail, and fleet headline remain visible without horizontal scrolling. Their exact grid budget is 18 rows: nightly `y=0,h=9`; CI/ferry `y=9,h=3`; health rail `y=12,h=3`; fleet headline `y=15,h=3`. With Grafana's 30 px grid row and 8 px row gap, this content is 676 px before dashboard chrome. Playwright asserts the fleet headline's bottom edge is inside the 900 px viewport and the dashboard has no horizontal overflow.

```text
┌ Infra · links · refresh ───────────────────────────────────────────────┐
│ Nightly regression matrix (custom panel, y0 h9)                      │
├ CI strip (y9 h3) ──────────────────┬ Ferry status / failures ─────────┤
├ 4-service health ┬ workers ┬ CPU ┬ memory ┬ chips ┬ probes (y12 h3) ┤
├ Region headline ───────────────────┬ Provisioning headline (y15 h3) ─┤
├ Workers by region / 24h ───────────┬ Provisioning success / 24h ─────┤
├ Jobs now / 24h ────────────────────┬ Provisioning pool outcomes ─────┤
└ Control-plane latency ─────────────┴ Synthetic probe state ──────────┘
```

Below the first viewport, history/table panels use six grid rows and W&B charts use seven. Density relies only on verified panel options: transparent background, exact grid positions, table cell-height `sm`, hidden table footer, explicit field widths, compact legends, units, thresholds, value mappings, and data links. It does not assume a global 11 px font option. Raw identifiers and URL helper fields stay hidden but remain selected for links.

Dashboard links use stable UIDs (`k8s`, `iris`, `fleet`, `training`, `pipelines`), preserve the current time range, and open in the same tab. GitHub Actions, the exact W&B report, and `https://iris.oa.dev` are external links that open in a new tab.

## Validation and artifacts

- Python behavior tests for long nightly rows and any restored bridge projection.
- Plugin unit tests for normalization, all status/duration presentations, accessibility names, and links.
- `infra/grafana` pytest suite.
- production Docker image smoke: Grafana registers the plugin and provisions `infra.json`.
- repository pre-commit and lint-catalog review.
- Playwright against Grafana 13.1.1. A deterministic fixture bridge is mounted into the production image for the test container. The test waits for named rendered content, opens `/d/infra`, checks the compact overview and deep-dive sections, and captures three 1280×720 viewports.
- Startup smoke checks the Grafana log/API and rendered panels so a missing plugin, rejected signature, or wrong unsigned allowlist fails even though dashboard JSON provisioning itself would succeed.
- Screenshots are published as Weaver image artifacts.
- final implementation report artifact containing the before/after comparison, test results, known signing caveat, screenshot links, commit, and PR.

## Scope limits

- Keep the existing Infinity bridge and provisioned alerting. The redesign changes presentation, not alert evaluation.
- Do not restore a separate Hono/React service. The W&B hero charts are restored because `training.json` is not equivalent.
- Do not add user-triggered control-plane actions. This remains a read-only operational view.
- Do not add a generic rendering DSL or configuration system to the three-view plugin.

## Peer review disposition

Two independent Codex reviews checked revision 1 against the current Grafana service and the deleted status page. The requested Claude Fable lane produced no output and was stopped; a second Codex reviewer replaced it.

Accepted findings changed the implementation materially: full nightly attempt/recovery metadata, W&B parity, a custom equal-width commit strip, fixed latest-cycle EAV projections, four-service health semantics, exact grid math, semantic-table accessibility, explicit unsigned-plugin mechanics, and an executable fixture-server Playwright harness. The design also drops the inaccurate claim that `training.json` replaces the W&B report.

One direction remains rejected: rebuilding the entire status page as a Grafana app/Scenes page. Three custom panel views are enough for the layouts native panels cannot express. The rest stays in normal provisioned panels, which preserves Grafana time range, refresh, dashboard links, data-source execution, and drill-down behavior.
