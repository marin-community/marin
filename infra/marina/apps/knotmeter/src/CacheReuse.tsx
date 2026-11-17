import {
  AggregateChart,
  type AggregatePoint,
} from "@/components/inference/agentic-point/aggregate-chart";
import { ExpandableChart } from "@/components/inference/agentic-point/expandable-chart";
import { CHART_SIZES } from "@/components/inference/agentic-point/chart-shared";
import {
  type Run,
  modelName,
  sourceLabel,
  measurementLabel,
  topologyLabel,
} from "./types";
import { setParameters, useRoute } from "./navigation";

interface CacheShare {
  hbm: number;
  host: number;
  unreused: number;
  theoretical: number | null;
  reportedTotal: number;
}
const finite = (n: unknown): n is number =>
  typeof n === "number" && Number.isFinite(n);
// The same tier clamping as InferenceX cacheShareOf; the source is the saved profiling aggregate.
export function cacheShare(run: Run): CacheShare | null {
  const cache = run.aggregate.server_metrics?.cache;
  if (!cache) return null;
  const gpu = cache.gpu_cache_hit_rate;
  const hostRaw = finite(cache.cpu_cache_hit_rate)
    ? cache.cpu_cache_hit_rate
    : cache.external_cache_hit_rate;
  if (!finite(gpu) || !finite(hostRaw)) return null;
  const hbm = Math.max(0, Math.min(1, gpu));
  const host = Math.max(0, Math.min(1 - hbm, hostRaw));
  const theoretical = run.metrics.cache?.theoretical_cache_hit_rate;
  return {
    hbm,
    host,
    unreused: Math.max(0, 1 - hbm - host),
    theoretical: finite(theoretical) ? theoretical : null,
    reportedTotal: gpu + hostRaw,
  };
}

export function CacheReuse({ runs }: { runs: Run[] }) {
  const { params } = useRoute();
  const models = [...new Set(runs.map((r) => r.model))];
  const model =
    params.get("model") ||
    models.find((m) => m.includes("Snowball")) ||
    models[0];
  const hardware = params.get("hardware") || "";
  const date = params.get("date") || "";
  const filtered = runs
    .filter(
      (r) =>
        r.model === model &&
        (!hardware || r.hardware === hardware) &&
        (!date || r.date === date),
    )
    .toSorted(
      (a, b) =>
        a.hardware.localeCompare(b.hardware) ||
        a.date.localeCompare(b.date) ||
        a.concurrency - b.concurrency,
    );
  const points = (
    metric: "isl" | "osl" | "kvCacheUtil" | "prefixCacheHitRate",
  ): AggregatePoint[] =>
    filtered.map((r) => ({
      label: `${r.hardware} C${r.concurrency} · ${topologyLabel(r)} · ${r.date} · ${sourceLabel(r)} · ${measurementLabel(r)}`,
      values: r.diagnostic_aggregates?.[metric] || {},
    }));
  return (
    <>
      <h1>Cache reuse and configuration statistics</h1>
      <p className="muted">
        Measured shares use the saved processor’s profiling-window counters. The
        theoretical ceiling comes from corpus prefix cacheability. In-flight
        estimates and per-interval hit-rate charts use separate definitions.
      </p>
      <div className="controls">
        <label>
          Model
          <select
            aria-label="Model"
            value={model}
            onChange={(e) =>
              setParameters({ model: e.target.value, hardware: "", date: "" })
            }
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m.split("/").at(-1)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Hardware
          <select
            aria-label="Hardware"
            value={hardware}
            onChange={(e) => setParameters({ hardware: e.target.value })}
          >
            <option value="">All hardware</option>
            {[
              ...new Set(
                runs.filter((r) => r.model === model).map((r) => r.hardware),
              ),
            ].map((h) => (
              <option key={h}>{h}</option>
            ))}
          </select>
        </label>
        <label>
          UTC date
          <select
            aria-label="UTC date"
            value={date}
            onChange={(e) => setParameters({ date: e.target.value })}
          >
            <option value="">All dates</option>
            {[
              ...new Set(
                runs.filter((r) => r.model === model).map((r) => r.date),
              ),
            ]
              .sort()
              .reverse()
              .map((d) => (
                <option key={d}>{d}</option>
              ))}
          </select>
        </label>
      </div>
      <section className="panel">
        <h2>Measured cache tiers and theoretical ceiling</h2>
        <div className="toolbar">
          <span className="tier-hbm">HBM cache hit</span>
          <span className="tier-host">Host / external cache hit</span>
          <span className="tier-unreused">Not reused</span>
          <span>Theoretical ceiling │</span>
        </div>
        {filtered.map((r) => {
          const share = cacheShare(r);
          return (
            <div className="cache-row" key={r.key}>
              <a href={`#/run/${r.key}`}>
                {r.hardware} C{r.concurrency}
                <small>
                  {r.date} · {sourceLabel(r)} · {topologyLabel(r)} ·{" "}
                  {measurementLabel(r)}
                </small>
              </a>
              {share ? (
                <div>
                  <div
                    className="cache-bar"
                    role="img"
                    aria-label={`HBM ${(share.hbm * 100).toFixed(3)}%, host ${(share.host * 100).toFixed(3)}%, not reused ${(share.unreused * 100).toFixed(3)}%`}
                  >
                    <div
                      style={{
                        width: `${share.hbm * 100}%`,
                        background: "#3f9fe0",
                      }}
                    />
                    <div
                      style={{
                        width: `${share.host * 100}%`,
                        background: "#f2b13a",
                      }}
                    />
                    <div
                      style={{
                        width: `${share.unreused * 100}%`,
                        background: "#5f6672",
                      }}
                    />
                    {share.theoretical !== null && (
                      <span
                        className="ceiling"
                        style={{ left: `${share.theoretical * 100}%` }}
                        title={`Theoretical ceiling ${(share.theoretical * 100).toFixed(3)}%`}
                      />
                    )}
                  </div>
                  <p>
                    HBM {(share.hbm * 100).toFixed(3)}% · host{" "}
                    {(share.host * 100).toFixed(3)}% · not reused{" "}
                    {(share.unreused * 100).toFixed(3)}% · theoretical{" "}
                    {share.theoretical === null
                      ? "not available"
                      : `${(share.theoretical * 100).toFixed(3)}%`}
                  </p>
                  {share.reportedTotal > 1 && (
                    <p className="notice">
                      The reported tier rates sum to more than 100%; the stacked
                      display is clipped.
                    </p>
                  )}
                </div>
              ) : (
                <p>Measured tier shares are unavailable for this run.</p>
              )}
            </div>
          );
        })}
      </section>
      <h2>Aggregate token and cache statistics across configurations</h2>
      <p className="muted">
        Input/output lengths use non-cancelled profiling requests. KV
        utilization and prefix-hit percentiles use the pinned upstream
        server-sample calculation. An absent series has no percentile markers.
      </p>
      <div className="diagnostic-grid">
        {(
          [
            ["isl", "ISL distribution (across configs)", "tokens"],
            ["osl", "OSL distribution (across configs)", "tokens"],
            ["kvCacheUtil", "KV cache utilization (across configs)", "%"],
            [
              "prefixCacheHitRate",
              "Prefix cache hit rate (across configs)",
              "%",
            ],
          ] as const
        ).map(([metric, title, unit]) => (
          <ExpandableChart
            key={metric}
            title={title}
            render={(expanded) => (
              <AggregateChart
                points={points(metric)}
                unit={unit}
                yMax={unit === "%" ? 1 : undefined}
                yFmt={
                  unit === "%" ? (v) => `${(v * 100).toFixed(1)}%` : undefined
                }
                {...(expanded ? CHART_SIZES.expanded : CHART_SIZES.inline)}
              />
            )}
          />
        ))}
      </div>
      <section className="panel table-scroll">
        <h2>Measured profiling totals</h2>
        <p className="muted">
          Calibration token totals use the saved worker-endpoint counters.
          Server output includes work on unfinished requests; benchmark
          throughput uses completed client requests. The original aggregates
          retain the mirrored router and worker sums in provenance.
        </p>
        <table>
          <thead>
            <tr>
              <th>Configuration</th>
              <th>Prompt tokens</th>
              <th>Output tokens</th>
              <th>Completed requests</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => {
              const tokens = r.calibration
                ? r.diagnostic_totals
                : r.aggregate.server_metrics?.tokens;
              return (
                <tr key={r.key}>
                  <td>
                    <a href={`#/run/${r.key}`}>
                      {modelName(r)} · {r.hardware} C{r.concurrency} · {r.date}
                      <small>
                        {topologyLabel(r)} · {sourceLabel(r)} ·{" "}
                        {measurementLabel(r)}
                      </small>
                    </a>
                  </td>
                  <td>
                    {tokens?.prompt_total?.toLocaleString() ?? "Not measured"}
                  </td>
                  <td>
                    {tokens?.generation_total?.toLocaleString() ??
                      "Not measured"}
                  </td>
                  <td>
                    {tokens?.requests_completed?.toLocaleString() ??
                      "Not measured"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    </>
  );
}
