import { useEffect, useMemo, useState } from "react";
import {
  SequenceMetricCard,
  RequestMetricOverTime,
} from "@/components/inference/agentic-point/request-metric-cards";
import {
  KvCacheUtilizationCard,
  RequestActivityCard,
  PrefixCacheHitRateCard,
  ThroughputCard,
  PromptTokenSourceCard,
  InflightUniqueTokensCard,
  type RequestActivityView,
} from "@/components/inference/agentic-point/server-metric-cards";
import { RequestTimelineView } from "@/components/inference/agentic-point/request-timeline";
import {
  phaseBoundarySec,
  sliceRequestChartDataByPhase,
  sliceServerSeriesByPhase,
  type StagePhase,
} from "@/components/inference/agentic-point/phase-slice";
import type { ThroughputSeriesKey } from "@/components/inference/agentic-point/time-series-math";
import {
  api,
  type Run,
  modelName,
  sourceUrl,
  sourceCommit,
  measurementLabel,
  topologyLabel,
} from "./types";
import { seriesKey } from "./metrics";
import { requestTimeline, requestChartData, type RunData } from "./data";
import { setParameters, useRoute } from "./navigation";

export function RunView({ run: listedRun, runs }: { run: Run; runs: Run[] }) {
  const { params } = useRoute();
  const view = params.get("view") || "diagnostics";
  const phase = (params.get("phase") || "profiling") as StagePhase;
  const [data, setData] = useState<RunData | null>(null);
  const run = data?.run || listedRun;
  const server = data?.series || null;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activity, setActivity] = useState<RequestActivityView>("queue");
  const [throughput, setThroughput] = useState<
    ReadonlySet<ThroughputSeriesKey>
  >(new Set(["input", "decode"]));
  const [source, setSource] = useState("all");
  useEffect(() => {
    let active = true;
    api<RunData>(`runs/${listedRun.key}/data`)
      .then((value) => {
        if (active) {
          setData(value);
          setLoading(false);
        }
      })
      .catch((e: Error) => {
        if (active) {
          setError(e.message);
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [listedRun.key]);
  const timeline = useMemo(() => (data ? requestTimeline(data) : null), [data]);
  const chartData = useMemo(
    () => (timeline ? requestChartData(timeline) : null),
    [timeline],
  );
  const phaseData = useMemo(
    () => (chartData ? sliceRequestChartDataByPhase(chartData, phase) : null),
    [chartData, phase],
  );
  const selectedSource = server?.metricSources?.find(
    (s) => s.source.id === source,
  );
  const sourceSeries = selectedSource
    ? {
        ...selectedSource,
        prefillTps: selectedSource.promptTps,
        decodeTps: selectedSource.generationTps,
      }
    : server;
  const sliced = useMemo(
    () =>
      server && sourceSeries
        ? sliceServerSeriesByPhase(
            sourceSeries,
            phase,
            phaseBoundarySec(server, chartData),
            server.durationS,
          )
        : null,
    [server, sourceSeries, chartData, phase],
  );
  const siblings = runs
    .filter((r) => seriesKey(r) === seriesKey(run) && r.model === run.model)
    .toSorted((a, b) => a.concurrency - b.concurrency);
  const pool =
    typeof run.aggregate.kv_cache_pool_tokens === "number"
      ? run.aggregate.kv_cache_pool_tokens
      : null;
  return (
    <>
      <a href="#/chart">← Performance chart</a>
      <h1>
        {modelName(run)} · {run.hardware} · C{run.concurrency}
      </h1>
      <p className="muted">
        {run.date} UTC · {run.framework} · {run.precision.toUpperCase()} ·{" "}
        {run.physical_gpus} physical GPUs on {run.hosts} hosts
        {" · "}
        {measurementLabel(run)} · {topologyLabel(run)}
      </p>
      {run.calibration && (
        <section className="panel" aria-label="Calibration window">
          <h2>15-minute calibration</h2>
          <p>
            When a calibration ends, the harness cancels unfinished requests.
            Cancellation share rises with load, so full-hour throughput and
            rankings may differ. Use full-hour measurements to choose a final
            operating load.
          </p>
          <p>
            {run.calibration.profiling_sent} profiling requests sent ·{" "}
            {run.profiling_requests} completed ·{" "}
            {run.calibration.profiling_cancelled} tail cancellations ·{" "}
            {run.calibration.profiling_errors} profiling errors.
          </p>
          <p>
            The native request export omits cancelled tail requests. Their
            counts come from the phase log; the timeline contains exported
            requests only.
          </p>
        </section>
      )}
      {run.model.includes("Hero") &&
        run.configuration.enable_prefix_caching === false && (
          <p className="notice">
            Provisional Hero run: prefix caching disabled, max batched tokens{" "}
            {String(run.configuration.max_num_batched_tokens)}, max sequences{" "}
            {String(run.configuration.max_num_seqs)}. These are the recorded
            settings.
          </p>
        )}
      <div className="toolbar">
        Sibling loads/configurations:
        {siblings.map((r) => (
          <a
            aria-current={r.key === run.key ? "page" : undefined}
            key={r.key}
            href={`#/run/${r.key}?view=${view}`}
          >
            {r.hardware} C{r.concurrency}
          </a>
        ))}
      </div>
      <div className="stats-grid">
        <div className="panel">
          <small>Output tokens/s/GPU</small>
          <strong data-testid="run-output">
            {run.metrics.throughput.per_gpu.output_tput_tps.toFixed(5)}
          </strong>
        </div>
        <div className="panel">
          <small>P90 TTFT</small>
          <strong>{run.metrics.latency.ttft.p90.toFixed(5)} s</strong>
        </div>
        <div className="panel">
          <small>P90 full-response ITL</small>
          <strong>
            {run.metrics.latency.full_response_itl?.p90.toFixed(5)} s
          </strong>
        </div>
        <div className="panel">
          <small>Profiling / all requests</small>
          <strong>
            {run.profiling_requests} / {run.request_count}
          </strong>
        </div>
      </div>
      <div className="toolbar">
        <button
          aria-pressed={view === "diagnostics"}
          onClick={() => setParameters({ view: "diagnostics" })}
        >
          Per-point diagnostics
        </button>
        <button
          aria-pressed={view === "timeline"}
          onClick={() => setParameters({ view: "timeline" })}
        >
          Request timeline
        </button>
        {view === "diagnostics" && (
          <label>
            Stage
            <select
              aria-label="Stage"
              value={phase}
              onChange={(e) => setParameters({ phase: e.target.value })}
            >
              <option value="profiling">Profiling</option>
              <option value="warmup">Warmup</option>
            </select>
          </label>
        )}
        {view === "diagnostics" && !!server?.metricSources.length && (
          <label>
            Server metrics source
            <select
              aria-label="Server metrics source"
              value={source}
              onChange={(e) => setSource(e.target.value)}
            >
              <option value="all">All endpoints</option>
              {server.metricSources.map((s) => (
                <option key={s.source.id} value={s.source.id}>
                  {s.source.role} ·{" "}
                  {s.source.workerId || s.source.endpointUrl || s.source.engine}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {view === "diagnostics" && phase === "warmup" && (
        <p className="notice">
          Warmup is a cache-warming pass capped at one output token. Decode and
          interactivity can be unavailable.
        </p>
      )}
      {error && <p role="alert">{error}</p>}
      {view === "diagnostics" && (
        <p className="muted">
          In-flight views estimate tokens from request lifetimes and shared
          conversation prefixes. Server KV utilization uses measured gauges.
        </p>
      )}
      {view === "timeline" ? (
        timeline ? (
          <section className="panel">
            <RequestTimelineView
              data={timeline}
              datasetSlug={`${encodeURIComponent(run.dataset)}@${run.dataset_revision}`}
              pointId={run.key}
            />
          </section>
        ) : loading ? (
          <p>Loading timeline…</p>
        ) : (
          <p>No request timeline is available.</p>
        )
      ) : (
        <div className="diagnostic-grid">
          <SequenceMetricCard
            metric="isl"
            timeline={phaseData}
            timelineLoading={loading}
          />
          <SequenceMetricCard
            metric="osl"
            timeline={phaseData}
            timelineLoading={loading}
          />
          <RequestMetricOverTime
            title="Interactivity over time"
            metric="interactivity"
            timeline={phaseData}
            isLoading={loading}
          />
          <RequestMetricOverTime
            title="TTFT and E2E over time"
            metric="ttft"
            latencySelector
            timeline={phaseData}
            isLoading={loading}
          />
          <RequestActivityCard
            sliced={sliced}
            phaseTimeline={phaseData}
            timelineLoading={loading}
            view={activity}
            onViewChange={setActivity}
          />
          {server ? (
            <>
              <KvCacheUtilizationCard sliced={sliced} />
              <PrefixCacheHitRateCard sliced={sliced} />
              <ThroughputCard
                sliced={sliced}
                selectedSource={selectedSource}
                selected={throughput}
                onSelectedChange={setThroughput}
              />
              <PromptTokenSourceCard sliced={sliced} />
            </>
          ) : (
            !loading && <p>No server metrics were collected for this run.</p>
          )}
          <InflightUniqueTokensCard
            phaseTimeline={phaseData}
            timelineLoading={loading}
            kvCachePoolTokens={pool}
          />
        </div>
      )}
      <section className="panel">
        <h2>Runtime, topology and provenance</h2>
        <p>
          {topologyLabel(run)} · {run.physical_gpus} GPUs · {run.hosts} hosts ·
          KV offload {String(run.aggregate.kv_offloading ?? "not reported")}
        </p>
        <p>
          <a href={sourceUrl(run)} target="_blank" rel="noreferrer">
            Original {run.calibration ? "artifacts" : "GitHub attempt"}
          </a>{" "}
          · source commit <code>{sourceCommit(run)}</code>
        </p>
        {run.source.kind === "private-calibration" && (
          <p>
            Iris job <code>{run.source.job_id}</code> · attempt{" "}
            <code>{run.source.attempt_uid}</code>
            <br />
            Raw archive <code>{run.source.artifact_uri}</code>
          </p>
        )}
        <p>
          Runtime <code>{run.runtime}</code>
        </p>
        <p>
          Recipe <code>{run.recipe}</code>
        </p>
        <p>
          Workload{" "}
          <code>
            {run.workload_key ||
              "Unknown: reimport the saved bundle for matched comparisons"}
          </code>
        </p>
        <p>
          Model revision <code>{run.model_revision}</code>
        </p>
        <p>
          Corpus {run.dataset} @ <code>{run.dataset_revision}</code>
        </p>
        <details>
          <summary>Full configuration and source hashes</summary>
          <pre>{JSON.stringify(run, null, 2)}</pre>
        </details>
      </section>
    </>
  );
}
