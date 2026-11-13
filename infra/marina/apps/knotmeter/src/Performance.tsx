import { useMemo, useRef, useState } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as d3 from "d3";
import { usePerfRulers } from "./usePerfRulers";
import { D3Chart } from "@/lib/d3-chart/D3Chart";
import type {
  D3ChartHandle,
  LayerConfig,
  CustomLayerConfig,
  RenderContext,
} from "@/lib/d3-chart/D3Chart/types";
import type { ContinuousScale } from "@/lib/d3-chart/types";
import {
  placeLineLabels,
  renderLineLabels,
  avoidPointLabelCollisions,
} from "@/components/inference/ui/line-label-layer";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { globalParetoFrontier } from "@/lib/charts/pareto-frontier";
import {
  type Run,
  modelName,
  sourceUrl,
  sourceLabel,
  seriesLabel,
  topologyLabel,
  measurementSeconds,
  measurementLabel,
} from "./types";
import { setParameters, useRoute } from "./navigation";
import {
  METRICS,
  AXES,
  type Metric,
  type Axis,
  type TokenBasis,
  metricValue,
  axisValue,
  seriesKey,
  comparisonScope,
} from "./metrics";

const COLORS = [
  "#3b82f6",
  "#e87722",
  "#10b981",
  "#a855f7",
  "#ef4444",
  "#0e9fa8",
  "#d946a2",
];
interface Point {
  x: number;
  y: number;
  precision: string;
  series: string;
  run: Run;
}

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const csvValue = (value: unknown) =>
  `"${String(value ?? "").replaceAll('"', '""')}"`;

export function Performance({ runs }: { runs: Run[] }) {
  const { params } = useRoute();
  const models = [...new Set(runs.map((r) => r.model))];
  const selected =
    params.get("model") ||
    models.find((m) => m.includes("Snowball")) ||
    models[0];
  const metric = (params.get("metric") || "output") as Metric;
  const axis = (params.get("axis") || "ttft") as Axis;
  const basis = (params.get("basis") || "total") as TokenBasis;
  const percentile = params.get("percentile") || "p90";
  const hardware = params.get("hardware") || "";
  const runSelection = params.get("runs");
  const windows = [
    ...new Set(
      runs.filter((r) => r.model === selected).map(measurementSeconds),
    ),
  ].sort((a, b) => b - a);
  const pickedWindow = runs.find((r) =>
    runSelection?.split(",").includes(r.key),
  );
  const measurement = Number(
    params.get("window") ||
      (pickedWindow ? measurementSeconds(pickedWindow) : windows[0]),
  );
  const dates = [
    ...new Set(
      runs
        .filter(
          (r) =>
            r.model === selected &&
            measurementSeconds(r) === measurement &&
            (!hardware || r.hardware === hardware),
        )
        .map((r) => r.date),
    ),
  ]
    .sort()
    .reverse();
  const dateSelection = params.get("date") || (runSelection ? "all" : "latest");
  const date =
    dateSelection === "latest"
      ? dates[0] || ""
      : dateSelection === "all"
        ? ""
        : dateSelection;
  const log = params.get("scale") === "log";
  const optimal = params.get("optimal") === "1";
  const showLabels = params.get("labels") !== "0";
  const labelStyle = params.get("labelstyle") || "concurrency";
  const showLines = params.get("linelabels") === "1";
  const showLegend = params.get("legend") !== "0";
  const contrast = params.get("contrast") === "1";
  const table = params.get("view") === "table";
  const selectedRuns = runSelection ? new Set(runSelection.split(",")) : null;
  const hidden = new Set((params.get("hidden") || "").split(","));
  const [point, setPoint] = useState<Point | null>(null);
  const [inspected, setInspected] = useState<string | null>(null);
  const [legendSearch, setLegendSearch] = useState("");
  const [sort, setSort] = useState("x");
  const [page, setPage] = useState(0);
  const [message, setMessage] = useState("");
  const chart = useRef<D3ChartHandle>(null);
  const rulerMode = params.get("ruler") === "1";
  const filtered = runs.filter(
    (r) =>
      r.model === selected &&
      measurementSeconds(r) === measurement &&
      (!hardware || r.hardware === hardware) &&
      (!date || r.date === date) &&
      (!selectedRuns || selectedRuns.has(r.key)),
  );
  const series = [
    ...new Map(
      filtered.map((run) => [seriesKey(run), { key: seriesKey(run), run }]),
    ).values(),
  ];
  const seriesOf = seriesKey;
  const color = (key: string) => {
    const i = series.findIndex((s) => s.key === key);
    return contrast
      ? ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"][i % 5]
      : COLORS[i % COLORS.length];
  };
  const rulers = usePerfRulers(
    rulerMode,
    params.get("rulers"),
    Object.fromEntries(
      series.map((s) => [`roofline-${s.key}`, comparisonScope(s.run)]),
    ),
  );
  const points = useMemo<Point[]>(
    () =>
      filtered.flatMap((run) => {
        const x = axisValue(run, axis, percentile);
        const y = metricValue(run, metric, basis);
        return x !== null && y !== null && y > 0
          ? [{ x, y, precision: run.precision, series: seriesOf(run), run }]
          : [];
      }),
    [
      runs,
      selected,
      hardware,
      date,
      axis,
      metric,
      percentile,
      runSelection,
      basis,
      measurement,
    ],
  );
  const visible = points.filter((p) => !hidden.has(p.series));
  const frontier = globalParetoFrontier(
    visible,
    axis === "interactivity" || axis === "normalized",
    true,
  );
  const plotted = optimal ? frontier : visible;
  const lines = Object.fromEntries(
    series.map((s) => [
      s.key,
      globalParetoFrontier(
        plotted.filter((p) => p.series === s.key),
        axis === "interactivity" || axis === "normalized",
        true,
      ),
    ]),
  );
  const drawLabels = (
    group: d3.Selection<SVGGElement, unknown, null, undefined>,
    ctx: RenderContext,
  ) => {
    const x = ctx.xScale as ContinuousScale;
    const y = ctx.yScale as ContinuousScale;
    const labels = showLines
      ? placeLineLabels(
          series
            .filter((s) => lines[s.key].length > 0)
            .map((s) => ({
              key: s.key,
              seriesId: s.key,
              label: seriesLabel(s.run),
              color: color(s.key),
              points: lines[s.key],
            })),
          x,
          y,
          { collisionWidth: 220 },
        )
      : [];
    renderLineLabels(group, labels, { seriesAttribute: "data-series-id" });
    if (showLabels) avoidPointLabelCollisions(group);
  };
  const labelsLayer: CustomLayerConfig = {
    type: "custom",
    key: "labels",
    render: drawLabels,
    onDisplayUpdate: drawLabels,
    onZoom: (group, ctx) =>
      drawLabels(group, {
        ...ctx,
        xScale: ctx.newXScale,
        yScale: ctx.newYScale,
      }),
  };
  const domain = (values: number[]): [number, number] => {
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (!values.length) return log ? [0.1, 1] : [0, 1];
    return log ? [min / 1.25, Math.max(max * 1.25, min * 2)] : [0, max * 1.2];
  };
  const layers: LayerConfig<Point>[] = [
    {
      type: "line",
      lines,
      config: { getColor: color, curve: d3.curveMonotoneX },
    },
    {
      type: "scatter",
      data: plotted,
      keyFn: (p) => p.run.key,
      config: {
        getColor: (p) => color(p.series),
        getLabelText: (p) =>
          labelStyle === "parallelism"
            ? topologyLabel(p.run)
            : labelStyle === "throughput"
              ? p.y.toFixed(2)
              : `C${p.run.concurrency}`,
        foreground: "var(--ink)",
        hideLabels: !showLabels,
        dataAttrs: { "run-key": (p) => p.run.key },
      },
    },
    rulers.layer,
    labelsLayer,
  ];
  const update = (values: Record<string, string>) => {
    if ("model" in values) {
      setLegendSearch("");
      values = {
        hardware: "",
        date: "",
        runs: "",
        hidden: "",
        window: "",
        ...values,
      };
    } else if ("window" in values) {
      values = { date: "all", runs: "", hidden: "", ...values };
    } else if ("hardware" in values) {
      setLegendSearch("");
      values = { date: "", runs: "", hidden: "", ...values };
    } else if (values.runs) {
      values = { date: "all", ...values };
    }
    setParameters({
      ...("axis" in values ||
      "metric" in values ||
      "model" in values ||
      "hardware" in values ||
      "date" in values ||
      "runs" in values ||
      "window" in values
        ? { rulers: "" }
        : {}),
      ...values,
    });
    setPoint(null);
    setPage(0);
  };
  const yLabel =
    basis === "uncached" && metric !== "output"
      ? metric === "input"
        ? "Uncached input throughput / GPU"
        : "Uncached input + output throughput / GPU"
      : METRICS[metric];
  const title = `${yLabel} vs ${AXES[axis]}`;
  const sorted = [...visible].sort((a, b) =>
    sort === "x"
      ? a.x - b.x
      : sort === "y"
        ? b.y - a.y
        : a.run.date.localeCompare(b.run.date),
  );
  async function png() {
    const svg = chart.current?.getSvgElement();
    if (!svg) return;
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute(
      "style",
      `background:${getComputedStyle(document.documentElement).getPropertyValue("--paper")};color:${getComputedStyle(document.documentElement).getPropertyValue("--ink")}`,
    );
    const originals = svg.querySelectorAll("*");
    clone.querySelectorAll("*").forEach((el, i) => {
      const style = getComputedStyle(originals[i]);
      for (const prop of [
        "fill",
        "stroke",
        "color",
        "font-family",
        "font-size",
      ])
        (el as SVGElement).style.setProperty(
          prop,
          style.getPropertyValue(prop),
        );
    });
    const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(clone))}`;
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      const exportSeries = series.filter((s) => !hidden.has(s.key));
      canvas.width = svg.clientWidth * 2;
      canvas.height = (svg.clientHeight + 86 + exportSeries.length * 22) * 2;
      const ctx = canvas.getContext("2d")!;
      ctx.scale(2, 2);
      const root = getComputedStyle(document.documentElement);
      ctx.fillStyle = root.getPropertyValue("--paper");
      ctx.fillRect(0, 0, canvas.width / 2, canvas.height / 2);
      ctx.fillStyle = root.getPropertyValue("--ink");
      ctx.font = "bold 16px sans-serif";
      ctx.fillText(selected, 18, 24);
      ctx.font = "13px sans-serif";
      ctx.fillText(
        `${title} · ${percentile.toUpperCase()} · AgentX · ${basis === "uncached" ? "Horace He accounting" : "all tokens"}`,
        18,
        48,
      );
      ctx.drawImage(image, 0, 66, svg.clientWidth, svg.clientHeight);
      exportSeries.forEach((s, i) => {
        ctx.fillStyle = color(s.key);
        ctx.fillText(
          `${seriesLabel(s.run)} · ${s.run.physical_gpus} physical GPUs`,
          18,
          svg.clientHeight + 84 + i * 22,
        );
      });
      canvas.toBlob((blob) => {
        if (blob) download(blob, "knotmeter-performance.png");
      }, "image/png");
    };
    image.onerror = () => {
      setMessage("PNG export failed");
    };
    image.src = url;
  }
  function csv() {
    const rows: unknown[][] = [
      [
        "run",
        "date_utc",
        "model",
        "hardware",
        "physical_gpus",
        "concurrency",
        "measurement",
        "topology",
        AXES[axis],
        yLabel,
        "percentile",
        "token_basis",
        "source",
      ],
    ];
    visible.forEach((p) =>
      rows.push([
        p.run.key,
        p.run.date,
        p.run.model,
        p.run.hardware,
        p.run.physical_gpus,
        p.run.concurrency,
        measurementLabel(p.run),
        topologyLabel(p.run),
        p.x,
        p.y,
        percentile,
        basis,
        sourceUrl(p.run),
      ]),
    );
    download(
      new Blob([rows.map((row) => row.map(csvValue).join(",")).join("\n")], {
        type: "text/csv",
      }),
      "knotmeter-performance.csv",
    );
  }
  async function share() {
    const url = new URL(window.location.href);
    url.pathname = url.pathname.replace(/\/v\/\d+\/$/, "/");
    const selection = new URLSearchParams(params);
    for (const [key, value] of Object.entries({
      model: selected,
      metric,
      axis,
      percentile,
      basis,
      date: dateSelection === "latest" ? date : dateSelection,
      runs: filtered.map((r) => r.key).join(",") || "none",
      window: String(measurement),
    }))
      selection.set(key, value);
    url.hash = `/chart?${selection}`;
    await navigator.clipboard.writeText(url.href);
    setMessage("Link copied");
  }
  return (
    <>
      <h1>Inference performance</h1>
      <p className="muted">
        Measured AgentX results. Throughput is normalized by the physical GPU
        allocation.
      </p>
      <div className="controls">
        <label>
          Model
          <select
            aria-label="Model"
            value={selected}
            onChange={(e) => update({ model: e.target.value })}
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
            onChange={(e) => update({ hardware: e.target.value })}
          >
            <option value="">All hardware</option>
            {[
              ...new Set(
                runs.filter((r) => r.model === selected).map((r) => r.hardware),
              ),
            ].map((h) => (
              <option key={h}>{h}</option>
            ))}
          </select>
        </label>
        <label>
          Measurement
          <select
            aria-label="Measurement"
            value={measurement}
            onChange={(e) => update({ window: e.target.value })}
          >
            {windows.map((seconds) => (
              <option key={seconds} value={seconds}>
                {measurementLabel(
                  runs.find(
                    (r) =>
                      r.model === selected && measurementSeconds(r) === seconds,
                  )!,
                )}
              </option>
            ))}
          </select>
        </label>
        <label>
          UTC date
          <select
            aria-label="UTC date"
            value={dateSelection}
            onChange={(e) => update({ date: e.target.value })}
          >
            <option value="latest">Latest date ({dates[0] || "none"})</option>
            <option value="all">All dates</option>
            {dates.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label>
          Y metric
          <select
            aria-label="Y metric"
            value={metric}
            onChange={(e) => update({ metric: e.target.value })}
          >
            {Object.entries(METRICS).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>
        {metric !== "output" && (
          <label>
            Token basis
            <select
              aria-label="Token basis"
              value={basis}
              onChange={(e) => update({ basis: e.target.value })}
            >
              <option value="total">All input tokens</option>
              <option value="uncached">
                Horace He — uncached input + output
              </option>
            </select>
          </label>
        )}
        <label>
          X axis
          <select
            aria-label="X axis"
            value={axis}
            onChange={(e) => update({ axis: e.target.value })}
          >
            {Object.entries(AXES).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label>
          Latency percentile
          <select
            aria-label="Latency percentile"
            value={percentile}
            onChange={(e) => update({ percentile: e.target.value })}
          >
            {["p50", "p75", "p90", "p95"].map((p) => (
              <option key={p} value={p}>
                {p.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
      </div>
      {basis === "uncached" && metric !== "output" && (
        <p className="notice">
          Horace He accounting excludes each run’s theoretically cacheable
          prompt prefix. It is a corpus-based estimate; measured cache hits are
          shown in diagnostics and cache reuse.
        </p>
      )}
      <details className="panel">
        <summary>
          Select individual runs and date overlays ({filtered.length}{" "}
          configurations)
        </summary>
        <button onClick={() => update({ runs: "" })}>All matching runs</button>
        <div className="run-selection">
          {runs
            .filter(
              (r) =>
                r.model === selected && measurementSeconds(r) === measurement,
            )
            .map((r) => (
              <label key={r.key}>
                <input
                  type="checkbox"
                  aria-label={`Overlay ${r.key}`}
                  checked={!selectedRuns || selectedRuns.has(r.key)}
                  onChange={(e) => {
                    const keys = selectedRuns
                      ? new Set(selectedRuns)
                      : new Set(
                          runs
                            .filter(
                              (r) =>
                                r.model === selected &&
                                measurementSeconds(r) === measurement,
                            )
                            .map((r) => r.key),
                        );
                    if (e.target.checked) keys.add(r.key);
                    else keys.delete(r.key);
                    update({ runs: [...keys].join(",") || "none" });
                  }}
                />
                {r.date} · {r.hardware} · C{r.concurrency} · {sourceLabel(r)} ·{" "}
                {measurementLabel(r)}
              </label>
            ))}
        </div>
      </details>
      <section className="panel">
        <div className="toolbar">
          <button onClick={() => update({ view: table ? "chart" : "table" })}>
            {table ? "Chart" : "Numeric table"}
          </button>
          <label>
            <input
              type="checkbox"
              checked={log}
              onChange={(e) =>
                update({ scale: e.target.checked ? "log" : "linear" })
              }
            />
            Log scales
          </label>
          <label>
            <input
              type="checkbox"
              checked={optimal}
              onChange={(e) => update({ optimal: e.target.checked ? "1" : "" })}
            />
            Optimal only
          </label>
          <label>
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => update({ labels: e.target.checked ? "1" : "0" })}
            />
            Point labels
          </label>
          <label>
            Label content
            <select
              aria-label="Label content"
              value={labelStyle}
              onChange={(e) => update({ labelstyle: e.target.value })}
            >
              <option value="concurrency">Concurrency</option>
              <option value="parallelism">Parallelism</option>
              <option value="throughput">Throughput</option>
            </select>
          </label>
          <label>
            <input
              type="checkbox"
              checked={showLines}
              onChange={(e) =>
                update({ linelabels: e.target.checked ? "1" : "" })
              }
            />
            Series labels
          </label>
          <label>
            <input
              type="checkbox"
              checked={showLegend}
              onChange={(e) => update({ legend: e.target.checked ? "1" : "0" })}
            />
            Legend
          </label>
          <label>
            <input
              type="checkbox"
              checked={contrast}
              onChange={(e) =>
                update({ contrast: e.target.checked ? "1" : "" })
              }
            />
            High contrast
          </label>
          <button
            onClick={() =>
              window.dispatchEvent(new Event("d3chart_zoom_reset_performance"))
            }
          >
            Reset zoom
          </button>
          <label>
            <input
              type="checkbox"
              checked={rulerMode}
              onChange={(e) => update({ ruler: e.target.checked ? "1" : "" })}
            />
            Perf Ruler
          </label>
          <button onClick={rulers.clear}>Clear rulers</button>
          <button onClick={csv}>CSV</button>
          <button onClick={png} disabled={table}>
            PNG
          </button>
          <button onClick={share}>Share</button>
          <span role="status">{message}</span>
        </div>
        <h2>{title}</h2>
        {rulerMode && (
          <p className="muted">
            Select two measured curves or points, then drag the ruler
            horizontally. Ratios use only overlapping measured ranges.{" "}
            {rulers.message}
          </p>
        )}
        {rulerMode &&
          rulers.state.rulers.map((r) => (
            <button key={r.id} onClick={() => rulers.remove(r.id)}>
              Remove ruler {r.id}
            </button>
          ))}
        {table ? (
          <>
            <label>
              Sort by
              <select
                aria-label="Sort by"
                value={sort}
                onChange={(e) => setSort(e.target.value)}
              >
                <option value="x">X ascending</option>
                <option value="y">Y descending</option>
                <option value="date">Date</option>
              </select>
            </label>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Run / UTC date</th>
                    <th>Hardware</th>
                    <th>GPUs</th>
                    <th>Concurrency</th>
                    <th>{AXES[axis]}</th>
                    <th>{yLabel}</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.slice(page * 25, page * 25 + 25).map((p) => (
                    <tr key={p.run.key}>
                      <td>
                        <button onClick={() => setPoint(p)}>{p.run.key}</button>
                        <small>{p.run.date}</small>
                      </td>
                      <td>{p.run.hardware}</td>
                      <td>{p.run.physical_gpus}</td>
                      <td>{p.run.concurrency}</td>
                      <td>{p.x.toFixed(5)}</td>
                      <td>{p.y.toFixed(5)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="toolbar">
              <button disabled={!page} onClick={() => setPage(page - 1)}>
                Previous
              </button>
              <span>
                Page {page + 1} / {Math.max(1, Math.ceil(sorted.length / 25))}
              </span>
              <button
                disabled={(page + 1) * 25 >= sorted.length}
                onClick={() => setPage(page + 1)}
              >
                Next
              </button>
            </div>
          </>
        ) : (
          <D3Chart<Point>
            ref={chart}
            chartId="performance"
            height={530}
            data={plotted}
            layers={layers}
            watermark="none"
            xScale={{
              type: log ? "log" : "linear",
              domain: domain(plotted.map((p) => p.x)),
            }}
            yScale={{
              type: log ? "log" : "linear",
              domain: domain(plotted.map((p) => p.y)),
            }}
            xAxis={{ label: AXES[axis] }}
            yAxis={{ label: `${yLabel} (tokens/s)` }}
            zoom={{ enabled: true, axes: "both", onZoom: rulers.raise }}
            onRender={rulers.raise}
            onDisplayUpdate={rulers.raise}
            tooltip={{
              rulerType: "crosshair",
              content: (p) =>
                renderToStaticMarkup(
                  <div className="point-tooltip">
                    <strong>{modelName(p.run)}</strong>
                    <p>
                      {p.run.hardware} · {p.run.physical_gpus} GPUs · C
                      {p.run.concurrency} · {p.run.date}
                    </p>
                    <p>
                      {AXES[axis]}: {p.x.toFixed(5)}
                    </p>
                    <p>
                      {yLabel}: {p.y.toFixed(5)}
                    </p>
                    <a href={`#/run/${p.run.key}`}>Open run diagnostics</a>
                  </div>,
                ),
              onPointClick: (p) => {
                if (rulerMode) {
                  rulers.selectPoint(p.series, p.x);
                  chart.current?.dismissTooltip();
                  chart.current?.hideTooltip();
                } else setPoint(p);
              },
              attachToLayer: 1,
            }}
            legendElement={
              showLegend ? (
                <div className="legend">
                  <input
                    aria-label="Search chart legend"
                    placeholder="Search runs or dates"
                    value={legendSearch}
                    onChange={(e) => setLegendSearch(e.target.value)}
                  />
                  {series
                    .filter((s) =>
                      `${seriesLabel(s.run)} ${s.run.date} ${sourceLabel(s.run)} ${s.run.framework}`
                        .toLowerCase()
                        .includes(legendSearch.toLowerCase()),
                    )
                    .map((s) => (
                      <label
                        key={s.key}
                        style={{ borderLeft: `4px solid ${color(s.key)}` }}
                      >
                        <input
                          type="checkbox"
                          checked={!hidden.has(s.key)}
                          onChange={(e) => {
                            const next = new Set(hidden);
                            if (e.target.checked) next.delete(s.key);
                            else next.add(s.key);
                            update({
                              hidden: [...next].filter(Boolean).join(","),
                            });
                          }}
                        />
                        {seriesLabel(s.run)}
                        <button
                          aria-label={`Inspect series ${seriesLabel(s.run)}`}
                          onClick={() => setInspected(s.key)}
                        >
                          Inspect
                        </button>
                      </label>
                    ))}
                </div>
              ) : undefined
            }
            noDataOverlay={
              !plotted.length ? (
                <p className="empty-chart">
                  No measured points match these filters.
                </p>
              ) : undefined
            }
          />
        )}
        {point && (
          <aside className="point-detail">
            <button
              aria-label="Close point details"
              onClick={() => setPoint(null)}
            >
              ×
            </button>
            <strong>
              {modelName(point.run)} · {point.run.hardware} · C
              {point.run.concurrency}
            </strong>
            <p>
              {point.run.date} · {point.run.physical_gpus} GPUs on{" "}
              {point.run.hosts} hosts · {point.run.framework} ·{" "}
              {point.run.precision.toUpperCase()}
              {" · "}
              {measurementLabel(point.run)} · {topologyLabel(point.run)} ·{" "}
              {sourceLabel(point.run)}
            </p>
            <p>
              {AXES[axis]}: {point.x.toFixed(5)} · {yLabel}:{" "}
              {point.y.toFixed(5)}
            </p>
            <a href={`#/run/${point.run.key}`}>Run diagnostics</a> ·{" "}
            <a href={sourceUrl(point.run)} target="_blank" rel="noreferrer">
              Original {point.run.calibration ? "artifacts" : "GitHub attempt"}
            </a>
          </aside>
        )}
      </section>
      <Dialog
        open={inspected !== null}
        onOpenChange={(open) => {
          if (!open) setInspected(null);
        }}
      >
        <DialogContent className="max-w-[95vw] w-[1000px]">
          <DialogHeader>
            <DialogTitle>Measured points in series</DialogTitle>
          </DialogHeader>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>UTC date / source</th>
                  <th>Hardware / GPUs</th>
                  <th>Load / parallelism</th>
                  <th>{AXES[axis]}</th>
                  <th>{yLabel}</th>
                  <th>Inspect</th>
                </tr>
              </thead>
              <tbody>
                {points
                  .filter((p) => p.series === inspected)
                  .toSorted((a, b) => a.run.concurrency - b.run.concurrency)
                  .map((p) => (
                    <tr key={p.run.key}>
                      <td>
                        {p.run.date}
                        <small>{p.run.key}</small>
                      </td>
                      <td>
                        {p.run.hardware} · {p.run.physical_gpus} GPUs
                      </td>
                      <td>
                        C{p.run.concurrency} · {topologyLabel(p.run)}
                        <small>{measurementLabel(p.run)}</small>
                      </td>
                      <td>{p.x.toFixed(5)}</td>
                      <td>{p.y.toFixed(5)}</td>
                      <td>
                        <button
                          onClick={() => {
                            if (rulerMode) rulers.selectPoint(p.series, p.x);
                            else setPoint(p);
                            setInspected(null);
                          }}
                        >
                          {rulerMode ? "Select for ruler" : "Pin point"}
                        </button>{" "}
                        <a href={`#/run/${p.run.key}`}>Diagnostics</a>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </DialogContent>
      </Dialog>
      <p className="muted">
        Curves connect loads with matching workload, runtime and serving
        controls. Repeats remain separate observations. Fifteen-minute
        calibrations and one-hour baselines have separate selections and ruler
        scopes.
      </p>
    </>
  );
}
