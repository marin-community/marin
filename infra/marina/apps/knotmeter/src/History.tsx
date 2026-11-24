import { useState } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { D3Chart } from "@/lib/d3-chart/D3Chart";
import type { CustomLayerConfig } from "@/lib/d3-chart/D3Chart/types";
import type * as d3 from "d3";
import { METRICS, axisValue, type Metric, type TokenBasis } from "./metrics";
import {
  matchedHistory,
  improvementWaterfall,
  type ImprovementStep,
} from "./history-math";
import {
  type Run,
  sourceUrl,
  measurementLabel,
  measurementSeconds,
} from "./types";
import { setParameters, useRoute } from "./navigation";

export function History({ runs }: { runs: Run[] }) {
  const { params } = useRoute();
  const view = params.get("view") || "trend";
  const [message, setMessage] = useState("");
  const models = [...new Set(runs.map((r) => r.model))];
  const model =
    params.get("model") ||
    models.find((m) => m.includes("Snowball")) ||
    models[0];
  const hardware = params.get("hardware") || "";
  const windows = [
    ...new Set(runs.filter((r) => r.model === model).map(measurementSeconds)),
  ].sort((a, b) => b - a);
  const measurement = Number(params.get("window") || windows[0]);
  const metric = (params.get("metric") || "output") as Metric;
  const basis = (params.get("basis") || "total") as TokenBasis;
  const filtered = runs.filter(
    (r) =>
      r.model === model &&
      measurementSeconds(r) === measurement &&
      (!hardware || r.hardware === hardware),
  );
  const xs = filtered
    .map((r) => axisValue(r, "interactivity", "p90"))
    .filter((v): v is number => v !== null);
  const target = params.has("target")
    ? Number(params.get("target"))
    : xs[0] || 1;
  const series = matchedHistory(filtered, target, metric, basis);
  const observations = series.flatMap((s, i) =>
    s.points.map((p) => ({
      ...p,
      x: Date.parse(p.date),
      y: p.value,
      precision: "bf16",
      series: `s${i}`,
      label: s.label,
    })),
  );
  const colors = ["#3b82f6", "#e87722", "#10b981", "#a855f7"];
  const dates = observations.map((p) => p.x);
  const minDate = Math.min(...dates);
  const maxDate = Math.max(...dates);
  const chartDays: [Date, Date] = dates.length
    ? [new Date(minDate - 86400000 / 2), new Date(maxDate + 86400000 / 2)]
    : [new Date(0), new Date(86400000)];
  async function share() {
    const url = new URL(window.location.href);
    url.pathname = url.pathname.replace(/\/v\/\d+\/$/, "/");
    url.hash = `/history?${new URLSearchParams({ model, hardware, metric, basis, target: String(target), view, window: String(measurement) })}`;
    await navigator.clipboard.writeText(url.href);
    setMessage("Link copied");
  }
  return (
    <>
      <h1>Historical performance</h1>
      <p className="muted">
        Throughput at one matched P90 full-response interactivity. Each dated
        baseline uses the latest submission that day and the upstream throughput
        Pareto/Hermite calculation. Calibrations group matching controls within
        each date. Measurement windows, profiles and model/corpus revisions
        remain separate.
      </p>
      <div className="controls">
        <label>
          Model
          <select
            aria-label="Model"
            value={model}
            onChange={(e) =>
              setParameters({
                model: e.target.value,
                target: "",
                hardware: "",
                window: "",
              })
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
          Measurement
          <select
            aria-label="Measurement"
            value={measurement}
            onChange={(e) =>
              setParameters({ window: e.target.value, target: "" })
            }
          >
            {windows.map((seconds) => (
              <option key={seconds} value={seconds}>
                {measurementLabel(
                  runs.find(
                    (r) =>
                      r.model === model && measurementSeconds(r) === seconds,
                  )!,
                )}
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
          Throughput
          <select
            aria-label="Throughput"
            value={metric}
            onChange={(e) => setParameters({ metric: e.target.value })}
          >
            {Object.entries(METRICS).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
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
              onChange={(e) => setParameters({ basis: e.target.value })}
            >
              <option value="total">All input</option>
              <option value="uncached">
                Horace He — uncached input + output
              </option>
            </select>
          </label>
        )}
        <label>
          Target interactivity
          <input
            aria-label="Target interactivity"
            type="number"
            step="any"
            min="0"
            value={target}
            onChange={(e) => setParameters({ target: e.target.value })}
          />
        </label>
        {!!xs.length && (
          <label>
            Operating point
            <input
              aria-label="Operating point"
              type="range"
              min={Math.min(...xs)}
              max={Math.max(...xs)}
              step="any"
              value={target}
              onChange={(e) => setParameters({ target: e.target.value })}
            />
          </label>
        )}
      </div>
      <div className="toolbar">
        <button
          aria-pressed={view === "trend"}
          onClick={() => setParameters({ view: "trend" })}
        >
          Trend line
        </button>
        <button
          aria-pressed={view === "waterfall"}
          onClick={() => setParameters({ view: "waterfall" })}
        >
          Improvement waterfall
        </button>
        <button onClick={share}>Share matched operating point</button>
        <span role="status">{message}</span>
      </div>
      <p>
        P90 interactivity target: {target.toFixed(5)} tokens/s/user. Values
        outside each measured curve’s range are unavailable.
      </p>
      {!observations.length ? (
        <section className="panel">
          <p>
            No curve covers this operating point. Select a measured target or
            import a completed sweep.
          </p>
        </section>
      ) : view === "trend" ? (
        <section className="panel">
          <D3Chart
            chartId="history"
            height={460}
            data={observations}
            watermark="none"
            xScale={{ type: "time", domain: chartDays }}
            yScale={{
              type: "linear",
              domain: [0, Math.max(...observations.map((p) => p.value)) * 1.15],
            }}
            layers={[
              {
                type: "line",
                lines: Object.fromEntries(
                  series.map((s, i) => [
                    `s${i}`,
                    s.points.length > 1
                      ? s.points.map((p) => ({
                          x: Date.parse(p.date),
                          y: p.value,
                        }))
                      : [],
                  ]),
                ),
                config: {
                  getColor: (key) =>
                    colors[Number(key.slice(1)) % colors.length],
                },
              },
              {
                type: "scatter",
                data: observations,
                config: {
                  getColor: (p) =>
                    colors[Number(p.series.slice(1)) % colors.length],
                },
              },
            ]}
            xAxis={{
              label: "Measured UTC date",
              tickFormat: (value) =>
                new Date(value as number).toISOString().slice(0, 10),
            }}
            yAxis={{ label: METRICS[metric] + " (tokens/s)" }}
            zoom={{ enabled: true, axes: "both" }}
            tooltip={{
              rulerType: "crosshair",
              attachToLayer: 1,
              content: (p) =>
                renderToStaticMarkup(
                  <div className="point-tooltip">
                    <strong>{p.label}</strong>
                    <p>
                      {p.date}: {p.value.toFixed(5)} tokens/s/GPU
                    </p>
                    <p>
                      Matched target {p.operatingPoint.toFixed(5)} ·{" "}
                      {p.sources.length} frontier observations
                    </p>
                    {p.sources.map((r) => (
                      <p key={r.key}>
                        <a href={`#/run/${r.key}`}>
                          C{r.concurrency} · {r.key}
                        </a>
                      </p>
                    ))}
                  </div>,
                ),
            }}
          />
          {series
            .filter((s) => s.points.length > 0)
            .map((s, i) => (
              <p key={s.key} style={{ color: colors[i % colors.length] }}>
                {s.label} · {s.points.length} dated values
                {s.points.length === 1
                  ? " (one measured date; no historical trend)"
                  : ""}
              </p>
            ))}
        </section>
      ) : (
        <div>
          {series.map((s) => (
            <Waterfall
              key={s.key}
              label={s.label}
              steps={improvementWaterfall(s.points)}
            />
          ))}
        </div>
      )}
      <section className="panel table-scroll">
        <h2>Matched values and source observations</h2>
        <table>
          <thead>
            <tr>
              <th>Hardware / identity</th>
              <th>Date</th>
              <th>Matched tokens/s/GPU</th>
              <th>Measured source configurations</th>
            </tr>
          </thead>
          <tbody>
            {series.flatMap((s) =>
              s.points.map((p) => (
                <tr key={s.key + p.date}>
                  <td>{s.label}</td>
                  <td>{p.date}</td>
                  <td>{p.value.toFixed(5)}</td>
                  <td>
                    {p.sources.map((r) => (
                      <span key={r.key}>
                        <a href={`#/run/${r.key}`}>C{r.concurrency}</a> ·{" "}
                        <a href={sourceUrl(r)}>original attempt</a>{" "}
                      </span>
                    ))}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </section>
    </>
  );
}

function Waterfall({
  label,
  steps,
}: {
  label: string;
  steps: ImprovementStep[];
}) {
  const data = steps.map((s, i) => ({ ...s, key: `w${i}` }));
  const layer: CustomLayerConfig = {
    type: "custom",
    render(group, ctx) {
      const x = ctx.xScale as d3.ScaleBand<string>;
      const y = ctx.yScale as d3.ScaleLinear<number, number>;
      group
        .selectAll<SVGRectElement, (typeof data)[number]>(".waterfall-step")
        .data(data)
        .join("rect")
        .attr("class", "waterfall-step")
        .attr("x", (d) => x(d.key)!)
        .attr("width", x.bandwidth())
        .attr("y", (d) => Math.min(y(d.from), y(d.to)))
        .attr("height", (d) => Math.abs(y(d.from) - y(d.to)))
        .attr("fill", (d) => (d.kind === "gain" ? "#10b981" : "#3b82f6"));
    },
  };
  return (
    <section className="panel">
      <h2>{label}</h2>
      {steps.length ? (
        <>
          <D3Chart
            chartId={`waterfall-${label.replace(/\W/g, "-")}`}
            data={data}
            height={350}
            watermark="none"
            xScale={{
              type: "band",
              domain: data.map((d) => d.key),
              padding: 0.3,
            }}
            yScale={{
              type: "linear",
              domain: [0, Math.max(...steps.map((s) => s.to)) * 1.1],
            }}
            layers={[layer]}
            xAxis={{
              label: "Record improvements",
              tickFormat: (key) => {
                const d = data.find((d) => d.key === String(key));
                return d ? (d.kind === "best" ? "Best" : d.date) : "";
              },
            }}
            yAxis={{ label: "Tokens/s/GPU" }}
          />
          <div className="toolbar">
            {steps.map((s, i) => (
              <span key={i}>
                {s.kind} {s.date}:{" "}
                {s.kind === "gain" ? "+" + s.delta.toFixed(5) : s.to.toFixed(5)}
              </span>
            ))}
          </div>
        </>
      ) : (
        <p>No matched observations.</p>
      )}
    </section>
  );
}
