import { useState } from "react";
import { D3Chart } from "@/lib/d3-chart/D3Chart";
import {
  type Run,
  modelName,
  sourceUrl,
  submissionIdentity,
  measurementLabel,
  measurementSeconds,
  topologyLabel,
} from "./types";
import { comparisonScope } from "./metrics";
import { navigate, setParameters, useRoute } from "./navigation";

const WEEK = 7 * 86400_000;
function mondayOf(date: string) {
  const d = new Date(date);
  return Date.UTC(
    d.getUTCFullYear(),
    d.getUTCMonth(),
    d.getUTCDate() - ((d.getUTCDay() + 6) % 7),
  );
}
export function Runs({ runs }: { runs: Run[] }) {
  const { params } = useRoute();
  const search = params.get("search") || "";
  const sort = params.get("sort") || "date";
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activity, setActivity] = useState("weekly");
  const filtered = runs
    .filter((r) =>
      [
        r.key,
        r.model,
        r.hardware,
        r.runtime,
        r.recipe,
        r.date,
        r.framework,
        r.precision,
        measurementLabel(r),
        topologyLabel(r),
        submissionIdentity(r),
      ]
        .join(" ")
        .toLowerCase()
        .includes(search.toLowerCase()),
    )
    .toSorted((a, b) =>
      sort === "model"
        ? a.model.localeCompare(b.model) ||
          a.hardware.localeCompare(b.hardware) ||
          a.concurrency - b.concurrency
        : sort === "concurrency"
          ? a.concurrency - b.concurrency
          : sort === "throughput"
            ? b.metrics.throughput.per_gpu.output_tput_tps -
              a.metrics.throughput.per_gpu.output_tput_tps
            : b.source_created_at.localeCompare(a.source_created_at),
    );
  const submissions = [
    ...new Map(runs.map((r) => [submissionIdentity(r), r])).values(),
  ];
  const weekCounts = new Map<number, number>();
  for (const r of submissions) {
    const week = mondayOf(r.source_created_at);
    weekCounts.set(week, (weekCounts.get(week) || 0) + 1);
  }
  const weeks = [...weekCounts.keys()].sort();
  let cumulative = 0;
  const activityRows = [];
  for (let week = weeks[0]; week <= weeks.at(-1)!; week += WEEK) {
    const count = weekCounts.get(week) || 0;
    cumulative += count;
    activityRows.push({
      week,
      label: new Date(week).toISOString().slice(0, 10),
      count,
      cumulative,
    });
  }
  const picked = runs.filter((r) => selected.has(r.key));
  const canCompare =
    picked.length > 0 && new Set(picked.map(comparisonScope)).size === 1;
  return (
    <>
      <h1>Run inventory and submissions</h1>
      <p className="muted">
        Completed GitHub attempts and native private calibrations. One
        submission can contain several measured configurations.
      </p>
      {[...new Set(runs.filter((r) => r.calibration).map((r) => r.model))].map(
        (model) => (
          <p key={model}>
            <a
              href={`#/chart?${new URLSearchParams({ model, window: "900", date: "all", optimal: "0" })}`}
            >
              Compare all 15-minute {model.split("/").at(-1)} calibrations,
              including repeats
            </a>
          </p>
        ),
      )}
      <div className="stats-grid">
        <div className="panel">
          <small>Imported submissions</small>
          <strong>{submissions.length}</strong>
        </div>
        <div className="panel">
          <small>Measured configurations</small>
          <strong>{runs.length}</strong>
        </div>
        <div className="panel">
          <small>UTC dates</small>
          <strong>{new Set(runs.map((r) => r.date)).size}</strong>
        </div>
        <div className="panel">
          <small>Models / hardware types</small>
          <strong>
            {new Set(runs.map((r) => r.model)).size} /{" "}
            {new Set(runs.map((r) => r.hardware)).size}
          </strong>
        </div>
      </div>
      <section className="panel">
        <div className="toolbar">
          <h2>Imported submission activity</h2>
          <button
            aria-pressed={activity === "weekly"}
            onClick={() => setActivity("weekly")}
          >
            Weekly
          </button>
          <button
            aria-pressed={activity === "cumulative"}
            onClick={() => setActivity("cumulative")}
          >
            Cumulative
          </button>
        </div>
        {activity === "weekly" ? (
          <D3Chart
            chartId="weekly-submissions"
            height={260}
            data={activityRows}
            watermark="none"
            xScale={{
              type: "band",
              domain: activityRows.map((r) => r.label),
              padding: 0.4,
            }}
            yScale={{
              type: "linear",
              domain: [
                0,
                Math.max(1, ...activityRows.map((r) => r.count)) * 1.1,
              ],
            }}
            layers={[
              {
                type: "bar",
                data: activityRows,
                config: {
                  getX: (r) => r.label,
                  getY: (r) => r.count,
                  getColor: () => "#3b82f6",
                  getForeground: () => "var(--ink)",
                },
              },
            ]}
            xAxis={{ label: "UTC week beginning Monday" }}
            yAxis={{
              label: "Imported submissions",
              tickFormat: (value) =>
                Number.isInteger(Number(value)) ? String(value) : "",
            }}
          />
        ) : (
          <D3Chart
            chartId="cumulative-submissions"
            height={260}
            data={activityRows}
            watermark="none"
            xScale={{
              type: "time",
              domain: [
                new Date(weeks[0] - WEEK / 3),
                new Date(weeks.at(-1)! + WEEK / 3),
              ],
            }}
            yScale={{ type: "linear", domain: [0, cumulative * 1.1 || 1] }}
            layers={[
              {
                type: "line",
                lines: {
                  submissions: activityRows.map((r) => ({
                    x: r.week,
                    y: r.cumulative,
                  })),
                },
                config: { getColor: () => "#3b82f6" },
              },
              {
                type: "point",
                data: activityRows,
                config: {
                  getCx: (r) => r.week,
                  getCy: (r) => r.cumulative,
                  getX: (r) => r.week,
                  getY: (r) => r.cumulative,
                  getColor: () => "#3b82f6",
                },
              },
            ]}
            xAxis={{
              label: "UTC week",
              tickFormat: (date) =>
                new Date(date as number).toISOString().slice(0, 10),
            }}
            yAxis={{ label: "Cumulative imported submissions" }}
          />
        )}
      </section>
      <div className="controls">
        <label>
          Search runs
          <input
            aria-label="Search runs"
            placeholder="Model, hardware, runtime, recipe, date or run ID"
            value={search}
            onChange={(e) => {
              setParameters({ search: e.target.value });
              setPage(0);
            }}
          />
        </label>
        <label>
          Sort by
          <select
            aria-label="Run sort"
            value={sort}
            onChange={(e) => {
              setParameters({ sort: e.target.value });
              setPage(0);
            }}
          >
            <option value="date">Newest submission</option>
            <option value="model">Model / hardware</option>
            <option value="concurrency">Concurrency</option>
            <option value="throughput">Output throughput / GPU</option>
          </select>
        </label>
      </div>
      <div className="toolbar">
        <button
          disabled={!canCompare}
          onClick={() =>
            navigate(
              `/chart?${new URLSearchParams({ model: picked[0].model, window: String(measurementSeconds(picked[0])), date: "all", optimal: "0", runs: picked.map((r) => r.key).join(",") })}`,
            )
          }
        >
          Compare selected runs ({selected.size})
        </button>
        <button onClick={() => setSelected(new Set())}>Clear selection</button>
        {picked.length > 1 && !canCompare && (
          <span>
            Select matching model, workload and measurement windows to compare
            operating curves.
          </span>
        )}
      </div>
      <section className="panel table-scroll">
        <table>
          <thead>
            <tr>
              <th>Select</th>
              <th>Model / run</th>
              <th>UTC submission</th>
              <th>Hardware / topology</th>
              <th>Load</th>
              <th>Output tokens/s/GPU</th>
              <th>Configuration</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(page * 25, page * 25 + 25).map((r) => (
              <tr key={r.key}>
                <td>
                  <input
                    type="checkbox"
                    aria-label={`Compare ${r.key}`}
                    checked={selected.has(r.key)}
                    onChange={(e) =>
                      setSelected((previous) => {
                        const next = new Set(previous);
                        if (e.target.checked) next.add(r.key);
                        else next.delete(r.key);
                        return next;
                      })
                    }
                  />
                </td>
                <td>
                  <a href={`#/run/${r.key}`}>{modelName(r)}</a>
                  <small>{r.key}</small>
                  <small>{measurementLabel(r)}</small>
                </td>
                <td>
                  {r.date}
                  <small>{r.source_created_at.slice(11, 19)} UTC</small>
                  <a href={sourceUrl(r)} target="_blank" rel="noreferrer">
                    Original attempt
                  </a>
                </td>
                <td>
                  {r.hardware} · {r.physical_gpus} GPUs / {r.hosts} hosts
                  <small>
                    {r.framework} · {r.precision.toUpperCase()} ·{" "}
                    {topologyLabel(r)}
                  </small>
                </td>
                <td>C{r.concurrency}</td>
                <td>
                  {r.metrics.throughput.per_gpu.output_tput_tps.toFixed(5)}
                </td>
                <td>
                  <details>
                    <summary>Settings and provenance</summary>
                    <p>
                      KV offload:{" "}
                      {String(r.aggregate.kv_offloading ?? "not reported")}
                    </p>
                    <p>
                      Runtime <code>{r.runtime}</code>
                    </p>
                    <p>
                      Recipe <code>{r.recipe}</code>
                    </p>
                    <pre>
                      {JSON.stringify(
                        {
                          source: r.source,
                          configuration: r.configuration,
                          manifest: r.manifest,
                          source_files: r.source_files,
                        },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="toolbar">
          <button disabled={!page} onClick={() => setPage(page - 1)}>
            Previous
          </button>
          <span>
            {filtered.length} configurations · page {page + 1} /{" "}
            {Math.max(1, Math.ceil(filtered.length / 25))}
          </span>
          <button
            disabled={(page + 1) * 25 >= filtered.length}
            onClick={() => setPage(page + 1)}
          >
            Next
          </button>
        </div>
      </section>
    </>
  );
}
