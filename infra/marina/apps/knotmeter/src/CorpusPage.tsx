import { useEffect, useState } from "react";
import { TraceFlamegraph } from "@/components/datasets/trace-flamegraph";
import type { ConversationStructure } from "@/hooks/api/use-datasets";
import { D3Chart } from "@/lib/d3-chart/D3Chart";
import { api } from "./types";
import { useRoute, setParameters } from "./navigation";

interface Summary {
  in: number;
  out: number;
  cached: number;
  uncached: number;
  numTurns: number;
  numSubagentGroups: number;
  requests: number;
  subagent_requests: number;
  models: Record<string, number>;
  conversations?: number;
}
interface Bin {
  x0: number;
  x1: number;
  count: number;
}
interface Distribution {
  bins: Bin[];
  stats: {
    count: number;
    min: number;
    max: number;
    mean: number;
    median: number;
    p90: number;
    p95: number;
  } | null;
}
interface Dataset {
  dataset: string;
  revision: string;
  summary: Summary;
  distributions: Record<string, Distribution>;
}
interface Conversation {
  conversation_id: string;
  summary: Summary;
}
interface Detail {
  conversation_id: string;
  structure: ConversationStructure;
}
const n = (value: number) =>
  value.toLocaleString("en-US", { maximumFractionDigits: 2 });

function SummaryCards({ value }: { value: Summary }) {
  return (
    <div className="stats-grid">
      {[
        ["Requests", value.requests],
        ["Input tokens", value.in],
        ["Output tokens", value.out],
        [
          "Corpus cached input",
          value.in
            ? `${((100 * value.cached) / value.in).toFixed(2)}%`
            : "Not available",
        ],
        ["Subagent groups", value.numSubagentGroups],
        ["Subagent requests", value.subagent_requests],
      ].map(([label, v]) => (
        <div key={String(label)} className="panel">
          <small>{label}</small>
          <strong>{typeof v === "number" ? n(v) : v}</strong>
        </div>
      ))}
    </div>
  );
}

function DatasetDistribution({
  title,
  value,
}: {
  title: string;
  value: Distribution;
}) {
  const bins = value.bins.map((b, i) => ({ ...b, key: `b${i}` }));
  return (
    <section className="panel">
      <h2>{title}</h2>
      {value.stats ? (
        <>
          <D3Chart
            chartId={`dataset-${title.replaceAll(/\W/g, "-")}`}
            height={250}
            data={bins}
            watermark="none"
            xScale={{
              type: "band",
              domain: bins.map((b) => b.key),
              padding: 0.05,
            }}
            yScale={{
              type: "linear",
              domain: [0, Math.max(...bins.map((b) => b.count)) * 1.1 || 1],
            }}
            layers={[
              {
                type: "bar",
                data: bins,
                config: {
                  getX: (b) => b.key,
                  getY: (b) => b.count,
                  getColor: () => "#3b82f6",
                  getForeground: () => "var(--ink)",
                },
              },
            ]}
            xAxis={{
              label: title,
              tickFormat: (key) => {
                const b = bins.find((b) => b.key === String(key));
                return b && bins.indexOf(b) % 6 === 0 ? n(b.x0) : "";
              },
            }}
            yAxis={{ label: "Requests / conversations" }}
          />
          <p className="muted">
            n={n(value.stats.count)} · mean {n(value.stats.mean)} · median{" "}
            {n(value.stats.median)} · P90 {n(value.stats.p90)} · P95{" "}
            {n(value.stats.p95)}
          </p>
        </>
      ) : (
        <p>No observations</p>
      )}
    </section>
  );
}

export function CorpusPage() {
  const { path, params } = useRoute();
  const match = /^\/agentx\/(.+)@([^/]+)\/conversations\/([^/]+)$/.exec(path);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(true);
  const [rows, setRows] = useState<Conversation[]>([]);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState("");
  const [page, setPage] = useState(0);
  const [view, setView] = useState("flamegraph");
  const dataset = match
    ? decodeURIComponent(match[1])
    : params.get("dataset") || datasets[0]?.dataset;
  const revision = match
    ? match[2]
    : params.get("revision") || datasets[0]?.revision;
  const cid = match ? decodeURIComponent(match[3]) : null;
  const active = datasets.find(
    (d) => d.dataset === dataset && d.revision === revision,
  );
  useEffect(() => {
    api<Dataset[]>("datasets")
      .then(setDatasets)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoadingDatasets(false));
  }, []);
  useEffect(() => {
    if (!dataset || !revision) return;
    let alive = true;
    setError("");
    setDetail(null);
    setRows([]);
    const query = new URLSearchParams({ dataset, revision });
    if (cid)
      api<Detail>(`corpus/${encodeURIComponent(cid)}?${query}`)
        .then((r) => {
          if (alive) setDetail(r);
        })
        .catch((e: Error) => setError(e.message));
    else
      api<Conversation[]>(`corpus?${query}`)
        .then((r) => {
          if (alive) setRows(r);
        })
        .catch((e: Error) => setError(e.message));
    return () => {
      alive = false;
    };
  }, [dataset, revision, cid]);
  const search = params.get("search") || "";
  const sort = params.get("sort") || "tokens";
  const filtered = rows
    .filter((r) =>
      `${r.conversation_id} ${Object.keys(r.summary.models).join(" ")}`
        .toLowerCase()
        .includes(search.toLowerCase()),
    )
    .sort((a, b) =>
      sort === "id"
        ? a.conversation_id.localeCompare(b.conversation_id)
        : sort === "turns"
          ? b.summary.requests - a.summary.requests
          : sort === "subagents"
            ? b.summary.numSubagentGroups - a.summary.numSubagentGroups
            : b.summary.in - a.summary.in,
    );
  const back = params.get("back");
  return (
    <>
      {cid && (
        <div className="toolbar">
          <a href={`#${back || "/corpus"}`}>
            {back?.startsWith("/run/")
              ? "← Return to request timeline"
              : "← Corpus browser"}
          </a>
          <button onClick={() => window.history.back()}>Back</button>
        </div>
      )}
      <h1>{cid ? "Original corpus conversation" : "Corpus browser"}</h1>
      {dataset ? (
        <p>
          {dataset} @ <code>{revision}</code>
        </p>
      ) : (
        <p>
          {loadingDatasets
            ? "Loading datasets…"
            : "No corpus imported. Load original corpus structures to browse conversations."}
        </p>
      )}
      <p className="muted">
        Original source timing and token structure. Cached prefixes describe
        corpus cacheability; actual server cache hits are measured separately.
      </p>
      {error && <p role="alert">{error}</p>}
      {cid ? (
        detail ? (
          <>
            <h2>{detail.conversation_id}</h2>
            <SummaryCards
              value={{
                ...detail.structure.totals,
                requests: detail.structure.nodes.reduce(
                  (total, node) =>
                    total +
                    (node.kind === "subagent" ? node.children.length : 1),
                  0,
                ),
                subagent_requests: detail.structure.nodes.reduce(
                  (total, node) =>
                    total +
                    (node.kind === "subagent" ? node.children.length : 0),
                  0,
                ),
                models: {},
              }}
            />
            <div className="toolbar">
              <button
                aria-pressed={view === "flamegraph"}
                onClick={() => setView("flamegraph")}
              >
                Token flamegraph
              </button>
              <button
                aria-pressed={view === "structure"}
                onClick={() => setView("structure")}
              >
                Original structure
              </button>
            </div>
            {view === "flamegraph" ? (
              <section className="panel">
                <TraceFlamegraph
                  structure={detail.structure}
                  highlightTurn={
                    params.has("turn") ? Number(params.get("turn")) : null
                  }
                  highlightRawIndex={
                    params.has("raw") ? Number(params.get("raw")) : null
                  }
                  highlightInnerIndex={
                    params.has("inner") ? Number(params.get("inner")) : null
                  }
                  highlightAgentId={params.get("sa")}
                />
              </section>
            ) : (
              <section className="panel">
                <pre>{JSON.stringify(detail.structure, null, 2)}</pre>
              </section>
            )}
          </>
        ) : (
          <p>Loading conversation…</p>
        )
      ) : (
        <>
          {active && (
            <>
              <p>
                {active.summary.conversations} conversations ·{" "}
                {n(active.summary.numTurns)} main turns
              </p>
              <SummaryCards value={active.summary} />
              <section className="panel">
                <h2>Original models</h2>
                <div className="toolbar">
                  {Object.entries(active.summary.models)
                    .sort((a, b) => b[1] - a[1])
                    .map(([model, count]) => (
                      <span key={model}>
                        {model}: {n(count)} requests
                      </span>
                    ))}
                </div>
              </section>
              <details>
                <summary>
                  Dataset token, turn, cache and subagent distributions
                </summary>
                <div className="diagnostic-grid">
                  {Object.entries(active.distributions).map(
                    ([title, value]) => (
                      <DatasetDistribution
                        key={title}
                        title={title}
                        value={value}
                      />
                    ),
                  )}
                </div>
              </details>
            </>
          )}
          <div className="controls">
            <label>
              Dataset
              <select
                aria-label="Dataset"
                value={active ? `${active.dataset}@${active.revision}` : ""}
                onChange={(e) => {
                  const at = e.target.value.lastIndexOf("@");
                  setParameters({
                    dataset: e.target.value.slice(0, at),
                    revision: e.target.value.slice(at + 1),
                  });
                  setPage(0);
                }}
              >
                {datasets.map((d) => (
                  <option
                    key={`${d.dataset}@${d.revision}`}
                    value={`${d.dataset}@${d.revision}`}
                  >
                    {d.dataset} @ {d.revision.slice(0, 8)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Search corpus
              <input
                aria-label="Search corpus"
                value={search}
                placeholder="Conversation ID or model"
                onChange={(e) => {
                  setParameters({ search: e.target.value });
                  setPage(0);
                }}
              />
            </label>
            <label>
              Sort by
              <select
                aria-label="Corpus sort"
                value={sort}
                onChange={(e) => {
                  setParameters({ sort: e.target.value });
                  setPage(0);
                }}
              >
                <option value="tokens">Input tokens</option>
                <option value="turns">Requests</option>
                <option value="subagents">Subagent groups</option>
                <option value="id">Conversation ID</option>
              </select>
            </label>
          </div>
          <section className="panel table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Conversation</th>
                  <th>Models</th>
                  <th>Requests</th>
                  <th>Subagent groups</th>
                  <th>Input / output tokens</th>
                  <th>Cached input</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(page * 25, page * 25 + 25).map((r) => (
                  <tr key={r.conversation_id}>
                    <td>
                      <a
                        href={`#/agentx/${encodeURIComponent(dataset!)}@${revision}/conversations/${encodeURIComponent(r.conversation_id)}`}
                      >
                        {r.conversation_id}
                      </a>
                    </td>
                    <td>{Object.keys(r.summary.models).join(", ")}</td>
                    <td>{r.summary.requests}</td>
                    <td>{r.summary.numSubagentGroups}</td>
                    <td>
                      {n(r.summary.in)} / {n(r.summary.out)}
                    </td>
                    <td>
                      {r.summary.in
                        ? ((r.summary.cached / r.summary.in) * 100).toFixed(1) +
                          "%"
                        : "—"}
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
                {filtered.length} conversations · page {page + 1} /{" "}
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
      )}
    </>
  );
}
