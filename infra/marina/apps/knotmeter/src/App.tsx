import { useEffect, useState } from "react";
import { api, type Run } from "./types";
import { useRoute } from "./navigation";
import { Performance } from "./Performance";
import { RunView } from "./RunView";
import { CorpusPage } from "./CorpusPage";
import { Runs } from "./Runs";
import { CacheReuse } from "./CacheReuse";
import { History } from "./History";

export function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const { path } = useRoute();
  const runKey = path.startsWith("/run/")
    ? decodeURIComponent(path.slice(5))
    : null;
  const run = runs.find((r) => r.key === runKey);
  useEffect(() => {
    api<Run[]>("runs")
      .then(setRuns)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);
  return (
    <>
      <header className="app-header">
        <a href="#/chart" className="brand">
          Knotmeter
        </a>
        <span>Saved AgentX measurements</span>
        <a href="https://marina.oa.dev/" className="directory-link">
          Marina apps
        </a>
      </header>
      <main>
        <nav className="toolbar">
          <a href="#/chart">Performance</a>
          <a href="#/runs">Runs and submissions</a>
          <a href="#/history">History</a>
          <a href="#/cache">Cache and aggregates</a>
          <a href="#/corpus">Corpus</a>
        </nav>
        {error ? (
          <p role="alert">{error}</p>
        ) : loading ? (
          <p>Loading saved runs…</p>
        ) : path.startsWith("/corpus") || path.startsWith("/agentx/") ? (
          <CorpusPage key={path} />
        ) : path === "/runs" ? (
          <Runs runs={runs} />
        ) : path === "/cache" ? (
          <CacheReuse runs={runs} />
        ) : path === "/history" ? (
          <History runs={runs} />
        ) : run ? (
          <RunView run={run} runs={runs} key={run.key} />
        ) : runKey ? (
          <p>Run has not been imported.</p>
        ) : runs.length ? (
          <Performance runs={runs} key={path} />
        ) : (
          <p>
            No runs imported. Load a saved collector bundle to browse
            measurements.
          </p>
        )}
      </main>
      <footer>
        <a href="attribution.txt">Source attribution</a> ·{" "}
        <a href="inferencex-license.txt">GPL v3</a> ·{" "}
        <a href="https://github.com/marin-community/marin/pull/11020">
          Application source
        </a>
      </footer>
    </>
  );
}
