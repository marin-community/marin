import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useWorkers } from "../hooks/useWorkers";
import { useWorkersHistory } from "../hooks/useWorkersHistory";

function formatClock(ms: number): string {
  const d = new Date(ms);
  const hh = d.getHours().toString().padStart(2, "0");
  const mm = d.getMinutes().toString().padStart(2, "0");
  return `${hh}:${mm}`;
}

export function WorkersPanel() {
  const { data, isLoading, error } = useWorkers();
  const history = useWorkersHistory();
  const samples = history.data?.samples ?? [];

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Workers
        </h3>
        <span className="text-xs text-slate-500">
          {samples.length > 1
            ? `${samples.length} samples · last 24h`
            : "history warming up"}
        </span>
      </div>
      <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        {isLoading && <div className="text-slate-400">loading…</div>}
        {error && (
          <div className="text-rose-400">failed to load: {(error as Error).message}</div>
        )}
        {data?.error && <div className="text-sm text-rose-400">{data.error}</div>}
        {data && !data.error && (
          <>
            <div className="flex items-baseline gap-3">
              <span className="text-4xl font-bold text-emerald-300">{data.available}</span>
              <span className="text-slate-400">available</span>
              <span className="text-slate-600">·</span>
              <span className="text-2xl font-semibold text-slate-200">{data.total}</span>
              <span className="text-slate-400">total</span>
            </div>
            <div className="mt-4 h-48">
              {samples.length > 1 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={samples}
                    margin={{ top: 4, right: 8, bottom: 4, left: -16 }}
                  >
                    <CartesianGrid stroke="#1e293b" strokeDasharray="2 4" />
                    <XAxis
                      dataKey="t"
                      type="number"
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={formatClock}
                      stroke="#475569"
                      fontSize={11}
                    />
                    <YAxis stroke="#475569" fontSize={11} />
                    <Tooltip
                      contentStyle={{
                        background: "#0f172a",
                        border: "1px solid #1e293b",
                        borderRadius: 4,
                        fontSize: 12,
                      }}
                      labelFormatter={(value) =>
                        new Date(value as number).toLocaleString()
                      }
                    />
                    <Line
                      type="monotone"
                      dataKey="total"
                      stroke="#475569"
                      strokeWidth={1}
                      strokeDasharray="4 4"
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="available"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-500">
                  history warming up — samples collected every 30s, chart appears once
                  we have two points
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
