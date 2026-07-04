import type { ProbeCheck, ProvisionFleet, ProvisionPool, ProvisioningSnapshot } from "../api";
import { useProbes } from "../hooks/useProbes";
import { formatRelative } from "./chartUtils";

// create→ready latency, reported in seconds; minutes once it crosses 90s.
function formatLatencySeconds(value: number | null): string {
  if (value === null) return "—";
  if (value < 90) return `${Math.round(value)}s`;
  return `${(value / 60).toFixed(1)}m`;
}

function ratioColor(ratio: number | null): string {
  if (ratio === null) return "text-slate-400";
  if (ratio >= 0.9) return "text-emerald-400";
  if (ratio >= 0.5) return "text-amber-400";
  return "text-rose-400";
}

function formatPercent(ratio: number | null): string {
  return ratio === null ? "—" : `${Math.round(ratio * 100)}%`;
}

function CheckPill({ check }: { check: ProbeCheck }) {
  return (
    <div
      className="flex items-center gap-2 rounded-md border border-slate-800 bg-slate-900/60 px-3 py-1.5"
      title={`updated ${formatRelative(check.collectedAt)}`}
    >
      <span
        className={`h-2.5 w-2.5 shrink-0 rounded-full ${check.up ? "bg-emerald-500" : "bg-rose-500"}`}
      />
      <span className="font-mono text-xs text-slate-300">{check.probe}</span>
      {check.latencyMs !== null && (
        <span className="text-xs text-slate-500">{check.latencyMs}ms</span>
      )}
    </div>
  );
}

// One outcome count rendered as a labelled chip; colour cues whether the
// outcome is good (ready) or a problem (stockout/error/preempted).
function OutcomeChip({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className={`font-mono text-sm ${tone}`}>{value}</span>
      <span className="text-xs text-slate-500">{label}</span>
    </div>
  );
}

function FleetSummary({ fleet }: { fleet: ProvisionFleet }) {
  return (
    <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
      <div>
        <div className={`text-3xl font-bold ${ratioColor(fleet.successRatio)}`}>
          {formatPercent(fleet.successRatio)}
        </div>
        <div className="text-xs text-slate-500">
          create success · {fleet.ready}/{fleet.outcomes} attempts
        </div>
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-1">
        <OutcomeChip label="ready" value={fleet.ready} tone="text-emerald-400" />
        <OutcomeChip label="stockout" value={fleet.stockout} tone="text-amber-400" />
        <OutcomeChip label="error" value={fleet.error} tone="text-rose-400" />
        <OutcomeChip label="preempted" value={fleet.preempted} tone="text-slate-300" />
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-1">
        <OutcomeChip label="pools placing" value={fleet.poolsPlacing} tone="text-emerald-400" />
        <OutcomeChip
          label="pools stuck"
          value={fleet.poolsStockoutDead}
          tone={fleet.poolsStockoutDead > 0 ? "text-rose-400" : "text-slate-300"}
        />
      </div>
      <div className="text-xs text-slate-500">
        latency p50 {formatLatencySeconds(fleet.latencyP50Seconds)} · p95{" "}
        {formatLatencySeconds(fleet.latencyP95Seconds)}
      </div>
    </div>
  );
}

function PoolRow({ pool }: { pool: ProvisionPool }) {
  return (
    <tr className="border-t border-slate-800/70">
      <td className="py-1.5 pr-3">
        <span className="font-mono text-xs text-slate-300">{pool.resourceType}</span>
        <span className="text-xs text-slate-500">
          {" "}
          · {pool.scaleGroup} · {pool.zone}
        </span>
      </td>
      <td className="px-2 text-right font-mono text-xs text-slate-400">{pool.outcomes}</td>
      <td className="px-2 text-right font-mono text-xs text-emerald-400">{pool.ready}</td>
      <td className="px-2 text-right font-mono text-xs text-amber-400">{pool.stockout}</td>
      <td className="px-2 text-right font-mono text-xs text-rose-400">{pool.error}</td>
      <td className="px-2 text-right font-mono text-xs text-slate-400">{pool.preempted}</td>
      <td className={`px-2 text-right font-mono text-xs ${ratioColor(pool.successRatio)}`}>
        {formatPercent(pool.successRatio)}
      </td>
      <td className="pl-2 text-right font-mono text-xs text-slate-400">
        {formatLatencySeconds(pool.latencyP50Seconds)}
      </td>
    </tr>
  );
}

function ProvisioningSection({ provisioning }: { provisioning: ProvisioningSnapshot }) {
  const { fleet, pools, windowHours, collectedAt } = provisioning;
  return (
    <section>
      <div className="mb-3 flex items-baseline gap-2">
        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Provisioning
        </h4>
        {windowHours !== null && (
          <span className="text-xs text-slate-500">trailing {windowHours}h window</span>
        )}
        {collectedAt && (
          <span className="ml-auto text-xs text-slate-600">
            collected {formatRelative(collectedAt)}
          </span>
        )}
      </div>
      {!fleet || collectedAt === null ? (
        <div className="text-sm text-slate-500">no provisioning data yet</div>
      ) : (
        <div className="space-y-4">
          <FleetSummary fleet={fleet} />
          {pools.length > 0 && (
            <table className="w-full">
              <thead>
                <tr className="text-left text-[10px] uppercase tracking-wider text-slate-600">
                  <th className="pb-1 pr-3 font-medium">pool</th>
                  <th className="px-2 pb-1 text-right font-medium">attempts</th>
                  <th className="px-2 pb-1 text-right font-medium">ready</th>
                  <th className="px-2 pb-1 text-right font-medium">stockout</th>
                  <th className="px-2 pb-1 text-right font-medium">error</th>
                  <th className="px-2 pb-1 text-right font-medium">preempt</th>
                  <th className="px-2 pb-1 text-right font-medium">success</th>
                  <th className="pl-2 pb-1 text-right font-medium">p50</th>
                </tr>
              </thead>
              <tbody>
                {pools.map((pool) => (
                  <PoolRow key={`${pool.resourceType} ${pool.scaleGroup} ${pool.zone}`} pool={pool} />
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </section>
  );
}

export function ProbesPanel() {
  const { data, isLoading, error } = useProbes();

  return (
    <section>
      <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-xl font-semibold text-slate-200">Probes</h2>
        <span className="text-xs text-slate-500">synthetic canary · infra/probes</span>
      </div>
      <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        {isLoading && <div className="text-slate-400">loading…</div>}
        {error && (
          <div className="text-rose-400">failed to load: {(error as Error).message}</div>
        )}
        {data?.error && <div className="text-sm text-rose-400">{data.error}</div>}
        {data && !data.error && (
          <div className="space-y-5">
            <section>
              <h4 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Health checks
              </h4>
              {data.checks.length === 0 ? (
                <div className="text-sm text-slate-500">no probe samples in the last 2h</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {data.checks.map((check) => (
                    <CheckPill key={check.probe} check={check} />
                  ))}
                </div>
              )}
            </section>
            <div className="border-t border-slate-800" />
            <ProvisioningSection provisioning={data.provisioning} />
          </div>
        )}
      </div>
    </section>
  );
}
