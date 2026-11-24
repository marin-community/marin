// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type { StructureNode } from "@/hooks/api/use-datasets";
import type { Locale } from "@/lib/i18n/i18n";

export const OVERLAP_COLORS = [
  "#06b6d4",
  "#ec4899",
  "#6366f1",
  "#84cc16",
  "#f97316",
] as const;

export const MAX_LANES = 6;

export interface TimedRequest {
  key: string;
  startS?: number;
  endS?: number;
}

export interface RequestOverlapGroup {
  id: string;
  requestKeys: string[];
  startS: number;
  endS: number;
}

export function findRequestOverlapGroups(
  requests: TimedRequest[],
  scopeKey = "scope",
): RequestOverlapGroup[] {
  const valid = requests.filter(
    (request): request is TimedRequest & { startS: number; endS: number } =>
      Number.isFinite(request.startS) &&
      Number.isFinite(request.endS) &&
      request.endS! > request.startS!,
  );
  const boundaries = [
    ...new Set(valid.flatMap((request) => [request.startS, request.endS])),
  ].toSorted((a, b) => a - b);
  const candidates = new Map<string, Omit<RequestOverlapGroup, "id">>();

  for (let i = 0; i < boundaries.length - 1; i++) {
    const startS = boundaries[i]!;
    const endS = boundaries[i + 1]!;
    if (endS <= startS) continue;
    const requestKeys = valid
      .filter((request) => request.startS <= startS && request.endS >= endS)
      .map((request) => request.key)
      .toSorted();
    if (requestKeys.length < 2) continue;
    const key = requestKeys.join("\u0000");
    const existing = candidates.get(key);
    candidates.set(key, {
      requestKeys,
      startS: existing ? Math.min(existing.startS, startS) : startS,
      endS: existing ? Math.max(existing.endS, endS) : endS,
    });
  }

  const maximal = [...candidates.values()].filter(
    (candidate, _, all) =>
      !all.some(
        (other) =>
          other.requestKeys.length > candidate.requestKeys.length &&
          candidate.requestKeys.every((key) => other.requestKeys.includes(key)),
      ),
  );

  return maximal
    .toSorted(
      (a, b) =>
        a.startS - b.startS ||
        a.endS - b.endS ||
        a.requestKeys.join(",").localeCompare(b.requestKeys.join(",")),
    )
    .map((group, index) => ({ ...group, id: `${scopeKey}-${index + 1}` }));
}

export interface RowOverlap {
  id: string;
  label: string;
  color: string;
  startS: number;
  endS: number;
  peerCount: number;
}

export interface VisibleRow {
  key: string;
  label: string;
  sublabel?: string;
  timeLabel?: string;
  cached: number;
  uncached: number;
  output: number;
  total: number;
  indent: number;
  isGroup: boolean;
  isExpanded: boolean;
  groupIndex?: number;
  overlaps: RowOverlap[];
}

export function formatElapsedTime(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const mm = String(minutes).padStart(2, "0");
  const ss = String(secs).padStart(2, "0");
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function timeLabel(startS?: number, endS?: number): string | undefined {
  if (startS === undefined || !Number.isFinite(startS)) return undefined;
  const start = formatElapsedTime(startS);
  if (endS === undefined || !Number.isFinite(endS) || endS <= startS)
    return `+${start}`;
  return `+${start}–${formatElapsedTime(endS)}`;
}

export interface DeepLinkHighlight {
  turn?: number | null;
  raw?: number | null;
  inner?: number | null;
  agent?: string | null;
}

export interface DeepLinkTarget {
  rowKey: string;
  expandGroup: number | null;
}

export function resolveDeepLinkTarget(
  nodes: readonly StructureNode[],
  highlight: DeepLinkHighlight,
): DeepLinkTarget | null {
  const { turn, raw, inner, agent } = highlight;
  if (typeof raw === "number" && raw >= 0) {
    if (typeof inner === "number" && inner >= 0) {
      const gi = nodes.findIndex(
        (node, i) => node.kind === "subagent" && (node.rawIndex ?? i) === raw,
      );
      if (gi === -1) return null;
      const group = nodes[gi] as Extract<StructureNode, { kind: "subagent" }>;
      const ci = group.children.findIndex(
        (child, i) => (child.innerIndex ?? i) === inner,
      );
      if (ci === -1) return null;
      return { rowKey: `g-${gi}-c-${ci}`, expandGroup: gi };
    }
    const i = nodes.findIndex(
      (node, idx) => node.kind === "turn" && (node.rawIndex ?? idx) === raw,
    );
    if (i !== -1) return { rowKey: `t-${i}`, expandGroup: null };
    return null;
  }
  if (typeof turn !== "number" || turn < 0) return null;
  if (agent) {
    const gi = nodes.findIndex(
      (n) => n.kind === "subagent" && n.agentId === agent,
    );
    if (gi === -1) return null;
    const group = nodes[gi] as Extract<StructureNode, { kind: "subagent" }>;
    if (turn >= group.children.length) return null;
    return { rowKey: `g-${gi}-c-${turn}`, expandGroup: gi };
  }
  let ordinal = 0;
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i].kind === "turn") {
      if (ordinal === turn) return { rowKey: `t-${i}`, expandGroup: null };
      ordinal += 1;
    }
  }
  return null;
}

export function buildRowOverlaps(
  nodes: readonly StructureNode[],
): Map<string, RowOverlap[]> {
  const mainGroups = findRequestOverlapGroups(
    nodes.flatMap((node, i) =>
      node.kind === "turn"
        ? [{ key: `t-${i}`, startS: node.startS, endS: node.endS }]
        : [],
    ),
    "main",
  );
  const subagentGroups = nodes.flatMap((node, i) =>
    node.kind === "subagent"
      ? findRequestOverlapGroups(
          node.children.map((child, ci) => ({
            key: `g-${i}-c-${ci}`,
            startS: child.startS,
            endS: child.endS,
          })),
          `subagent-${i}`,
        )
      : [],
  );
  const groups: RequestOverlapGroup[] = [...mainGroups, ...subagentGroups];

  const byRow = new Map<string, RowOverlap[]>();
  groups.forEach((group, groupIndex) => {
    const overlap = {
      id: group.id,
      label: `P${groupIndex + 1}`,
      color: OVERLAP_COLORS[groupIndex % OVERLAP_COLORS.length]!,
      startS: group.startS,
      endS: group.endS,
      peerCount: group.requestKeys.length - 1,
    };
    group.requestKeys.forEach((key) =>
      byRow.set(key, [...(byRow.get(key) ?? []), overlap]),
    );
  });
  return byRow;
}

export function buildVisibleRows(
  nodes: readonly StructureNode[],
  expanded: ReadonlySet<number>,
  overlapsByRow: ReadonlyMap<string, RowOverlap[]>,
  locale: Locale = "en",
): VisibleRow[] {
  const turnLabel = (turn: number) =>
    locale === "zh" ? `第 ${turn} 轮` : `Turn ${turn}`;
  const turnCount = (count: number) =>
    locale === "zh" ? `${count} 轮` : `${count} turn${count === 1 ? "" : "s"}`;
  const subturnLabel = (turn: number) =>
    locale === "zh" ? `↳ 子轮次 ${turn}` : `↳ subturn ${turn}`;
  const out: VisibleRow[] = [];
  let turnNo = 0;
  nodes.forEach((node: StructureNode, i) => {
    if (node.kind === "turn") {
      turnNo += 1;
      out.push({
        key: `t-${i}`,
        label: turnLabel(turnNo),
        sublabel: node.model ?? undefined,
        timeLabel: timeLabel(node.startS, node.endS),
        cached: node.cached,
        uncached: node.uncached,
        output: node.out,
        total: node.in + node.out,
        indent: 0,
        isGroup: false,
        isExpanded: false,
        overlaps: overlapsByRow.get(`t-${i}`) ?? [],
      });
    } else {
      const isExpanded = expanded.has(i);
      out.push({
        key: `g-${i}`,
        label: `${node.label}`,
        sublabel: `${turnCount(node.children.length)}${
          node.durationMs ? ` · ${(node.durationMs / 1000).toFixed(0)}s` : ""
        }`,
        timeLabel: timeLabel(node.startS, node.endS),
        cached: node.cached,
        uncached: node.uncached,
        output: node.out,
        total: node.in + node.out,
        indent: 0,
        isGroup: true,
        isExpanded,
        groupIndex: i,
        overlaps: [],
      });
      if (isExpanded) {
        node.children.forEach((child, ci) => {
          out.push({
            key: `g-${i}-c-${ci}`,
            label: subturnLabel(ci + 1),
            sublabel: child.model ?? undefined,
            timeLabel: timeLabel(child.startS, child.endS),
            cached: child.cached,
            uncached: child.uncached,
            output: child.out,
            total: child.in + child.out,
            indent: 1,
            isGroup: false,
            isExpanded: false,
            overlaps: overlapsByRow.get(`g-${i}-c-${ci}`) ?? [],
          });
        });
      }
    }
  });
  return out;
}

export interface BraceSeg {
  role: "first" | "middle" | "last" | "through";
  isMember: boolean;
  color: string;
  groupId: string;
  peerCount: number;
  startS: number;
  endS: number;
}

export interface BraceLayout {
  laneCount: number;
  overflowLanes: number;
  rowSegs: { lane: number; seg: BraceSeg }[][];
}

export function computeBraceLayout(rows: readonly VisibleRow[]): BraceLayout {
  const groupMap = new Map<
    string,
    {
      id: string;
      color: string;
      peerCount: number;
      startS: number;
      endS: number;
      idxs: number[];
    }
  >();
  rows.forEach((r, idx) => {
    for (const ov of r.overlaps) {
      const g = groupMap.get(ov.id) ?? {
        id: ov.id,
        color: ov.color,
        peerCount: ov.peerCount,
        startS: ov.startS,
        endS: ov.endS,
        idxs: [],
      };
      g.idxs.push(idx);
      groupMap.set(ov.id, g);
    }
  });
  const groups = [...groupMap.values()]
    .filter((g) => g.idxs.length >= 2)
    .map((g) => ({
      ...g,
      min: Math.min(...g.idxs),
      max: Math.max(...g.idxs),
      members: new Set(g.idxs),
    }))
    .toSorted((a, b) => a.min - b.min || a.max - b.max);

  const laneEnd: number[] = [];
  const laneOf = new Map<string, number>();
  for (const g of groups) {
    let lane = laneEnd.findIndex((end) => end < g.min);
    if (lane === -1) {
      lane = laneEnd.length;
      laneEnd.push(g.max);
    } else {
      laneEnd[lane] = g.max;
    }
    laneOf.set(g.id, lane);
  }
  const rawLaneCount = laneEnd.length;
  const laneCount = Math.min(rawLaneCount, MAX_LANES);
  const displayLane = (lane: number) => Math.min(lane, laneCount - 1);

  const rowSegs: { lane: number; seg: BraceSeg }[][] = rows.map(() => []);
  for (const g of groups) {
    const lane = displayLane(laneOf.get(g.id)!);
    for (let idx = g.min; idx <= g.max; idx++) {
      const isMember = g.members.has(idx);
      const role =
        idx === g.min
          ? "first"
          : idx === g.max
            ? "last"
            : isMember
              ? "middle"
              : "through";
      const seg: BraceSeg = {
        role,
        isMember,
        color: g.color,
        groupId: g.id,
        peerCount: g.peerCount,
        startS: g.startS,
        endS: g.endS,
      };
      const cell = rowSegs[idx]!;
      const existing = cell.find((c) => c.lane === lane);
      if (!existing) cell.push({ lane, seg });
      else if (seg.isMember && !existing.seg.isMember) existing.seg = seg;
    }
  }
  return { laneCount, overflowLanes: rawLaneCount - laneCount, rowSegs };
}
