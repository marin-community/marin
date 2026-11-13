// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type { RequestRecord } from "@/hooks/api/use-request-timeline";
import { localePath, type Locale } from "@/lib/i18n/i18n";

export type RowMode = "conversation" | "worker";

export function datasetConvId(cid: string): string {
  const i = cid.indexOf("::");
  return i === -1 ? cid : cid.slice(0, i);
}

export function subagentIdOf(cid: string): string | null {
  const i = cid.indexOf("::sa:");
  if (i === -1) return null;
  const raw = cid.slice(i + "::sa:".length);
  const colon = raw.indexOf(":");
  return colon === -1 ? raw : raw.slice(0, colon);
}

export function conversationHref(
  datasetSlug: string,
  req: RequestRecord,
  locale: Locale = "en",
): string {
  const convId = req.srcTrace ?? datasetConvId(req.cid);
  const params = new URLSearchParams({ turn: String(req.ti) });
  if (
    typeof req.srcOuter === "number" &&
    Number.isInteger(req.srcOuter) &&
    req.srcOuter >= 0
  ) {
    params.set("raw", String(req.srcOuter));
    if (
      typeof req.srcInner === "number" &&
      Number.isInteger(req.srcInner) &&
      req.srcInner >= 0
    ) {
      params.set("inner", String(req.srcInner));
    }
  }
  const sa = subagentIdOf(req.cid);
  if (sa && !params.has("inner")) params.set("sa", sa);
  return localePath(
    `#/agentx/${datasetSlug}/conversations/${encodeURIComponent(convId)}?${params.toString()}`,
    locale,
  );
}

export function requestSourceLabel(
  req: RequestRecord,
  locale: Locale = "en",
): string {
  if (typeof req.srcOuter === "number") {
    if (typeof req.srcInner === "number")
      return locale === "zh"
        ? `原始记录 ${req.srcOuter} / 子项 ${req.srcInner}`
        : `raw ${req.srcOuter} / child ${req.srcInner}`;
    return locale === "zh" ? `原始记录 ${req.srcOuter}` : `raw ${req.srcOuter}`;
  }
  return locale === "zh"
    ? `回放轮次 ${req.ti + 1}`
    : `replay turn ${req.ti + 1}`;
}

export interface RequestIdleStats {
  idleNs: number;
  spanNs: number;
}

export function requestIdleStats(
  requests: readonly RequestRecord[],
): RequestIdleStats {
  const intervals = requests
    .filter(
      ({ start, end }) =>
        Number.isFinite(start) && Number.isFinite(end) && end >= start,
    )
    .map(({ start, end }) => ({ start, end }))
    .toSorted((a, b) => a.start - b.start || a.end - b.end);
  if (intervals.length === 0) return { idleNs: 0, spanNs: 0 };

  const firstStart = intervals[0]!.start;
  let mergedEnd = intervals[0]!.end;
  let idleNs = 0;
  for (let i = 1; i < intervals.length; i++) {
    const interval = intervals[i]!;
    if (interval.start > mergedEnd) idleNs += interval.start - mergedEnd;
    if (interval.end > mergedEnd) mergedEnd = interval.end;
  }
  return { idleNs, spanNs: mergedEnd - firstStart };
}

const ROW_COLORS = [
  "#3b82f6",
  "#ef4444",
  "#10b981",
  "#f59e0b",
  "#a855f7",
  "#06b6d4",
  "#f97316",
  "#84cc16",
  "#ec4899",
  "#14b8a6",
  "#8b5cf6",
  "#eab308",
];

type RowKind = "parent" | "worker" | "subagent" | "stream" | "aux";

export interface RequestTimelineRow {
  key: string;
  label: string;
  color: string;
  requests: RequestRecord[];
  depth: number;
  kind: RowKind;
  streamCount?: number;
  parentRowKey?: string;
  auxCount?: number;
}

export function splitTimelineCid(cid: string): {
  parent: string;
  subagentBase: string | null;
  stream: number | null;
  aux: string | null;
} {
  const sep = cid.indexOf("::sa:");
  if (sep === -1) {
    const auxSep = cid.indexOf("::aux:");
    if (auxSep !== -1) {
      return {
        parent: cid.slice(0, auxSep),
        subagentBase: null,
        stream: null,
        aux: cid.slice(auxSep + "::aux:".length),
      };
    }
    return { parent: cid, subagentBase: null, stream: null, aux: null };
  }
  const parent = cid.slice(0, sep);
  const raw = cid.slice(sep + 5);
  const auxMatch = /^(?<base>[^:]+):aux:(?<aux>.+)$/.exec(raw);
  if (auxMatch) {
    return {
      parent,
      subagentBase: auxMatch.groups!.base!,
      stream: null,
      aux: auxMatch.groups!.aux!,
    };
  }
  const m = /^(?<base>.*):s(?<stream>\d+)$/.exec(raw);
  if (m)
    return { parent, subagentBase: m[1]!, stream: Number(m[2]), aux: null };
  return { parent, subagentBase: raw, stream: null, aux: null };
}

export function conversationReplayKey(request: RequestRecord): string {
  const parent = splitTimelineCid(request.cid).parent;
  return request.ri === undefined
    ? parent
    : `${parent}::replay-lane:${request.ri}`;
}

export function computeStableRowIndex(
  requests: readonly RequestRecord[],
  mode: RowMode,
): Map<string, number> {
  const firstStart = new Map<string, number>();
  const inProfiling = new Set<string>();
  const inWarmup = new Set<string>();
  for (const r of requests) {
    const key = mode === "conversation" ? conversationReplayKey(r) : r.wid;
    const cur = firstStart.get(key);
    if (cur === undefined || r.start < cur) firstStart.set(key, r.start);
    if (r.phase === "profiling") inProfiling.add(key);
    else inWarmup.add(key);
  }
  const spansBoth = (key: string) => inProfiling.has(key) && inWarmup.has(key);
  const keys = [...firstStart.keys()].toSorted(
    (a, b) =>
      Number(spansBoth(b)) - Number(spansBoth(a)) ||
      firstStart.get(a)! - firstStart.get(b)! ||
      (a < b ? -1 : a > b ? 1 : 0),
  );
  const index = new Map<string, number>();
  keys.forEach((key, i) => index.set(key, i));
  return index;
}

export function buildRequestTimelineRows(
  requests: RequestRecord[],
  mode: RowMode,
  expandedSubagents: ReadonlySet<string>,
  stableRowIndex?: ReadonlyMap<string, number>,
  locale: Locale = "en",
): RequestTimelineRow[] {
  const index = stableRowIndex ?? computeStableRowIndex(requests, mode);
  const colorFor = (key: string) =>
    ROW_COLORS[
      (((index.get(key) ?? 0) % ROW_COLORS.length) + ROW_COLORS.length) %
        ROW_COLORS.length
    ]!;
  const orderOf = (key: string) => index.get(key) ?? Number.POSITIVE_INFINITY;
  if (mode !== "conversation") {
    const groups = new Map<string, RequestRecord[]>();
    for (const r of requests) {
      let list = groups.get(r.wid);
      if (!list) {
        list = [];
        groups.set(r.wid, list);
      }
      list.push(r);
    }
    const rows: RequestTimelineRow[] = [];
    for (const [key, list] of groups) {
      list.sort((a, b) => a.start - b.start);
      rows.push({
        key,
        label: shortenWid(key),
        color: colorFor(key),
        requests: list,
        depth: 0,
        kind: "worker",
      });
    }
    rows.sort(
      (a, b) =>
        orderOf(a.key) - orderOf(b.key) ||
        a.requests[0]!.start - b.requests[0]!.start,
    );
    return rows;
  }

  interface SubagentLanes {
    streams: Map<number | null, RequestRecord[]>;
    aux: Map<string, RequestRecord[]>;
  }
  interface Tree {
    groupKey: string;
    parentCid: string;
    replayIndex?: number;
    parentReqs: RequestRecord[];
    parentAux: Map<string, RequestRecord[]>;
    subagents: Map<string, SubagentLanes>;
    firstStart: number;
  }
  const trees = new Map<string, Tree>();
  for (const r of requests) {
    const { parent, subagentBase, stream, aux } = splitTimelineCid(r.cid);
    const groupKey = conversationReplayKey(r);
    let tree = trees.get(groupKey);
    if (!tree) {
      tree = {
        groupKey,
        parentCid: parent,
        replayIndex: r.ri,
        parentReqs: [],
        parentAux: new Map(),
        subagents: new Map(),
        firstStart: Number.POSITIVE_INFINITY,
      };
      trees.set(groupKey, tree);
    }
    if (subagentBase === null && aux !== null) {
      const list = tree.parentAux.get(aux);
      if (list) list.push(r);
      else tree.parentAux.set(aux, [r]);
    } else if (subagentBase === null) {
      tree.parentReqs.push(r);
    } else {
      let lanes = tree.subagents.get(subagentBase);
      if (!lanes) {
        lanes = { streams: new Map(), aux: new Map() };
        tree.subagents.set(subagentBase, lanes);
      }
      if (aux === null) {
        const list = lanes.streams.get(stream);
        if (list) list.push(r);
        else lanes.streams.set(stream, [r]);
      } else {
        const list = lanes.aux.get(aux);
        if (list) list.push(r);
        else lanes.aux.set(aux, [r]);
      }
    }
    if (r.start < tree.firstStart) tree.firstStart = r.start;
  }

  const sortedTrees = [...trees.values()].toSorted(
    (a, b) =>
      orderOf(a.groupKey) - orderOf(b.groupKey) || a.firstStart - b.firstStart,
  );
  const rows: RequestTimelineRow[] = [];
  for (const tree of sortedTrees) {
    const color = colorFor(tree.groupKey);
    tree.parentReqs.sort((a, b) => a.start - b.start);
    const parentRowKey =
      tree.parentReqs.length > 0 ? tree.groupKey : `__parent_${tree.groupKey}`;
    const replayLabel =
      tree.replayIndex === undefined
        ? tree.parentCid
        : `${tree.parentCid} · ${locale === "zh" ? "重放" : "replay"} ${tree.replayIndex + 1}`;
    rows.push({
      key: parentRowKey,
      label: replayLabel,
      color,
      requests: tree.parentReqs,
      depth: 0,
      kind: "parent",
    });

    const parentAuxEntries = [...tree.parentAux.entries()].toSorted(
      (a, b) =>
        (a[1][0]?.start ?? Number.POSITIVE_INFINITY) -
        (b[1][0]?.start ?? Number.POSITIVE_INFINITY),
    );
    for (const [auxId, reqs] of parentAuxEntries) {
      reqs.sort((a, b) => a.start - b.start);
      rows.push({
        key: `${tree.groupKey}::aux:${auxId}`,
        label: `aux ${auxId} · ${locale === "zh" ? "并行" : "parallel"}`,
        color,
        requests: reqs,
        depth: 1,
        kind: "aux",
        parentRowKey,
      });
    }

    const subagentEntries = [...tree.subagents.entries()].toSorted((a, b) => {
      const aStart = Math.min(
        ...[...a[1].streams.values(), ...a[1].aux.values()].map(
          (reqs) => reqs[0]?.start ?? Number.POSITIVE_INFINITY,
        ),
      );
      const bStart = Math.min(
        ...[...b[1].streams.values(), ...b[1].aux.values()].map(
          (reqs) => reqs[0]?.start ?? Number.POSITIVE_INFINITY,
        ),
      );
      return aStart - bStart;
    });
    for (const [saBase, lanes] of subagentEntries) {
      const subagentKey = `${tree.groupKey}::sa:${saBase}`;
      const allReqs: RequestRecord[] = [];
      for (const reqs of lanes.streams.values()) {
        for (const request of reqs) allReqs.push(request);
      }
      allReqs.sort((a, b) => a.start - b.start);
      const streamCount = lanes.streams.size;
      rows.push({
        key: subagentKey,
        label: `↳ ${formatSubagentLabel(saBase, locale)}`,
        color,
        requests: allReqs,
        depth: 1,
        kind: "subagent",
        streamCount,
        auxCount: lanes.aux.size,
      });

      if (streamCount > 1 && expandedSubagents.has(subagentKey)) {
        const streamEntries = [...lanes.streams.entries()].toSorted((a, b) => {
          const ai = a[0] ?? -1;
          const bi = b[0] ?? -1;
          return ai - bi;
        });
        for (const [streamIdx, reqs] of streamEntries) {
          reqs.sort((a, b) => a.start - b.start);
          rows.push({
            key: `${subagentKey}:s${streamIdx ?? "∅"}`,
            label: `${locale === "zh" ? "流" : "stream"} ${streamIdx ?? "∅"}`,
            color,
            requests: reqs,
            depth: 2,
            kind: "stream",
            parentRowKey: subagentKey,
          });
        }
      }

      const auxEntries = [...lanes.aux.entries()].toSorted(
        (a, b) =>
          (a[1][0]?.start ?? Number.POSITIVE_INFINITY) -
          (b[1][0]?.start ?? Number.POSITIVE_INFINITY),
      );
      for (const [auxId, reqs] of auxEntries) {
        reqs.sort((a, b) => a.start - b.start);
        rows.push({
          key: `${subagentKey}:aux:${auxId}`,
          label: `aux ${auxId} · ${locale === "zh" ? "并行" : "parallel"}`,
          color,
          requests: reqs,
          depth: 2,
          kind: "aux",
          parentRowKey: subagentKey,
        });
      }
    }
  }
  return rows;
}

export function formatSubagentLabel(raw: string, _locale: Locale): string {
  const m = /^subagent_(?<index>\d+)_(?<hash>[0-9a-f]+)$/iu.exec(raw);
  if (!m) return raw;
  return `subagent ${m[1]} · ${m[2]!.slice(0, 4)}`;
}

export function shortenWid(wid: string): string {
  return wid.replace(/^worker_/, "w_").slice(0, 12);
}
