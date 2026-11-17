// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type { StagePhase } from "./phase-slice";
import type { RowMode } from "./timeline-rows";

export type PhaseFilter = StagePhase;

export interface TimelineViewSnapshot {
  viewStart: number;
  viewEnd: number | null;
  rowMode: RowMode;
  phaseFilter: PhaseFilter;
  expanded: string[];
  scrollTop: number;
  scrollLeft: number;
}

const TIMELINE_VIEW_SNAPSHOT_PREFIX = "agentic-timeline-view:";
const ROW_MODE_VALUES: readonly RowMode[] = ["conversation", "worker"];
const PHASE_FILTER_VALUES: readonly PhaseFilter[] = ["warmup", "profiling"];

const finiteOr = (value: unknown, fallback: number): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

export function parseTimelineViewSnapshot(
  raw: string | null,
): TimelineViewSnapshot | null {
  if (!raw) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const record = parsed as Record<string, unknown>;
  const rowMode = ROW_MODE_VALUES.includes(record.rowMode as RowMode)
    ? (record.rowMode as RowMode)
    : "conversation";
  const phaseFilter = PHASE_FILTER_VALUES.includes(
    record.phaseFilter as PhaseFilter,
  )
    ? (record.phaseFilter as PhaseFilter)
    : "profiling";
  const viewEnd =
    typeof record.viewEnd === "number" && Number.isFinite(record.viewEnd)
      ? record.viewEnd
      : null;
  const expanded = Array.isArray(record.expanded)
    ? record.expanded.filter(
        (entry): entry is string => typeof entry === "string",
      )
    : [];
  return {
    viewStart: finiteOr(record.viewStart, 0),
    viewEnd,
    rowMode,
    phaseFilter,
    expanded,
    scrollTop: finiteOr(record.scrollTop, 0),
    scrollLeft: finiteOr(record.scrollLeft, 0),
  };
}

function timelineSnapshotKey(pointId: string): string {
  return `${TIMELINE_VIEW_SNAPSHOT_PREFIX}${pointId}`;
}

export function saveTimelineViewSnapshot(
  pointId: string,
  snapshot: TimelineViewSnapshot,
): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(
      timelineSnapshotKey(pointId),
      JSON.stringify(snapshot),
    );
  } catch {}
}

export function consumeTimelineViewSnapshot(
  pointId: string,
): TimelineViewSnapshot | null {
  if (typeof window === "undefined") return null;
  try {
    const key = timelineSnapshotKey(pointId);
    const raw = window.sessionStorage.getItem(key);
    window.sessionStorage.removeItem(key);
    return parseTimelineViewSnapshot(raw);
  } catch {
    return null;
  }
}
