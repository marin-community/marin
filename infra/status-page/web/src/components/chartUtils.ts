import { useCallback, useEffect, useState } from "react";

// Palette for per-region chart lines, picked for contrast on the dark
// background; cycles if there are more regions than entries. Shared by the
// worker-availability and provisioning charts so a region keeps one color
// across both.
export const REGION_COLORS = [
  "#10b981", // emerald-500
  "#06b6d4", // cyan-500
  "#8b5cf6", // violet-500
  "#f59e0b", // amber-500
  "#ec4899", // pink-500
  "#f43f5e", // rose-500
  "#14b8a6", // teal-500
  "#3b82f6", // blue-500
];

// Map region name → stable color. Keyed off the alphabetically-sorted region
// set (not display order) so a region's color doesn't shift when counts
// reorder a legend, and so both charts agree on the same region's color.
export function regionColorMap(regions: string[]): Map<string, string> {
  const map = new Map<string, string>();
  [...regions].sort().forEach((region, i) => map.set(region, REGION_COLORS[i % REGION_COLORS.length]));
  return map;
}

// Human "N{s,m,h,d} ago" for an ISO timestamp. Non-parseable input is
// echoed back. Shared by every panel's "updated …" / "collected …" line.
export function formatRelative(iso: string): string {
  const delta = Date.now() - Date.parse(iso);
  if (!Number.isFinite(delta)) return iso;
  const seconds = Math.round(delta / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

export function formatClock(ms: number): string {
  const d = new Date(ms);
  const hh = d.getHours().toString().padStart(2, "0");
  const mm = d.getMinutes().toString().padStart(2, "0");
  return `${hh}:${mm}`;
}

export function formatDuration(ms: number): string {
  if (ms < 60_000) return `${Math.max(1, Math.round(ms / 1000))}s`;
  const minutes = Math.round(ms / 60_000);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const remMinutes = minutes % 60;
  if (hours < 24) {
    return remMinutes === 0 ? `${hours}h` : `${hours}h ${remMinutes}m`;
  }
  const days = Math.floor(hours / 24);
  const remHours = hours % 24;
  return remHours === 0 ? `${days}d` : `${days}d ${remHours}h`;
}

export function displayedSpanLabel(samples: { t: number }[]): string {
  if (samples.length < 2) return "history warming up";
  const first = samples[0]?.t;
  const last = samples[samples.length - 1]?.t;
  if (first === undefined || last === undefined || last <= first) {
    return "history warming up";
  }
  return `last ${formatDuration(last - first)}`;
}

export function useContainerSize<T extends HTMLElement>() {
  const [node, setNode] = useState<T | null>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);

  const ref = useCallback((el: T | null) => setNode(el), []);

  useEffect(() => {
    if (!node) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setSize({ width, height });
        }
      }
    });
    obs.observe(node);
    return () => obs.disconnect();
  }, [node]);

  return { ref, size };
}
