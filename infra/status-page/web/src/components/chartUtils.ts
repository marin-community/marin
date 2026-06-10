import { useCallback, useEffect, useState } from "react";

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
