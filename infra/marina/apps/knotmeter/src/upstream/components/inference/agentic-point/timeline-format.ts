// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export function formatTickLabel(ns: number): string {
  const ms = ns / 1e6;
  if (ms < 1000) return `+${ms.toFixed(0)}ms`;
  if (ms < 60_000) return `+${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`;
  return `+${(ms / 60_000).toFixed(1)}m`;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)}s`;
  return `${(ms / 60_000).toFixed(2)}m`;
}
