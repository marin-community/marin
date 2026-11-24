// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type { Locale } from "@/lib/i18n/i18n";

function localeTag(locale: Locale): "en-US" | "zh-CN" {
  return locale === "zh" ? "zh-CN" : "en-US";
}

export function localeNumber(n: number, locale: Locale): string {
  return n.toLocaleString(localeTag(locale));
}

export function compact(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  if (abs > 0 && abs < 1) return n.toFixed(2);
  return String(Math.round(n));
}

export function perConversation(
  n: number | undefined,
  locale: Locale = "en",
): string {
  if (typeof n !== "number" || !Number.isFinite(n)) return "—";
  return n.toLocaleString(localeTag(locale), { maximumFractionDigits: 1 });
}

export function formatPct(fraction: number | undefined): string {
  return typeof fraction === "number" ? `${(fraction * 100).toFixed(0)}%` : "—";
}

export function formatShare(part: number, total: number): string {
  return total > 0 ? `${((part / total) * 100).toFixed(0)}%` : "—";
}
