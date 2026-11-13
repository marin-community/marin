// SPDX-License-Identifier: GPL-3.0-only
// InferenceX 315128ce wire contracts.
export interface RequestRecord {
  cid: string;
  ri?: number;
  ti: number;
  srcTrace?: string;
  srcOuter?: number;
  srcInner?: number;
  srcKind?: string;
  wid: string;
  ad: number;
  phase: string;
  credit: number;
  start: number;
  ack: number | null;
  end: number;
  ttftMs: number | null;
  tpotMs: number | null;
  isl: number | null;
  osl: number | null;
  cancelled: boolean;
}

export interface RequestTimeline {
  version: number;
  startNs: number;
  endNs: number;
  durationS: number;
  requests: RequestRecord[];
}
