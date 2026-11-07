// SPDX-License-Identifier: GPL-3.0-only
// InferenceX 315128ce wire contracts.
export interface RequestChartRecord {
  cid: string;
  phase: string;
  start: number;
  end: number;
  ttftMs: number | null;
  tpotMs: number | null;
  isl: number | null;
  osl: number | null;
  cancelled: boolean;
}

export interface RequestChartData {
  version: number;
  timelineVersion: number;
  startNs: number;
  endNs: number;
  durationS: number;
  requests: RequestChartRecord[];
}
