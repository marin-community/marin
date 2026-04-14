// Fixed-capacity circular buffer for worker-count history.
//
// Capacity is sized so the buffer holds 24h of samples at a 30s cadence
// (2880 slots). Each append overwrites the oldest slot once full, so
// memory stays bounded regardless of how long the server runs. History
// is in-process and lost on restart — see infra/status-page/README.md
// "Known limitations" for the follow-up plan (persist to GCS or grow a
// worker_count_history table in the controller).

import type { WorkerSample } from "./sources/workers.js";

export class WorkerHistory {
  private readonly capacity: number;
  private readonly buffer: (WorkerSample | undefined)[];
  private head = 0;
  private size = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }

  push(sample: WorkerSample): void {
    this.buffer[this.head] = sample;
    this.head = (this.head + 1) % this.capacity;
    if (this.size < this.capacity) {
      this.size += 1;
    }
  }

  /** Snapshot of samples in chronological order. */
  samples(): WorkerSample[] {
    const out: WorkerSample[] = [];
    const start = this.size < this.capacity ? 0 : this.head;
    for (let i = 0; i < this.size; i++) {
      const sample = this.buffer[(start + i) % this.capacity];
      if (sample !== undefined) {
        out.push(sample);
      }
    }
    return out;
  }
}
