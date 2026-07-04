// Fixed-capacity circular buffers for the in-process sample histories.
//
// Capacity is sized so a buffer holds 24h of samples at its sampler's cadence.
// Each append overwrites the oldest slot once full, so memory stays bounded
// regardless of how long the server runs. These histories are in-process and
// lost on restart; worker-count history moved out to finelog (see
// server/sources/clusterHistory.ts), but the iris-ping and control-plane
// latency series are still sampled in-process here.

import type { IrisPingSample } from "./sources/iris.js";
import type { ServiceHealthHistorySample } from "./sources/serviceHealth.js";

export class RingBuffer<T> {
  private readonly capacity: number;
  private readonly buffer: (T | undefined)[];
  private head = 0;
  private size = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }

  push(sample: T): void {
    this.buffer[this.head] = sample;
    this.head = (this.head + 1) % this.capacity;
    if (this.size < this.capacity) {
      this.size += 1;
    }
  }

  /** Snapshot of samples in chronological order. */
  samples(): T[] {
    const out: T[] = [];
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

export class IrisPingHistory extends RingBuffer<IrisPingSample> {}

export class ServiceHealthHistory extends RingBuffer<ServiceHealthHistorySample> {}
