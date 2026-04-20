// In-process TTL cache with request coalescing.
//
// Concurrent callers for the same key share a single in-flight promise,
// so the upstream (GitHub, the iris controller) sees at most one request
// per key per TTL window even under a thundering herd.

interface Entry<T> {
  value: T;
  expiresAt: number;
}

export class TTLCache<T> {
  private readonly entries = new Map<string, Entry<T>>();
  private readonly inflight = new Map<string, Promise<T>>();

  constructor(private readonly ttlMs: number) {}

  async get(key: string, loader: () => Promise<T>): Promise<T> {
    const now = Date.now();
    const hit = this.entries.get(key);
    if (hit && hit.expiresAt > now) {
      return hit.value;
    }

    const existing = this.inflight.get(key);
    if (existing) {
      return existing;
    }

    const promise = (async () => {
      try {
        const value = await loader();
        this.entries.set(key, { value, expiresAt: Date.now() + this.ttlMs });
        return value;
      } finally {
        this.inflight.delete(key);
      }
    })();

    this.inflight.set(key, promise);
    return promise;
  }
}
