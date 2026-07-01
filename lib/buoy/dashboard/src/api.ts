// Thin typed fetch wrapper over buoy's JSON API. Paths are RELATIVE (no leading
// '/') so they resolve against <base href> — i.e. the proxy sub-path /proxy/buoy/.

export async function api<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${path} -> ${res.status}`)
  return (await res.json()) as T
}

export async function apiOr<T>(path: string, fallback: T): Promise<T> {
  try {
    return await api<T>(path)
  } catch {
    return fallback
  }
}

export function qs(params: Record<string, string | number>): string {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) p.set(k, String(v))
  return p.toString()
}
