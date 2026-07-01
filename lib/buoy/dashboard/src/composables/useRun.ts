import { ref, shallowRef } from 'vue'
import { api, apiOr, qs } from '../api'
import type { Manifest } from '../types'

export interface RunRef {
  entity: string
  project: string
  run_id: string
}

export type FlatValue = string | number | boolean | null
export type Flat = Record<string, FlatValue>

interface MirrorStatus {
  state: string
  detail?: string
  error?: string
}

const LIVE_INTERVAL = 30_000
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

// The query params buoy's API expects (it accepts either `run` or `run_id`).
const query = (r: RunRef) => qs({ entity: r.entity, project: r.project, run: r.run_id })

// Flatten nested config/summary into dotted keys for the summary tables.
function flatten(obj: unknown, prefix = ''): Flat {
  const out: Flat = {}
  if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      const key = prefix ? `${prefix}.${k}` : k
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        Object.assign(out, flatten(v, key))
      } else {
        out[key] = Array.isArray(v) ? JSON.stringify(v) : (v as FlatValue)
      }
    }
  }
  return out
}

// Loads one run: POST /api/mirror, poll status to done, then read manifest/config/
// summary. A newer load() supersedes an in-flight one (seq guard). While a run is
// running, a browser-driven timer re-mirrors it every LIVE_INTERVAL (no server
// watcher) and refreshes the manifest in place.
export function useRun() {
  const manifest = shallowRef<Manifest | null>(null)
  const config = ref<Flat>({})
  const summary = ref<Flat>({})
  const loading = ref<{ verb: string; detail: string } | null>(null)
  const error = ref<string | null>(null)
  let seq = 0
  let current: RunRef | null = null
  let liveTimer: ReturnType<typeof setTimeout> | null = null

  async function mirrorJob(r: RunRef, refresh: boolean, verb: string, silent = false): Promise<string> {
    await fetch('api/mirror', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ entity: r.entity, project: r.project, run_id: r.run_id, refresh }),
    })
    for (;;) {
      const s = await api<MirrorStatus>(`api/mirror_status?${query(r)}`)
      if (s.state !== 'running') return s.state
      if (!silent) loading.value = { verb, detail: s.detail ?? '…' }
      await sleep(1500)
    }
  }

  async function readSnapshot(r: RunRef): Promise<Manifest> {
    const [m, cfg, summ] = await Promise.all([
      api<Manifest>(`api/manifest?${query(r)}`),
      apiOr<Record<string, unknown>>(`api/config?${query(r)}`, {}),
      apiOr<Record<string, unknown>>(`api/summary?${query(r)}`, {}),
    ])
    manifest.value = m
    config.value = flatten(cfg)
    summary.value = flatten(summ)
    return m
  }

  function scheduleLive() {
    if (liveTimer) clearTimeout(liveTimer)
    liveTimer = null
    if (current && manifest.value?.state === 'running') {
      liveTimer = setTimeout(refreshLive, LIVE_INTERVAL)
    }
  }

  async function refreshLive() {
    if (!current) return
    const r = current
    const mine = seq
    const state = await mirrorJob(r, false, 'mirroring', true) // silent: no overlay
    if (mine !== seq) return
    if (state === 'done') await readSnapshot(r)
    scheduleLive()
  }

  async function load(r: RunRef, refresh = false) {
    const mine = ++seq
    current = r
    if (liveTimer) clearTimeout(liveTimer)
    error.value = null
    const verb = refresh ? 'refetching' : 'mirroring'
    loading.value = { verb, detail: 'starting…' }
    const state = await mirrorJob(r, refresh, verb)
    if (mine !== seq) return // superseded by a newer selection
    if (state === 'error' || state === 'absent') {
      error.value = state === 'absent' ? 'run not found' : 'mirror failed'
      loading.value = null
      return
    }
    await readSnapshot(r)
    if (mine !== seq) return
    loading.value = null
    scheduleLive()
  }

  const refetch = () => (current ? load(current, true) : undefined)

  return { manifest, config, summary, loading, error, load, refetch }
}
