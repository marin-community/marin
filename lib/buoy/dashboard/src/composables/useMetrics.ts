import { ref } from 'vue'
import { api, qs } from '../api'
import type { MetricSeries } from '../types'
import type { RunRef } from './useRun'

export interface Series {
  x: number[]
  y: number[]
}

// Holds the selected metrics + a per-key columnar cache, so toggling chart options
// never refetches. `ensure` fetches only the not-yet-cached selected keys.
export function useMetrics() {
  const selected = ref<string[]>([])
  const data = ref<Record<string, Series>>({})

  function reset(defaults: string[]) {
    selected.value = [...defaults]
    data.value = {}
  }

  async function ensure(r: RunRef) {
    const need = selected.value.filter((k) => !(k in data.value))
    if (!need.length) return
    const res = await api<{ metrics: MetricSeries }>(
      `api/metrics?${qs({ entity: r.entity, project: r.project, run: r.run_id, keys: need.join(',') })}`,
    )
    for (const k of need) data.value[k] = res.metrics[k] ?? { x: [], y: [] }
  }

  function add(key: string) {
    if (!selected.value.includes(key)) selected.value.push(key)
  }

  function remove(key: string) {
    selected.value = selected.value.filter((k) => k !== key)
  }

  return { selected, data, reset, ensure, add, remove }
}
