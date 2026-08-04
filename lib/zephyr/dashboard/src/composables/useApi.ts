import { ref, type Ref } from 'vue'

export type QueryParams = Record<string, string | number | boolean>
export type QuerySource = QueryParams | (() => QueryParams)

export interface ApiState<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  refresh: () => Promise<void>
}

function url(path: string, query: QuerySource | undefined): string {
  const resolved = typeof query === 'function' ? query() : (query ?? {})
  const params = new URLSearchParams()
  for (const [name, value] of Object.entries(resolved)) params.set(name, String(value))
  const search = params.toString()
  return search ? `api/${path}?${search}` : `api/${path}`
}

async function errorMessage(path: string, response: Response): Promise<string> {
  const text = (await response.text().catch(() => '')).trim()
  if (text) return `${path}: ${response.status} ${text}`
  return `${path}: ${response.status} ${response.statusText}`
}

export function useDashboardApi<T>(path: string, query?: QuerySource): ApiState<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const loading = ref(false)
  const error = ref<string | null>(null)
  let generation = 0

  async function refresh() {
    const currentGeneration = ++generation
    loading.value = true
    error.value = null
    try {
      const response = await fetch(url(path, query), { headers: { Accept: 'application/json' } })
      if (currentGeneration !== generation) return
      if (!response.ok) throw new Error(await errorMessage(path, response))
      data.value = (await response.json()) as T
    } catch (caught) {
      if (currentGeneration !== generation) return
      error.value = caught instanceof Error ? caught.message : String(caught)
    } finally {
      if (currentGeneration === generation) loading.value = false
    }
  }

  return { data, loading, error, refresh }
}
