import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

export function useCatalogQuery() {
  const route = useRoute()
  const router = useRouter()

  function queryValue(key: string): string {
    const value = route.query[key]
    return typeof value === 'string' ? value : ''
  }

  function location(values: Record<string, string>) {
    const query = { ...route.query }
    for (const [key, value] of Object.entries(values)) {
      if (value) query[key] = value
      else delete query[key]
    }
    return { path: route.path, query }
  }

  const pushQuery = (values: Record<string, string>) => router.push(location(values))
  const replaceQuery = (values: Record<string, string>) => router.replace(location(values))
  const offset = computed(() => {
    const value = Number(queryValue('offset'))
    return Number.isSafeInteger(value) && value >= 0 ? value : 0
  })

  function changePage(value: number) {
    void pushQuery({ offset: value > 0 ? String(value) : '' })
  }

  function correctOffset(page: { total: number; limit: number } | null) {
    if (page && offset.value > 0 && offset.value >= page.total) {
      const lastOffset = Math.floor(Math.max(0, page.total - 1) / page.limit) * page.limit
      void replaceQuery({ offset: lastOffset ? String(lastOffset) : '' })
    }
  }

  return { queryValue, pushQuery, replaceQuery, offset, changePage, correctOffset }
}
