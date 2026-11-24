import { computed, watch, type Ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

export const CATALOG_SEARCH_DELAY = 300

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

  function pagination(page: Ref<{ total: number; limit: number } | null>) {
    const limit = computed(() => page.value?.limit ?? 0)
    const hasNext = computed(() => !!page.value && offset.value + limit.value < page.value.total)
    const showPagination = computed((previous: boolean | undefined) =>
      offset.value > 0 || (page.value ? hasNext.value : previous ?? false),
    )
    watch(page, (result) => {
      if (result && offset.value > 0 && offset.value >= result.total) {
        const lastOffset = Math.floor(Math.max(0, result.total - 1) / result.limit) * result.limit
        void replaceQuery({ offset: lastOffset ? String(lastOffset) : '' })
      }
    })
    return { limit, hasNext, showPagination }
  }

  return { queryValue, location, pushQuery, replaceQuery, offset, changePage, pagination }
}
