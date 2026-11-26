import { computed, onScopeDispose, ref, watch, type Ref } from 'vue'
import { useApi } from '@/composables/useApi'
import { useRouteQuery } from '@/composables/useRouteQuery'
import type { CatalogPage } from '@/types/api'

const SEARCH_DELAY = 300

export function useCatalogQuery() {
  const { route, queryValue, pushQuery, replaceQuery } = useRouteQuery()
  const offset = computed(() => {
    const value = Number(queryValue('offset'))
    return Number.isSafeInteger(value) && value >= 0 ? value : 0
  })

  const query = computed(() => queryValue('q'))
  const draft = ref(query.value)
  let searchTimer: ReturnType<typeof setTimeout> | undefined
  const cancelSearch = () => clearTimeout(searchTimer)

  function submitSearch() {
    cancelSearch()
    const value = draft.value.trim()
    if (value !== query.value) void replaceQuery({ q: value, offset: '' })
  }

  const search = computed({
    get: () => draft.value,
    set(value: string) {
      draft.value = value
      cancelSearch()
      if (value.trim() !== query.value) searchTimer = setTimeout(submitSearch, SEARCH_DELAY)
    },
  })

  // History can change the offset or filters while leaving q unchanged.
  watch(() => route.fullPath, () => {
    cancelSearch()
    if (draft.value.trim() !== query.value) draft.value = query.value
  }, { flush: 'sync' })
  onScopeDispose(cancelSearch)

  function pushSearchQuery(values: Record<string, string>) {
    cancelSearch()
    if (values.q !== undefined) {
      draft.value = values.q
      return pushQuery(values)
    }
    const value = draft.value.trim()
    return pushQuery(value === query.value ? values : { ...values, q: value, offset: '' })
  }

  function changePage(value: number) {
    void pushSearchQuery({ offset: value > 0 ? String(value) : '' })
  }

  function updateCatalogQuery(values: Record<string, string>) {
    return pushSearchQuery({ ...values, offset: '' })
  }

  return { queryValue, offset, query, search, submitSearch, changePage, updateCatalogQuery }
}

export function useCatalogPage<Row>(path: () => string, offset: Readonly<Ref<number>>) {
  const { replaceQuery } = useRouteQuery()
  const api = useApi<CatalogPage<Row>>(path)
  const page = api.data
  watch(path, () => {
    page.value = null
    void api.refresh()
  }, { immediate: true, flush: 'sync' })
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
  return { ...api, limit, hasNext, showPagination }
}
