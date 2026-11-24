import { computed, watch, type ComputedRef } from 'vue'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { COMPARISON_MODELS_ENDPOINT, MAX_COMPARE } from '@/constants'
import type { Comparison, Meta } from '@/types/api'

type ComparisonModels = Pick<Meta, 'models' | 'default_cohort' | 'verified_cohorts' | 'versions'>

export function useComparisonData(selection: ComputedRef<string>, requested: ComputedRef<string[]>) {
  const choices = useApi<ComparisonModels>(() => `${COMPARISON_MODELS_ENDPOINT}?${selection.value}`)
  const available = computed(() => choices.data.value?.models ?? [])
  const eligible = computed(() => new Set(available.value))
  const selected = computed(() => requested.value.filter((model) => eligible.value.has(model)).slice(0, MAX_COMPARE))
  const remaining = computed(() => available.value.filter((model) => !selected.value.includes(model)))
  const excluded = computed(() => requested.value.filter((model) => !eligible.value.has(model)))
  const comparing = computed(() => selected.value.length >= 2)

  const path = computed(() => comparing.value
    ? `api/compare?${selection.value}&models=${encodeURIComponent(selected.value.join(','))}`
    : '',
  )
  const results = useApi<Comparison>(() => path.value)

  watch(selection, () => {
    choices.data.value = null
    void choices.refresh()
  }, { immediate: true })
  watch(path, (request) => {
    results.data.value = null
    if (request) void results.refresh()
  })
  onViewRefresh(() => {
    void choices.refresh()
    if (comparing.value) void results.refresh()
  })

  return { choices, results, selected, remaining, excluded, comparing }
}
