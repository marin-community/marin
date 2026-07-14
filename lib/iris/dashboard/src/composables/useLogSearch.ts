/**
 * Query state for find-in-loaded-lines: the text, its modifiers, and the matcher
 * compiled from them.
 *
 * The query is debounced because every keystroke re-runs the matcher over every
 * loaded line. `applied` is the settled query — components key their match lists
 * off it, not off `query`.
 */
import { computed, type ComputedRef, onUnmounted, type Ref, ref, watch } from 'vue'
import { compileSearch, type SearchMatcher } from '@/utils/logSearch'

export interface LogSearch {
  query: Ref<string>
  caseSensitive: Ref<boolean>
  useRegex: Ref<boolean>
  /** The debounced query the matcher was built from. */
  applied: Ref<string>
  /** Null when the query is empty or does not compile. */
  matcher: ComputedRef<SearchMatcher | null>
  /** The regex compile error, when the query does not parse. */
  error: ComputedRef<string | null>
}

export function useLogSearch(debounceMs: number): LogSearch {
  const query = ref('')
  const caseSensitive = ref(false)
  const useRegex = ref(false)
  const applied = ref('')

  let timer: number | undefined
  watch(query, (value) => {
    window.clearTimeout(timer)
    timer = window.setTimeout(() => { applied.value = value }, debounceMs)
  })
  onUnmounted(() => window.clearTimeout(timer))

  const compiled = computed(() =>
    compileSearch(applied.value, { caseSensitive: caseSensitive.value, useRegex: useRegex.value }),
  )
  const matcher = computed(() => (compiled.value.kind === 'ok' ? compiled.value.matcher : null))
  const error = computed(() => (compiled.value.kind === 'invalid' ? compiled.value.message : null))

  return { query, caseSensitive, useRegex, applied, matcher, error }
}
