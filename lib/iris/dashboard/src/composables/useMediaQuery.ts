import { ref, onMounted, onUnmounted } from 'vue'

/**
 * Reactive boolean tied to a CSS media query. The returned ref updates as the
 * viewport crosses the query threshold (e.g. on resize or device rotation).
 *
 * SSR-safe: defaults to false on the server and during the very first render,
 * then syncs from `window.matchMedia` on mount.
 */
export function useMediaQuery(query: string) {
  const matches = ref(false)
  let mql: MediaQueryList | null = null
  const listener = (e: MediaQueryListEvent) => { matches.value = e.matches }

  onMounted(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return
    mql = window.matchMedia(query)
    matches.value = mql.matches
    mql.addEventListener('change', listener)
  })

  onUnmounted(() => {
    if (mql) mql.removeEventListener('change', listener)
  })

  return matches
}
