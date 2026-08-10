/**
 * A wrapping cursor over a list of row indices.
 *
 * Backs the "next/previous match" and "next/previous exception" controls, which
 * differ only in which rows they step through. `position` is an offset into
 * `indices`, not a row index; -1 means nothing is selected yet, so the first
 * forward step lands on the first entry and the first backward step on the last.
 */
import { computed, type ComputedRef, type Ref, ref } from 'vue'

export interface IndexCursor {
  /** Offset into `indices`; -1 when nothing is selected. */
  position: Ref<number>
  /** The row index the cursor currently points at, or -1. */
  target: ComputedRef<number>
  /** Step `delta` entries, wrapping. Returns the row index, or null if empty. */
  step: (delta: number) => number | null
  reset: () => void
}

export function useIndexCursor(indices: Ref<number[]>): IndexCursor {
  const position = ref(-1)

  // The list shrinks as the query changes, so a stale position reads as "none".
  const target = computed(() => indices.value[position.value] ?? -1)

  function step(delta: number): number | null {
    const list = indices.value
    if (list.length === 0) return null
    position.value = position.value < 0
      ? (delta > 0 ? 0 : list.length - 1)
      : (position.value + delta + list.length) % list.length
    return list[position.value]
  }

  function reset() {
    position.value = -1
  }

  return { position, target, step, reset }
}
