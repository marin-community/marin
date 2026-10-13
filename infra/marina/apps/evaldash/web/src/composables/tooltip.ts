/**
 * A single shared cursor-following tooltip. Marks bind it by spreading the handlers
 * from `tip(content)` onto an element (`v-on="tip({ title, lines })"`); `TooltipHost`,
 * mounted once at the app root, renders the active content. Content is structured
 * (title + lines) rather than raw HTML so values are auto-escaped by Vue.
 */
import { reactive } from 'vue'

export interface TipLine {
  label: string
  value: string
  tone?: 'default' | 'muted' | 'best'
}

export interface TipContent {
  title: string
  lines?: TipLine[]
}

interface TipState {
  show: boolean
  content: TipContent | null
  x: number
  y: number
}

export const tipState = reactive<TipState>({ show: false, content: null, x: 0, y: 0 })

function track(e: MouseEvent) {
  tipState.x = e.clientX
  tipState.y = e.clientY
}

export function tip(content: TipContent) {
  return {
    mouseenter: (e: MouseEvent) => {
      tipState.content = content
      tipState.show = true
      track(e)
    },
    mousemove: track,
    mouseleave: () => {
      tipState.show = false
    },
  }
}
