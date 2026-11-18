// Touch keyboards have no Shift+Enter, so on touch-first devices Enter inserts
// a newline. Cmd/Ctrl+Enter still sends, for touch devices with a hardware keyboard.
const TOUCH_INPUT = typeof window !== 'undefined' && window.matchMedia('(pointer: coarse)').matches

export const COMPOSER_HINT = TOUCH_INPUT ? '' : ' (Enter to send, Shift+Enter for a newline)'

export function isSendKey(event: KeyboardEvent): boolean {
  if (event.isComposing || event.key !== 'Enter') return false
  if (event.metaKey || event.ctrlKey) return true
  return !TOUCH_INPUT && !event.shiftKey
}
