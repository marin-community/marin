// Touch keyboards have no Shift+Enter, so on touch-first devices Enter inserts
// a newline. Cmd/Ctrl+Enter still sends, for touch devices with a hardware keyboard.
const TOUCH_INPUT = typeof window !== 'undefined' && window.matchMedia('(pointer: coarse)').matches
// A touch device that also reports a mouse or trackpad (e.g. an iPad with a
// keyboard case) likely has a hardware keyboard, so it gets the shortcut hint.
const TOUCH_WITH_POINTER = TOUCH_INPUT && window.matchMedia('(any-pointer: fine)').matches

function composerHint(): string {
  if (!TOUCH_INPUT) return ' (Enter to send, Shift+Enter for a newline)'
  if (TOUCH_WITH_POINTER) return ' (Cmd/Ctrl+Enter to send)'
  return ''
}

export const COMPOSER_HINT = composerHint()

export function isSendKey(event: KeyboardEvent): boolean {
  if (event.isComposing || event.key !== 'Enter') return false
  if (event.metaKey || event.ctrlKey) return true
  return !TOUCH_INPUT && !event.shiftKey
}
