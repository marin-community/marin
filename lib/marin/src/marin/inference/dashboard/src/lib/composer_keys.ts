// Touch keyboards have no Shift+Enter, so on touch-first devices Enter inserts
// a newline and the send button submits.
const TOUCH_INPUT = typeof window !== 'undefined' && window.matchMedia('(pointer: coarse)').matches

export const COMPOSER_HINT = TOUCH_INPUT ? '' : ' (Enter to send, Shift+Enter for a newline)'

export function isSendKey(event: KeyboardEvent): boolean {
  return !TOUCH_INPUT && !event.isComposing && event.key === 'Enter' && !event.shiftKey
}
