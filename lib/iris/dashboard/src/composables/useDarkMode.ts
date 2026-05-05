import { ref, watch } from 'vue'

const STORAGE_KEY = 'iris-dark-mode'

function getInitial(): boolean {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored !== null) return stored === 'true'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  } catch {
    return false
  }
}

const isDark = ref(getInitial())

function apply(dark: boolean) {
  document.documentElement.classList.toggle('dark', dark)
  try { localStorage.setItem(STORAGE_KEY, String(dark)) } catch {}
}

// Sync on first import
apply(isDark.value)

watch(isDark, apply)

export function useDarkMode() {
  function toggle() { isDark.value = !isDark.value }
  return { isDark, toggle }
}
