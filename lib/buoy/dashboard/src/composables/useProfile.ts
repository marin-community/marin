import { ref } from 'vue'
import { api, qs } from '../api'
import type { RunRef } from './useRun'

interface ProfileStatus {
  state: string
  error?: string
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

// Drives the async xprof prepare: POST /api/profile_prepare, then poll
// /api/profile_status until the xprof process is ready (or errors).
export function useProfile() {
  const state = ref<'preparing' | 'ready' | 'error'>('preparing')
  const error = ref('')
  let seq = 0

  async function prepare(r: RunRef) {
    const mine = ++seq
    state.value = 'preparing'
    error.value = ''
    await fetch('api/profile_prepare', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ entity: r.entity, project: r.project, run_id: r.run_id }),
    })
    for (;;) {
      const s = await api<ProfileStatus>(`api/profile_status?${qs({ entity: r.entity, project: r.project, run: r.run_id })}`)
      if (mine !== seq) return
      if (s.state === 'ready') {
        state.value = 'ready'
        return
      }
      if (s.state === 'error') {
        state.value = 'error'
        error.value = s.error ?? 'profile failed'
        return
      }
      await sleep(1500)
    }
  }

  return { state, error, prepare }
}
