/**
 * Composable for profiling a single target (process or task) via the ProfileTask RPC.
 *
 * Encapsulates profile-type construction, the RPC call, and result handling
 * (file download for CPU/memory, new-window display for thread dumps).
 */
import { ref } from 'vue'

type RpcCall = (method: string, body?: Record<string, unknown>) => Promise<{ profileData?: string; error?: string }>

export type ProfilerType = 'cpu' | 'memory' | 'threads'

function buildProfileType(profilerType: ProfilerType): Record<string, unknown> {
  if (profilerType === 'cpu') return { cpu: { format: 'SPEEDSCOPE' } }
  if (profilerType === 'memory') return { memory: { format: 'FLAMEGRAPH' } }
  return { threads: {} }
}

function handleProfileResult(decoded: string, profilerType: ProfilerType, label: string) {
  if (profilerType === 'threads') {
    const w = window.open('', '_blank')
    if (w) {
      const escaped = decoded.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      w.document.open()
      w.document.write(
        `<html><head><title>Thread Dump \u2013 ${label}</title></head><body><pre>${escaped}</pre></body></html>`,
      )
      w.document.close()
    }
  } else {
    const blob = new Blob([decoded], { type: 'application/octet-stream' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const ts = new Date().toISOString().replace(/[T]/g, '_').replace(/:/g, '-').replace(/\.\d+Z$/, '')
    a.download = `${ts}_profile-${label.replace(/\//g, '_')}.out`
    a.click()
    URL.revokeObjectURL(url)
  }
}

/**
 * Returns reactive profiling state and a `profile()` function.
 *
 * @param rpcCall  One-shot RPC caller (controllerRpcCall or workerRpcCall)
 * @param target   The profiling target — a string or getter for reactive targets (e.g. task IDs)
 */
export function useProfileAction(rpcCall: RpcCall, target: string | (() => string)) {
  const profiling = ref(false)
  const resolveTarget = typeof target === 'function' ? target : () => target

  async function profile(profilerType: ProfilerType) {
    const currentTarget = resolveTarget()
    profiling.value = true
    try {
      const body = {
        target: currentTarget,
        durationSeconds: 10,
        profileType: buildProfileType(profilerType),
      }
      const resp = await rpcCall('ProfileTask', body)
      if (resp.error) {
        alert(`${profilerType.toUpperCase()} profile failed: ${resp.error}`)
        return
      }
      if (resp.profileData) {
        handleProfileResult(atob(resp.profileData), profilerType, currentTarget)
      }
    } catch (e) {
      alert(`${profilerType.toUpperCase()} profile failed: ${e instanceof Error ? e.message : e}`)
    } finally {
      profiling.value = false
    }
  }

  return { profiling, profile }
}
