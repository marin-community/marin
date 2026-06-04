/**
 * Composable for profiling a single target (process or task) via the ProfileTask RPC.
 *
 * Encapsulates profile-type construction, the RPC call, and result handling:
 * CPU profiles open in the bundled speedscope viewer, memory profiles download,
 * and thread dumps render in a new window.
 */
import { ref } from 'vue'
import { openSpeedscopeWindow } from '@/utils/speedscope'

type RpcCall = (method: string, body?: Record<string, unknown>) => Promise<{ profileData?: string; error?: string }>

export type ProfilerType = 'cpu' | 'memory' | 'threads'

function buildProfileType(profilerType: ProfilerType): Record<string, unknown> {
  if (profilerType === 'cpu') return { cpu: { format: 'SPEEDSCOPE' } }
  if (profilerType === 'memory') return { memory: { format: 'RAW' } }
  return { threads: {} }
}

/** Decode a base64 string to a Uint8Array without UTF-8 corruption. */
function base64ToBytes(base64: string): Uint8Array {
  const bin = atob(base64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) {
    bytes[i] = bin.charCodeAt(i)
  }
  return bytes
}

function showThreadDump(raw: string, label: string) {
  const decoded = atob(raw)
  const w = window.open('', '_blank')
  if (!w) return
  const escaped = decoded.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  w.document.open()
  w.document.write(
    `<html><head><title>Thread Dump \u2013 ${label}</title></head><body><pre>${escaped}</pre></body></html>`,
  )
  w.document.close()
}

function downloadBytes(bytes: Uint8Array, label: string, ext: string) {
  const blob = new Blob([bytes], { type: 'application/octet-stream' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  const ts = new Date().toISOString().replace(/[T]/g, '_').replace(/:/g, '-').replace(/\.\d+Z$/, '')
  a.download = `${ts}_profile-${label.replace(/\//g, '_')}.${ext}`
  a.click()
  URL.revokeObjectURL(url)
}

function handleProfileResult(raw: string, profilerType: ProfilerType, label: string) {
  if (profilerType === 'threads') {
    showThreadDump(raw, label)
  } else {
    downloadBytes(base64ToBytes(raw), label, 'bin')
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
    // Open the viewer synchronously within the click gesture; CPU profiling
    // takes ~10s, after which a fresh window.open would be blocked as a popup.
    const pending = profilerType === 'cpu' ? openSpeedscopeWindow() : null
    try {
      const body = {
        target: currentTarget,
        durationSeconds: 10,
        profileType: buildProfileType(profilerType),
      }
      const resp = await rpcCall('ProfileTask', body)
      if (resp.error) {
        pending?.cancel()
        alert(`${profilerType.toUpperCase()} profile failed: ${resp.error}`)
        return
      }
      if (!resp.profileData) {
        pending?.cancel()
        return
      }
      if (pending) {
        pending.show(base64ToBytes(resp.profileData), currentTarget)
      } else {
        handleProfileResult(resp.profileData, profilerType, currentTarget)
      }
    } catch (e) {
      pending?.cancel()
      alert(`${profilerType.toUpperCase()} profile failed: ${e instanceof Error ? e.message : e}`)
    } finally {
      profiling.value = false
    }
  }

  return { profiling, profile }
}
