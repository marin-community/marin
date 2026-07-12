import { ref } from 'vue'

/** Reachability of the (preemptible) ducky query backend, shared app-wide:
 * App.vue polls it periodically for the warning banner, and useQuery re-checks
 * on any query failure. */
export const duckyAvailable = ref(true)

export async function checkDuckyStatus(): Promise<void> {
  try {
    const resp = await fetch('api/ducky-status')
    const data = await resp.json()
    duckyAvailable.value = !!data.available
  } catch (e) {
    /* status check itself failed; leave the banner as-is */
  }
}
