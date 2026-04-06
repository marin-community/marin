async function fetchJSON(url) {
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`${url}: ${resp.status} ${resp.statusText}`)
  return resp.json()
}

export function fetchOverview() {
  return fetchJSON('/api/overview')
}

export function fetchSavings() {
  return fetchJSON('/api/savings')
}

export function fetchRules() {
  return fetchJSON('/api/rules')
}

export function fetchSimulate(excludeIds) {
  const params = excludeIds.map(id => `exclude=${id}`).join('&')
  return fetchJSON(`/api/rules/simulate?${params}`)
}

export function fetchExplore(bucket, prefix = '') {
  const params = new URLSearchParams({ bucket, prefix })
  return fetchJSON(`/api/explore?${params}`)
}

export async function fetchDeleteEstimate(patterns) {
  const resp = await fetch('/api/delete-patterns/estimate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patterns }),
  })
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json()
}

// Delete rules CRUD
export async function fetchDeleteRules() {
  return fetchJSON('/api/delete-rules')
}

export async function createDeleteRule(rule) {
  const resp = await fetch('/api/delete-rules', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rule),
  })
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json()
}

export async function removeDeleteRule(id) {
  const resp = await fetch(`/api/delete-rules/${id}`, { method: 'DELETE' })
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json()
}

export async function recalculateDeleteRules() {
  const resp = await fetch('/api/delete-rules/recalculate', { method: 'POST' })
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json()
}

// Protect rule mutations
export async function createProtectRule(rule) {
  const resp = await fetch('/api/rules', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rule),
  })
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json()
}

export async function removeProtectRule(id) {
  const resp = await fetch(`/api/rules/${id}`, { method: 'DELETE' })
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json()
}

export async function recalculateProtectRules() {
  const resp = await fetch('/api/rules/recalculate', { method: 'POST' })
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json()
}

// Unified explorer
export async function fetchUnifiedExplore(prefix = '') {
  const params = new URLSearchParams()
  if (prefix) params.set('prefix', prefix)
  return fetchJSON(`/api/explore/unified?${params}`)
}

// Sync
export async function triggerSync() {
  const resp = await fetch('/api/sync', { method: 'POST' })
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json()
}

export async function fetchSyncStatus() {
  return fetchJSON('/api/sync/status')
}
