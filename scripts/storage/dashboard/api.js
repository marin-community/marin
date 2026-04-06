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
