// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

export interface ActivityHit {
  type: 'activity'
  id: number
  source: string
  kind: string
  date: string | null
  author: string | null
  title: string | null
  url: string
  snippet: string
  score: number
}

export interface WikiHit {
  type: 'wiki'
  id: number
  created_at: string
  updated_at: string
  author: string
  title: string
  use_when: string
  tags: string[]
  snippet: string
  reference_count: number
  score: number
  body?: string
}

export type SearchDomain = string

export interface SearchDomainOption {
  value: SearchDomain
  label: string
}

export interface SearchConfiguration {
  domains: SearchDomainOption[]
  default_domains: SearchDomain[]
  display_sha_characters: number
}

export interface FederatedResult {
  id: string
  domain: SearchDomain
  title: string
  subtitle: string
  url: string
  snippet: string
  score: number
  distance: number | null
  lexical_score: number | null
  references: SearchReference[]
}

export interface SearchReference {
  line: number
  text: string
  url: string
}

export interface RepositoryIndexStatus {
  repository: string
  branch: string
  status: 'empty' | 'building' | 'ready'
  commit_sha: string | null
  completed_files: number | null
  total_files: number | null
  started_at: string | null
  indexed_at: string | null
}

// The full chunk behind an activity hit: GET /api/chunks/{id} adds the untruncated text.
export interface ActivityDetail extends ActivityHit {
  text: string | null
  ref: string | null
  parent: string | null
}

export async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export function formatDate(value: string | null): string {
  if (!value) return 'Date unknown'
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(value))
}
