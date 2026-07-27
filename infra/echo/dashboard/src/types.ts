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
  snippet: string
  reference_count: number
  score: number
  body?: string
}

// The full chunk behind an activity hit: GET /api/chunks/{id} adds the untruncated text.
export interface ActivityDetail extends ActivityHit {
  text: string | null
  ref: string | null
  parent: string | null
}

export type Result = ActivityHit | WikiHit

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
