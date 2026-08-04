import type { DashboardDefinition, DashboardListResponse, SavedDashboard } from '@/types/dashboard'

async function checkedDashboardResponse(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as {
      error?: { message?: unknown }
    } | null
    const message = payload?.error?.message
    throw new Error(typeof message === 'string' ? message : `${response.status} ${response.statusText}`)
  }
  return response
}

async function dashboardRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await checkedDashboardResponse(path, init)
  return response.json() as Promise<T>
}

export async function listSavedDashboards(): Promise<SavedDashboard[]> {
  return (await dashboardRequest<DashboardListResponse>('v1/dashboards')).dashboards
}

export function saveDashboard(definition: DashboardDefinition): Promise<SavedDashboard> {
  return dashboardRequest<SavedDashboard>(`v1/dashboards/${encodeURIComponent(definition.id)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(definition),
  })
}

export async function deleteSavedDashboard(dashboardId: string): Promise<void> {
  await checkedDashboardResponse(`v1/dashboards/${encodeURIComponent(dashboardId)}`, {
    method: 'DELETE',
  })
}
