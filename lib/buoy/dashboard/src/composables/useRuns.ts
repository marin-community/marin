import { ref, watch } from 'vue'
import { api, apiOr, qs } from '../api'
import type { RunRow } from '../types'

// Drives the sidebar pickers + run list. Runs are filtered server-side (user +
// name search) so any run is findable regardless of recency.
export function useRuns() {
  const entity = ref('')
  const project = ref('')
  const user = ref('')
  const search = ref('')
  const entities = ref<string[]>([])
  const projects = ref<string[]>([])
  const users = ref<string[]>([])
  const runs = ref<RunRow[]>([])
  const loading = ref(false)

  async function loadEntities() {
    entity.value = (await apiOr('api/defaults', { entity: '' })).entity
    entities.value = (await apiOr('api/entities', { entities: [] as string[] })).entities
  }

  async function loadProjects() {
    projects.value = entity.value
      ? (await apiOr(`api/projects?${qs({ entity: entity.value })}`, { projects: [] as string[] })).projects
      : []
  }

  async function loadUsers() {
    users.value =
      entity.value && project.value
        ? (await apiOr(`api/users?${qs({ entity: entity.value, project: project.value })}`, { users: [] as string[] }))
            .users
        : []
  }

  async function fetchRuns() {
    if (!entity.value || !project.value) {
      runs.value = []
      return
    }
    loading.value = true
    const params: Record<string, string | number> = { entity: entity.value, project: project.value, limit: 100 }
    if (user.value) params.user = user.value
    if (search.value) params.search = search.value
    try {
      runs.value = (await api<{ runs: RunRow[] }>(`api/runs?${qs(params)}`)).runs ?? []
    } finally {
      loading.value = false
    }
  }

  watch(entity, loadProjects)
  watch(project, () => {
    loadUsers()
    fetchRuns()
  })

  return { entity, project, user, search, entities, projects, users, runs, loading, loadEntities, fetchRuns }
}
