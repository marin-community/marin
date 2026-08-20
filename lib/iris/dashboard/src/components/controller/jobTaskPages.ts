import type { ResourceListTasksResponse, ResourceTaskSummary } from '@/types/rpc'

export type FetchJobTaskPage = (pageToken?: string) => Promise<ResourceListTasksResponse>

/** Load the bounded Task collection declared by a Job summary. */
export async function loadJobTasks(
  expectedTaskCount: number,
  fetchPage: FetchJobTaskPage,
): Promise<ResourceTaskSummary[]> {
  if (expectedTaskCount <= 0) return []

  const tasks: ResourceTaskSummary[] = []
  const seenPageTokens = new Set<string>()
  let pageToken: string | undefined

  while (tasks.length < expectedTaskCount) {
    const response = await fetchPage(pageToken)
    const pageTasks = response.tasks ?? []
    tasks.push(...pageTasks)

    const nextPageToken = response.page?.nextPageToken
    if (pageTasks.length === 0 || !nextPageToken || seenPageTokens.has(nextPageToken)) break
    seenPageTokens.add(nextPageToken)
    pageToken = nextPageToken
  }

  return tasks
}
