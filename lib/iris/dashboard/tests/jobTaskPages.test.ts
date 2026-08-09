import assert from 'node:assert/strict'
import test from 'node:test'

import { loadJobTasks } from '../src/components/controller/jobTaskPages.ts'
import type { ResourceListTasksResponse, ResourceTaskSummary } from '../src/types/rpc.ts'

function task(index: number): ResourceTaskSummary {
  const resourceId = `/owner/job/${index}`
  return {
    identity: {
      key: { clusterId: 'cluster-a', kind: 'RESOURCE_KIND_TASK', resourceId },
      taskUid: `task-uid-${index}`,
    },
    job: {
      key: { clusterId: 'cluster-a', kind: 'RESOURCE_KIND_JOB', resourceId: '/owner/job' },
      jobUid: 'job-uid',
    },
    taskIndex: index,
    state: 'RESOURCE_STATE_RUNNING',
    executionClusterId: 'cluster-a',
    backendId: 'local',
    failureCount: 0,
    preemptionCount: 0,
  }
}

test('job task collection includes tasks beyond the first page', async () => {
  const allTasks = Array.from({ length: 125 }, (_, index) => task(index))
  const pages = new Map<string | undefined, ResourceListTasksResponse>([
    [undefined, { tasks: allTasks.slice(0, 100), page: { nextPageToken: 'second-page' } }],
    ['second-page', { tasks: allTasks.slice(100), page: {} }],
  ])

  const tasks = await loadJobTasks(allTasks.length, async pageToken => pages.get(pageToken) ?? {})

  assert.equal(tasks.length, 125)
  assert.equal(tasks.at(100)?.identity.key.resourceId, '/owner/job/100')
  assert.equal(tasks.at(-1)?.identity.key.resourceId, '/owner/job/124')
})
