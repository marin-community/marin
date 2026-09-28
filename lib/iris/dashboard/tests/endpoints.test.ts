import assert from 'node:assert/strict'
import test from 'node:test'

import { filterEndpoints, groupEndpointsByOwner, sortEndpointsByName } from '../src/utils/endpoints.ts'

const endpoints = [
  {
    endpointId: 'zeta',
    name: '/serve/zeta-10',
    address: 'zeta.internal:80',
    taskId: '/zoe/training/0',
    metadata: { chart: 'inference' },
  },
  {
    endpointId: 'alpha',
    name: '/serve/alpha',
    address: 'alpha.internal:80',
    taskId: '/alice/helm-deploy/0',
    metadata: { chart: 'frontend' },
  },
  {
    endpointId: 'zeta-2',
    name: '/serve/zeta-2',
    address: 'zeta-2.internal:80',
    taskId: '/zoe/training/1',
  },
  {
    endpointId: 'evaluation',
    name: '/serve/evaluation',
    address: 'evaluation.internal:80',
    taskId: '/zoe/evaluation/0',
  },
  {
    endpointId: 'system',
    name: '/system/log-server',
    address: 'logs.internal:80',
  },
]

test('endpoint search matches names, addresses, jobs, and metadata', () => {
  assert.deepEqual(filterEndpoints(endpoints, 'ALPHA').map(endpoint => endpoint.endpointId), ['alpha'])
  assert.deepEqual(filterEndpoints(endpoints, 'logs.internal').map(endpoint => endpoint.endpointId), ['system'])
  assert.deepEqual(filterEndpoints(endpoints, 'helm').map(endpoint => endpoint.endpointId), ['alpha'])
  assert.deepEqual(filterEndpoints(endpoints, 'chart=frontend').map(endpoint => endpoint.endpointId), ['alpha'])
})

test('endpoint groups sort users, jobs, and endpoint names', () => {
  const groups = groupEndpointsByOwner(endpoints)

  assert.deepEqual(groups.map(group => group.user), [null, 'alice', 'zoe'])
  assert.deepEqual(groups[0]?.jobs[0]?.endpoints.map(endpoint => endpoint.endpointId), ['system'])
  assert.deepEqual(groups[1]?.jobs.map(job => job.jobId), ['/alice/helm-deploy'])
  assert.deepEqual(groups[2]?.jobs.map(job => job.jobId), ['/zoe/evaluation', '/zoe/training'])
  assert.deepEqual(groups[2]?.jobs[1]?.endpoints.map(endpoint => endpoint.endpointId), ['zeta-2', 'zeta'])
})

test('endpoint sorting uses natural name order', () => {
  assert.deepEqual(
    sortEndpointsByName(endpoints).map(endpoint => endpoint.endpointId),
    ['alpha', 'evaluation', 'zeta-2', 'zeta', 'system'],
  )
})
