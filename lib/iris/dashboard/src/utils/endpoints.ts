import type { EndpointInfo } from '@/types/rpc'

// Controller endpoint-proxy helpers.
//
// A registered endpoint named `/tunix/inference/server` is reachable through
// the controller's reverse proxy at `/proxy/tunix.inference.server`: the proxy
// decodes the path component by replacing '.' with '/' and resolving both the
// slash-prefixed and bare forms. The encoding has no escape for a literal '.',
// so a name containing one cannot be turned into a working proxy link.

/** Whether `name` can be encoded into a working proxy path (no literal '.'). */
export function canProxyEndpoint(name: string): boolean {
  return name.length > 0 && !name.includes('.')
}

/**
 * Proxy path for an endpoint name, e.g. `/tunix/inference/server` ->
 * `/proxy/tunix.inference.server`. The leading slash is dropped and remaining
 * slashes become dots to match the proxy's decoding. Gate on
 * {@link canProxyEndpoint} before rendering the result as a link.
 */
export function proxyPathForEndpoint(name: string): string {
  const trimmed = name.replace(/^\/+/, '')
  return `/proxy/${trimmed.split('/').join('.')}`
}

/** Short label for an endpoint: its last path segment (or the whole name). */
export function endpointLabel(name: string): string {
  const trimmed = name.replace(/\/+$/, '')
  const slash = trimmed.lastIndexOf('/')
  return slash >= 0 ? trimmed.slice(slash + 1) : trimmed
}

export interface EndpointJobGroup {
  jobId: string | null
  endpoints: EndpointInfo[]
}

export interface EndpointUserGroup {
  user: string | null
  endpointCount: number
  jobs: EndpointJobGroup[]
}

const NAME_COLLATOR = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' })

export function jobIdFromTaskId(taskId?: string): string | null {
  if (!taskId) return null
  const slash = taskId.lastIndexOf('/')
  return slash > 0 ? taskId.slice(0, slash) : taskId
}

function userFromJobId(jobId: string | null): string | null {
  if (!jobId?.startsWith('/')) return null
  return jobId.split('/')[1] || null
}

function endpointSearchText(endpoint: EndpointInfo): string {
  const metadata = Object.entries(endpoint.metadata ?? {}).map(([key, value]) => `${key}=${value}`)
  return [endpoint.name, endpoint.address, endpoint.taskId ?? '', ...metadata].join('\n').toLocaleLowerCase()
}

/** Match endpoints by name, address, task, or metadata. */
export function filterEndpoints(endpoints: EndpointInfo[], query: string): EndpointInfo[] {
  const normalizedQuery = query.trim().toLocaleLowerCase()
  if (!normalizedQuery) return endpoints
  return endpoints.filter(endpoint => endpointSearchText(endpoint).includes(normalizedQuery))
}

function compareEndpointNames(left: EndpointInfo, right: EndpointInfo): number {
  return NAME_COLLATOR.compare(left.name, right.name)
    || NAME_COLLATOR.compare(left.address, right.address)
    || NAME_COLLATOR.compare(left.endpointId ?? '', right.endpointId ?? '')
}

export function sortEndpointsByName(endpoints: EndpointInfo[]): EndpointInfo[] {
  return [...endpoints].sort(compareEndpointNames)
}

/** Group endpoint rows by their task owner's user and job, sorting every level by name. */
export function groupEndpointsByOwner(endpoints: EndpointInfo[]): EndpointUserGroup[] {
  const grouped = new Map<string | null, Map<string | null, EndpointInfo[]>>()

  for (const endpoint of endpoints) {
    const jobId = jobIdFromTaskId(endpoint.taskId)
    const user = userFromJobId(jobId)
    let jobs = grouped.get(user)
    if (!jobs) {
      jobs = new Map()
      grouped.set(user, jobs)
    }
    const jobEndpoints = jobs.get(jobId)
    if (jobEndpoints) {
      jobEndpoints.push(endpoint)
    } else {
      jobs.set(jobId, [endpoint])
    }
  }

  return [...grouped.entries()]
    .sort(([left], [right]) => {
      if (left === null) return right === null ? 0 : -1
      if (right === null) return 1
      return NAME_COLLATOR.compare(left, right)
    })
    .map(([user, jobs]) => {
      const jobGroups = [...jobs.entries()]
        .sort(([left], [right]) => {
          if (left === null) return right === null ? 0 : -1
          if (right === null) return 1
          return NAME_COLLATOR.compare(left, right)
        })
        .map(([jobId, jobEndpoints]) => ({
          jobId,
          endpoints: sortEndpointsByName(jobEndpoints),
        }))
      return {
        user,
        endpointCount: jobGroups.reduce((total, job) => total + job.endpoints.length, 0),
        jobs: jobGroups,
      }
    })
}
