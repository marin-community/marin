import type { JobStatus } from '@/types/rpc'

export interface JobTreeNode {
  job: JobStatus
  depth: number
}

export function getParentJobName(jobName: string): string | null {
  if (!jobName) return null
  const lastSlash = jobName.lastIndexOf('/')
  if (lastSlash <= 0) return null
  return jobName.slice(0, lastSlash)
}

export function getLeafJobName(jobName: string): string {
  if (!jobName) return jobName
  const lastSlash = jobName.lastIndexOf('/')
  return lastSlash >= 0 ? jobName.slice(lastSlash + 1) : jobName
}

export function flattenJobTree(
  jobList: JobStatus[],
  expandedJobNames: ReadonlySet<string>,
  comparator?: (a: JobStatus, b: JobStatus) => number,
): JobTreeNode[] {
  const jobByName = new Map(jobList.map(job => [job.name, job]))
  const childrenMap = new Map<string, JobStatus[]>()
  const rootJobs: JobStatus[] = []

  for (const job of jobList) {
    const parentName = getParentJobName(job.name)
    if (parentName && jobByName.has(parentName)) {
      const children = childrenMap.get(parentName)
      if (children) {
        children.push(job)
      } else {
        childrenMap.set(parentName, [job])
      }
      continue
    }
    rootJobs.push(job)
  }

  const result: JobTreeNode[] = []

  function walk(list: JobStatus[], depth: number) {
    const sorted = comparator ? [...list].sort(comparator) : list
    for (const job of sorted) {
      result.push({ job, depth })
      const children = childrenMap.get(job.name)
      if (children && expandedJobNames.has(job.name)) {
        walk(children, depth + 1)
      }
    }
  }

  walk(rootJobs, 0)
  return result
}

export function jobsWithChildren(jobList: JobStatus[]): Set<string> {
  const parents = new Set<string>()
  for (const job of jobList) {
    if (job.hasChildren) {
      parents.add(job.name)
    }
  }
  return parents
}

export function flattenLoadedJobTree(
  rootJobs: JobStatus[],
  childJobsByParent: ReadonlyMap<string, JobStatus[]>,
  expandedJobIds: ReadonlySet<string>,
): JobTreeNode[] {
  const result: JobTreeNode[] = []

  function walk(jobs: JobStatus[], depth: number) {
    for (const job of jobs) {
      result.push({ job, depth })
      if (expandedJobIds.has(job.jobId)) {
        const children = childJobsByParent.get(job.jobId) ?? []
        walk(children, depth + 1)
      }
    }
  }

  walk(rootJobs, 0)
  return result
}
