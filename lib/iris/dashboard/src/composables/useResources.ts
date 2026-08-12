/** Generic ResourceService transport helpers.
 *
 * Resource-specific components still own their typed bodies and queries. This
 * module only builds the common ResourceService envelope and unwraps it again.
 */
import { computed } from 'vue'
import {
  resourceRpcCall,
  useResourceRpc,
  type RpcBody,
  type RpcState,
} from '@/composables/useRpc'
import type { ResourcePageInfo, ResourceSourceStatus } from '@/types/rpc'

export const RESOURCE_TYPES = {
  job: 'iris/job',
  task: 'iris/task',
  attempt: 'iris/attempt',
  node: 'iris/node',
  endpoint: 'iris/endpoint',
  capacity: 'iris/capacity',
  userSummary: 'iris/user-summary',
  activityEntry: 'iris/activity-entry',
  profileCapture: 'iris/profile-capture',
} as const

export const RESOURCE_MESSAGES = {
  jobQuery: 'iris.resource.JobQuery',
  taskQuery: 'iris.resource.TaskQuery',
  nodeQuery: 'iris.resource.NodeQuery',
  endpointQuery: 'iris.resource.EndpointQuery',
  listUsersRequest: 'iris.resource.ListUsersRequest',
  activityQuery: 'iris.resource.ActivityQuery',
  jobUpdate: 'iris.resource.JobUpdate',
  taskUpdate: 'iris.resource.TaskUpdate',
  attemptUpdate: 'iris.resource.AttemptUpdate',
  profileAttemptRequest: 'iris.resource.ProfileAttemptRequest',
} as const

export interface ResourceRef {
  authorityClusterId: string
  type: string
  id: string
  uid?: string
}

interface ResourceEnvelope<T> {
  ref: ResourceRef
  body?: T & { '@type'?: string }
  etag?: string
}

interface GetResourceResponse<T> {
  resource?: ResourceEnvelope<T>
  sourceStatuses?: ResourceSourceStatus[]
}

interface ListResourcesResponse<T> {
  resources?: Array<ResourceEnvelope<T>>
  nextPageToken?: string
  sourceStatuses?: ResourceSourceStatus[]
}

export interface ResourcePage<T> {
  items: T[]
  page: ResourcePageInfo
}

export interface ResourceOperation<T> {
  ref: ResourceRef
  phase: string
  verb: string
  requestedRef?: ResourceRef
  resolvedRef?: ResourceRef
  affected?: ResourceRef[]
  result?: T & { '@type'?: string }
}

export interface MutationOptions {
  requestId?: string
  reason?: string
}

function anyMessage(typeName: string, value: Record<string, unknown> = {}): Record<string, unknown> {
  return { '@type': `type.googleapis.com/${typeName}`, ...value }
}

function bodyValue<T>(resource?: ResourceEnvelope<T>): T | null {
  return resource?.body as T | undefined ?? null
}

function mapRpcState<Input, Output>(state: RpcState<Input>, map: (value: Input) => Output): RpcState<Output> {
  return {
    data: computed(() => state.data.value === null ? null : map(state.data.value)),
    loading: state.loading,
    error: state.error,
    refresh: state.refresh,
  }
}

export function useGetResource<T>(ref: ResourceRef | (() => ResourceRef), view: 'BASIC' | 'FULL'): RpcState<T> {
  const body: RpcBody = () => ({
    ref: typeof ref === 'function' ? ref() : ref,
    view: `RESOURCE_VIEW_${view}`,
  })
  const state = useResourceRpc<GetResourceResponse<T>>('GetResource', body)
  return mapRpcState(state, response => bodyValue(response.resource) as T)
}

export async function getResource<T>(ref: ResourceRef, view: 'BASIC' | 'FULL'): Promise<T> {
  const response = await resourceRpcCall<GetResourceResponse<T>>('GetResource', {
    ref,
    view: `RESOURCE_VIEW_${view}`,
  })
  const body = bodyValue(response.resource)
  if (body === null) throw new Error(`GetResource returned no body for ${ref.type} ${ref.id}`)
  return body
}

export function useListResources<T>(
  type: string,
  queryType: string,
  query: Record<string, unknown> | (() => Record<string, unknown>),
  view: 'BASIC' | 'FULL',
): RpcState<ResourcePage<T>> {
  const body: RpcBody = () => ({
    type,
    query: anyMessage(queryType, typeof query === 'function' ? query() : query),
    view: `RESOURCE_VIEW_${view}`,
  })
  const state = useResourceRpc<ListResourcesResponse<T>>('ListResources', body)
  return mapRpcState(state, response => ({
    items: (response.resources ?? []).map(resource => bodyValue(resource) as T),
    page: {
      nextPageToken: response.nextPageToken,
      sourceStatuses: response.sourceStatuses,
    },
  }))
}

export async function listResources<T>(
  type: string,
  queryType: string,
  query: Record<string, unknown>,
  view: 'BASIC' | 'FULL',
): Promise<ResourcePage<T>> {
  const response = await resourceRpcCall<ListResourcesResponse<T>>('ListResources', {
    type,
    query: anyMessage(queryType, query),
    view: `RESOURCE_VIEW_${view}`,
  })
  return {
    items: (response.resources ?? []).map(resource => bodyValue(resource) as T),
    page: {
      nextPageToken: response.nextPageToken,
      sourceStatuses: response.sourceStatuses,
    },
  }
}

export async function updateResource<T>(
  ref: ResourceRef,
  updateType: string,
  update: Record<string, unknown>,
  options: MutationOptions = {},
): Promise<ResourceOperation<T>> {
  return resourceRpcCall<ResourceOperation<T>>('UpdateResource', {
    mutation: {
      requestId: options.requestId ?? crypto.randomUUID(),
      reason: options.reason,
    },
    ref,
    update: anyMessage(updateType, update),
  })
}

export async function createResource<T>(
  type: string,
  parent: ResourceRef,
  bodyType: string,
  body: Record<string, unknown>,
  options: MutationOptions = {},
): Promise<ResourceOperation<T>> {
  return resourceRpcCall<ResourceOperation<T>>('CreateResource', {
    mutation: {
      requestId: options.requestId ?? crypto.randomUUID(),
      reason: options.reason,
    },
    type,
    parent,
    body: anyMessage(bodyType, body),
  })
}
