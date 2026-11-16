import { inject, onUnmounted, provide, ref, watch, type InjectionKey, type Ref } from 'vue'

export type AgentContextValue =
  | null
  | boolean
  | number
  | string
  | AgentContextValue[]
  | { [key: string]: AgentContextValue }

export interface AppAgentContext {
  version: 1
  contextKey: string
  label: string
  state: Record<string, AgentContextValue>
}

const agentContextKey: InjectionKey<Ref<AppAgentContext | null>> = Symbol('marina-agent-context')

/** Create the page-context slot owned by the shared Marina shell. */
export function provideAgentContext(): Ref<AppAgentContext | null> {
  const context = ref<AppAgentContext | null>(null)
  provide(agentContextKey, context)
  return context
}

/** Publish a view's current agent context until that view unmounts. */
export function useAgentContext(context: Ref<AppAgentContext | null>): void {
  const target = inject(agentContextKey)
  if (!target) throw new Error('useAgentContext must run below Marina Shell')
  watch(context, (value) => (target.value = value), { immediate: true })
  onUnmounted(() => {
    if (target.value?.contextKey === context.value?.contextKey) target.value = null
  })
}
