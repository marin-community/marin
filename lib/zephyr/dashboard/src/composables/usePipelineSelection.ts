import { ref } from 'vue'

const QUERY_PARAMETER = 'execution_id'

export const selectedExecutionId = ref(new URL(window.location.href).searchParams.get(QUERY_PARAMETER) ?? '')

export function selectPipeline(executionId: string) {
  selectedExecutionId.value = executionId
  const url = new URL(window.location.href)
  if (executionId) url.searchParams.set(QUERY_PARAMETER, executionId)
  else url.searchParams.delete(QUERY_PARAMETER)
  window.history.replaceState(window.history.state, '', url)
}
