import { useRoute, useRouter } from 'vue-router'

export function useRouteQuery() {
  const route = useRoute()
  const router = useRouter()

  function queryValue(key: string): string {
    const value = route.query[key]
    return typeof value === 'string' ? value : ''
  }

  function location(values: Record<string, string>) {
    const query = { ...route.query }
    for (const [key, value] of Object.entries(values)) {
      if (value) query[key] = value
      else delete query[key]
    }
    return { path: route.path, query }
  }

  const pushQuery = (values: Record<string, string>) => router.push(location(values))
  const replaceQuery = (values: Record<string, string>) => router.replace(location(values))
  return { route, location, queryValue, pushQuery, replaceQuery }
}
