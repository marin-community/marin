// Calling another app's API on this origin.
//
// Every app is served under `/{app}/` and answers its API under `/api/{app}/`,
// so a call is a path and a JSON body. `marina.echo.search.query({ q })` posts
// to `/api/echo/search/query`; the property names are the path segments and the
// call is the last one.
//
// A path-shaped proxy rather than a generated client: the kernel adds an app by
// mounting a prefix, and nothing here has to be regenerated when it does.

/** What a call answers with, once the response is parsed. */
export type Answer = unknown

/** Any depth of path segments, ending in a call. */
type Path = { [segment: string]: Path } & ((body?: unknown) => Promise<Answer>)

async function post(path: string[], body: unknown): Promise<Answer> {
  const response = await fetch(`/api/${path.join('/')}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
  const text = await response.text()
  if (!response.ok) throw new Error(`/api/${path.join('/')}: ${response.status} ${text}`)
  return text ? JSON.parse(text) : null
}

function at(path: string[]): Path {
  return new Proxy((() => {}) as unknown as Path, {
    get: (_, segment: string) => at([...path, segment]),
    apply: (_, __, args: unknown[]) => post(path, args[0]),
  })
}

/** The root: `marina.{app}.{...}.{op}(body)`. */
export const marina = at([])
