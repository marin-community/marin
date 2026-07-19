import DOMPurify from 'dompurify'
import { marked } from 'marked'

marked.setOptions({ gfm: true, breaks: true })

/** Render model output as sanitized HTML — model text is untrusted, and the
 * dashboard shares its origin with the Iris controller proxy. */
export function renderMarkdown(text: string): string {
  return DOMPurify.sanitize(marked.parse(text, { async: false }))
}
