import DOMPurify from 'dompurify'
import { marked, Renderer } from 'marked'

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

const renderer = new Renderer()
renderer.html = ({ text, block }) => {
  const escaped = escapeHtml(text)
  return block ? `<pre><code>${escaped}</code></pre>\n` : escaped
}

marked.setOptions({ gfm: true, breaks: true, renderer })

/** Convert model Markdown to HTML while displaying raw HTML and XML literally. */
export function markdownHtml(text: string): string {
  return marked.parse(text, { async: false })
}

/** Render model output as sanitized HTML — model text is untrusted, and the
 * dashboard shares its origin with the Iris controller proxy. */
export function renderMarkdown(text: string): string {
  return DOMPurify.sanitize(markdownHtml(text))
}
