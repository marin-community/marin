import assert from 'node:assert/strict'
import test from 'node:test'

import { markdownHtml } from '../src/lib/markdown.ts'

test('renders unfenced XML as literal model output', () => {
  const output = markdownHtml(`<ticket_analysis>
  <category>billing</category>
  <summary>Customer won&apos;t accept a duplicate charge.</summary>
</ticket_analysis>`)

  assert.match(output, /^<pre><code>&lt;ticket_analysis&gt;/)
  assert.match(output, /&lt;category&gt;billing&lt;\/category&gt;/)
  assert.match(output, /Customer won&amp;apos;t accept a duplicate charge\./)
  assert.match(output, /&lt;\/ticket_analysis&gt;<\/code><\/pre>/)
  assert.doesNotMatch(output, /<ticket_analysis>/)
})
