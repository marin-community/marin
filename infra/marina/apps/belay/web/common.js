// Shared by the Belay admin and public views: DOM and Markdown helpers, formatting, and the idea timeline.
// Nothing from the API is parsed as HTML.

const REFRESH_MS = 8000
const TERMINAL = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'unschedulable'])
const WANDB = 'https://wandb.ai/marin-community/marin_moe/runs/'
const STAGES = [
  ['pending', 'Suggestions', 'waiting for a maintainer'],
  ['approved', 'Approved', 'waiting for a session to plan'],
  ['planned', 'Planned', 'plan posted, runs submitting'],
  ['running', 'Running', 'runs on the cluster'],
  ['recorded', 'Recorded', 'result in the registry'],
]

// The last state read from the API; `now` anchors relative times.
let state = null
// The app's root (`/belay/`) whether the page was opened at `/belay/` or `/belay/admin`; also works for a static copy.
const BELAY_BASE = location.pathname.replace(/admin\/?$/, '').replace(/[^/]*$/, '')

// Each view (public, admin, denied) is a <template>; showing one clones it into #view-root, so element ids are
// unique on the page.
function showView(name) {
  const template = document.getElementById(`tpl-${name}`)
  document.getElementById('view-root').replaceChildren(template.content.cloneNode(true))
}

// ---------- tiny DOM helpers (text only; nothing from the API is parsed as HTML) ----------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag)
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue
    if (key === 'class') node.className = value
    else if (key === 'text') node.textContent = value
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value)
    else if (key === 'dataset') Object.assign(node.dataset, value)
    else node.setAttribute(key, value === true ? '' : value)
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue
    node.append(child instanceof Node ? child : document.createTextNode(String(child)))
  }
  return node
}

function link(href, text) {
  return el('a', { href, target: '_blank', rel: 'noopener' }, text)
}

function setChildren(id, ...children) {
  document.getElementById(id).replaceChildren(...children.flat().filter((c) => c !== null && c !== undefined && c !== false))
}

// ---------- a Markdown subset (paragraphs, lists, tables, `code`, **bold**, links), built as DOM nodes ----------

function inline(text) {
  const nodes = []
  const pattern = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\[[^\]]+\]\((https?:\/\/[^)\s]+)\))/g
  let last = 0
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) nodes.push(text.slice(last, match.index))
    const token = match[0]
    if (match[1]) nodes.push(el('code', {}, token.slice(1, -1)))
    else if (match[2]) nodes.push(el('strong', {}, ...inline(token.slice(2, -2))))
    else nodes.push(link(match[4], token.slice(1, token.indexOf(']'))))
    last = match.index + token.length
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

function markdown(source) {
  const lines = source.replace(/\r/g, '').split('\n')
  const blocks = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) {
      i += 1
    } else if (line.trim().startsWith('|')) {
      const rows = []
      while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(lines[i++].trim())
      const cells = (row) => row.replace(/^\||\|$/g, '').split('|').map((c) => c.trim())
      const body = rows.filter((r, n) => !(n === 1 && /^\|[\s|:-]+\|?$/.test(r)))
      const [head, ...rest] = body.map(cells)
      blocks.push(
        el('div', { class: 'scroll' }, el('table', {},
          el('thead', {}, el('tr', {}, head.map((c) => el('th', {}, ...inline(c))))),
          el('tbody', {}, rest.map((r) => el('tr', {}, r.map((c) => el('td', {}, ...inline(c)))))),
        )),
      )
    } else if (/^\s*([-*]|\d+\.)\s/.test(line)) {
      const ordered = /^\s*\d+\./.test(line)
      const items = []
      while (i < lines.length && /^\s*([-*]|\d+\.)\s/.test(lines[i])) {
        let item = lines[i++].replace(/^\s*([-*]|\d+\.)\s/, '')
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s/.test(lines[i])) item += ' ' + lines[i++].trim()
        items.push(el('li', {}, ...inline(item)))
      }
      blocks.push(el(ordered ? 'ol' : 'ul', {}, items))
    } else if (/^#{1,4}\s/.test(line)) {
      blocks.push(el('h4', {}, ...inline(line.replace(/^#+\s/, ''))))
      i += 1
    } else {
      const para = []
      while (i < lines.length && lines[i].trim() && !/^\s*([-*]|\d+\.)\s|^\s*\||^#{1,4}\s/.test(lines[i])) para.push(lines[i++].trim())
      blocks.push(el('p', {}, ...inline(para.join(' '))))
    }
  }
  return el('div', { class: 'md' }, blocks)
}

// ---------- formatting ----------

function relative(iso) {
  if (!iso) return ''
  const seconds = Math.round((Date.parse(state?.now ?? new Date().toISOString()) - Date.parse(iso)) / 1000)
  const future = seconds < 0
  const s = Math.abs(seconds)
  let text
  if (s < 45) text = 'just now'
  else if (s < 90) text = '1 min'
  else if (s < 3600) text = `${Math.round(s / 60)} min`
  else if (s < 86400 * 1.5) text = `${(s / 3600).toFixed(s < 36000 ? 1 : 0)} h`
  else text = `${Math.round(s / 86400)} days`
  if (text === 'just now') return text
  return future ? `in ${text}` : `${text} ago`
}

function absolute(iso) {
  const d = new Date(iso)
  const sameDay = new Date().toDateString() === d.toDateString()
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  return sameDay ? time : `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`
}

function when(iso) {
  if (!iso) return el('time', { text: '—' })
  return el('time', { datetime: iso, title: new Date(iso).toString() }, `${relative(iso)} · ${absolute(iso)}`)
}

function duration(start, end) {
  const s = Math.max(0, Math.round((Date.parse(end ?? state.now) - Date.parse(start)) / 1000))
  if (s < 60) return `${s} s`
  if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`
}

function bytes(n) {
  if (n === null || n === undefined) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = n
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000
    i += 1
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}

function loss(v) {
  return v === null || v === undefined ? '—' : Number(v).toFixed(4)
}

function delta(v) {
  if (v === null || v === undefined) return el('span', { class: 'num' }, '—')
  const cls = Math.abs(v) <= 0.002 ? 'num' : v < 0 ? 'delta-better' : 'delta-worse'
  return el('span', { class: cls }, `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(4)}`)
}

const RUN_WORDS = {
  pending: 'waiting for a node',
  running: 'training',
  succeeded: 'finished',
  failed: 'failed',
  killed: 'stopped',
  worker_failed: 'node failed',
  unschedulable: 'cannot be scheduled',
}

function runPill(run) {
  const good = run.state === 'succeeded' ? 'true' : TERMINAL.has(run.state) ? 'false' : undefined
  return el('span', { class: 'pill', dataset: good ? { good } : {} }, RUN_WORDS[run.state] ?? run.state)
}

function progress(run) {
  if (!run.total_steps) {
    if (run.state === 'pending') return el('span', { class: 'hint' }, 'not started')
    if (TERMINAL.has(run.state)) return el('span', { class: 'num' }, run.step ? `${run.step.toLocaleString()} steps` : '—')
    return el('span', { class: 'hint' }, run.step ? `step ${run.step.toLocaleString()}` : 'starting…')
  }
  const step = Math.min(run.step ?? 0, run.total_steps)
  const bar = el('span')
  bar.style.width = `${(100 * step) / run.total_steps}%`
  return el(
    'div',
    { class: 'progress' },
    el('div', { class: 'bar' }, bar),
    el('span', { class: 'num' }, `${step.toLocaleString()} / ${run.total_steps.toLocaleString()}`),
  )
}

function wandbUrl(run) {
  return run.wandb_url ?? `${WANDB}${run.name}`
}

// ---------- shared sections ----------

function step(stateName, title, sub, ...body) {
  return el(
    'li',
    { class: 'step', dataset: { state: stateName } },
    el('span', { class: 'step-mark', 'aria-hidden': 'true' }, stateName === 'done' ? '✓' : stateName === 'failed' ? '!' : ''),
    el('div', { class: 'step-title' }, title, sub ? el('span', { class: 'step-sub' }, sub) : null),
    body.length ? el('div', { class: 'step-body' }, ...body) : null,
  )
}

function runsTable(runs) {
  return el(
    'div',
    { class: 'scroll' },
    el(
      'table',
      {},
      el('thead', {}, el('tr', {}, ['Run', 'State', 'Progress', 'Paloma macro', ''].map((h) => el('th', {}, h)))),
      el(
        'tbody',
        {},
        runs.map((r) =>
          el(
            'tr',
            {},
            el('td', {}, el('code', {}, r.name)),
            el('td', {}, runPill(r)),
            el('td', {}, progress(r)),
            el('td', { class: 'num' }, loss(r.loss)),
            el('td', {}, link(wandbUrl(r), 'W&B')),
          ),
        ),
      ),
    ),
  )
}

// One idea's five-step timeline. `planWaiting` says what the plan step is waiting on, which only the admin view
// knows in detail.
function renderIdea(idea, planWaiting) {
  const plan = idea.plan
  const result = idea.result
  const runs = state.runs.filter((r) => idea.registry_id && r.registry_id === idea.registry_id)
  const live = runs.filter((r) => !TERMINAL.has(r.state))
  const label = idea.title ?? idea.body.split('\n')[0].slice(0, 120)

  const approved = step(
    'done',
    'Approved',
    idea.origin === 'intake' ? `suggested on #9514 by ${idea.author}` : `proposed by ${idea.author ?? 'a maintainer'}`,
    el('span', { class: 'hint' }, idea.decided_by ? `approved by ${idea.decided_by} · ` : 'approved · '),
    when(idea.decided_at ?? idea.approved_at),
    idea.url ? [' · ', link(idea.url, 'original comment')] : null,
  )

  let planStep
  if (plan) {
    planStep = step(
      'done',
      'Plan',
      plan.protocol,
      markdown(plan.summary ?? ''),
      el(
        'dl',
        { class: 'facts' },
        el('dt', {}, 'Cost'), el('dd', {}, plan.cost),
        el('dt', {}, 'Code'),
        el('dd', {}, plan.commit ? link(`https://github.com/marin-community/marin/commit/${plan.commit}`, el('code', {}, plan.commit.slice(0, 10))) : '—', plan.branch ? [' on ', el('code', {}, plan.branch)] : null),
        plan.spec_url ? [el('dt', {}, 'Run spec'), el('dd', {}, link(plan.spec_url, 'runs/' + idea.registry_id + '.yaml'))] : null,
        el('dt', {}, 'Runs'),
        el('dd', {}, (plan.runs ?? []).map((r) => el('div', {}, el('code', {}, r.name), ' — ', r.variant, r.control ? ' (control)' : ''))),
      ),
    )
  } else if (idea.state === 'declined') {
    planStep = step('failed', 'Plan', 'declined by the agent', markdown(idea.note ?? ''))
  } else {
    planStep = step('current', 'Plan', planWaiting)
  }

  let runStep
  if (runs.length) {
    const finished = runs.length - live.length
    runStep = step(live.length ? 'current' : 'done', 'Runs', `${finished} of ${runs.length} finished`, runsTable(runs))
  } else {
    runStep = step(plan ? 'current' : 'todo', 'Runs', plan ? 'submitting…' : 'not submitted yet')
  }

  let resultStep
  if (result) {
    resultStep = step(
      'done',
      'Result',
      'Δ Paloma macro loss vs the same-seed control (negative is better)',
      el(
        'div',
        { class: 'scroll' },
        el(
          'table',
          {},
          el('thead', {}, el('tr', {}, ['Run', 'Variant', 'Paloma macro', 'Δ vs control'].map((h) => el('th', {}, h)))),
          el(
            'tbody',
            {},
            (result.rows ?? []).map((r) =>
              el('tr', {}, el('td', {}, el('code', {}, r.name)), el('td', {}, r.variant), el('td', { class: 'num' }, loss(r.loss)), el('td', {}, r.control ? el('span', { class: 'hint' }, 'control') : delta(r.delta))),
            ),
          ),
        ),
      ),
      el('div', { class: 'verdict', dataset: { kind: result.kind ?? 'neutral' } }, result.verdict),
      result.summary ? markdown(result.summary) : null,
    )
  } else {
    resultStep = step(live.length ? 'todo' : runs.length ? 'current' : 'todo', 'Result', runs.length && !live.length ? 'being recorded' : 'after the runs finish')
  }

  const registryStep = step(
    idea.state === 'recorded' ? 'done' : idea.registry_id ? 'current' : 'todo',
    'Registry',
    idea.registry_id ? (idea.state === 'recorded' ? 'trial recorded' : 'entry created, trial pending') : 'gets an id when planned',
    idea.registry_id
      ? idea.registry_url
        ? link(idea.registry_url, el('code', {}, idea.registry_id))
        : el('code', {}, idea.registry_id)
      : null,
  )

  const statePill = el('span', { class: 'pill', dataset: idea.state === 'recorded' ? { good: 'true' } : idea.state === 'declined' ? { warn: 'true' } : {} }, idea.state)
  return el(
    'article',
    { class: 'idea', id: `idea-${idea.id}` },
    el('div', { class: 'idea-head' }, idea.registry_id ? el('code', {}, idea.registry_id) : null, el('h3', {}, label), statePill, el('span', { class: 'hint' }, 'updated ', when(idea.updated_at))),
    el('ol', { class: 'steps' }, approved, planStep, runStep, resultStep, registryStep),
  )
}

function renderPipeline(pipeline) {
  setChildren(
    'pipeline',
    STAGES.map(([key, label, what]) =>
      el(
        'li',
        { dataset: { hot: String(pipeline[key] > 0) } },
        el('div', { class: 'count' }, String(pipeline[key])),
        el('div', {}, el('div', { class: 'stage' }, label), el('div', { class: 'what' }, what)),
      ),
    ),
  )
}

function renderNext(next) {
  document.getElementById('next').dataset.tone = next.tone
  document.getElementById('next-text').textContent = next.text
}
