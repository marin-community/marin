// Shared by the Belay pages (front, status, Hero MFU, admin): routing, data loading, the page nav, and small DOM and
// formatting helpers. Nothing from the API is parsed as HTML here.

const REFRESH_MS = 10000
const TERMINAL = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'unschedulable'])
const WANDB = 'https://wandb.ai/marin-community/marin_moe/runs/'
// The app's root (`/a/<id>/`) for the front page, status, mfu and admin.
const BELAY_BASE = location.pathname.replace(/(status|admin|mfu)\/?$/, '').replace(/[^/]*$/, '')
// The release's frontend digest, stamped into the page by build.py; asset() adds it to every static file URL.
const VERSION = document.querySelector('meta[name="belay-version"]').content

function asset(name) {
  return `${BELAY_BASE}${name}?v=${VERSION}`
}

const PAGES = {
  front: BELAY_BASE,
  status: `${BELAY_BASE}status`,
  mfu: `${BELAY_BASE}mfu`,
  admin: `${BELAY_BASE}admin`,
}
const SOURCES = {
  results: `${BELAY_BASE}api/public/results`,
  state: `${BELAY_BASE}api/public/state`,
}


function pageName() {
  if (/\/admin\/?$/.test(location.pathname)) return 'admin'
  if (/\/status(\.html)?\/?$/.test(location.pathname)) return 'status'
  if (/\/mfu(\.html)?\/?$/.test(location.pathname)) return 'mfu'
  return 'front'
}

// Each page is a <template>; showing one clones it into #view-root, so element ids stay unique.
function showView(name) {
  const template = document.getElementById(`tpl-${name}`)
  document.getElementById('view-root').replaceChildren(template.content.cloneNode(true))
}

async function loadJson(kind) {
  const response = await fetch(SOURCES[kind], { headers: { accept: 'application/json' } })
  if (!response.ok) throw new Error(`${kind}: ${response.status}`)
  return response.json()
}

// Belay · Status · Hero MFU · Admin. Admin appears only for callers the server accepts as approvers.
async function renderNav(current) {
  const links = [['front', 'Belay'], ['status', 'Status'], ['mfu', 'Hero MFU'], ['admin', 'Admin']]
  const nav = document.getElementById('pages')
  const show = (canAdmin) =>
    nav.replaceChildren(
      ...links
        .filter(([key]) => key !== 'admin' || canAdmin || current === 'admin')
        .map(([key, label]) => el('a', { href: PAGES[key], 'aria-current': key === current ? 'page' : null }, label)),
    )
  show(false)
  if (current === 'admin') return
  try {
    const response = await fetch(`${BELAY_BASE}api/admin/me`, { headers: { accept: 'application/json' } })
    show(response.ok)
  } catch {
    show(false)
  }
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
  const seconds = Math.round((Date.now() - Date.parse(iso)) / 1000)
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
  return el('time', { datetime: iso, title: `${absolute(iso)} · ${new Date(iso).toString()}` }, relative(iso))
}

function duration(start, end) {
  const s = Math.max(0, Math.round(((end ? Date.parse(end) : Date.now()) - Date.parse(start)) / 1000))
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
  pending: 'queued',
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


function eta(run) {
  if (!run.steps_per_sec || !run.total_steps || run.step === null) return null
  const seconds = (run.total_steps - run.step) / run.steps_per_sec
  if (seconds < 60) return '<1 min'
  return seconds < 3600 ? `${Math.round(seconds / 60)} min` : `${(seconds / 3600).toFixed(1)} h`
}

const STAGES = [
  ['pending', 'Waiting'],
  ['approved', 'Approved'],
  ['planned', 'Planned'],
  ['running', 'Running'],
  ['recorded', 'Recorded'],
]

function renderPipeline(pipeline) {
  setChildren(
    'pipeline',
    STAGES.map(([key, label]) =>
      el('li', { dataset: { hot: String(pipeline[key] > 0) } }, el('b', {}, String(pipeline[key])), el('span', {}, label)),
    ),
  )
}

function renderNext(next) {
  const box = document.getElementById('next')
  box.dataset.tone = next.tone
  setChildren('next', el('span', { class: 'dot', 'aria-hidden': 'true' }), el('span', {}, next.text))
}
