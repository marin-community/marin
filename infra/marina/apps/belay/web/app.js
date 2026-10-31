// Marin Belay admin page. Reads /belay/api/state every few seconds and renders it; the only writes are
// approve, dismiss, propose and the dispatch switch.

const API = 'api'
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

let state = null
let busy = false
const expanded = new Set()

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
  if (!run.total_steps) return el('span', { class: 'hint' }, run.state === 'pending' ? 'not started' : 'compiling…')
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

// ---------- network ----------

async function api(path, body) {
  const options = body === undefined ? {} : { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }
  const response = await fetch(`${API}/${path}`, options)
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`)
  return response.json()
}

function toast(text) {
  const node = document.getElementById('toast')
  node.textContent = text
  node.dataset.show = 'true'
  clearTimeout(toast.timer)
  toast.timer = setTimeout(() => (node.dataset.show = 'false'), 3500)
}

async function act(path, body, message) {
  if (busy) return
  busy = true
  try {
    await api(path, body)
    toast(message)
    await refresh()
  } catch (error) {
    toast(`Failed: ${error.message}`)
  } finally {
    busy = false
  }
}

// ---------- sections ----------

function renderOverview() {
  const { next, settings, watcher, queued, pipeline } = state
  document.getElementById('next').dataset.tone = next.tone
  document.getElementById('next-text').textContent = next.text

  const dispatch = document.getElementById('dispatch')
  dispatch.checked = settings.dispatch_enabled
  const word = document.getElementById('dispatch-word')
  word.textContent = settings.dispatch_enabled ? 'on' : 'off'
  word.className = settings.dispatch_enabled ? 'pill' : 'chip'
  if (settings.dispatch_enabled) word.dataset.good = 'true'
  else delete word.dataset.good

  const stale = watcher.ticked_at && Date.parse(state.now) - Date.parse(watcher.ticked_at) > 6 * 60 * 1000
  setChildren(
    'watcher',
    watcher.ticked_at
      ? el('span', { class: stale ? 'pill' : '' , dataset: stale ? { warn: 'true' } : {} }, `checked ${relative(watcher.ticked_at)}`)
      : el('span', { class: 'pill', dataset: { warn: 'true' } }, 'not running'),
    watcher.last_error ? el('span', { class: 'problem' }, ` ${watcher.last_error}`) : null,
  )
  const github = document.getElementById('github')
  github.textContent = settings.github_posting ? 'on' : 'off (activity feed only)'
  setChildren('queued', queued.length ? `${queued.length}: ${queued.map((e) => e.summary).join('; ')}` : 'none')

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

function renderPending() {
  const pending = state.ideas.filter((i) => i.state === 'pending')
  if (!pending.length) {
    setChildren('pending', el('div', { class: 'empty' }, 'No suggestions are waiting. New #9514 comments and proposals appear here.'))
    return
  }
  setChildren(
    'pending',
    pending.map((idea) => {
      const open = expanded.has(idea.id)
      const long = idea.body.length > 520 || idea.body.split('\n').length > 6
      return el(
        'article',
        { class: 'card' },
        el(
          'div',
          { class: 'card-head' },
          el('span', { class: 'author' }, idea.author),
          el('span', { class: 'chip' }, idea.origin === 'intake' ? '#9514 comment' : 'proposal'),
          when(idea.posted_at),
          idea.url ? link(idea.url, 'open on GitHub') : null,
        ),
        idea.title ? el('h3', {}, idea.title) : null,
        el('p', { class: `body${long && !open ? ' clamped' : ''}` }, idea.body),
        el(
          'div',
          { class: 'card-actions' },
          el('button', { class: 'primary', type: 'button', onclick: () => act(`ideas/${idea.id}/approve`, {}, 'Approved: queued for the next session.') }, 'Approve'),
          el('button', { class: 'quiet', type: 'button', onclick: () => act(`ideas/${idea.id}/dismiss`, {}, 'Dismissed.') }, 'Dismiss'),
          long
            ? el('button', { class: 'quiet', type: 'button', onclick: () => { open ? expanded.delete(idea.id) : expanded.add(idea.id); render() } }, open ? 'Show less' : 'Show all')
            : null,
        ),
      )
    }),
  )
}

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

function renderIdea(idea) {
  const plan = idea.plan
  const result = idea.result
  const runs = state.runs.filter((r) => idea.registry_id && r.registry_id === idea.registry_id)
  const activeSession = state.sessions.find((s) => s.state === 'running')
  const live = runs.filter((r) => !TERMINAL.has(r.state))
  const label = idea.title ?? idea.body.split('\n')[0].slice(0, 120)

  const approved = step(
    'done',
    'Approved',
    `${idea.origin === 'intake' ? 'suggested on #9514' : 'proposed'} by ${idea.author}`,
    el('span', { class: 'hint' }, 'approved by ', idea.decided_by ?? '—', ' · '),
    when(idea.decided_at),
    idea.url ? [' · ', link(idea.url, 'original comment')] : null,
  )

  let planStep
  if (plan) {
    planStep = step(
      'done',
      'Plan',
      plan.protocol,
      el('p', { class: 'body' }, plan.summary),
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
    planStep = step('failed', 'Plan', 'declined by the agent', el('p', { class: 'body' }, idea.note ?? ''))
  } else {
    planStep = step(
      'current',
      'Plan',
      activeSession ? `session ${activeSession.id} is planning it now` : state.settings.dispatch_enabled ? 'a session starts within a minute' : 'waiting: dispatch is off',
    )
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
      result.summary ? el('p', { class: 'body' }, result.summary) : null,
    )
  } else {
    resultStep = step(live.length ? 'todo' : runs.length ? 'current' : 'todo', 'Result', runs.length && !live.length ? 'the next session records it' : 'after the runs finish')
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

function renderIdeas() {
  const ideas = state.ideas.filter((i) => !['pending', 'dismissed'].includes(i.state))
  if (!ideas.length) {
    setChildren('idea-list', el('div', { class: 'empty' }, 'No approved ideas yet. Approve a suggestion above to start one.'))
    return
  }
  setChildren('idea-list', ideas.map(renderIdea))
}

function renderRuns() {
  const cap = state.settings.node_cap
  const used = state.nodes_requested
  document.getElementById('nodes-value').textContent = `${state.nodes_in_use} running, ${used} of ${cap} reserved`
  const nodesBar = document.getElementById('nodes-bar')
  nodesBar.style.width = `${Math.min(100, (100 * used) / cap)}%`
  nodesBar.dataset.warn = String(used >= cap)
  document.getElementById('nodes-hint').textContent =
    used >= cap ? 'At the soft cap: new runs are held until one finishes.' : `${cap - used} node(s) free under the soft cap (checked by the submit helper).`

  const w = state.watcher
  const limit = state.settings.storage_limit_bytes
  document.getElementById('storage-value').textContent = `${bytes(w.storage_bytes)} of ${bytes(limit)}`
  const storageBar = document.getElementById('storage-bar')
  storageBar.style.width = `${Math.min(100, (100 * (w.storage_bytes ?? 0)) / limit)}%`
  storageBar.dataset.warn = String((w.storage_bytes ?? 0) > 0.9 * limit)
  document.getElementById('storage-hint').textContent = w.storage_at
    ? `${(w.storage_objects ?? 0).toLocaleString()} objects, summed ${relative(w.storage_at)}. Objects expire 30 days after they are written.`
    : 'Not measured yet.'

  if (!state.runs.length) {
    setChildren('run-rows', el('tr', {}, el('td', { colspan: '6', class: 'hint' }, 'No Belay runs yet.')))
    return
  }
  setChildren(
    'run-rows',
    state.runs.map((r) =>
      el(
        'tr',
        {},
        el('td', {}, el('code', {}, r.name), el('div', { class: 'hint' }, r.job_id)),
        el('td', {}, runPill(r)),
        el('td', {}, progress(r)),
        el('td', { class: 'num' }, loss(r.loss)),
        el('td', {}, when(r.submitted_at)),
        el('td', {}, link(wandbUrl(r), 'W&B')),
      ),
    ),
  )
}

function renderSessions() {
  if (!state.sessions.length) {
    setChildren('session-list', el('div', { class: 'empty' }, 'No sessions have run yet. One starts when dispatch is on and an event is queued.'))
    return
  }
  setChildren(
    'session-list',
    state.sessions.map((s) => {
      const good = s.state === 'succeeded' ? { good: 'true' } : s.state === 'running' ? {} : { warn: 'true' }
      return el(
        'article',
        { class: 'session' },
        el(
          'div',
          { class: 'session-head' },
          el('strong', {}, el('code', {}, s.id)),
          el('span', { class: 'pill', dataset: good }, s.state === 'running' ? 'working' : s.state),
          when(s.started_at),
          el('span', { class: 'hint' }, s.ended_at ? `took ${duration(s.started_at, s.ended_at)}` : `running for ${duration(s.started_at)}`),
          s.turns ? el('span', { class: 'hint' }, `${s.turns} turns`) : null,
          s.cost_usd ? el('span', { class: 'hint' }, `$${s.cost_usd.toFixed(2)}`) : null,
          link(`${API}/sessions/${encodeURIComponent(s.id)}/transcript`, s.transcript_bytes ? 'transcript' : 'transcript (pending)'),
        ),
        s.events.length ? el('ul', {}, s.events.map((e) => el('li', {}, e.summary))) : null,
        s.summary ? el('p', { class: 'summary' }, s.summary) : null,
      )
    }),
  )
}

function renderActivity() {
  if (!state.activity.length) {
    setChildren('feed', el('li', { class: 'hint' }, 'Nothing has happened yet.'))
    return
  }
  setChildren(
    'feed',
    state.activity.map((a) =>
      el(
        'li',
        { dataset: { kind: a.kind } },
        el('div', { class: 'feed-title' }, a.registry_id ? [el('code', {}, a.registry_id), ' '] : null, a.title),
        el('div', {}, when(a.at), a.github_url ? [' · ', link(a.github_url, 'on GitHub')] : null),
        a.body ? el('pre', { class: 'feed-body' }, a.body) : null,
      ),
    ),
  )
}

function render() {
  if (!state) return
  renderOverview()
  renderPending()
  renderIdeas()
  renderRuns()
  renderSessions()
  renderActivity()
}

async function refresh() {
  try {
    state = await api('state')
    render()
  } catch (error) {
    document.getElementById('next-text').textContent = `Cannot reach the Belay API: ${error.message}`
    document.getElementById('next').dataset.tone = 'blocked'
  }
}

document.getElementById('dispatch').addEventListener('change', (event) => {
  const on = event.target.checked
  act('settings', { dispatch_enabled: on }, on ? 'Dispatch on: queued events start a session.' : 'Dispatch off: no new sessions start.')
})

document.getElementById('propose-form').addEventListener('submit', (event) => {
  event.preventDefault()
  const form = event.target
  const data = new FormData(form)
  act('ideas', { title: data.get('title'), body: data.get('body') }, 'Added as a pending suggestion.').then(() => {
    form.reset()
    form.closest('details').open = false
  })
})

fetch('/api/marina/me')
  .then((r) => (r.ok ? r.json() : null))
  .then((me) => {
    document.getElementById('who').textContent = me && me.user !== 'anonymous' ? me.user.split('@')[0] : 'local preview'
  })
  .catch(() => {})

refresh()
setInterval(refresh, REFRESH_MS)
