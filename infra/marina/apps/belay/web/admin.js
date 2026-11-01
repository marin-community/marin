// Marin Belay admin view: everything with controls or internal detail. Reads api/admin/state every few seconds;
// the writes are approve, dismiss, propose and the dispatch switch. The server admits only listed approvers, and
// anyone else sees the 403 page. Loaded after common.js.

const API = `${BELAY_BASE}api/admin`
let busy = false
const expanded = new Set()

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
  renderNext(next)

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

  renderPipeline(pipeline)
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


function renderIdeas() {
  const ideas = state.ideas.filter((i) => !['pending', 'dismissed'].includes(i.state))
  if (!ideas.length) {
    setChildren('idea-list', el('div', { class: 'empty' }, 'No approved ideas yet. Approve a suggestion above to start one.'))
    return
  }
  const active = state.sessions.find((s) => s.state === 'running')
  const waiting = active
    ? `session ${active.id} is planning it now`
    : state.settings.dispatch_enabled
      ? 'a session starts within a minute'
      : 'waiting: dispatch is off'
  setChildren('idea-list', ideas.map((idea) => renderIdea(idea, waiting)))
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
        s.summary ? el('div', { class: 'summary' }, markdown(s.summary)) : null,
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
        el('div', { class: 'feed-title' }, a.registry_id && !a.title.startsWith(a.registry_id) ? [el('code', {}, a.registry_id), ' '] : null, a.title),
        el('div', {}, when(a.at), a.github_url ? [' · ', link(a.github_url, 'on GitHub')] : null),
        a.body ? el('div', { class: 'feed-body' }, markdown(a.body)) : null,
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

function wireControls() {
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
}

function deny(status, message) {
  showView('denied')
  document.title = `${status} · Marin Belay admin`
  document.getElementById('denied-code').textContent = String(status)
  document.getElementById('denied-text').textContent = message
}

async function start() {
  const response = await fetch(`${API}/me`, { headers: { accept: 'application/json' } })
  if (!response.ok) {
    const detail = await response.json().then((b) => b.detail).catch(() => '')
    deny(response.status, detail || 'You do not have access to the Belay admin view.')
    return
  }
  const me = await response.json()
  showView('admin')
  document.getElementById('who').textContent = me.user
  wireControls()
  await refresh()
  setInterval(refresh, REFRESH_MS)
}

start()
