// Marin Belay admin page: approvals and controls. The server admits only listed approvers (401 signed out, 403
// otherwise); everyone else sees the denied page. Loaded after common.js.

const API = `${BELAY_BASE}api/admin`
const openBodies = new Set()
let busy = false
let settingsLoaded = false

async function api(path, body) {
  const options =
    body === undefined
      ? {}
      : { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }
  const response = await fetch(`${API}/${path}`, options)
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`)
  return response.json()
}

function toast(text) {
  const node = document.getElementById('toast')
  node.textContent = text
  node.dataset.show = 'true'
  clearTimeout(toast.timer)
  toast.timer = setTimeout(() => (node.dataset.show = 'false'), 3000)
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

function meter(id, used, total, text) {
  document.getElementById(`${id}-v`).textContent = text
  const bar = document.getElementById(`${id}-bar`)
  bar.style.width = `${Math.min(100, (100 * used) / Math.max(total, 1))}%`
  bar.dataset.warn = String(used >= 0.9 * total)
}

function renderControls() {
  const { settings, watcher, queued } = state
  document.getElementById('dispatch').checked = settings.dispatch_enabled
  document.getElementById('dispatch-word').textContent = settings.dispatch_enabled ? 'on' : 'off'
  meter('nodes', state.nodes_requested, settings.node_cap, `${state.nodes_requested} / ${settings.node_cap}`)
  meter(
    'storage',
    watcher.storage_bytes ?? 0,
    settings.storage_limit_bytes,
    `${bytes(watcher.storage_bytes)} / ${bytes(settings.storage_limit_bytes)}`,
  )
  const stale = !watcher.ticked_at || Date.parse(state.now) - Date.parse(watcher.ticked_at) > 6 * 60 * 1000
  setChildren(
    'watcher-v',
    el('span', { class: stale ? 'warn' : 'ok' }, watcher.ticked_at ? relative(watcher.ticked_at) : 'not running'),
  )
  document.getElementById('queued-v').textContent = queued.length ? `${queued.length} queued` : 'queue empty'
}

function renderPending() {
  // Pending ideas wait for a decision. "upstream" ideas were approved on GitHub before the cutover, so nothing
  // queued them; Approve queues one (an idea approved on both paths still runs once).
  const escalated = (i) => i.state === 'approved' && (i.note ?? '').startsWith('Escalated:')
  const pending = state.ideas.filter((i) => i.state === 'pending' || i.state === 'upstream' || escalated(i))
  document.getElementById('pending-n').textContent = String(pending.filter((i) => i.state === 'pending').length)
  if (!pending.length) {
    setChildren('pending', el('div', { class: 'none' }, 'Nothing to review'))
    return
  }
  setChildren(
    'pending',
    pending.map((idea) => {
      const first = (idea.title ?? idea.body.trim().split('\n')[0]).replace(/\*\*/g, '')
      const expanded = openBodies.has(idea.id)
      return el(
        'div',
        { class: 'line-wrap' },
        el(
          'div',
          { class: 'line pending' },
          el(
            'button',
            {
              type: 'button',
              class: 'nm linkish',
              'aria-expanded': String(expanded),
              onclick: () => {
                expanded ? openBodies.delete(idea.id) : openBodies.add(idea.id)
                render()
              },
            },
            el('b', {}, first),
            el(
              'small',
              {},
              escalated(idea)
                ? `needs maintainer: ${idea.note.replace('Escalated: ', '')} · `
                : `${idea.state === 'upstream' ? 'approved on GitHub, not queued · ' : 'not approved · '}`,
              `${idea.origin === 'intake' ? '#9514' : 'proposal'} · ${idea.author} · `,
              when(idea.posted_at),
            ),
          ),
          idea.url ? link(idea.url, '↗') : el('span'),
          escalated(idea)
            ? el('button', { type: 'button', class: 'btn primary', onclick: () => {
                const note = window.prompt('Note for the session (for example "retest on current recipe"), or leave empty', '')
                if (note !== null) act(`ideas/${idea.id}/requeue`, { note }, 'Re-queued')
              } }, 'Re-queue')
            : el(
                'button',
                { type: 'button', class: 'btn primary', onclick: () => act(`ideas/${idea.id}/approve`, {}, 'Approved and queued') },
                idea.state === 'upstream' ? 'Queue' : 'Approve',
              ),
          idea.state === 'pending'
            ? el('button', { type: 'button', class: 'btn', onclick: () => act(`ideas/${idea.id}/dismiss`, {}, 'Dismissed') }, 'Dismiss')
            : escalated(idea)
              ? el('button', { type: 'button', class: 'btn', onclick: () => {
                  const note = window.prompt('Why decline?', '')
                  if (note !== null) act(`ideas/${idea.id}/decline`, { note }, 'Declined')
                } }, 'Decline')
              : el('span'),
        ),
        expanded ? el('div', { class: 'detail-c' }, markdown(idea.body)) : null,
      )
    }),
    alreadyRun(),
  )
}

// #9514 suggestions the registry already holds: done, never queued again unless a maintainer re-queues one.
function alreadyRun() {
  const done = state.ideas.filter((i) => i.state === 'recorded' && (i.note ?? '').startsWith('Already run as'))
  if (!done.length) return null
  return el(
    'details',
    { class: 'fold' },
    el('summary', {}, `Already run (${done.length})`),
    ...done.map((idea) =>
      el(
        'div',
        { class: 'line pending' },
        el(
          'span',
          { class: 'nm' },
          el('b', {}, (idea.title ?? idea.body.trim().split('\n')[0]).replace(/\*\*/g, '').slice(0, 90)),
          el('small', {}, `#9514 · ${idea.author} · done → `, link(`${PAGES.status}#${idea.registry_id}`, idea.registry_id)),
        ),
        idea.url ? link(idea.url, '↗') : el('span'),
        el('button', { type: 'button', class: 'btn', onclick: () => {
          const note = window.prompt('Re-queue note for the session (for example "retest on current recipe")', 'retest on current recipe')
          if (note) act(`ideas/${idea.id}/requeue`, { note }, 'Re-queued')
        } }, 'Re-queue'),
      ),
    ),
  )
}

const SESSION_CLASS = { succeeded: 'won', running: 'tested', failed: 'lost', lost: 'lost' }

function renderSessions() {
  if (!state.sessions.length) {
    setChildren('sessions', el('div', { class: 'none' }, 'No sessions yet'))
    return
  }
  setChildren(
    'sessions',
    state.sessions.map((s) =>
      el(
        'div',
        { class: 'line static session' },
        when(s.started_at),
        el('span', { class: `st ${SESSION_CLASS[s.state] ?? 'tested'}` }, s.state === 'running' ? 'working' : s.state),
        el('span', { class: 'nm' }, el('b', {}, s.events.map((e) => e.summary).join('; ') || '—'), el('small', {}, s.id)),
        el(
          'span',
          { class: 'mono muted' },
          [
            s.ended_at ? duration(s.started_at, s.ended_at) : duration(s.started_at),
            s.turns ? `${s.turns}t` : null,
            s.cost_usd ? `$${s.cost_usd.toFixed(2)}` : null,
          ]
            .filter(Boolean)
            .join(' · '),
        ),
        link(`${API}/sessions/${encodeURIComponent(s.id)}/transcript`, 'log ↗'),
      ),
    ),
  )
}

function fillSettings() {
  if (settingsLoaded) return
  settingsLoaded = true
  const { settings } = state
  document.getElementById('set-node-cap').value = settings.node_cap
  document.getElementById('set-github').checked = settings.github_posting
  document.getElementById('set-approvers').value = settings.approvers
    .map((a) => `${a.github} ${a.identity ?? ''}`.trim())
    .join('\n')
}

function render() {
  renderNext(state.next)
  renderControls()
  renderPending()
  renderSessions()
  fillSettings()
}

async function refresh() {
  try {
    state = await api('state')
    render()
  } catch (error) {
    renderNext({ tone: 'blocked', text: `Belay API unavailable (${error.message})` })
  }
}

function wireControls() {
  document.getElementById('dispatch').addEventListener('change', (event) => {
    const on = event.target.checked
    act('settings', { dispatch_enabled: on }, on ? 'Dispatch on' : 'Dispatch off')
  })
  document.getElementById('settings-form').addEventListener('submit', (event) => {
    event.preventDefault()
    const previous = new Map(state.settings.approvers.map((a) => [a.github, a]))
    const approvers = document
      .getElementById('set-approvers')
      .value.split('\n')
      .map((line) => line.trim().split(/\s+/))
      .filter(([github]) => github)
      .map(([github, identity]) => ({ ...(previous.get(github) ?? { github }), identity: identity ?? null }))
    act(
      'settings',
      {
        node_cap: Number(document.getElementById('set-node-cap').value),
        github_posting: document.getElementById('set-github').checked,
        approvers,
      },
      'Settings saved',
    ).then(() => (settingsLoaded = false))
  })
  document.getElementById('propose-form').addEventListener('submit', (event) => {
    event.preventDefault()
    const title = document.getElementById('propose-title')
    const body = document.getElementById('propose-body')
    act('ideas', { title: title.value, body: body.value }, 'Added to pending').then(() => {
      title.value = ''
      body.value = ''
    })
  })
}

function deny(status, message) {
  showView('denied')
  document.title = `${status} · Marin Belay`
  document.getElementById('denied-code').textContent = String(status)
  document.getElementById('denied-text').textContent = message
  document.getElementById('denied-status').href = PAGES.status
  document.getElementById('denied-front').href = PAGES.front
}

async function start() {
  const response = await fetch(`${API}/me`, { headers: { accept: 'application/json' } })
  if (!response.ok) {
    const detail = await response.json().then((b) => b.detail).catch(() => '')
    deny(response.status, detail || 'You do not have access to Belay admin.')
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
