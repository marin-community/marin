// Marin Belay admin page: approvals and controls. The server admits only listed approvers (401 signed out, 403
// otherwise); everyone else sees the denied page. Loaded after common.js.

const API = `${BELAY_BASE}api/admin`
const openBodies = new Set()
let state = null // the admin state last read from the API
let busy = false
let settingsLoaded = false
const INTAKE_ISSUE = 'https://api.github.com/repos/marin-community/marin/issues/9514'
const BOTS = new Set(['marin-belay-bot', 'loom-oa-dev[bot]'])
let intakeComments = null // the newest #9514 comments, read from GitHub in this browser on load and on Reload
let intakeError = null

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
  const cap = document.getElementById('node-cap')
  if (document.activeElement !== cap) cap.value = settings.node_cap
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

// An optional node grant beside an approval: blank keeps the idea under the global cap.
function nodesInput() {
  return el('input', {
    type: 'number',
    min: '1',
    max: '64',
    class: 'nodes-in',
    placeholder: 'nodes',
    title: 'Node grant: this idea may hold up to this many nodes, past the global cap. Blank: global cap.',
    'aria-label': 'Node grant',
  })
}

// Optional study hours beside the node grant: for a custom-track study, the two are its budget.
function hoursInput() {
  return el('input', {
    type: 'number',
    min: '0.25',
    step: '0.25',
    class: 'nodes-in',
    placeholder: 'hours',
    title: 'Study budget (custom track): with the node grant, the most its runs may use. Blank: no study budget.',
    'aria-label': 'Study hours',
  })
}

// An optional Iris priority band for the idea's runs, above its Track's band (the empty choice keeps the Track's).
const PRIORITY_BANDS = ['batch', 'interactive', 'production']

function priorityInput() {
  return el(
    'select',
    { class: 'priority-in', title: "Iris priority for this idea's runs; only raises the Track's band", 'aria-label': 'Priority' },
    el('option', { value: '' }, 'track band'),
    ...PRIORITY_BANDS.map((band) => el('option', { value: band }, band)),
  )
}

// What an approver has typed into a row's controls, by row key and field, so a refresh (which rebuilds every row)
// keeps a draft, and the focused field with its cursor.
const drafts = new Map()

function drafted(node, key, field, initial = '') {
  node.dataset.draft = `${key}:${field}`
  node.value = drafts.get(key)?.[field] ?? initial
  const save = () => drafts.set(key, { ...drafts.get(key), [field]: node.value })
  node.addEventListener('input', save)
  node.addEventListener('change', save)
  return node
}

function withFocusKept(renderAll) {
  const active = document.activeElement
  const field = active?.dataset?.draft
  const caret = field && 'selectionStart' in active ? [active.selectionStart, active.selectionEnd] : null
  renderAll()
  if (!field) return
  const again = document.querySelector(`[data-draft="${CSS.escape(field)}"]`)
  if (!again) return
  again.focus()
  if (caret && again.setSelectionRange) again.setSelectionRange(...caret)
}

function grantOf(nodes, hours, priority) {
  return {
    ...(nodes.value ? { nodes: Number(nodes.value) } : {}),
    ...(hours.value ? { hours: Number(hours.value) } : {}),
    ...(priority.value ? { priority: priority.value } : {}),
  }
}

function renderPending() {
  // Pending ideas wait for a decision; approval happens only here.
  const escalated = (i) => i.state === 'approved' && (i.note ?? '').startsWith('Escalated:')
  const pending = state.ideas.filter((i) => i.state === 'pending' || escalated(i))
  document.getElementById('pending-n').textContent = String(pending.filter((i) => i.state === 'pending').length)
  if (!pending.length) {
    setChildren('pending', el('div', { class: 'none' }, 'Nothing to review'), alreadyRun())
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
                : 'not approved · ',
              `${idea.origin === 'intake' ? '#9514' : 'proposal'} · ${idea.author} · `,
              when(idea.posted_at),
            ),
          ),
          idea.url ? link(idea.url, '↗') : el('span'),
          escalated(idea)
            ? requeueWithGrant(idea, '', `idea${idea.id}`)
            : approveWithGrant(`idea${idea.id}`, 'Approve', (body) =>
                act(`ideas/${idea.id}/approve`, body, 'Approved and queued'),
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
        expanded ? storedText(idea.body, idea.hidden_removed) : null,
      )
    }),
    alreadyRun(),
  )
}

// The text a session would read, shown raw (not rendered) with what was stripped from the comment flagged.
function storedText(text, removed) {
  return el(
    'div',
    { class: 'detail-c' },
    removed?.length ? el('div', { class: 'warn small' }, `Stripped: ${removed.join('; ')}`) : null,
    el('pre', { class: 'raw' }, text),
  )
}

// A #9514 comment as GitHub serves it plus the visible text this page showed: what an intake decision sends.
function review(c) {
  return {
    comment_id: String(c.id),
    url: c.html_url,
    author: c.user.login,
    body: c.body,
    shown: c.visible.text,
    posted_at: c.created_at,
  }
}

// Decline with the approver's reason, which the declined idea shows; the button waits for a reason.
function declineWithReason(key, run) {
  const reason = el('input', { type: 'text', class: 'reason', placeholder: 'reason to decline', 'aria-label': 'Reason to decline' })
  drafted(reason, key, 'reason')
  const button = el('button', { type: 'button', class: 'btn' }, 'Decline')
  const update = () => {
    button.disabled = !reason.value.trim()
  }
  update()
  reason.addEventListener('input', update)
  button.addEventListener('click', () => {
    drafts.delete(key)
    run(reason.value.trim())
  })
  return el('span', { class: 'grant' }, reason, button)
}

// A primary action with its grant fields (nodes, study hours, priority) and the approver's instructions for the
// session: the action gets those that are filled. The session follows the instructions as the maintainer note.
function approveWithGrant(key, label, run, suggested = '') {
  const nodes = drafted(nodesInput(), key, 'nodes')
  const hours = drafted(hoursInput(), key, 'hours')
  const priority = drafted(priorityInput(), key, 'priority')
  const note = el('textarea', {
    class: 'instructions',
    rows: '2',
    placeholder: 'instructions for the session (optional): track, settings, what to compare',
    'aria-label': 'Instructions for the session',
  })
  drafted(note, key, 'note', suggested)
  const decision = () => ({ ...grantOf(nodes, hours, priority), ...(note.value.trim() ? { note: note.value.trim() } : {}) })
  return el(
    'span',
    { class: 'grant' },
    note,
    nodes,
    hours,
    priority,
    el('button', {
      type: 'button',
      class: 'btn primary',
      onclick: () => {
        const body = decision()
        drafts.delete(key)
        run(body)
      },
    }, label),
  )
}

function requeueWithGrant(idea, suggested, key) {
  return approveWithGrant(key, 'Re-queue', (grant) => act(`ideas/${idea.id}/requeue`, { note: '', ...grant }, 'Re-queued'), suggested)
}

// The newest #9514 comments, read from GitHub's public API in this browser (no server keys), with the visible text
// the server would store for each. Approving one sends the comment as GitHub serves it plus the text shown here;
// the server stores exactly that text, refuses a comment that no longer strips to it, and refuses one the registry
// already holds.
async function loadIntake() {
  try {
    const github = async (url) => {
      const response = await fetch(url)
      if (!response.ok) throw new Error(response.status === 403 ? 'GitHub rate limit for this browser; Reload later' : `${response.status}`)
      return response.json()
    }
    const issue = await github(INTAKE_ISSUE)
    const last = Math.max(1, Math.ceil(issue.comments / 30))
    const page = await github(`${INTAKE_ISSUE}/comments?per_page=30&page=${last}`)
    const comments = page.filter((c) => c.user.type !== 'Bot' && !BOTS.has(c.user.login)).reverse()
    const visible = await api('intake/visible', { bodies: comments.map((c) => c.body) })
    intakeComments = comments.map((c, i) => ({ ...c, visible: visible[i] }))
    intakeError = null
  } catch (error) {
    intakeComments = []
    intakeError = error.message
  }
  renderIntakeLive()
}

function renderIntakeLive() {
  if (intakeComments === null) {
    setChildren('intake-live', el('div', { class: 'none' }, 'Loading…'))
    return
  }
  if (!intakeComments.length) {
    setChildren('intake-live', el('div', { class: 'none' }, intakeError ? `GitHub unavailable: ${intakeError}` : 'No comments'))
    return
  }
  const known = new Map(state.ideas.filter((i) => i.comment_id).map((i) => [String(i.comment_id), i]))
  const rows = intakeComments.map((c) => {
    const idea = known.get(String(c.id))
    const registered = state.registered_comments[String(c.id)]
    return { c, idea, registered, waiting: !registered && (!idea || idea.state === 'pending') }
  })
  // Comments still waiting for a decision first; each group stays newest first.
  rows.sort((a, b) => Number(b.waiting) - Number(a.waiting))
  setChildren(
    'intake-live',
    rows.map(({ c, idea, registered, waiting }) => {
      const first = c.visible.text.trim().split('\n')[0].replace(/\*\*/g, '').slice(0, 110)
      const key = `c${c.id}`
      const expanded = openBodies.has(key)
      const flag = c.visible.removed.length ? el('span', { class: 'warn' }, ' · stripped hidden text') : null
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
                expanded ? openBodies.delete(key) : openBodies.add(key)
                renderIntakeLive()
              },
            },
            el('b', {}, first),
            el('small', {}, `${c.user.login} · `, when(c.created_at), flag),
          ),
          link(c.html_url, '↗'),
          waiting
            ? approveWithGrant(`c${c.id}`, 'Approve', (grant) => act('intake/approve', { ...review(c), ...grant }, 'Approved and queued'))
            : registered && !idea?.registry_id
              ? el('span', { class: 'muted small' }, 'done → ', link(`${PAGES.status}/${registered}`, registered))
              : el(
                  'span',
                  { class: 'muted small' },
                  idea.registry_id ? `${idea.state} → ${idea.registry_id}` : idea.state === 'declined' ? idea.note : idea.state,
                ),
          waiting ? declineWithReason(`c${c.id}`, (note) => act('intake/decline', { ...review(c), note }, 'Declined')) : null,
        ),
        expanded ? storedText(c.visible.text, c.visible.removed) : null,
      )
    }),
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
          el('small', {}, `#9514 · ${idea.author} · done → `, link(`${PAGES.status}/${idea.registry_id}`, idea.registry_id)),
        ),
        idea.url ? link(idea.url, '↗') : el('span'),
        requeueWithGrant(idea, 'retest on current recipe', `run${idea.id}`),
      ),
    ),
  )
}

// A token count as 1.2k / 3.4M; sessions run on a subscription, so tokens, not dollars, show their size.
function tokens(n) {
  if (n === undefined || n === null) return '—'
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n)
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
            s.turns ? `${s.turns} turns` : null,
            s.input_tokens !== null ? `in ${tokens(s.input_tokens)}` : null,
            s.output_tokens !== null ? `out ${tokens(s.output_tokens)}` : null,
            s.cache_read_tokens !== null ? `cache ${tokens(s.cache_read_tokens)}` : null,
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
    withFocusKept(() => {
      render()
      renderIntakeLive()
    })
  } catch (error) {
    renderNext({ tone: 'blocked', text: `Belay API unavailable (${error.message})` })
  }
}

function wireControls() {
  document.getElementById('dispatch').addEventListener('change', (event) => {
    const on = event.target.checked
    act('settings', { dispatch_enabled: on }, on ? 'Dispatch on' : 'Dispatch off')
  })
  document.getElementById('node-cap-form').addEventListener('submit', (event) => {
    event.preventDefault()
    const cap = Number(document.getElementById('node-cap').value)
    act('settings', { node_cap: cap }, `Node cap ${cap}`)
  })
  document.getElementById('intake-reload').addEventListener('click', () => {
    refresh()
    loadIntake()
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
  loadIntake()
  // No timed refresh here (the public Status page has one): rebuilding the lists mid-edit costs an approver their
  // place, and admin data changes slowly. The page refreshes after each action, on returning to the tab, and on Reload.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') refresh()
  })
}

start()
