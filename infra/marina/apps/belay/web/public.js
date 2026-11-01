// Marin Belay public view: read-only status for anyone. It reads api/public/state from the app, or the JSON
// snapshot named by <body data-source> in a static export (GitHub Pages). It has no controls and shows no
// transcripts, internal paths, job ids, or node and storage figures; the server leaves those out of the data too.
// Loaded after common.js.

const SOURCE = document.body.dataset.source || `${BELAY_BASE}api/public/state`
const STATUS_WORDS = {
  pending: ['waiting for review', undefined],
  approved: ['approved, plan pending', 'true'],
  planned: ['planned', 'true'],
  running: ['running', 'true'],
  recorded: ['result recorded', 'true'],
  declined: ['declined', 'false'],
}

function statusPill(idea) {
  const [word, good] = STATUS_WORDS[idea.state] ?? [idea.state, undefined]
  return el('span', { class: 'pill', dataset: good ? { good } : {} }, word)
}

function renderSuggestions() {
  const suggestions = state.ideas.filter((i) => i.origin === 'intake')
  if (!suggestions.length) {
    setChildren('public-suggestions', el('div', { class: 'empty' }, 'No suggestions yet. Post one on #9514.'))
    return
  }
  setChildren(
    'public-suggestions',
    suggestions.map((idea) =>
      el(
        'article',
        { class: 'card' },
        el(
          'div',
          { class: 'card-head' },
          el('span', { class: 'author' }, idea.author),
          statusPill(idea),
          when(idea.posted_at),
          idea.url ? link(idea.url, 'open on GitHub') : null,
          idea.registry_id ? [' · ', el('a', { href: `#idea-${idea.id}` }, idea.registry_id)] : null,
        ),
        el('p', { class: 'body clamped' }, idea.body),
      ),
    ),
  )
}

function renderPublicIdeas() {
  const ideas = state.ideas.filter((i) => i.state !== 'pending')
  if (!ideas.length) {
    setChildren('public-ideas', el('div', { class: 'empty' }, 'No ideas have been approved yet.'))
    return
  }
  setChildren('public-ideas', ideas.map((idea) => renderIdea(idea, 'waiting for the Belay agent to plan it')))
}

function renderPublicRuns() {
  if (!state.runs.length) {
    setChildren('public-run-rows', el('tr', {}, el('td', { colspan: '5', class: 'hint' }, 'No Belay runs yet.')))
    return
  }
  setChildren(
    'public-run-rows',
    state.runs.map((r) =>
      el(
        'tr',
        {},
        el('td', {}, el('code', {}, r.name)),
        el('td', {}, runPill(r)),
        el('td', {}, progress(r)),
        el('td', { class: 'num' }, loss(r.loss)),
        el('td', {}, when(r.submitted_at)),
        el('td', {}, link(wandbUrl(r), 'W&B')),
      ),
    ),
  )
}

function renderPublic() {
  renderNext(state.next)
  renderPipeline(state.pipeline)
  renderSuggestions()
  renderPublicIdeas()
  renderPublicRuns()
  setChildren('public-updated', 'Data as of ', when(state.now))
}

async function refreshPublic() {
  try {
    const response = await fetch(SOURCE, { headers: { accept: 'application/json' } })
    if (!response.ok) throw new Error(`${response.status}`)
    state = await response.json()
    renderPublic()
  } catch (error) {
    document.getElementById('next-text').textContent = `Belay status is unavailable right now (${error.message}).`
    document.getElementById('next').dataset.tone = 'blocked'
  }
}

showView('public')
document.getElementById('who').textContent = 'public · read-only'
refreshPublic()
setInterval(refreshPublic, REFRESH_MS)
