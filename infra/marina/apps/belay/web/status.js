// Marin Belay status page: what is running, recent results and activity, from the public state (read-only, the same
// filtered data the static export publishes). Loaded after common.js.

const expandedIdeas = new Set()
const KIND_CLASS = { better: 'won', worse: 'lost', neutral: 'noise' }
const KIND_WORD = { better: 'better', worse: 'worse', neutral: 'noise' }
const STATE_CLASS = { recorded: 'in', running: 'won', planned: 'tested', approved: 'tested', pending: 'tested', declined: 'lost' }
const STATE_WORD = {
  pending: 'waiting',
  approved: 'approved, queued',
  upstream: 'approved on GitHub, not queued',
  planned: 'planned',
  running: 'running',
  recorded: 'recorded',
  declined: 'declined',
}
const FEED_ICON = { intake: '＋', approved: '✓', plan: '✎', started: '▶', finished: '■', result: '◆', declined: '✕' }

function verdictChip(result) {
  const kind = result.kind ?? 'neutral'
  return el('span', { class: `st ${KIND_CLASS[kind] ?? 'noise'}` }, KIND_WORD[kind] ?? kind)
}

function variantDelta(idea) {
  const row = (idea.result?.rows ?? []).find((r) => !r.control)
  return row ? delta(row.delta) : el('span', { class: 'muted' }, '—')
}

function ideaRuns(idea) {
  return state.runs.filter((r) => idea.registry_id && r.registry_id === idea.registry_id)
}

function detail(idea) {
  const plan = idea.plan
  const runs = ideaRuns(idea)
  const facts = [
    idea.url ? ['Source', link(idea.url, idea.author ? `#9514 · ${idea.author}` : 'GitHub')] : ['Source', 'maintainer'],
    plan?.cost ? ['Cost', plan.cost] : null,
    plan?.commit ? ['Code', link(`https://github.com/marin-community/marin/commit/${plan.commit}`, el('code', {}, plan.commit.slice(0, 10)))] : null,
    plan?.spec_url ? ['Spec', link(plan.spec_url, `runs/${idea.registry_id}.yaml`)] : null,
    idea.registry_url ? ['Registry', link(idea.registry_url, idea.registry_id)] : null,
  ].filter(Boolean)
  return el(
    'div',
    { class: 'detail-c' },
    el('div', { class: 'kv' }, facts.map(([k, v]) => el('span', { class: 'kvp' }, el('span', { class: 'k' }, k), v))),
    runs.length
      ? el(
          'div',
          { class: 'mini-runs' },
          runs.map((r) =>
            el(
              'div',
              { class: 'mini-run' },
              el('code', {}, r.name),
              runPill(r),
              progress(r),
              el('span', { class: 'mono' }, loss(r.loss)),
              link(wandbUrl(r), 'W&B ↗'),
            ),
          ),
        )
      : null,
    idea.result ? el('p', { class: 'verdict-line' }, verdictChip(idea.result), ' ', idea.result.verdict) : null,
    idea.note ? el('p', { class: 'muted small' }, idea.note) : null,
    !idea.result && !idea.note ? el('p', { class: 'muted small clamp' }, plan?.summary ?? idea.body) : null,
  )
}

function ideaLine(idea, cells) {
  const key = idea.id
  const expanded = expandedIdeas.has(key)
  const button = el(
    'button',
    {
      type: 'button',
      class: 'line',
      'aria-expanded': String(expanded),
      onclick: () => {
        expanded ? expandedIdeas.delete(key) : expandedIdeas.add(key)
        render()
      },
    },
    ...cells,
  )
  return el('div', { class: 'line-wrap' }, button, expanded ? detail(idea) : null)
}

function idCell(idea) {
  return el('span', { class: 'xid mono' }, idea.registry_id ?? `#${idea.id}`)
}

function renderRunning() {
  const live = state.runs.filter((r) => !TERMINAL.has(r.state))
  if (!live.length) {
    setChildren('running', el('div', { class: 'none' }, 'Nothing running'))
    return
  }
  setChildren(
    'running',
    live.map((r) => {
      const idea = state.ideas.find((i) => i.registry_id && i.registry_id === r.registry_id)
      const time = eta(r)
      const cells = [
        el('span', { class: 'xid mono' }, r.registry_id ?? ''),
        el('span', { class: 'nm' }, el('b', {}, idea?.title ?? r.name), el('small', {}, r.name)),
        progress(r),
        el('span', { class: 'eta mono' }, r.state === 'pending' ? 'queued' : time ? `${time} left` : '—'),
      ]
      return idea ? ideaLine(idea, cells) : el('div', { class: 'line static' }, ...cells)
    }),
  )
}

function renderRecent() {
  const done = state.ideas.filter((i) => i.result || i.state === 'declined').slice(0, 8)
  if (!done.length) {
    setChildren('recent', el('div', { class: 'none' }, 'No results yet'))
    return
  }
  setChildren(
    'recent',
    done.map((idea) =>
      ideaLine(idea, [
        idCell(idea),
        el('span', { class: 'nm' }, el('b', {}, idea.title)),
        el('span', { class: 'd mono' }, idea.result ? variantDelta(idea) : ''),
        idea.result ? verdictChip(idea.result) : el('span', { class: 'st lost' }, 'declined'),
      ]),
    ),
  )
}

function renderIdeas() {
  if (!state.ideas.length) {
    setChildren('ideas', el('div', { class: 'none' }, 'No ideas yet'))
    return
  }
  setChildren(
    'ideas',
    state.ideas.map((idea) =>
      ideaLine(idea, [
        idCell(idea),
        el('span', { class: 'nm' }, el('b', {}, idea.title), el('small', {}, idea.author ? `#9514 · ${idea.author}` : 'maintainer')),
        el('span', { class: `st ${STATE_CLASS[idea.state] ?? 'tested'}` }, STATE_WORD[idea.state] ?? idea.state),
        when(idea.updated_at),
      ]),
    ),
  )
}

function renderFeed() {
  const items = state.activity.slice(0, 15)
  if (!items.length) {
    setChildren('feed', el('li', { class: 'none' }, 'Nothing yet'))
    return
  }
  setChildren(
    'feed',
    items.map((a) =>
      el(
        'li',
        { dataset: { kind: a.kind } },
        el('span', { class: 'ic', 'aria-hidden': 'true' }, FEED_ICON[a.kind] ?? '•'),
        el('span', { class: 'ft' }, a.registry_id ? [el('code', {}, a.registry_id), ' '] : null, a.text),
        when(a.at),
      ),
    ),
  )
}

function render() {
  renderNext(state.next)
  renderPipeline(state.pipeline)
  renderRunning()
  renderRecent()
  renderIdeas()
  renderFeed()
  setChildren('updated', 'updated ', when(state.now))
}

async function refresh() {
  try {
    state = await loadJson('state')
    render()
  } catch (error) {
    renderNext({ tone: 'blocked', text: `Status unavailable (${error.message})` })
  }
}

showView('status')
refresh()
if (!STATIC) setInterval(refresh, REFRESH_MS)
