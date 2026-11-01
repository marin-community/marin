// Marin Belay status page: the idea checklist (one row per registry idea, its gates, recipe status and potential),
// then what is running and the activity feed, from the registry results plus the public state (read-only, the same
// data the static export publishes). Loaded after common.js.

let registry = null // the registry's results JSON (ideas with trials, events, potential)
const checklistView = { sort: 'potential', query: '' }

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

// Gate 3, hero-scale MFU: the hero-mfu track's relative step-time Δ, else the speedrun step-time cost as a hint.
function heroGate(idea) {
  const hero = counted(idea.trials.filter((t) => t.rung === 'hero-mfu')).at(-1)
  if (hero) {
    const d = hero.delta[0]
    return { kind: verdictOf(d, trackInfo('hero-mfu').noise ?? 0.005), text: `s/step ${SIGN(d * 100, 1)}%`, title: hero.variant }
  }
  const stc = idea.trials.find((t) => t.rung === 'speedrun' && t.protocol === 'equal_steps' && t.step_time_change)
  return { kind: 'none', text: stc ? `not run · speedrun ${stc.step_time_change} step time` : 'not run', title: '' }
}

// Gate 4, the hero-scale projection: the competitive ladder's Δ per rung (legacy wider rungs count too).
function scalingGate(idea) {
  const rungs = counted(idea.trials.filter((t) => t.rung?.startsWith('competitive-') || ['d768', 'd1024', 'd1280'].includes(t.rung)))
  if (!rungs.length) return { kind: 'none', text: 'not run', title: '' }
  const latest = new Map(rungs.map((t) => [t.rung.replace('competitive-', ''), t]))
  const parts = [...latest].map(([r, t]) => `${r} ${SIGN(t.delta[0])}`)
  const worst = [...latest.values()].map((t) => verdictOf(t.delta[0], 0.006))
  const kind = worst.includes('loss') ? 'loss' : worst.every((v) => v === 'win') ? 'win' : 'noise'
  return { kind, text: `${parts.join(' · ')} · projection pending`, title: parts.join('\n') }
}

function recipeCell(idea) {
  const event = recipeEvent(idea)
  const day = event ? new Date(`${event.date}T12:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : ''
  if (event?.type === 'adopted') return el('span', { class: 'gate-chip won', title: event.note ?? '' }, el('b', {}, '✓'), ` ${recipeTag}`)
  if (event?.type === 'removed') return el('span', { class: 'gate-chip noise', title: event.note ?? '' }, el('b', {}, '↺'), ` removed at ${day}`)
  return el('span', { class: 'gate-chip tested' }, el('b', {}, '–'))
}

function approvalDate(idea, pageIdea) {
  const dates = [pageIdea?.decided_at?.slice(0, 10), ...(idea.trials ?? []).map((t) => t.date), ...(idea.events ?? []).map((e) => e.date)]
  return dates.filter(Boolean).sort()[0] ?? ''
}

function checklistRows() {
  const live = new Set(state.runs.filter((r) => !TERMINAL.has(r.state)).map((r) => r.registry_id))
  return registry.ideas.map((idea) => {
    const pageIdea = state.ideas.find((i) => i.registry_id === idea.xid)
    const comment = (idea.trials ?? []).map((t) => t.comment_url).find((u) => (u ?? '').includes('issuecomment-'))
    return {
      idea,
      comment,
      date: approvalDate(idea, pageIdea),
      author: idea.intake?.author ?? null,
      accepted: inLiveRecipe(idea),
      score: idea.potential?.score ?? null,
      speedrun: speedrunGate(idea, live.has(idea.xid)),
      hero: heroGate(idea),
      scaling: scalingGate(idea),
    }
  })
}

const SORTS = {
  potential: (a, b) => a.accepted - b.accepted || (b.score ?? -1) - (a.score ?? -1) || b.date.localeCompare(a.date),
  date: (a, b) => b.date.localeCompare(a.date),
  accepted: (a, b) => b.accepted - a.accepted || b.date.localeCompare(a.date),
}

function renderChecklist() {
  if (!registry) return
  const q = checklistView.query.toLowerCase()
  const rows = checklistRows()
    .filter((r) => !q || `${r.idea.xid} ${r.idea.name} ${r.author ?? ''} ${r.idea.category}`.toLowerCase().includes(q))
    .sort(SORTS[checklistView.sort])
  document.getElementById('checklist-n').textContent = String(rows.length)
  const registryUrl = (x) => `https://github.com/marin-community/marin/blob/marin_belay_v1_registry/experiments/belay/registry/${x.xid}-${x.id}.yaml`
  const head = el(
    'div',
    { class: 'ck-row ck-head', role: 'row' },
    ['Idea', 'Author', 'Date', 'Speedrun', 'Hero MFU', 'Projection', 'Recipe', 'Potential'].map((h) => el('span', { role: 'columnheader' }, h)),
  )
  setChildren(
    'checklist',
    head,
    rows.map((r) =>
      el(
        'div',
        { class: `ck-row${r.accepted ? ' live' : ''}`, role: 'row', title: r.accepted ? `In the live recipe, ${recipeTag}` : '' },
        el(
          'span',
          { class: 'ck-idea', 'data-label': 'Idea' },
          el('span', { class: 'xid mono' }, r.comment ? link(r.comment, r.idea.xid) : r.idea.xid),
          ' ',
          link(registryUrl(r.idea), r.idea.name),
        ),
        el('span', { 'data-label': 'Author' }, r.author ? link(`https://github.com/${r.author}`, `@${r.author}`) : el('span', { class: 'muted' }, 'maintainer')),
        el('span', { class: 'mono', 'data-label': 'Date' }, r.date),
        el('span', { 'data-label': 'Speedrun' }, gateChip(r.speedrun)),
        el('span', { 'data-label': 'Hero MFU' }, gateChip(r.hero)),
        el('span', { 'data-label': 'Projection' }, gateChip(r.scaling)),
        el('span', { 'data-label': 'Recipe' }, recipeCell(r.idea)),
        el(
          'span',
          { 'data-label': 'Potential' },
          r.score === null
            ? el('span', { class: 'muted' }, '–')
            : el('span', { class: 'score', title: `${r.idea.potential.rationale} (${r.idea.potential.model}, ${r.idea.potential.date})` }, `${r.score}/10`),
        ),
      ),
    ),
  )
}

function wireChecklist() {
  document.getElementById('ck-sort').addEventListener('change', (e) => {
    checklistView.sort = e.target.value
    renderChecklist()
  })
  document.getElementById('ck-filter').addEventListener('input', (e) => {
    checklistView.query = e.target.value
    renderChecklist()
  })
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
  renderChecklist()
  renderFeed()
  setChildren('updated', 'updated ', when(state.now))
}

async function refresh() {
  try {
    ;[state, registry, trackTable] = await Promise.all([
      loadJson('state'),
      loadJson('results'),
      fetch(asset('tracks.json')).then((r) => r.json()),
    ])
    recipeTag = trackTable.recipe_tag
    trackTable = trackTable.tracks
    render()
  } catch (error) {
    renderNext({ tone: 'blocked', text: `Status unavailable (${error.message})` })
  }
}

showView('status')
wireChecklist()
refresh()
if (!STATIC) setInterval(refresh, REFRESH_MS)
