// Marin Belay status page: the idea checklist (one row per registry idea, its gates, recipe status and potential),
// then what is running and the activity feed, from the registry results plus the public state (read-only, the same
// data the static export publishes). Loaded after common.js.

let registry = null // the registry's results JSON (ideas with trials, events, potential)
let trackTable = [] // tracks.json's tracks, generated from config.TRACKS
let recipeTag = '' // the live recipe's checkpoint tag (config.recipe_tag), from tracks.json
let defaultAuthor = '' // config.default_author: credited for internal proposals, from tracks.json

const GATE = {
  win: { icon: '✓', cls: 'won' },
  noise: { icon: '~', cls: 'noise' },
  loss: { icon: '✗', cls: 'lost' },
  running: { icon: '◷', cls: 'run' },
  none: { icon: '–', cls: 'tested' },
  na: { icon: '·', cls: 'tested' },
}
const SIGN = (v, digits = 4) => `${v > 0 ? '+' : v < 0 ? '\u2212' : '±'}${Math.abs(v).toFixed(digits)}`
const OUTSIDE = /outside the verdict|not part of the verdict/

function trackInfo(id) {
  return trackTable.find((t) => t.track === id) ?? {}
}

function verdictOf(d, noise) {
  if (d === undefined || d === null) return 'none'
  return d <= -noise ? 'win' : d >= noise ? 'loss' : 'noise'
}

// The registry build shows macro loss as placeholder BPB; gates judge the recorded macro loss (``loss_delta``).
function counted(trials) {
  return trials
    .filter((t) => !OUTSIDE.test(t.note ?? '') && (t.delta ?? []).length)
    .map((t) => ({ ...t, delta: t.loss_delta ?? t.delta }))
}

// Gate 1, beat a vanilla transformer: the latest vanilla-track trial (vanilla-d512 to vanilla-d1280).
function denseGate(idea) {
  const t = counted(idea.trials.filter((x) => x.rung?.startsWith('vanilla-'))).at(-1)
  if (!t) return { kind: 'none', text: 'not run', title: '' }
  const d = t.delta[0]
  const kind = verdictOf(d, trackInfo(t.rung).noise)
  const where = t.protocol === 'fixed_time' ? 'equal time' : 'equal data'
  return { kind, text: `${kind} at ${where}, Δ ${SIGN(d)} · ${t.rung.replace('vanilla-', '')}`, title: `${t.variant} (${t.rung}, ${t.date})` }
}

// Gate 2, the 8-minute speedrun: the speedrun track's stages, or a legacy d512 trial from before Tracks existed.
function speedrunGate(idea, live) {
  const sr = trackInfo('speedrun')
  const ours = counted(idea.trials.filter((t) => t.rung === 'speedrun'))
  const legacy = counted(idea.trials.filter((t) => t.rung === 'd512'))
  const data = ours.filter((t) => t.protocol === 'equal_steps').at(-1)
  const time = ours.filter((t) => t.protocol === 'fixed_time').at(-1)
  if (live) return { kind: 'running', text: 'running', title: '' }
  if (idea.track && idea.track !== 'speedrun' && !ours.length) return { kind: 'na', text: `n/a (track ${idea.track})`, title: '' }
  const final = time ?? data ?? legacy.at(-1)
  if (!final) return { kind: 'none', text: 'not run', title: '' }
  const d = final.delta[0]
  const kind = verdictOf(d, sr.noise)
  const where = time ? 'equal time' : data ? 'equal data' : `legacy ${final.protocol === 'fixed_time' ? 'fixed time' : 'equal steps'}`
  const steps = final.steps ? ` · ${final.steps.toLocaleString()}${sr.baseline_steps && (time || data) ? ` vs ${sr.baseline_steps.toLocaleString()}` : ''} steps` : ''
  const stages = [data, time].filter(Boolean).map(
    (t) => `${t.protocol === 'fixed_time' ? 'Equal time' : 'Equal data'}: Δ ${SIGN(t.delta[0])}${t.steps ? ` at ${t.steps.toLocaleString()} steps` : ''}${t.step_time_change ? `, step time ${t.step_time_change}` : ''}`,
  )
  return { kind, text: `${kind} at ${where}, Δ ${SIGN(d)}${steps}`, title: stages.join('\n') || `${final.variant} (${final.rung}, ${final.date})` }
}

// The idea's standing in the live recipe: its latest adopted/removed event for the recipe's checkpoint tag.
function recipeEvent(idea) {
  return (idea.events ?? []).filter((e) => e.checkpoint === recipeTag && ['adopted', 'removed'].includes(e.type)).at(-1)
}

function inLiveRecipe(idea) {
  return recipeEvent(idea)?.type === 'adopted'
}

function gateChip(g) {
  const look = GATE[g.kind] ?? GATE.none
  return el('span', { class: `gate-chip ${look.cls}`, title: g.title || g.text }, el('b', {}, look.icon), ` ${g.text}`)
}
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
    idea.url ? ['Source', link(idea.url, idea.author ? `#9514 · ${idea.author}` : 'GitHub')] : ['Source', link(`https://github.com/${defaultAuthor}`, `proposal by @${defaultAuthor}`)],
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
    return { kind: verdictOf(d, trackInfo('hero-mfu').noise), text: `s/step ${SIGN(d * 100, 1)}%`, title: hero.variant }
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
  const worst = [...latest.values()].map((t) => verdictOf(t.delta[0], trackInfo('competitive-d512').noise))
  const kind = worst.includes('loss') ? 'loss' : worst.every((v) => v === 'win') ? 'win' : 'noise'
  return { kind, text: `${parts.join(' · ')} · projection pending`, title: parts.join('\n') }
}

function recipeCell(idea) {
  const event = recipeEvent(idea)
  const day = event ? new Date(`${event.date}T12:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : ''
  if (event?.type === 'adopted') return el('span', { class: 'gate-chip won', title: event.note ?? '' }, el('b', {}, '✓'), ` ${recipeTag}`)
  if (event?.type === 'removed') {
    return el(
      'span',
      { class: 'recipe-cell' },
      el('span', { class: 'gate-chip noise', title: event.note ?? '' }, el('b', {}, '↺'), ` removed at ${day}`),
      el('span', { class: 'gate-chip run', title: 'In Belay Ckpt 1, not in the live recipe: retested on the current recipe' }, retestStage(idea, event)),
    )
  }
  return el('span', { class: 'gate-chip tested' }, el('b', {}, '–'))
}

// A Ckpt 1 idea the live recipe dropped is retested on the current recipe: "retest" until a speedrun trial lands.
function retestStage(idea, event) {
  return idea.trials.some((t) => t.rung === 'speedrun' && t.date >= event.date) ? 'retested' : 'retest'
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
      author: idea.author,
      accepted: inLiveRecipe(idea),
      score: idea.potential?.score ?? null,
      dense: denseGate(idea),
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
    .filter((r) => !q || `${r.idea.xid} ${r.idea.name} ${r.author ?? ''} ${r.idea.category} ${recipeEvent(r.idea)?.type === 'removed' ? 'retest' : ''}`.toLowerCase().includes(q))
    .sort(SORTS[checklistView.sort])
  document.getElementById('checklist-n').textContent = String(rows.length)
  const registryUrl = (x) => `https://github.com/marin-community/marin/blob/marin_belay_v1_registry/experiments/belay/registry/${x.xid}-${x.id}.yaml`
  const head = el(
    'div',
    { class: 'ck-row ck-head', role: 'row' },
    ['Idea', 'Author', 'Date', 'Dense', 'Speedrun', 'Hero MFU', 'Projection', 'Recipe', 'Potential'].map((h) => el('span', { role: 'columnheader' }, h)),
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
        el('span', { 'data-label': 'Author' }, link(`https://github.com/${r.author}`, `@${r.author}`)),
        el('span', { class: 'mono', 'data-label': 'Date' }, r.date),
        el('span', { 'data-label': 'Dense' }, gateChip(r.dense)),
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
    defaultAuthor = trackTable.default_author
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
