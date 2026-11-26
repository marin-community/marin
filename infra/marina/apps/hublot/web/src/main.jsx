import { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Flow } from './Flow.jsx';
import { InspectorCard, MetricsProgress } from './Inspector.jsx';
import { NodeInfo } from './NodeInfo.jsx';
import { ChartLegend, FlowLegend, TimeLegend } from './Legends.jsx';
import {
  BACKGROUND_READ_PAUSE_MS, cards, deferredCards, FIRST_SAMPLE_BATCH_SIZE, FLEET_CARD,
  flowGraph, GRAPH_SAMPLE_LIMIT, inferenceMetricCards, inferenceTabGroups, metricCards, METRIC_POINTS, MIN_WINDOW_MS,
  RAW_METRIC_WINDOW_MS, SAMPLE_BATCH_SIZE,
} from './contracts.js';
import { cardsWithSamples, elapsed, hasValues, shortName, utc } from './display.js';
import {
  BUCKET_MS, createDetailSeries, createSampledSeries, createSummarySeries, MAX_COUNTER_GAP_MS, sampleEnds,
} from './measurements.js';
import { createRunModel } from './run.js';
import './style.css';

const minRange = MIN_WINDOW_MS;
const appBase = import.meta.env.BASE_URL;
const metricSettleMs = 350;
const busyRetryLimitMs = 30_000;
const metricCacheSize = 3;
const sampledPrefix = 'sampled:';
const runReadSteps = 2;
const dashboardReadSteps = runReadSteps + metricCards.size;
const metricOrder = Object.values(inferenceTabGroups).flat();
const metricGroupByCard = Object.fromEntries(Object.entries(inferenceTabGroups)
  .flatMap(([group, ids]) => ids.map((id) => [id, group])));
const metricNames = {
  [inferenceMetricCards.s]: 'Prefill and decode',
  [inferenceMetricCards['%']]: 'KV cache',
  inferenceDefaults: 'Requests and token rates',
  inferenceOtherTabs: 'Timing and KV cache',
};

function runParams(run, window) {
  const params = new URLSearchParams({ run });
  if (window) {
    params.set('from', window.from);
    params.set('to', window.to);
  }
  return params;
}

class RunRequestError extends Error {
  constructor(message, status, choices, retryMs) {
    super(message);
    this.status = status;
    this.choices = choices;
    this.retryMs = retryMs;
  }
}

async function readApi(path, search, signal) {
  const response = await fetch(appBase + 'api/' + path + '?' + search, { signal });
  const data = await response.json();
  if (!response.ok) {
    const seconds = Number(response.headers.get('Retry-After'));
    const retryMs = [429, 503].includes(response.status) && Number.isFinite(seconds) ? seconds * 1000 : 0;
    throw new RunRequestError(data.error ?? 'Hublot could not load the data.', response.status, data.choices, retryMs);
  }
  return data;
}

function busyWait(error) {
  return error instanceof RunRequestError && error.retryMs > 0 ? error.retryMs + Math.random() * 250 : 0;
}

async function readForeground(path, search, { signal, active = () => true } = {}) {
  const retryUntil = performance.now() + busyRetryLimitMs;
  while (!signal?.aborted && active()) {
    try {
      return await readApi(path, search, signal);
    } catch (error) {
      const wait = busyWait(error);
      if (!wait || signal?.aborted || performance.now() + wait > retryUntil) throw error;
      await new Promise((resolve) => setTimeout(resolve, wait));
    }
  }
  return null;
}

async function readSamplePage(query, ends) {
  // Marina's proxy forwards only one value per query parameter.
  query.set('until', ends.join(','));
  const page = await readApi('samples', query);
  if (page.samples.length !== ends.length) throw new Error('Incomplete sample response.');
  return page;
}

function sameRange(left, right) {
  return left?.from === right?.from && left?.to === right?.to;
}

function summaryResolution(range) {
  return Math.max(2 * MAX_COUNTER_GAP_MS, Math.ceil((range.to - range.from) / (METRIC_POINTS * BUCKET_MS)) * BUCKET_MS);
}

function detailResolution(range) {
  return range.to - range.from <= RAW_METRIC_WINDOW_MS ? 0 : summaryResolution(range);
}

function covers(detail, range, resolution = Infinity) {
  return detail && range && detail.window.from <= range.from && detail.window.to >= range.to &&
    (detail.bucketMs ?? 0) <= resolution;
}

function sampledKey(card) {
  return sampledPrefix + card;
}

function sampledGroup(card) {
  return deferredCards.has(card) ? FLEET_CARD : card;
}

function sampledView(results, range) {
  return results?.find((result) => sameRange(result.window, range)) ??
    results?.find((result) => covers(result, range));
}

function samplesRead(view, range) {
  return sameRange(view?.window, range) ? view?.samples.length ?? 0 : 0;
}

function metricError(errors, key, range) {
  const error = errors[key];
  return sameRange(error?.range, range) ? error?.message : null;
}

function metricCard(id, inferenceUnit) {
  return id === 'inference' ? inferenceMetricCards[inferenceUnit] : id;
}

function OpeningRun() {
  return (
    <div className="empty-box">
      <h2>Opening run</h2>
    </div>
  );
}

function runLocation() {
  return {
    id: new URLSearchParams(window.location.search).get('run') ?? '',
    date: history.state?.hublotDate ?? '',
  };
}

function TimeBar({ loaded, range, setRange, cursor, waits, origin, hasRunStart, reset }) {
  const duration = loaded.to - loaded.from;
  const left = (range.from - loaded.from) / duration * 100;
  const right = (range.to - loaded.from) / duration * 100;
  const zoomed = range.from !== loaded.from || range.to !== loaded.to;
  const sliderEnd = Math.ceil(duration / 1000);
  const sliderValue = (time) => time === loaded.to ? sliderEnd : Math.round((time - loaded.from) / 1000);
  const timestamp = (event) => Math.min(loaded.to, loaded.from + Number(event.target.value) * 1000);
  const elapsedText = (time) => elapsed(time, origin) + ' elapsed. ' + utc(time);
  return (
    <section className="time-bar" aria-label="Shared time range">
      <div className="time-display">
        <strong>{hasRunStart ? 'Run time' : 'Window time'}</strong>
        {zoomed && <span data-testid="selected-range" title={utc(range.from) + ' to ' + utc(range.to)}>
          {elapsed(range.from, origin)}–{elapsed(range.to, origin)}
        </span>}
      </div>
      <div className="time-ruler">
        <div className="ruler-track" aria-hidden="true">
          <i className="ruler-range" style={{ left: left + '%', right: (100 - right) + '%' }} />
          {waits.map((span) => {
            const start = (Math.max(span.start, loaded.from) - loaded.from) / duration * 100;
            const end = (Math.min(span.end, loaded.to) - loaded.from) / duration * 100;
            return <i key={span.id} className="ruler-wait" style={{ left: start + '%', width: (end - start) + '%' }} />;
          })}
          {cursor !== null && <i className="ruler-cursor" style={{ left: ((cursor - loaded.from) / duration * 100) + '%' }} />}
        </div>
        <input type="range" aria-label="Range start" aria-valuetext={elapsedText(range.from)}
          min={0} max={sliderEnd} value={sliderValue(range.from)}
          onChange={(event) => setRange({ ...range, from: Math.min(timestamp(event), range.to - minRange) })} />
        <input type="range" aria-label="Range end" aria-valuetext={elapsedText(range.to)}
          min={0} max={sliderEnd} value={sliderValue(range.to)}
          onChange={(event) => setRange({ ...range, to: Math.max(timestamp(event), range.from + minRange) })} />
        <div className="ruler-endpoints" aria-hidden="true">
          <span>{elapsed(loaded.from, origin)}</span><span>{elapsed(loaded.to, origin)}</span>
        </div>
      </div>
      <div className="time-actions">
        <TimeLegend />
        <button type="button" onClick={reset} disabled={range.from === loaded.from && range.to === loaded.to}>Full window</button>
      </div>
    </section>
  );
}

function RunHeading({ model }) {
  const { start_ms, end_ms, inferenceGpus, policyGpus, batchPolicy, trainingRows } = model.run;
  return (
    <section className="run-heading" aria-label="Resolved execution"
      data-execution={model.run.execution_uid} data-cluster={model.run.cluster} data-job={model.run.job_id}>
      <div className="run-title">
        <time>{new Date(model.window.from).toISOString().slice(0, 10)}</time>
        <h2 title={model.run.run_id ?? model.run.job_id}>{shortName(model.run.run_id || model.run.job_id)}</h2>
      </div>
      <div className="run-origin">
        <span>{model.run.cluster}</span>
        <details>
          <summary>Run details</summary>
          <dl>
            <dt>Run ID</dt><dd>{model.run.run_id || 'Unavailable'}</dd>
            <dt>Execution UID</dt><dd>{model.run.execution_uid}</dd>
            <dt>Iris job</dt><dd>{model.run.job_id || 'Unavailable'}</dd>
            <dt>Source</dt><dd>Finelog · {model.deployment} · {model.namespace}</dd>
            {start_ms != null && <><dt>Run started</dt><dd>{utc(start_ms)}</dd></>}
            {end_ms != null && <><dt>Run ended</dt><dd>{utc(end_ms)} · {elapsed(end_ms, start_ms)} elapsed</dd></>}
            {(inferenceGpus || policyGpus) && <>
              <dt>Configured GPUs</dt>
              <dd>{[inferenceGpus && inferenceGpus + ' inference', policyGpus && policyGpus + ' trainer']
                .filter(Boolean).join(' · ')}</dd>
            </>}
            {(batchPolicy || trainingRows != null || model.batchTarget != null) && <>
              <dt>Batching</dt><dd>{[
                batchPolicy === 'full_batch' ? 'Full batch' : batchPolicy === 'rolling' ? 'Rolling' : null,
                trainingRows != null && trainingRows + ' training rows',
                model.batchTarget != null && model.batchTarget + ' groups per batch',
              ].filter(Boolean).join(' · ')}</dd>
            </>}
            {model.maxStaleness != null && <><dt>Staleness limit</dt><dd>{model.maxStaleness} policy steps</dd></>}
            <dt>Loaded window</dt><dd>{utc(model.window.from)} to {utc(model.window.to)}</dd>
            {model.run.driverReadFailed && <><dt>Driver logs</dt><dd>Finelog could not load driver timings for this window.</dd></>}
          </dl>
        </details>
      </div>
    </section>
  );
}

function App() {
  const [navigation, setNavigation] = useState(runLocation);
  const [runInput, setRunInput] = useState(navigation.id);
  const [dateHint, setDateHint] = useState(navigation.date);
  const [inputError, setInputError] = useState('');
  const [overview, setOverview] = useState(null);
  const [load, setLoad] = useState(() => ({ state: navigation.id ? 'finding' : 'idle' }));
  const [metricCache, setMetricCache] = useState({});
  const [metricErrors, setMetricErrors] = useState({});
  const [detailRequests, setDetailRequests] = useState([]);
  const [detailLoad, setDetailLoad] = useState(null);
  const detailRequest = useRef(null);
  const detailInterest = useRef(null);
  const backgroundPace = useRef({ nextStart: 0, retry: null });
  const [expanded, setExpanded] = useState(false);
  const [range, setRange] = useState(null);
  const [hover, setHover] = useState(null);
  const [point, setPoint] = useState(null);
  const activeCursor = hover ?? point;
  const cursor = activeCursor?.time ?? null;
  const [selected, setSelected] = useState(null);
  const [inferenceUnit, setInferenceUnit] = useState('requests');
  const [infoNode, setInfoNode] = useState(null);
  const [pins, setPins] = useState([]);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [sideWidth, setSideWidth] = useState(438);
  const [resizing, setResizing] = useState(false);
  const workspace = useRef(null);
  const cardRefs = useRef(new Map());
  const chartOrder = useMemo(() => selected && !pins.includes(selected) ? [selected, ...pins] : pins, [selected, pins]);
  const fineViews = useMemo(() => {
    if (!range) return {};
    const resolution = detailResolution(range);
    const complete = (id) => metricCache[id]?.find((entry) => entry.next == null && covers(entry, range, resolution));
    return Object.fromEntries([...metricCards].map((id) => {
      const shared = complete(metricGroupByCard[id]);
      return [id, complete(id) ?? (shared && {
        window: shared.window, bucketMs: shared.bucketMs, series: shared.seriesByCard[id],
      })];
    }));
  }, [metricCache, range]);
  const partialDetails = useMemo(() => {
    if (!range) return {};
    const readSoFar = (id) => metricCache[id]?.find((entry) => entry.next != null && sameRange(entry.window, range));
    return Object.fromEntries([...Object.keys(inferenceTabGroups), ...metricCards].map((id) => {
      const shared = readSoFar(metricGroupByCard[id]);
      const own = readSoFar(id);
      return [id, shared && (!own || shared.next > own.next) ? shared : own];
    }));
  }, [metricCache, range]);
  const sampledViews = useMemo(() => range ? Object.fromEntries(
    [FLEET_CARD, inferenceMetricCards.s, inferenceMetricCards['%']].map((id) =>
      [id, sampledView(metricCache[sampledKey(id)], range)]),
  ) : {}, [metricCache, range]);
  const metricViews = useMemo(() => {
    return Object.fromEntries([...metricCards].flatMap((id) => {
      const preview = sampledViews[sampledGroup(id)];
      const view = fineViews[id] ?? (preview && cards[id] ? {
        ...preview,
        series: Object.fromEntries(cards[id].series.filter((name) => name in preview.series)
          .map((name) => [name, preview.series[name]])),
      } : preview);
      return view ? [[id, view]] : [];
    }));
  }, [fineViews, sampledViews]);
  const model = useMemo(() => {
    if (!overview) return null;
    const series = Object.assign({ ...overview.series }, ...Object.values(metricViews).map((result) => result.series));
    return { ...overview, series };
  }, [overview, metricViews]);
  const partialViews = useMemo(() => {
    if (!overview || !range) return {};
    return Object.fromEntries([...metricCards].flatMap((id) => {
      const page = partialDetails[id];
      if (!page || fineViews[id]) return [];
      const loaded = { from: range.from, to: page.next };
      const series = createSummarySeries(page.buckets, loaded, page.bucketMs, id, overview.run.inferenceGpus);
      if (!Object.values(series).some((curves) => curves.some((curve) => hasValues(curve, loaded)))) return [];
      return [[id, { window: range, bucketMs: page.bucketMs, series, partial: true }]];
    }));
  }, [fineViews, overview, partialDetails, range]);
  const inspectorModel = useMemo(() => model && ({
    ...model, series: Object.assign({ ...model.series }, ...Object.values(partialViews).map((view) => view.series)),
  }), [model, partialViews]);
  const run = overview?.run;
  const samplePlan = useMemo(() => overview && range ? sampleEnds(range, overview.phases) : [], [overview, range]);
  const fleetKey = sampledKey(FLEET_CARD);
  const fleetReady = [...deferredCards].every((id) => fineViews[id]);
  const fleetError = !fleetReady && metricError(metricErrors, fleetKey, range);
  const detailBusy = detailLoad != null;
  const openMetrics = inspectorOpen ? chartOrder.map((id) => metricCard(id, inferenceUnit))
    .filter((id) => metricCards.has(id)) : [];
  const missingMetrics = metricOrder.filter((id) => !fineViews[id]);
  const focusedMetric = inspectorOpen ? metricCard(selected, inferenceUnit) : null;
  const nextAutomaticMetric = [focusedMetric, ...openMetrics, ...missingMetrics].find((id) =>
    metricCards.has(id) && !fineViews[id] && !metricError(metricErrors, id, range));
  const automaticGroup = metricGroupByCard[nextAutomaticMetric];
  const readTogether = automaticGroup && range && range.to - range.from > RAW_METRIC_WINDOW_MS &&
    !metricError(metricErrors, automaticGroup, range);
  const nextAutomaticDetail = readTogether ? automaticGroup : nextAutomaticMetric;
  const graphSampleLimit = fleetReady ? 0 : Math.min(GRAPH_SAMPLE_LIMIT, samplePlan.length);
  const fleetSamplesRead = samplesRead(sampledViews[FLEET_CARD], range);
  const chartUsesFleet = !readTogether && openMetrics.some((id) =>
    sampledGroup(id) === FLEET_CARD && !fineViews[id] && !partialViews[id]);
  const fleetSampleLimit = chartUsesFleet ? samplePlan.length : graphSampleLimit;
  const loadingFleet = fleetSamplesRead < fleetSampleLimit && !fleetError;
  const nextDetail = detailRequests.find(({ card, window }) =>
    !fineViews[card] && sameRange(window, range) &&
    !metricError(metricErrors, card, range))?.card;
  const nextSample = readTogether ? undefined : openMetrics.filter((id) => !fineViews[id] && !partialViews[id])
    .map(sampledGroup).find((id) => id !== FLEET_CARD && samplesRead(sampledViews[id], range) < samplePlan.length &&
      !metricError(metricErrors, sampledKey(id), range));
  const nextRead = nextDetail ?? (loadingFleet && fleetSamplesRead === 0 ? fleetKey : null)
    ?? (nextSample && sampledKey(nextSample)) ?? (loadingFleet ? fleetKey : null) ?? nextAutomaticDetail;
  const loadingCards = loadingFleet && fleetSamplesRead < graphSampleLimit ? deferredCards
    : new Set(detailLoad ? [detailLoad.card] : []);
  const availableCards = useMemo(() => {
    if (!model) return new Set();
    return new Set([...cardsWithSamples(model), ...deferredCards]);
  }, [model]);
  const edgeMap = useMemo(() => model
    ? new Map(flowGraph(expanded, model.run.separateGrading).edges.map((edge) => [edge.id, edge]))
    : new Map(), [expanded, model]);

  useEffect(() => {
    const pop = () => setNavigation(runLocation());
    window.addEventListener('popstate', pop);
    return () => window.removeEventListener('popstate', pop);
  }, []);

  useEffect(() => {
    const { id, date } = navigation;
    setRunInput(id);
    setDateHint(date);
    setInputError('');
    setOverview(null);
    setMetricCache({});
    setMetricErrors({});
    setDetailRequests([]);
    backgroundPace.current.retry = null;
    setDetailLoad(null);
    setHover(null);
    setPoint(null);
    setExpanded(false);
    setPins([]);
    setSelected(null);
    setInferenceUnit('requests');
    setInfoNode(null);
    setInspectorOpen(false);
    if (!id) {
      setLoad({ state: 'idle' });
      return;
    }
    const params = runParams(id);
    const link = '?' + params;
    if (window.location.search !== link) history.replaceState({ hublotDate: date }, '', link);
    if (date) params.set('date', date);
    setLoad({ state: 'finding' });
    const pending = new AbortController();
    (async () => {
      let locating = true;
      try {
        await detailRequest.current?.done;
        if (pending.signal.aborted) return;
        const result = await readForeground('run', params, { signal: pending.signal });
        if (pending.signal.aborted) return;
        locating = false;
        const execution = result.run;
        const linkId = id === execution.execution_uid ? id : execution.id;
        history.replaceState({ hublotDate: date }, '', '?' + runParams(linkId));
        setLoad({ state: 'overview' });
        const windowQuery = runParams(execution.id, result.window);
        if (execution.start_ms != null) windowQuery.set('run_start', execution.start_ms);
        if (execution.end_ms != null) windowQuery.set('run_end', execution.end_ms);
        const timeline = await readForeground('window', windowQuery, { signal: pending.signal });
        if (pending.signal.aborted) return;
        setOverview(createRunModel({ ...result, ...timeline, run: { ...execution, ...timeline.runSettings } }));
        setRange(result.window);
        setLoad({ state: 'ready' });
      } catch (error) {
        if (!pending.signal.aborted) {
          setLoad({
            state: 'error',
            message: error instanceof RunRequestError ? error.message : 'Hublot could not load the run from Marina.',
            choices: error.choices,
            canSearchDate: locating && [404, 504].includes(error.status),
          });
        }
      }
    })();
    return () => {
      pending.abort();
      // Let Finelog finish the current page before resolving a different run.
      if (detailRequest.current) detailRequest.current.cancelled = true;
    };
  }, [navigation]);

  useEffect(() => {
    detailInterest.current = { range, cards: inspectorOpen ? chartOrder.map((id) => metricCard(id, inferenceUnit)) : [] };
  }, [chartOrder, inferenceUnit, inspectorOpen, range]);

  useEffect(() => {
    if (!overview || !range || !nextRead || detailBusy) return;
    const sampled = nextRead.startsWith(sampledPrefix);
    const background = nextRead !== nextDetail;
    const card = sampled ? nextRead.slice(sampledPrefix.length) : nextRead;
    const view = sampledViews[card];
    const offset = sampled ? samplesRead(view, range) : 0;
    const partial = partialDetails[card];
    const delay = background
      ? Math.max(offset || partial ? 0 : metricSettleMs, backgroundPace.current.nextStart - performance.now())
      : 0;
    const timer = setTimeout(() => {
      if (detailRequest.current) return;
      const windowQuery = runParams(overview.run.id, range);
      windowQuery.set('card', card);
      const request = { cancelled: false, done: null };
      detailRequest.current = request;
      setDetailLoad({ key: nextRead, card });
      const active = () => {
        const interest = detailInterest.current;
        return !request.cancelled && sameRange(range, interest.range) &&
          (!sampled || card === FLEET_CARD || interest.cards.includes(card));
      };
      const remember = (entry) => setMetricCache((current) => ({
        ...current,
        [nextRead]: [entry, ...(current[nextRead] ?? []).filter((earlier) =>
          !sameRange(earlier.window, range))].slice(0, metricCacheSize),
      }));
      request.done = (async () => {
        try {
          const { inferenceGpus } = overview.run;
          if (sampled) {
            const batchSize = offset ? SAMPLE_BATCH_SIZE : FIRST_SAMPLE_BATCH_SIZE;
            const limit = card === FLEET_CARD ? fleetSampleLimit : samplePlan.length;
            const page = await readSamplePage(windowQuery, samplePlan.slice(offset, Math.min(offset + batchSize, limit)));
            backgroundPace.current.retry = null;
            if (!active()) return;
            const samples = offset ? [...view.samples, ...page.samples] : page.samples;
            const buckets = offset ? [...view.buckets, ...page.buckets] : page.buckets;
            const series = createSampledSeries(buckets, samples, card, inferenceGpus);
            remember({ window: range, sampled: true, samples, buckets, series });
            return;
          }
          if (partial) windowQuery.set('after', partial.next);
          if (background) windowQuery.set('background', 'true');
          const page = background ? await readApi('window', windowQuery)
            : await readForeground('window', windowQuery, { active });
          if (background) backgroundPace.current.retry = null;
          if (request.cancelled || !page) return;
          const { bucketMs, next } = page;
          const buckets = partial ? [...partial.buckets, ...page.buckets] : page.buckets;
          if (next != null) {
            remember({ window: range, bucketMs, next, buckets });
            return;
          }
          if (inferenceTabGroups[card]) {
            const seriesByCard = Object.fromEntries(inferenceTabGroups[card].map((id) =>
              [id, createSummarySeries(buckets, range, bucketMs, id, inferenceGpus)]));
            remember({ window: range, bucketMs, seriesByCard });
            return;
          }
          const series = page.rows ? createDetailSeries(page.rows, range, card, inferenceGpus)
            : createSummarySeries(buckets, range, bucketMs, card, inferenceGpus);
          if (active()) setPoint((current) => metricCard(current?.id, inferenceUnit) === card ? null : current);
          remember({ window: range, bucketMs, series });
        } catch (error) {
          const wait = background ? busyWait(error) : 0;
          if (wait && active()) {
            const now = performance.now();
            const pace = backgroundPace.current;
            const retry = pace.retry;
            const since = retry?.key === nextRead && sameRange(retry.window, range) ? retry.since : now;
            if (now + wait < since + busyRetryLimitMs) {
              pace.retry = { key: nextRead, window: range, since };
              pace.nextStart = Math.max(pace.nextStart, now + wait);
              return;
            }
          }
          if (!request.cancelled) setMetricErrors((current) => ({
            ...current,
            [nextRead]: { range, message: error instanceof RunRequestError ? error.message : 'Hublot could not load these metrics.' },
          }));
        } finally {
          if (background) {
            const pace = backgroundPace.current;
            pace.nextStart = Math.max(pace.nextStart, performance.now() + BACKGROUND_READ_PAUSE_MS);
          }
          detailRequest.current = null;
          if (!request.cancelled) setDetailLoad(null);
        }
      })();
    }, delay);
    return () => clearTimeout(timer);
  }, [detailBusy, fleetSampleLimit, inferenceUnit, nextDetail, nextRead, overview, partialDetails, range,
    sampledViews, samplePlan]);

  useEffect(() => {
    if (!inspectorOpen || !selected) return;
    const frame = requestAnimationFrame(() => cardRefs.current.get(selected)?.scrollIntoView({ block: 'nearest' }));
    return () => cancelAnimationFrame(frame);
  }, [inspectorOpen, selected]);

  function openRun(run, date = '') {
    const id = run.trim();
    if (!id) {
      setInputError('Enter a run ID.');
      return;
    }
    const link = '?' + runParams(id);
    const state = { hublotDate: date };
    if (window.location.search === link) history.replaceState(state, '', link);
    else history.pushState(state, '', link);
    setNavigation({ id, date });
  }

  function selectRange(next) {
    setHover(null);
    setPoint((current) => current && current.time >= next.from && current.time < next.to ? current : null);
    setRange((current) => sameRange(current, next) ? current : next);
  }

  function focusSpan(span) {
    const from = Math.max(model.window.from, Math.min(span.start, model.window.to - minRange));
    const to = Math.min(model.window.to, Math.max(span.end + 1, from + minRange));
    selectRange({ from, to });
  }

  function inspect(id) {
    if (!availableCards.has(id)) return;
    if (id === 'dispatch' || id === 'workers') setExpanded(true);
    setInspectorOpen(true);
    setSelected(id);
    if (point && point.id !== id && !pins.includes(point.id)) setPoint(null);
    if (inspectorOpen && selected === id) cardRefs.current.get(id)?.scrollIntoView({ block: 'nearest' });
  }

  function closeCard(id) {
    setPins((current) => current.filter((pin) => pin !== id));
    if (selected === id) setSelected(null);
    if (point?.id === id) setPoint(null);
  }

  function moveCard(id, direction) {
    setPins((current) => {
      const next = [...current];
      const index = next.indexOf(id);
      const destination = index + direction;
      if (destination < 0 || destination >= next.length) return current;
      [next[index], next[destination]] = [next[destination], next[index]];
      return next;
    });
  }

  function resizeSidebar(width) {
    const size = workspace.current.clientWidth;
    setSideWidth(Math.max(322, Math.min(width, size - 445)));
  }

  const waits = model?.phases.filter(({ kind, start, end }) =>
    kind === 'rollout_wait' && start < model.window.to && end >= model.window.from) ?? [];
  const origin = model?.run.start_ms ?? model?.window.from;
  function detailForCard(id) {
    if (!deferredCards.has(id)) return null;
    const card = metricCard(id, inferenceUnit);
    const view = partialViews[card] ?? metricViews[card];
    const fine = !!fineViews[card];
    const sampleCount = samplesRead(sampledViews[sampledGroup(card)], range);
    const covered = fine || sampleCount >= samplePlan.length;
    const wantsDetail = !fine && detailRequests.some((request) => request.card === card && sameRange(request.window, range));
    const key = sampledKey(sampledGroup(card));
    const detailError = fine ? null : metricError(metricErrors, card, range);
    const error = detailError ?? (fine || wantsDetail ? null : metricError(metricErrors, key, range));
    return {
      card, key, fine, covered, wantsDetail, error, detailError,
      unit: id === 'inference' ? inferenceUnit : null,
      window: view?.window,
      bucketMs: view?.bucketMs,
      sampled: !!view?.sampled,
      partial: !!view?.partial,
    };
  }

  function retryMetrics(key) {
    if (backgroundPace.current.retry?.key === key) backgroundPace.current.retry = null;
    setMetricErrors((current) => ({ ...current, [key]: null }));
  }

  function loadDetail(card) {
    retryMetrics(card);
    setDetailRequests((current) => [{ card, window: range }, ...current.filter((request) => request.card !== card)]);
  }

  function progressForMetrics() {
    if (!overview || !range) {
      if (load.state === 'finding') return { label: 'Finding execution', percent: 0 };
      if (load.state === 'overview') return { label: 'Loading timeline', percent: Math.floor(100 / dashboardReadSteps) };
      return null;
    }
    if (!missingMetrics.length) return { label: 'Chart data loaded', percent: 100 };
    // Each metric group contributes the share of the selected range that has been read.
    let completed = dashboardReadSteps - missingMetrics.length;
    for (const id of missingMetrics) {
      const page = partialDetails[id];
      if (page) completed += (page.next - range.from) / (range.to - range.from);
    }
    const percent = Math.min(99, Math.floor(100 * completed / dashboardReadSteps));
    if (nextDetail) {
      const queued = detailLoad && detailLoad.key !== nextDetail;
      const title = metricNames[nextDetail] ?? cards[nextDetail].title;
      return { label: (queued ? 'Detail queued · ' : 'Loading detail · ') + title, percent };
    }
    const loading = detailLoad?.key ?? nextRead;
    if (loading) {
      const title = metricNames[loading] ?? cards[loading]?.title;
      return { label: title ? 'Loading · ' + title : 'Loading chart data', percent };
    }
    return { label: 'Chart data incomplete', percent, onRetry: () => missingMetrics.forEach(retryMetrics) };
  }

  const metricsProgress = progressForMetrics();
  return (
    <div className="app">
      <header className="top-bar">
        <div className="brand">
          <img className="brand-mark" src={appBase + 'hublot.svg'} width="44" height="44" alt="" />
          <div><h1>Hublot</h1><p>RL performance</p></div>
        </div>
        <div className="top-controls">
          <form className="run-form" onSubmit={(event) => {
            event.preventDefault();
            openRun(runInput, runInput === navigation.id || runInput === run?.id ? navigation.date : '');
          }}>
            <label className="sr-only" htmlFor="run-id">Run ID</label>
            <input id="run-id" placeholder="Paste a Finelog run ID or Iris execution UID"
              spellCheck={false} autoComplete="off" value={runInput}
              onChange={(event) => setRunInput(event.target.value)} />
            <button type="submit" className="primary-button">Open run</button>
          </form>
        </div>
      </header>
      {inputError && <div className="form-error" role="alert">{inputError}</div>}
      {!model && <main className="empty">
        {load.state === 'idle' && <div className="empty-box">
          <h2>Open a run</h2>
          <p>Paste an exact run ID, Iris job path, or execution UID to inspect activity across the loop.</p>
        </div>}
        {(load.state === 'finding' || load.state === 'overview') && <OpeningRun />}
        {load.state === 'error' && <div className="empty-box">
          <div role="alert"><h2>Run unavailable</h2><p>{load.message}</p></div>
          {load.canSearchDate && <form className="lookup-date" onSubmit={(event) => {
            event.preventDefault();
            openRun(navigation.id, dateHint);
          }}>
            <label>Run start date (UTC)<input type="date" required value={dateHint}
              onChange={(event) => setDateHint(event.target.value)} /></label>
            <button type="submit" className="primary-button">Search by date</button>
          </form>}
          {load.choices?.length > 0 && <ul className="run-choices">
            {load.choices.map((choice) => <li key={choice.id}>
              <button type="button" onClick={() => openRun(choice.id, navigation.date)}>
                <strong>{choice.cluster} · {choice.execution_uid}</strong>
                <small>{choice.job_id} · {utc(choice.last_ms)}</small>
              </button>
            </li>)}
          </ul>}
        </div>}
      </main>}
      {model && range && <main className="dashboard">
        <RunHeading model={model} />
        <TimeBar loaded={model.window} range={range} setRange={selectRange} cursor={cursor}
          waits={waits} origin={origin} hasRunStart={model.run.start_ms != null}
          reset={() => selectRange(model.window)} />
        <div className={'workspace' + (resizing ? ' resizing' : '') + (inspectorOpen ? '' : ' inspector-folded')} ref={workspace}
          style={{ '--sidebar': sideWidth + 'px' }}>
          <section className="flow-pane" aria-label="Rollout flow">
            <div className="pane-heading">
              <h2>Flow</h2>
              <FlowLegend />
            </div>
            <Flow model={model} availableCards={availableCards} loadingCards={loadingCards} expanded={expanded} setExpanded={setExpanded} selected={selected}
              onInspect={inspect} onInfo={setInfoNode} range={range} cursor={cursor} resizing={resizing} />
            <div className="flow-foot">
              <span>{cursor === null ? 'Latest point in view' : 'At ' + elapsed(cursor, origin) + ' elapsed'}</span>
              {fleetError ? <div className="fleet-error">
                <span role="status" title={fleetError}>Inference summary unavailable</span>
                <button type="button" className="related-card"
                  onClick={() => retryMetrics(fleetKey)}>Retry summary</button>
              </div> : <span>Click a node or stream to inspect it</span>}
            </div>
          </section>
          {inspectorOpen && <div className="sidebar-handle" role="separator" aria-label="Resize inspector"
            aria-orientation="vertical" aria-valuemin={322} aria-valuenow={sideWidth}
            tabIndex={0} onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              setResizing(true);
            }} onPointerMove={(event) => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                resizeSidebar(workspace.current.getBoundingClientRect().right - event.clientX);
              }
            }} onLostPointerCapture={() => setResizing(false)}
            onKeyDown={(event) => {
              if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                event.preventDefault();
                resizeSidebar(sideWidth + (event.key === 'ArrowLeft' ? 24 : -24));
              }
            }} />}
          <aside className="inspector" aria-label="Inspector">
            <div className="pane-heading inspect-heading">
              <h2>Inspector</h2>
              {inspectorOpen && <ChartLegend phases={model.phases} />}
              <button type="button" className="fold-button"
                title={inspectorOpen ? 'Fold inspector' : 'Open inspector'}
                aria-label={inspectorOpen ? 'Fold inspector' : 'Open inspector'}
                aria-expanded={inspectorOpen} aria-controls="inspector-charts"
                onClick={() => { setHover(null); setInspectorOpen(!inspectorOpen); }}>
                <i className="panel-icon" aria-hidden="true" />
              </button>
            </div>
            <div id="inspector-charts" className="inspect-stack" data-testid="chart-stack" hidden={!inspectorOpen}>
              {!chartOrder.length && <p className="inspector-empty">Select a node or stream on the graph.</p>}
              {chartOrder.map((id) => {
                const detail = detailForCard(id);
                return <InspectorCard key={id} id={id} edge={edgeMap.get(id)}
                  model={inspectorModel} availableCards={availableCards} range={range} origin={origin} hover={hover} setHover={setHover}
                  detail={detail} metricsProgress={metricsProgress} onSelectUnit={setInferenceUnit}
                  onLoadDetail={() => loadDetail(detail.card)} onRetryMetrics={() => retryMetrics(detail.key)}
                  cursor={activeCursor} point={point?.id === id ? point : null} onPoint={setPoint} onInspect={inspect}
                  onPin={(pin) => setPins((current) => current.includes(pin) ? current : [...current, pin])}
                  onClose={closeCard} onMove={moveCard} onFocusSpan={focusSpan}
                  pinned={pins.includes(id)} pinIndex={pins.indexOf(id)} pinCount={pins.length}
                  active={selected === id} register={(node) => {
                    if (node) cardRefs.current.set(id, node);
                    else cardRefs.current.delete(id);
                  }} />;
              })}
            </div>
          </aside>
        </div>
        {infoNode && <NodeInfo id={infoNode} model={model} onClose={() => setInfoNode(null)} />}
      </main>}
      {metricsProgress && <MetricsProgress {...metricsProgress} />}
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
