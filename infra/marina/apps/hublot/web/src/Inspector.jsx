import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, LineChart, ScatterChart } from 'echarts/charts';
import { GridComponent, MarkAreaComponent, MarkLineComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { cards, nodes } from './contracts.js';
import {
  canceledAttemptCount, cardSeries, clock, elapsed, hasValues, inRange, number,
  entryOccupancyAtLongestWait, sampleAt, timingDistribution, utc,
} from './display.js';
import { ChartLegend } from './Legends.jsx';
import { phaseTypes } from './phases.js';

echarts.use([BarChart, LineChart, ScatterChart, GridComponent, MarkAreaComponent, MarkLineComponent, CanvasRenderer]);

const grid = { left: 49, right: 16, top: 13, bottom: 29 };
const colors = ['#055fa6', '#11826d', '#7651a7', '#a65443', '#416078', '#aa5a02'];
const axisNumber = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });

function option(series, range, origin, phases, events, unit) {
  const overlays = {
    markArea: {
      silent: true, animation: false, label: { show: false },
      data: phases.filter(({ kind, start, end }) =>
        phaseTypes[kind] && start < range.to && end > range.from)
        .map(({ kind, start, end }) => [
          { xAxis: Math.max(range.from, start) - origin, itemStyle: { color: phaseTypes[kind].color } },
          { xAxis: Math.min(range.to, end) - origin },
        ]),
    },
    markLine: {
      silent: true, animation: false, symbol: 'none', label: { show: false },
      lineStyle: { color: '#607d96', opacity: .60, type: 'dashed', width: 1 },
      data: events.filter((event) => inRange(event, range)).map((event) => ({
          xAxis: event.t - origin, lineStyle: { color: event.kind === 'step_complete' ? '#4e7494' : '#8560ab' },
      })),
    },
  };
  const plot = series.map((signal, i) => ({
    id: signal.id,
    name: signal.label,
    type: signal.kind,
    symbolSize: signal.kind === 'scatter' ? 7 : 4,
    showSymbol: false,
    connectNulls: false,
    itemStyle: { color: colors[i % colors.length] },
    lineStyle: { width: 2, color: colors[i % colors.length] },
    barMaxWidth: 12,
    data: signal.data.map((point) => [point.t - origin, point.value]),
    ...(i === 0 ? overlays : {}),
  }));
  return {
    useUTC: true, animation: false, grid,
    xAxis: {
      type: 'time', min: range.from - origin, max: range.to - origin, splitNumber: 4,
      axisLabel: { color: '#536a7e', fontSize: 12, formatter: (value) => elapsed(value, 0), hideOverlap: true },
      axisLine: { lineStyle: { color: '#a7b6c3' } }, axisTick: { show: false },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value', min: 0,
      max: unit === 's per 10 s' ? 10 : undefined,
      interval: unit === 's per 10 s' ? 5 : undefined,
      splitNumber: 3, minInterval: 0.01,
      axisLabel: { color: '#536a7e', fontSize: 11,
        formatter: (v) => v >= 1000 ? axisNumber.format(v) : number.format(v) },
      splitLine: { lineStyle: { color: '#e6edf3' } },
    },
    series: plot,
  };
}

function TimePlot({ id, series, range, origin, cursor, setHover, onPoint, phases, events, unit, cardTitle }) {
  const element = useRef(null);
  const instance = useRef(null);
  useEffect(() => {
    const chart = echarts.init(element.current, null, { renderer: 'canvas' });
    instance.current = chart;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0 && entry.contentRect.height > 0) chart.resize();
    });
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      chart.dispose();
      instance.current = null;
    };
  }, []);

  useEffect(() => {
    instance.current.setOption(option(series, range, origin, phases, events, unit), { notMerge: true });
  }, [series, range, origin, phases, events, unit]);

  function pointerTime(event) {
    const box = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - box.left - grid.left;
    const y = event.clientY - box.top;
    const width = box.width - grid.left - grid.right;
    if (x < 0 || x > width || y < grid.top || y > box.height - grid.bottom) return null;
    return Math.min(range.to - 1, Math.round(range.from + x / width * (range.to - range.from)));
  }

  function lockPoint(time) {
    if (time === null) return;
    const hits = series.map((signal) => pointAt(signal, time, range)).filter((p) => p?.value != null);
    const nearest = hits.sort((a, b) => Math.abs(a.t - time) - Math.abs(b.t - time))[0];
    setHover(null);
    onPoint({ id, time: nearest?.t ?? time });
  }

  function moveKeyboard(event) {
    if (event.key === 'Escape') {
      setHover(null);
      onPoint(null);
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      lockPoint(cursor ?? range.from);
      return;
    }
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    const step = Math.max(1, Math.round((range.to - range.from) / 100));
    const t = (cursor ?? range.from) + (event.key === 'ArrowLeft' ? -step : step);
    setHover({ id, time: Math.max(range.from, Math.min(range.to - 1, t)) });
  }

  const fraction = cursor === null ? null : (cursor - range.from) / (range.to - range.from);
  return (
    <div className="time-plot" role="group" tabIndex={0} aria-label={cardTitle + ' chart in ' + unit}
      aria-description="Click or press Enter for point details."
      data-range-from={range.from} data-range-to={range.to} data-cursor={cursor ?? ''}
      onPointerMove={(event) => {
        const time = pointerTime(event);
        if (time !== null) setHover({ id, time });
      }}
      onClick={(event) => lockPoint(pointerTime(event))} onKeyDown={moveKeyboard}>
      <div ref={element} className="plot-canvas" />
      {fraction !== null && fraction >= 0 && fraction <= 1 &&
        <div className="plot-guide" aria-hidden="true" style={grid}>
          <span style={{ left: (fraction * 100) + '%' }} />
        </div>}
    </div>
  );
}

const readable = (value) => value?.replaceAll('_', ' ');
const quantity = (value, unit) => number.format(value) + (unit === 's per 10 s' ? ' of 10 s' : ' ' + unit);
const pointAt = (signal, time, range) =>
  sampleAt(signal.data.filter((point) => inRange(point, range)), time, (range.to - range.from) / 140);

function TimingDistribution({ signal, range }) {
  const distribution = timingDistribution(signal.data, range);
  if (!distribution) return <p className="missing-line">No timing observations in this view.</p>;
  const { count, p50, p95, buckets } = distribution;
  const max = Math.max(...buckets.map((bucket) => bucket.count));
  return <section className="timing-distribution" aria-label={signal.label + ' distribution'}>
    <div className="metric-summary">
      <strong>{number.format(count)} {count === 1 ? 'observation' : 'observations'}</strong>
      <span>Median {quantity(p50, 's')} · P95 {quantity(p95, 's')}</span>
    </div>
    <ul aria-label="Duration buckets">
      {buckets.map(({ label, count: samples }) => <li key={label}>
        <span>{label}</span><i aria-hidden="true" style={{ '--bar-width': samples / max * 100 + '%' }} />
        <strong>{number.format(samples)}</strong>
      </li>)}
    </ul>
  </section>;
}

function hoverEvent(time, range, phases, events) {
  const tolerance = (range.to - range.from) / 140;
  const nearest = events.filter(({ t }) => Math.abs(time - t) <= tolerance)
    .sort((a, b) => Math.abs(a.t - time) - Math.abs(b.t - time))[0];
  if (nearest) return nearest;
  return phases.find(({ start, end }) => start <= time && time < end);
}

function eventLabel(event) {
  const step = event.step == null ? '' : 'Step ' + event.step;
  if (event.kind === 'step_complete') return step ? step + ' complete' : 'Step complete';
  if (event.kind === 'training') return step ? step + ' training' : 'Training';
  const duration = Number.isFinite(event.duration) ? quantity(event.duration, 's') : null;
  if (event.kind === 'rollout_wait') return [step, 'batch wait', duration].filter(Boolean).join(' · ');
  if (event.kind === 'weight_sync' || event.kind === 'sync_complete') return ['Weight sync', duration, readable(event.reason)].filter(Boolean).join(' · ');
  return [event.reason ?? phaseTypes[event.kind]?.label, duration].filter(Boolean).join(' · ');
}

function sampleTime(point, cursor) {
  if (point?.recorded != null) return ['Sample', point.recorded];
  if (point?.start != null) return ['Bin start', point.start];
  return ['Cursor', cursor];
}

function sampleTitle(label, time, point) {
  const title = label + ' · ' + utc(time);
  if (point?.first == null || point.last == null || point.first === point.last) return title;
  return title + '\nRecords ' + utc(point.first) + ' to ' + utc(point.last);
}

function ChartReadout({ time, showContext, origin, range, series, phases, events, caption }) {
  const samples = time === null ? [] : series.map((signal) => pointAt(signal, time, range));
  const sample = samples.find((point) => point?.value != null);
  const event = time !== null && showContext ? hoverEvent(time, range, phases, events) : null;
  const [timeLabel, at] = sampleTime(sample, time);
  return <div className="chart-readout" data-testid={time === null ? undefined : 'chart-readout'}>
    <div className="series-key">
      {series.map((signal, i) => <span key={signal.id}>
        <i style={{ background: colors[i % colors.length] }} />
        {signal.label}{signal.group ? ' · ' + readable(signal.group) : ''}
        {time !== null && <b>{samples[i]?.value != null ? quantity(samples[i].value, signal.unit) : 'No sample'}</b>}
      </span>)}
    </div>
    <div className="readout-meta">
      {time === null || !showContext ? <span>{caption}</span> : <>
        <time dateTime={new Date(time).toISOString()} title={sampleTitle(timeLabel, at, sample)}>
          {elapsed(time, origin)} elapsed · {clock(time, true)} UTC
        </time>
        {event && <span className={'readout-event event-' + event.kind}
          title={('start' in event ? utc(event.start) + ' to ' + utc(event.end) + '\n' : '') +
            'Recorded ' + utc(event.recorded) + (event.outcome ? ' · ' + event.outcome : '')}>
          {eventLabel(event)}
        </span>}
      </>}
    </div>
  </div>;
}

function ChartDialog({ title, endpoints, phases, children, onClose }) {
  const element = useRef(null);
  const titleId = useId();
  useLayoutEffect(() => { element.current.showModal(); }, []);
  return <dialog ref={element} className="chart-dialog" aria-labelledby={titleId} onClose={onClose}>
    <header className="chart-dialog-header">
      <div><h2 id={titleId}>{title}</h2>{endpoints}</div>
      <ChartLegend phases={phases} />
      <button type="button" className="dialog-close" onClick={() => element.current.close()}>Exit full screen</button>
    </header>
    <div className="chart-dialog-main">{children}</div>
  </dialog>;
}

const tabNames = {
  'entries/s': 'Entry rate', 'attempts/s': 'Attempt rate', 'groups/s': 'Group rate',
  'groups/batch': 'Groups per batch', 'rollouts/eval': 'Eval grades', requests: 'Requests',
  groups: 'Queue depth', 'policy steps': 'Policy lag', attempts: 'Active attempts', entries: 'Quarantine',
  'awaits/attempt': 'Model awaits', 'rollouts/group': 'Rollouts per group', 'tokens/step': 'Training tokens',
  'sequences/step': 'Training sequences', s: 'Timing', 's per 10 s': 'Waiting per 10 s',
};

function PointDetails({ point, series, model, range, origin, group, onClear }) {
  const records = series.map((signal) => [signal, pointAt(signal, point.time, range)])
    .filter(([, record]) => record?.value != null);
  const representative = records[0]?.[1];
  const [timeLabel, time] = sampleTime(representative, point.time);
  const companionVolume = records.flatMap(([signal, record]) => {
    const id = { batchGroups: 'batchTokens', batchTokens: 'batchGroups' }[signal.id];
    if (!id) return [];
    const [other] = model.series[id];
    const value = other.data.find((item) => item.t === record.t && item.step === record.step);
    return value ? [[other, value]] : [];
  });
  const entryFlow = series.some((signal) => signal.id === 'entryReleases');
  const bins = records.filter(([, record]) => record.count != null);
  const phase = model.phases.find(({ start, end }) => start <= point.time && point.time < end);
  const extraQueue = records.some(([signal]) => signal.unit === 'requests') ?
    ['waitingCapacity', 'waitingDeferred'].flatMap((name) => {
      const signal = model.series[name]?.find((item) => item.group === (group || null));
      const value = signal && pointAt(signal, point.time, range)?.value;
      return value == null ? [] : [[signal.label, String(value)]];
    }) : [];
  const evaluation = records.some(([signal]) => signal.id === 'evalGraded') &&
    model.evaluations?.find((row) => row.step === representative.step);
  const waitPhases = representative?.waitStart != null
    ? [...new Set(model.phases.filter(({ start, end }) => start < representative.t && end > representative.waitStart)
      .map((item) => item.reason ?? phaseTypes[item.kind]?.label).filter(Boolean))]
    : [];
  return <section className="point-details" aria-label="Point details">
    <header><h4>Point details</h4><button type="button" onClick={onClear}>Clear point</button></header>
    <dl>
      {[...records, ...companionVolume].map(([signal, record]) => <div key={signal.id}>
        <dt>{signal.label}</dt><dd>{quantity(record.value, signal.unit)}</dd>
      </div>)}
      <div><dt>{timeLabel}</dt><dd>{elapsed(time, origin)} elapsed · {utc(time)}</dd></div>
      {representative?.step != null && <div><dt>{representative.waitStart != null ? 'Policy step' : 'Step'}</dt><dd>{representative.step}</dd></div>}
      {representative?.waitStart != null && <div><dt>Estimated wait start</dt><dd>{elapsed(representative.waitStart, origin)} elapsed · {utc(representative.waitStart)}</dd></div>}
      {representative?.outstanding != null && <div><dt>Outstanding groups at start</dt>
        <dd title="Recorded rollout starts minus groups recorded leaving the Buffer">
          {representative.outstanding} estimated · {model.run.trainingRows} training rows
        </dd>
      </div>}
      {waitPhases.length > 0 && <div><dt>Phases during wait</dt><dd>{waitPhases.join(' · ')}</dd></div>}
      {bins.map(([signal, record]) => <div key={signal.id + '/count'}>
        <dt>{bins.length === 1 ? 'Count in this bin' : signal.label + ' in this bin'}</dt>
        <dd>{number.format(record.count)}{!entryFlow && ' across ' + record.samples + ' records'}
          {entryFlow && <small className="record-window">
            {clock(record.first, true)}{record.first !== record.last && ' to ' + clock(record.last, true)} UTC
          </small>}
        </dd>
      </div>)}
      {representative?.engines != null && <div><dt>Engines sampled</dt><dd>{representative.engines}</dd></div>}
      {extraQueue.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
      {evaluation?.attempted != null && <div><dt>Attempted across suites</dt><dd>{evaluation.attempted}</dd></div>}
      {evaluation?.submissionFailures != null && <div><dt>Submission failures</dt><dd>{evaluation.submissionFailures}</dd></div>}
      {representative?.reason && <div><dt>Recorded reason</dt><dd>{readable(representative.reason)}</dd></div>}
      {representative?.disposition && <div><dt>Buffer disposition</dt><dd>{readable(representative.disposition)}</dd></div>}
      {representative?.outcome && <div><dt>Recorded outcome</dt><dd>{representative.outcome}</dd></div>}
      {phase && <div><dt>Recorded phase</dt><dd>{eventLabel(phase)}{phase.source ? ' · ' + phase.source : ''}</dd></div>}
    </dl>
    {representative?.waitStart != null && <p>Finelog records this duration when the await ends. Its outcome is unavailable.</p>}
    {!records.length && <p>No chart sample is recorded at this time.</p>}
  </section>;
}

function cardSummary(definition, model, range, unit, cancellations) {
  if (definition.entryOccupancySummary) {
    const wait = entryOccupancyAtLongestWait(model.series.taskWait[0].data, range);
    if (!wait) return null;
    return {
      headline: number.format(wait.outstanding) + ' of ' + number.format(model.run.trainingRows) + ' training entries in use',
      context: 'Estimated at the start of the longest task selection wait in view (' + quantity(wait.value, 's') + ')',
    };
  }
  if (definition.cancellationSummary && cancellations !== null) {
    return {
      headline: number.format(cancellations) + (cancellations === 1 ? ' canceled attempt' : ' canceled attempts'),
      context: 'Recorded worker outcomes in this time range',
    };
  }
  if (definition.batchSummary && (unit === 'groups/batch' || unit === 'tokens/batch')) {
    const batches = model.series.batchGroups[0].data.filter((item) => inRange(item, range));
    if (!batches.length) return null;
    const tokens = model.series.batchTokens[0].data;
    const counts = batches.map((batch) => tokens.find((item) => item.t === batch.t && item.step === batch.step)?.value);
    const totalGroups = batches.reduce((sum, point) => sum + point.value, 0);
    const totalTokens = counts.every(Number.isFinite) ? counts.reduce((sum, count) => sum + count, 0) : null;
    return {
      headline: number.format(totalGroups) + ' groups handed off',
      context: batches.length + (batches.length === 1 ? ' batch' : ' batches') +
        (totalTokens == null ? '' : ' · ' + number.format(totalTokens) + ' generated tokens'),
    };
  }
  if (definition.evalSummary && unit === 'rollouts/eval') {
    const evals = (model.evaluations ?? []).filter((item) => inRange(item, range));
    if (!evals.length) return null;
    const add = (field) => evals.every((item) => item[field] != null)
      ? evals.reduce((total, item) => total + item[field], 0) : null;
    const attempted = add('attempted');
    const failed = add('submissionFailures');
    return {
      headline: add('graded') + (attempted == null ? '' : ' of ' + attempted) + ' graded',
      context: evals.length + (evals.length === 1 ? ' evaluation' : ' evaluations') +
        (failed == null ? '' : ' · ' + failed + ' submission failures'),
    };
  }
  if (definition.rejectionSummary && unit === 'groups/batch') {
    const rejected = model.series.rejected[0].data.filter((item) => inRange(item, range));
    if (!rejected.length) return null;
    const stale = model.series.staleRejected[0].data.filter((item) => inRange(item, range));
    const count = rejected.reduce((sum, item) => sum + item.value, 0);
    return {
      headline: count + ' groups rejected',
      context: rejected.length + (rejected.length === 1 ? ' batch summary' : ' batch summaries') +
        (stale.length === rejected.length ? ' · ' + stale.reduce((sum, item) => sum + item.value, 0) + ' stale' : ''),
    };
  }
  return null;
}

export function InspectorCard({
  id, edge, model, availableCards, range, origin, hover, setHover, cursor, point, onPoint, onInspect,
  onPin, onClose, onMove, onFocusSpan,
  pinned, pinIndex, pinCount, active, register,
}) {
  const definition = cards[id];
  const [unitChoice, setUnitChoice] = useState('');
  const [groupChoice, setGroupChoice] = useState('');
  const [distributionOpen, setDistributionOpen] = useState(false);
  const [distributionChoice, setDistributionChoice] = useState('');
  const [fullscreen, setFullscreen] = useState(false);
  const all = useMemo(() => cardSeries(id, model.series), [id, model.series]);
  const wholeWindow = model.window;
  const populated = useMemo(() => all.filter((signal) => hasValues(signal, wholeWindow)), [all, wholeWindow]);
  const groups = [...new Set(populated.map((signal) => signal.group).filter(Boolean))]
    .sort((a, b) => a.localeCompare(b, 'en', { numeric: true }));
  const hasAggregate = populated.some((signal) => !signal.group);
  const group = groups.includes(groupChoice) ? groupChoice : hasAggregate ? '' : groups[0];
  const byGroup = populated.filter((signal) => signal.group === (group || null));
  const units = [...new Set(byGroup.map((signal) => signal.unit))];
  const unit = units.includes(unitChoice) ? unitChoice : units[0];
  const perGpu = unit === 'tokens/s/GPU';
  const series = useMemo(() => populated.filter((signal) => signal.group === (group || null) && signal.unit === unit),
    [populated, group, unit]);
  const unavailable = all.filter((signal) => signal.group === (group || null) && !populated.includes(signal));
  const gapSignal = definition.gapSignal ?? definition.series[0];
  const showGap = definition.gap && !model.series[gapSignal]?.some((s) => hasValues(s, range));
  const cancellations = definition.cancellationSummary || definition.rejectionSummary
    ? canceledAttemptCount(model.series.attemptTime[0].data, range) : null;
  const summary = cardSummary(definition, model, range, unit, cancellations);
  const waits = definition.batchWaitZoom ? model.phases.filter(({ kind, end, start }) =>
    kind === 'rollout_wait' && start < range.to && end >= range.from) : [];
  const links = (definition.links ?? (showGap && definition.related ? [definition.related] : []))
    .filter((link) => availableCards.has(link.id));
  if (definition.rejectionSummary && cancellations !== null && availableCards.has('workers')) {
    links.push({ id: 'workers', label: 'View canceled attempts in Workers' });
  }
  const distributions = series.filter((signal) => signal.distribution);
  const distribution = distributionOpen ?
    distributions.find((signal) => signal.id === distributionChoice) ?? distributions[0] : null;
  const grouped = series.find((s) => definition.summaries?.[s.id]);
  const summaryCount = grouped?.data.filter((p) => inRange(p, range)).length;
  let caption = '';
  if (perGpu) caption = model.run.inferenceGpus + ' configured inference GPUs';
  if (unit === 's per 10 s') caption = '10 s means it waited through the whole bin.';
  if (unit === 's') caption = 'Duration in seconds';
  if (definition.lagCaption && unit === 'policy steps' && model.maxStaleness != null) caption = 'Configured staleness limit · ' + model.maxStaleness + ' steps';
  if (unit === 'attempts') caption = 'Intervals reconstructed from recorded worker calls.';
  if (id === 'workers' && unit === 's') caption = 'Wait totals within an attempt can overlap.';
  if (series.some((signal) => signal.id === 'entryReleases')) caption = 'Releases are counted when groups leave for training.';
  if (grouped) caption = summaryCount + ' ' + definition.summaries[grouped.id] +
    (summaryCount === 1 ? ' summary' : ' summaries') + ' in view';
  const endpoints = edge && <div className="card-endpoints" data-testid="card-endpoints">
    {nodes[edge.from[0]].label} <span aria-hidden="true">→</span><span className="sr-only"> to </span> {nodes[edge.to[0]].label}
  </div>;
  const inspectChart = <>
    {showGap && <p className="card-gap">{definition.gap}</p>}
    {summary && <div className="metric-summary"><strong>{summary.headline}</strong><span>{summary.context}</span></div>}
    {(units.length > 0 || groups.length > 0) &&
      <div className="series-controls">
        <div className="unit-options" role="group" aria-label="Chart units">
          {units.map((item) => <button type="button" key={item}
            title={item} aria-pressed={unit === item} onClick={() => { setUnitChoice(item); setHover(null); }}>
            {definition.tabs?.[item] ?? tabNames[item] ?? item}
          </button>)}
        </div>
        {!perGpu && groups.length + Number(hasAggregate) > 1 && (!definition.evalSummary || unit === 's') &&
          <select aria-label={definition.engines ? 'Inference engine' : definition.groupLabel ?? 'Breakdown'} value={group}
            onChange={(event) => { setGroupChoice(event.target.value); setHover(null); }}>
            {hasAggregate && <option value="">{definition.engines ? 'All engines' : definition.groupDefault ?? 'All reasons'}</option>}
            {groups.map((group) => <option key={group} value={group}>{readable(group)}</option>)}
          </select>}
      </div>}
    {distributions.length > 0 && <div className="series-controls timing-view">
      <div className="unit-options" role="group" aria-label="Timing view">
        <button type="button" aria-pressed={!distribution} onClick={() => setDistributionOpen(false)}>Timeline</button>
        <button type="button" aria-pressed={!!distribution} onClick={() => {
          setHover(null);
          if (point) onPoint(null);
          setDistributionOpen(true);
        }}>Distribution</button>
      </div>
      {distribution && distributions.length > 1 && <select aria-label="Timing measurement"
        value={distribution.id} onChange={(event) => setDistributionChoice(event.target.value)}>
        {distributions.map((signal) => <option key={signal.id} value={signal.id}>{signal.label}</option>)}
      </select>}
    </div>}
    {distribution ? <TimingDistribution signal={distribution} range={range} /> : series.length > 0 ? <>
      <div className="chart-exploration" onPointerLeave={() => { if (hover?.id === id) setHover(null); }}>
        <TimePlot id={id} series={series} range={range} origin={origin} cursor={cursor?.time ?? null} setHover={setHover}
          onPoint={onPoint} phases={model.phases} events={model.events} unit={unit} cardTitle={definition.title} />
        <ChartReadout time={cursor?.time ?? null} showContext={cursor?.id === id} origin={origin} series={series} range={range}
          phases={model.phases} events={model.events} caption={caption} />
      </div>
      {point && <PointDetails point={point} series={series} model={model} range={range} origin={origin}
        group={group} onClear={() => { setHover(null); onPoint(null); }} />}
      {series.every((s) => !hasValues(s, range)) && <p className="missing-line">No samples in this view.</p>}
    </> : !showGap && <p className="missing-line">Finelog has no samples for this card in the loaded window.</p>}
    {!showGap && series.length > 0 && unavailable.length > 0 &&
      <p className="missing-line">{definition.missing ?? 'No records for ' + unavailable.map((s) => s.label).join(', ') + '.'}</p>}
    {links.length > 0 && <div className="related-cards">{links.map((link) =>
      <button key={link.id} type="button" className="related-card"
        onClick={() => { if (fullscreen) setFullscreen(false); onInspect(link.id); }}>{link.label}</button>
    )}</div>}
    {waits.length > 0 && <div className="wait-buttons" role="group" aria-label="Zoom to a recorded batch wait">
      {waits.map((span) => <button type="button" key={span.id} onClick={() => onFocusSpan(span)}
        data-step={span.step} data-duration={span.duration} data-start={span.start} data-end={span.end}
        title={'Zoom to this wait · ' + quantity(span.duration, 's') + ' · ' + utc(span.start) + ' to ' + utc(span.end)}>
        Step {span.step} wait · {elapsed(span.duration * 1000, 0)}
      </button>)}
    </div>}
  </>;
  return (
    <article ref={register} className={'inspect-card' + (active ? ' inspect-active' : '')}
      data-card={id} data-pinned={pinned} data-unit={unit ?? ''} aria-label={definition.title}>
      <header className="card-heading">
        <div><h3>{definition.title}</h3>{endpoints}</div>
        <div className="card-actions">
          <details className="chart-help">
            <summary aria-label={'How to read ' + definition.title}>?</summary>
            <div className="help-popover">
              <p>{definition.helpByUnit?.[unit] ?? definition.help}</p>
              {perGpu && <p>Per GPU divides the fleet rate by the configured inference GPU count.</p>}
            </div>
          </details>
          {series.length > 0 && <button type="button" aria-label={'Full screen ' + definition.title} title="Full screen"
            onClick={() => { setHover(null); setFullscreen(true); }}><i className="fullscreen-icon" aria-hidden="true" /></button>}
          {pinned ? <>
            <button type="button" aria-label={'Move ' + definition.title + ' up'} disabled={pinIndex === 0} onClick={() => onMove(id, -1)}>Up</button>
            <button type="button" aria-label={'Move ' + definition.title + ' down'} disabled={pinIndex === pinCount - 1} onClick={() => onMove(id, 1)}>Down</button>
          </> : <button type="button" aria-label={'Pin ' + definition.title} onClick={() => onPin(id)}>Pin</button>}
          <button type="button" className="close-card" aria-label={'Close ' + definition.title} onClick={() => onClose(id)}>×</button>
        </div>
      </header>
      {fullscreen ? <ChartDialog title={definition.title} endpoints={endpoints} phases={model.phases} onClose={() => {
        setHover(null);
        setFullscreen(false);
      }}>{inspectChart}</ChartDialog> : inspectChart}
    </article>
  );
}
