import { useId } from 'react';
import { phaseTypes } from './phases.js';

function LegendItem({ kind, label, detail, color }) {
  const id = useId();
  return <span className={'legend-item legend-' + kind} tabIndex={0} aria-describedby={id}>
    <i aria-hidden="true" style={color ? { background: color } : undefined} />{label}
    <span id={id} role="tooltip" className="legend-tip">{detail}</span>
  </span>;
}

export function TimeLegend() {
  return <div className="chart-legend" aria-label="Time range colors">
    <LegendItem kind="wait" label={phaseTypes.rollout_wait.label} detail={phaseTypes.rollout_wait.detail} />
  </div>;
}

export function ChartLegend({ phases }) {
  const known = new Set(phases.map(({ kind }) => kind));
  return <div className="chart-legend" aria-label="Chart markers">
    {Object.entries(phaseTypes).filter(([kind]) => known.has(kind)).map(([kind, meta]) =>
      <LegendItem key={kind} kind={kind} {...meta} />)}
    <LegendItem kind="unmarked" color="#fff" label="No phase recorded"
      detail="White areas fall outside the recorded phase timers. Rollouts and inference can still be active." />
    <LegendItem kind="step" label="Step complete" detail="The trainer recorded a completed optimizer step here." />
  </div>;
}

export function FlowLegend() {
  return <div className="chart-legend" aria-label="Flow arrows">
    <LegendItem kind="work" label="Work and results" detail="Solid arrows carry work or results, such as prompts, model outputs, or training batches." />
    <LegendItem kind="coordination" label="Coordination" detail="Dashed arrows carry lease grants or updated policy weights." />
  </div>;
}
