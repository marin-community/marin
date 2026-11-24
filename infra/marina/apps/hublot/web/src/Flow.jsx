import { useLayoutEffect, useMemo } from 'react';
import {
  Background, BaseEdge, EdgeLabelRenderer, Handle, MarkerType, Position,
  ReactFlow, ReactFlowProvider, useReactFlow, useStore,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { cards, flowGraph, nodes as graphNodes } from './contracts.js';
import { recordedNodeSettings, summary } from './display.js';

const positions = { top: Position.Top, right: Position.Right, bottom: Position.Bottom, left: Position.Left };
const marker = { type: MarkerType.ArrowClosed, color: '#617991', width: 15, height: 15 };
const coordMarker = { ...marker, color: '#827394' };

function StageNode({ id, data }) {
  const { node, wires, value, selected, inspectable, hasSettings, onInspect, onInfo, onToggle } = data;
  const unit = value?.unit === 'attempts' && value.count === '1' ? 'attempt' : value?.unit;
  const Label = inspectable ? 'button' : 'div';
  const action = inspectable ? { type: 'button', onClick: (event) => { event.stopPropagation(); onInspect(id); } } : {};
  return (
    <div className={'stage-node stage-' + id + (selected ? ' stage-selected' : '') + (hasSettings ? ' stage-has-settings' : '')}
      style={{ width: node.size[0], height: node.size[1] }}>
      {wires.map(({ name, type, p }) => (
        <Handle key={type + ':' + name} type={type} id={name}
          position={positions[p.side]}
          style={['top', 'bottom'].includes(p.side) ? { left: p.offset + '%' } : { top: p.offset + '%' }}
          isConnectable={false} title={p.label} aria-label={p.label} />
      ))}
      <Label className="stage-inspect nodrag" {...action}
        title={inspectable ? undefined : 'No timeline samples in the loaded window.'}>
        <strong>{node.label}</strong>
        {value ? <>
          <span className="stage-value">{value.count}<small>{unit}</small></span>
          <span className="stage-detail">{value.label}</span>
        </>
          : <span className="stage-detail stage-unavailable">{inspectable ? 'No sample' : 'Unavailable'}</span>}
      </Label>
      {hasSettings && <button type="button" className="stage-info nodrag nopan"
        title={node.label + ' settings'} aria-label={node.label + ' settings'} aria-haspopup="dialog"
        onClick={(event) => { event.stopPropagation(); onInfo(id); }}>i</button>}
      {(id === 'rollouts' || id === 'dispatch') &&
        <button className="stage-toggle nodrag" type="button" onClick={(event) => { event.stopPropagation(); onToggle(); }}
          aria-expanded={id === 'dispatch'}>{id === 'rollouts' ? 'Expand Rollouts' : 'Collapse Rollouts'}</button>}
    </div>
  );
}

function roundedPath(points) {
  let path = 'M ' + points[0].join(' ');
  for (let i = 1; i < points.length - 1; i++) {
    const [x, y] = points[i];
    const [prevX, prevY] = points[i - 1];
    const [nextX, nextY] = points[i + 1];
    const radius = Math.min(12, Math.hypot(prevX - x, prevY - y) / 2, Math.hypot(nextX - x, nextY - y) / 2);
    const before = [x + Math.sign(prevX - x) * radius, y + Math.sign(prevY - y) * radius];
    const after = [x + Math.sign(nextX - x) * radius, y + Math.sign(nextY - y) * radius];
    path += ' L ' + before.join(' ') + ' Q ' + [x, y, ...after].join(' ');
  }
  return path + ' L ' + points.at(-1).join(' ');
}

function streamGeometry(edge, sx, sy, tx, ty) {
  const start = [sx, sy];
  const end = [tx, ty];
  const midX = (sx + tx) / 2;
  const midY = (sy + ty) / 2;
  if (edge.lane !== undefined) {
    return {
      path: roundedPath([start, [sx, edge.lane], [tx, edge.lane], end]),
      label: [midX, edge.lane],
    };
  }
  if (edge.id === 'leases') {
    const toDispatch = edge.to[0] === 'dispatch';
    const railX = toDispatch ? tx : sx - 144;
    const bends = toDispatch ? [[railX, sy]] : [[railX, sy], [railX, ty]];
    return { path: roundedPath([start, ...bends, end]), label: [railX, midY] };
  }
  if (edge.id === 'prompts' && edge.to[0] === 'rollouts') {
    return { path: roundedPath([start, [sx, ty], end]), label: [midX, ty] };
  }
  return { path: roundedPath([start, end]), label: [midX, midY] };
}

function FlowEdge({ id, sourceX, sourceY, targetX, targetY, data, markerEnd }) {
  const { edge, value, selected, inspectable, onInspect } = data;
  const { path, label: [x, y] } = streamGeometry(edge, sourceX, sourceY, targetX, targetY);
  const { gapLabel, gap } = cards[id];
  const readout = value?.value ?? (inspectable ? 'No sample' : gapLabel ?? 'Unavailable');
  const Label = inspectable ? 'button' : 'div';
  const action = inspectable ? { type: 'button', 'aria-label': 'Inspect ' + edge.label + '. ' + readout,
    onClick: () => onInspect(id) } : {};
  const description = inspectable ? edge.payload + (value ? ' · ' + value.label : '')
    : gap ?? 'No samples in the loaded window.';
  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd}
        style={{ stroke: edge.kind === 'coordination' ? '#827394' : '#617991',
          strokeWidth: selected ? 2.65 : 1.65,
          strokeDasharray: edge.kind === 'coordination' ? '6 5' : undefined }} />
      <EdgeLabelRenderer>
        <Label className={'edge-readout nodrag nopan ' + (selected ? 'edge-selected' : '')} {...action}
          style={{ transform: 'translate(-50%, -50%) translate(' + x + 'px, ' + y + 'px)' }}
          title={description}
          data-edge={id} data-from={edge.from.join('.')} data-to={edge.to.join('.')}
          data-payload={edge.payload}>
          <span>{edge.label}</span>
          <span className={value ? 'edge-value' : 'edge-unavailable'}>{readout}</span>
        </Label>
      </EdgeLabelRenderer>
    </>
  );
}

const nodeTypes = { stage: StageNode };
const edgeTypes = { stream: FlowEdge };

function FlowView({ model, availableCards, expanded, setExpanded, selected, onInspect, onInfo, range, cursor, resizing }) {
  const { fitView } = useReactFlow();
  const width = useStore((state) => state.width);
  const height = useStore((state) => state.height);
  const definition = useMemo(() => flowGraph(expanded, model.run.separateGrading), [expanded, model.run.separateGrading]);

  useLayoutEffect(() => {
    if (resizing || !width || !height) return;
    const frame = requestAnimationFrame(() => fitView({ padding: 0.14, maxZoom: 1.35 }));
    return () => cancelAnimationFrame(frame);
  }, [definition, fitView, height, resizing, width]);

  const flowNodes = definition.nodes.map(([id, node]) => {
    const [x, y] = node.position;
    const inspectable = availableCards.has(id);
    const wires = definition.edges.flatMap((edge) => {
      if (edge.from[0] === id) return [{ name: edge.from[1], p: node.ports[edge.from[1]], type: 'source' }];
      if (edge.to[0] === id) return [{ name: edge.to[1], p: node.ports[edge.to[1]], type: 'target' }];
      return [];
    });
    return { id, type: 'stage', position: { x, y }, width: node.size[0], height: node.size[1],
      selectable: inspectable, focusable: false,
      data: { node, wires, selected: id === selected, inspectable, onInspect, onInfo, onToggle: () => setExpanded(!expanded),
        hasSettings: recordedNodeSettings(id, model).length > 0,
        value: summary(id, model.series, range, cursor) } };
  });
  const flowEdges = definition.edges.map((edge) => ({
    id: edge.id, type: 'stream', source: edge.from[0], sourceHandle: edge.from[1],
    target: edge.to[0], targetHandle: edge.to[1], markerEnd: edge.kind === 'coordination' ? coordMarker : marker,
    selectable: availableCards.has(edge.id), focusable: false,
    ariaLabel: graphNodes[edge.from[0]].label + ' to ' + graphNodes[edge.to[0]].label + ' · ' + edge.label,
    data: { edge, selected: edge.id === selected, onInspect, inspectable: availableCards.has(edge.id),
      value: summary(edge.id, model.series, range, cursor) },
  }));
  return (
    <div className="flow-canvas" data-testid="flow-graph">
      <ReactFlow nodes={flowNodes} edges={flowEdges} nodeTypes={nodeTypes} edgeTypes={edgeTypes}
        nodesDraggable={false} nodesConnectable={false} edgesReconnectable={false}
        minZoom={.24} maxZoom={2.5} fitView fitViewOptions={{ padding: .14 }}
        onNodeClick={(_, node) => onInspect(node.id)}
        onEdgeClick={(_, edge) => onInspect(edge.id)}
        deleteKeyCode={null} proOptions={{ hideAttribution: true }}>
        <Background gap={18} size={1} color="#d5dfe9" />
      </ReactFlow>
      <button className="fit-button" type="button" onClick={() => fitView({ padding: .14, maxZoom: 1.35, duration: 200 })}>Fit graph</button>
    </div>
  );
}

export function Flow(props) {
  return <ReactFlowProvider><FlowView {...props} /></ReactFlowProvider>;
}
