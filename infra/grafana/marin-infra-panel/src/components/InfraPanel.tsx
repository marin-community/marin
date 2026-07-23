import React from 'react';
import { PanelProps } from '@grafana/data';
import { CommitStrip } from './CommitStrip';
import { NightlyMatrix } from './NightlyMatrix';
import { WandbChart } from './WandbChart';
import { InfraPanelOptions } from '../types';

export function InfraPanel({ options, data, width, height }: PanelProps<InfraPanelOptions>) {
  if (options.view === 'commits') {
    return <CommitStrip frames={data.series} width={width} height={height} />;
  }
  if (options.view === 'wandb') {
    return <WandbChart frames={data.series} width={width} height={height} />;
  }
  return <NightlyMatrix frames={data.series} width={width} height={height} />;
}
