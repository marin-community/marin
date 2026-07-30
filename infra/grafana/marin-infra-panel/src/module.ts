import { PanelPlugin } from '@grafana/data';
import { InfraPanel } from './components/InfraPanel';
import { InfraPanelOptions } from './types';

export const plugin = new PanelPlugin<InfraPanelOptions>(InfraPanel).setPanelOptions((builder) => {
  return builder.addRadio({
    path: 'view',
    name: 'View',
    defaultValue: 'status',
    settings: {
      options: [
        { value: 'status', label: 'Status page' },
        { value: 'nightlies', label: 'Nightly matrix' },
        { value: 'commits', label: 'Commit strip' },
        { value: 'wandb', label: 'W&B chart' },
      ],
    },
  });
});
